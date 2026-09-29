"""Discover openly licensed movie posters for MovieLens titles.

Sources (all open, no API key):
  * Wikidata  - maps an IMDb id to the film's poster file (property P3383).
  * Wikimedia Commons - supplies the license, author and a resized thumbnail.

Only images whose Commons license is reuse-friendly (public domain, CC0, CC BY,
CC BY-SA) are kept.  Results are merged into ``data/open_posters.json`` so the
app can show artwork *and* the required attribution.  Then run
``python fetch_posters.py`` to download the images into ``licensed_posters/``.

Most modern posters are copyrighted and are therefore not on Commons, so
coverage is naturally partial (classics, silent films, freely licensed and
government films).  Titles without a poster get a generated on-brand cover.

Usage:
    python open_posters.py                 # every movie in data/links.csv
    python open_posters.py --popular 1500  # only the 1,500 most-rated movies
    python open_posters.py --limit 200     # quick test run
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "open_posters.json"
SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MoctaleMovieLens/1.1 (open poster discovery; educational recommender project)"
THUMB_WIDTH = 600
SPARQL_BATCH = 150
COMMONS_BATCH = 40
OK_PREFIXES = ("public domain", "cc0", "cc by", "cc-by", "pd", "creative commons public domain")
BLOCKED = ("nc", "nd", "non-commercial", "noncommercial", "no derivatives", "fair use", "non-free")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("open-posters")


def is_reuse_friendly(license_name: str, copyrighted: str = "") -> bool:
    """True for public-domain and attribution-style licenses only."""
    value = license_name.strip().casefold()
    if not value:
        return False
    tokens = set(re.split(r"[^a-z0-9]+", value))
    if tokens & {"nc", "nd"} or any(word in value for word in BLOCKED[2:]):
        return False
    if copyrighted.strip().casefold() == "false":
        return True
    return value.startswith(OK_PREFIXES)


def clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def commons_title(file_url: str) -> str:
    """'http://commons.wikimedia.org/wiki/Special:FilePath/A%20b.jpg' -> 'File:A b.jpg'."""
    name = unquote(urlparse(file_url).path.rsplit("/", 1)[-1])
    return "File:" + name.replace("_", " ")


def wikidata_posters(session: requests.Session, imdb_ids: list[str]) -> dict[str, str]:
    """Return {imdb_id: commons file url} for the ids that have a poster on Wikidata."""
    values = " ".join(f'"{i}"' for i in imdb_ids)
    query = f"SELECT ?imdb ?poster WHERE {{ VALUES ?imdb {{ {values} }} ?film wdt:P345 ?imdb ; wdt:P3383 ?poster . }}"
    response = session.get(SPARQL_URL, params={"query": query, "format": "json"}, timeout=60)
    response.raise_for_status()
    found: dict[str, str] = {}
    for row in response.json()["results"]["bindings"]:
        found.setdefault(row["imdb"]["value"], row["poster"]["value"])
    return found


def commons_info(session: requests.Session, titles: list[str]) -> dict[str, dict]:
    """Return {'File:Name.jpg': metadata} using the Commons imageinfo API."""
    response = session.get(
        COMMONS_API,
        params={
            "action": "query", "format": "json", "formatversion": 2,
            "titles": "|".join(titles), "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime", "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Copyrighted|ImageDescription",
        },
        timeout=60,
    )
    response.raise_for_status()
    out: dict[str, dict] = {}
    for page in response.json().get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [None])[0]
        if info:
            out[page["title"].replace("_", " ")] = info
    return out


def build_entry(title: str, info: dict) -> dict | None:
    meta = info.get("extmetadata", {})
    license_name = clean_html(meta.get("LicenseShortName", {}).get("value", ""))
    copyrighted = meta.get("Copyrighted", {}).get("value", "")
    if not is_reuse_friendly(license_name, copyrighted):
        return None
    if not str(info.get("mime", "")).startswith("image/"):
        return None
    image_url = info.get("thumburl") or info.get("url")
    artist = clean_html(meta.get("Artist", {}).get("value", "")) or "Unknown"
    if not image_url:
        return None
    page_title = title.replace(" ", "_")
    return {
        "image_url": image_url,
        "source_url": f"https://commons.wikimedia.org/wiki/{page_title}",
        "artist": artist[:160],
        "license": license_name or "Public domain",
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "description": clean_html(meta.get("ImageDescription", {}).get("value", ""))[:240],
    }


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--popular", type=int, help="Only consider the N most-rated movies.")
    parser.add_argument("--limit", type=int, help="Only look at the first N candidate movies (testing).")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls (be polite).")
    args = parser.parse_args()

    links = pd.read_csv(ROOT / "data" / "links.csv", dtype={"imdbId": str})
    links = links.dropna(subset=["imdbId"])
    if args.popular:
        counts = pd.read_csv(ROOT / "data" / "ratings.csv", usecols=["movieId"])["movieId"].value_counts()
        links = links[links["movieId"].isin(counts.head(args.popular).index)]
    if args.limit:
        links = links.head(args.limit)
    imdb_to_movie = {f"tt{str(row.imdbId).zfill(7)}": int(row.movieId) for row in links.itertuples()}

    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    pending = [i for i, m in imdb_to_movie.items() if str(m) not in manifest]
    log.info("%d candidate movies, %d already in the manifest", len(imdb_to_movie), len(imdb_to_movie) - len(pending))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    added = 0
    for batch in chunks(pending, SPARQL_BATCH):
        try:
            posters = wikidata_posters(session, batch)
        except (requests.RequestException, KeyError, ValueError) as exc:
            log.warning("Wikidata batch failed (%s); continuing", exc)
            continue
        time.sleep(args.sleep)
        by_title = {commons_title(url): imdb for imdb, url in posters.items()}
        for title_batch in chunks(list(by_title), COMMONS_BATCH):
            try:
                infos = commons_info(session, title_batch)
            except (requests.RequestException, ValueError) as exc:
                log.warning("Commons batch failed (%s); continuing", exc)
                continue
            for title, info in infos.items():
                imdb = by_title.get(title)
                entry = build_entry(title, info) if imdb else None
                if entry:
                    manifest[str(imdb_to_movie[imdb])] = entry
                    added += 1
            time.sleep(args.sleep)
        log.info("Progress: %d open-licensed posters found so far", added)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Done. Added %d posters (manifest now has %d). Next: python fetch_posters.py", added, len(manifest))


if __name__ == "__main__":
    main()
