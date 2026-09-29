"""Film details, posters and critical reception from Wikidata + Wikipedia. No API key.

For each MovieLens film (matched through its IMDb ID):
  Wikidata   -> English Wikipedia article, runtime, director, Rotten Tomatoes /
                Metacritic / IMDb scores
  Wikipedia  -> lead image (the poster), lead paragraph (the overview) and the
                "Critical response" / "Reception" section (the review text)

  python wikipedia_source.py --popular 500
  python wikipedia_source.py --movie-ids 1 318 79132

Text is from Wikipedia under CC BY-SA 4.0; each film links to its article.
Poster images on English Wikipedia are usually non-free and hosted there under
fair use for identification; check that your use fits before publishing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import requests

from film_store import FilmStore

ROOT = Path(__file__).resolve().parent
SPARQL_URL = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1"
USER_AGENT = "moctale-movielens/1.0 (local film recommender for the MovieLens dataset; python-requests)"
CRITICS = {
    "http://www.wikidata.org/entity/Q105584": "Rotten Tomatoes",
    "http://www.wikidata.org/entity/Q150248": "Metacritic",
    "http://www.wikidata.org/entity/Q37312": "IMDb",
}
RECEPTION_HEADINGS = (
    re.compile(r"^critical (response|reception|reaction)", re.I),
    re.compile(r"^(reception|critical reception and legacy|response)$", re.I),
    re.compile(r"reception", re.I),
)


def _session() -> requests.Session:
    """Wikimedia asks API clients to identify themselves with a way to reach them.
    Put WIKIMEDIA_CONTACT=you@example.com (or a URL) in .env to add yours."""
    from tmdb import load_dotenv

    load_dotenv()
    contact = os.environ.get("WIKIMEDIA_CONTACT", "").strip()
    session = requests.Session()
    session.headers["User-Agent"] = f"{USER_AGENT[:-1]}; {contact})" if contact else USER_AGENT
    return session


def _request(session: requests.Session, url: str, params: dict | None = None, timeout: float = 20) -> requests.Response:
    """GET with backoff: Wikimedia answers bursts with 429 and a Retry-After header."""
    for attempt in range(5):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code in (429, 503):
            wait = float(response.headers.get("Retry-After") or 5 * 2 ** attempt)
            time.sleep(min(wait, 120))
            continue
        return response
    raise RuntimeError(f"Wikimedia kept rate-limiting {url}; try again later")


def _get(session: requests.Session, url: str, params: dict, timeout: float = 20) -> dict:
    response = _request(session, url, params, timeout)
    response.raise_for_status()
    return response.json()


def wikidata_facts(session: requests.Session, imdb_ids: list[str]) -> dict[str, dict]:
    """{'tt0111161': {'article': title, 'runtime': 142, 'director': ..., 'critic_scores': {...}}}"""
    values = " ".join(f'"{imdb}"' for imdb in imdb_ids)
    query = f"""
    SELECT ?imdb ?article ?dur ?dirLabel ?score ?by WHERE {{
      VALUES ?imdb {{ {values} }}
      ?item wdt:P345 ?imdb .
      ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
      OPTIONAL {{ ?item wdt:P2047 ?dur . }}
      OPTIONAL {{ ?item wdt:P57 ?dir . ?dir rdfs:label ?dirLabel . FILTER(LANG(?dirLabel) = "en") }}
      OPTIONAL {{ ?item p:P444 ?s . ?s ps:P444 ?score ; pq:P447 ?by .
                 FILTER(?by IN (wd:Q105584, wd:Q150248, wd:Q37312)) }}
    }}"""
    data = _get(session, SPARQL_URL, {"query": query, "format": "json"}, timeout=60)  # one query per 50 films
    facts: dict[str, dict] = {}
    for row in data["results"]["bindings"]:
        imdb = row["imdb"]["value"]
        entry = facts.setdefault(imdb, {"article": "", "runtime": None, "directors": [], "critic_scores": {}})
        entry["article"] = unquote(row["article"]["value"].rsplit("/wiki/", 1)[-1]).replace("_", " ")
        if "dur" in row and entry["runtime"] is None:
            try:
                entry["runtime"] = int(float(row["dur"]["value"]))
            except ValueError:
                pass
        if "dirLabel" in row and row["dirLabel"]["value"] not in entry["directors"]:
            entry["directors"].append(row["dirLabel"]["value"])
        if "score" in row:
            critic, score = CRITICS.get(row["by"]["value"]), row["score"]["value"].strip()
            if critic == "Rotten Tomatoes" and not score.endswith("%"):
                continue  # Rotten Tomatoes also lists an average rating; keep the Tomatometer
            if critic == "IMDb" and not score.endswith("/10"):
                continue
            if critic:
                entry["critic_scores"][critic] = score
    for entry in facts.values():
        entry["director"] = ", ".join(entry.pop("directors")[:2]) or None
    return facts


def page_images(session: requests.Session, titles: list[str]) -> dict[str, str]:
    """{requested title: original lead-image URL} for up to 50 articles in ONE API request.

    Much lighter on Wikimedia's rate limits than a summary request per film."""
    data = _get(session, WIKI_API, {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "pageimages", "piprop": "original", "pilicense": "any", "pilimit": "max",
        "titles": "|".join(titles),
    })
    query = data.get("query", {})
    renamed = {item["to"]: item["from"] for key in ("normalized", "redirects") for item in query.get(key, [])}
    images = {}
    for page in query.get("pages", []):
        source = (page.get("original") or {}).get("source")
        if not source:
            continue
        title = page["title"]
        while title in renamed:  # map back to the title we asked for
            title = renamed[title]
        images[title] = source.split("?", 1)[0]
    return images


def page_summary(session: requests.Session, title: str) -> dict | None:
    """Lead image (the poster), opening paragraph and URL from the edge-cached REST summary."""
    response = _request(session, f"{WIKI_REST}/page/summary/{quote(title.replace(' ', '_'), safe='')}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    clean = lambda url: url.split("?", 1)[0] if url else ""
    image = clean((data.get("originalimage") or {}).get("source"))
    return {
        "poster": clean((data.get("thumbnail") or {}).get("source")) or image,
        "poster_large": image,
        "overview": (data.get("extract") or "").strip() or None,
        "url": ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
               or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
    }


def article_text(session: requests.Session, title: str, max_paragraphs: int = 6) -> tuple[str, str, str]:
    """(first paragraph, critical reception text, reception section anchor); blanks when absent."""
    data = _get(session, WIKI_API, {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "titles": title,
    })
    pages = data.get("query", {}).get("pages", [])
    text = (pages[0].get("extract") if pages else "") or ""
    parts = re.split(r"^(=+) (.+?) =+\s*$", text, flags=re.M)
    intro = next((p.strip() for p in parts[0].split("\n") if len(p.strip()) > 40), "")
    reception, anchor = _reception(parts, max_paragraphs)
    return intro, reception, anchor


def _reception(parts: list[str], max_paragraphs: int) -> tuple[str, str]:
    sections = []  # (level, heading, body)
    for i in range(1, len(parts) - 2, 3):
        sections.append((len(parts[i]), parts[i + 1].strip(), parts[i + 2]))
    for pattern in RECEPTION_HEADINGS:
        for index, (level, heading, body) in enumerate(sections):
            if not pattern.search(heading) or "box office" in heading.lower():
                continue
            chunks = [body]
            for next_level, next_heading, next_body in sections[index + 1:]:
                if next_level <= level:
                    break
                if re.search(r"box office|accolade|award|list", next_heading, re.I):
                    continue
                chunks.append(next_body)
            paragraphs = [p.strip() for p in "\n".join(chunks).split("\n") if len(p.strip()) > 60]
            if paragraphs:
                return "\n\n".join(paragraphs[:max_paragraphs]), heading.replace(" ", "_")
    return "", ""


def sync(films: list[tuple[int, str]], store: FilmStore, sleep: float = 0.1, log=print) -> int:
    """films: [(movieId, 'tt0111161'), ...]. Returns how many were saved with details.

    One Wikidata query per 50 films, then two Wikipedia requests per film, one at a
    time as Wikimedia asks (parallel requests were throttled to ~5s each).
    """
    session = _session()
    saved = 0
    for start in range(0, len(films), 50):
        batch = films[start:start + 50]
        facts = wikidata_facts(session, [imdb for _movie_id, imdb in batch])
        for movie_id, imdb in batch:
            entry = facts.get(imdb)
            try:
                page = page_summary(session, entry["article"]) if entry else None
                if page is None:
                    store.save_film(movie_id, None)
                    continue
                _intro, reception, anchor = article_text(session, entry["article"])
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                log(f"  movieId {movie_id}: skipped ({exc})")
                continue
            time.sleep(sleep)
            reviews = []
            if reception:
                reviews.append({
                    "id": f"wikipedia:{movie_id}:reception",
                    "author": "Critical reception",
                    "content": reception,
                    "url": f"{page['url']}#{anchor}" if page.get("url") else None,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            store.save_film(movie_id, {
                "source": "wikipedia",
                "source_id": entry["article"],
                "overview": page["overview"],
                "runtime": entry["runtime"],
                "director": entry["director"],
                "critic_scores": entry["critic_scores"],
                "poster_url": page["poster"],
                "poster_url_large": page["poster_large"],
                "page_url": page["url"],
            }, reviews)
            saved += 1
        log(f"  {min(start + 50, len(films))}/{len(films)} checked, {saved} saved")
    return saved


def main() -> None:
    import pandas as pd

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--popular", type=int, help="Sync the N most-rated MovieLens films.")
    target.add_argument("--movie-ids", nargs="+", type=int, help="Sync these MovieLens movie IDs.")
    target.add_argument("--all", action="store_true", help="Sync every film with an IMDb ID.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch films that are already cached.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between films (be polite to Wikipedia).")
    args = parser.parse_args()

    links = pd.read_csv(ROOT / "data" / "links.csv", dtype={"imdbId": str}).dropna(subset=["imdbId"])
    if args.movie_ids:
        links = links[links["movieId"].isin(args.movie_ids)]
    elif args.popular:
        counts = pd.read_csv(ROOT / "data" / "ratings.csv", usecols=["movieId"])["movieId"].value_counts()
        links = links.assign(n=links["movieId"].map(counts).fillna(0)).sort_values("n", ascending=False).head(args.popular)

    store = FilmStore()
    done = set() if args.refresh else store.synced_ids("wikipedia")
    todo = [(int(r.movieId), f"tt{int(r.imdbId):07d}") for r in links.itertuples() if int(r.movieId) not in done]
    print(f"Fetching {len(todo)} films from Wikidata + Wikipedia ({len(links) - len(todo)} already cached).")
    sync(todo, store, args.sleep)
    counts = store.counts()
    print(f"Done. {counts['films']} films, {counts['with_poster']} with posters, {counts['reviews']} reception texts cached.")


if __name__ == "__main__":
    main()
