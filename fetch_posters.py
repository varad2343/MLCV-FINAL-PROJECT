"""Download the openly licensed posters listed in data/open_posters.json.

Run ``python open_posters.py`` first to discover posters on Wikidata/Commons."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

from open_posters import is_reuse_friendly


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "open_posters.json"
ALLOWED_HOSTS = {"upload.wikimedia.org", "commons.wikimedia.org"}
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("open-posters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-ids", nargs="+", type=int, help="Only download the listed MovieLens IDs.")
    parser.add_argument("--limit", type=int, help="Limit the number of manifest entries to download.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "licensed_posters")
    args = parser.parse_args()

    try:
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Could not read {MANIFEST}: {exc}")

    selected = list(entries.items())
    if args.movie_ids:
        requested = {str(movie_id) for movie_id in args.movie_ids}
        selected = [(movie_id, entry) for movie_id, entry in selected if movie_id in requested]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    if not selected:
        parser.error("No matching, curated posters were found in data/open_posters.json.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MoctaleMovieLens/1.0 (open poster attribution importer)"})
    for movie_id, entry in selected:
        license_name = str(entry.get("license", ""))
        image_url = str(entry.get("image_url", ""))
        host = urlparse(image_url).hostname
        suffix = Path(urlparse(image_url).path).suffix.lower()
        if host not in ALLOWED_HOSTS or suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            logger.warning("Skipping movie %s: unsupported image URL", movie_id)
            continue
        if not is_reuse_friendly(license_name) or not entry.get("source_url") or not entry.get("artist"):
            logger.warning("Skipping movie %s: source, creator, or a compatible license is missing", movie_id)
            continue

        destination = args.output_dir / f"{movie_id}{suffix}"
        if destination.is_file():
            logger.info("Already present: %s", destination)
            continue
        try:
            response = session.get(image_url, timeout=30)
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").startswith("image/") or len(response.content) > 15_000_000:
                logger.warning("Skipping movie %s: response was not a small image", movie_id)
                continue
            destination.write_bytes(response.content)
            logger.info("Saved %s · %s · %s", destination, entry["artist"], license_name)
        except requests.RequestException as exc:
            logger.warning("Could not fetch movie %s: %s", movie_id, exc)


if __name__ == "__main__":
    main()
