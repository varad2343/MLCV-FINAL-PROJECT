"""Optional: film details, images and user reviews from TMDB.

Wikipedia (wikipedia_source.py) works without a key. TMDB adds backdrops,
taglines and user reviews. Set one of these in the environment or in .env:
  TMDB_READ_TOKEN   the "API Read Access Token" (v4 bearer token), or
  TMDB_API_KEY      the v3 "API Key".

  python tmdb.py --popular 500
  python tmdb.py --movie-ids 1 318 79132 --review-pages 3

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

from film_store import FilmStore

ROOT = Path(__file__).resolve().parent
API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into os.environ without overriding real variables."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def image_url(path: str | None, size: str) -> str:
    return f"{IMAGE_BASE}/{size}{path}" if path else ""


class TmdbClient:
    def __init__(self, read_token: str = "", api_key: str = "", timeout: float = 15):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self.params: dict[str, str] = {}
        if read_token:
            self.session.headers["Authorization"] = f"Bearer {read_token}"
        elif api_key:
            self.params["api_key"] = api_key
        else:
            raise RuntimeError("Set TMDB_READ_TOKEN or TMDB_API_KEY in .env to use TMDB.")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "TmdbClient | None":
        load_dotenv()
        token, key = os.environ.get("TMDB_READ_TOKEN", ""), os.environ.get("TMDB_API_KEY", "")
        return cls(token, key) if token or key else None

    def get(self, path: str, **params) -> dict | None:
        for attempt in range(4):
            response = self.session.get(f"{API_BASE}{path}", params={**self.params, **params}, timeout=self.timeout)
            if response.status_code == 404:
                return None
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            if response.status_code == 401:
                raise RuntimeError("TMDB rejected the credentials. Check TMDB_READ_TOKEN / TMDB_API_KEY in .env.")
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"TMDB kept rate-limiting {path}")

    def movie(self, tmdb_id: int) -> dict | None:
        return self.get(f"/movie/{tmdb_id}", append_to_response="reviews,release_dates", language="en-US")

    def reviews(self, tmdb_id: int, page: int) -> dict | None:
        return self.get(f"/movie/{tmdb_id}/reviews", page=page, language="en-US")


def _us_certification(data: dict) -> str | None:
    for country in ((data.get("release_dates") or {}).get("results") or []):
        if country.get("iso_3166_1") == "US":
            for release in country.get("release_dates") or []:
                if release.get("certification"):
                    return release["certification"]
    return None


def to_film(tmdb_id: int, data: dict) -> dict:
    return {
        "source": "tmdb",
        "source_id": str(tmdb_id),
        "title": data.get("title"),
        "tagline": data.get("tagline") or None,
        "overview": data.get("overview") or None,
        "runtime": data.get("runtime") or None,
        "release_date": data.get("release_date") or None,
        "language": data.get("original_language"),
        "certification": _us_certification(data),
        "rating": data.get("vote_average"),
        "rating_count": data.get("vote_count"),
        "poster_url": image_url(data.get("poster_path"), "w342"),
        "poster_url_large": image_url(data.get("poster_path"), "w500"),
        "backdrop_url": image_url(data.get("backdrop_path"), "w1280"),
        "page_url": f"https://www.themoviedb.org/movie/{tmdb_id}",
        "review_total": int((data.get("reviews") or {}).get("total_results") or 0),
    }


def to_reviews(results: list[dict]) -> list[dict]:
    return [{
        "id": f"tmdb:{review['id']}",
        "author": review.get("author") or (review.get("author_details") or {}).get("username") or "TMDB reader",
        "rating": (review.get("author_details") or {}).get("rating"),
        "content": review.get("content") or "",
        "url": review.get("url"),
        "created_at": review.get("created_at"),
    } for review in results]


def sync_movie(client: TmdbClient, store: FilmStore, movie_id: int, tmdb_id: int, review_pages: int = 1) -> dict | None:
    data = client.movie(tmdb_id)
    if data is None:
        store.save_film(movie_id, None)
        return None
    results = list((data.get("reviews") or {}).get("results") or [])
    total_pages = int((data.get("reviews") or {}).get("total_pages") or 1)
    for page in range(2, min(total_pages, review_pages) + 1):
        results += (client.reviews(tmdb_id, page) or {}).get("results") or []
    store.save_film(movie_id, to_film(tmdb_id, data), to_reviews(results))
    return data


def main() -> None:
    import pandas as pd

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--popular", type=int, help="Sync the N most-rated MovieLens films.")
    target.add_argument("--movie-ids", nargs="+", type=int, help="Sync these MovieLens movie IDs.")
    target.add_argument("--all", action="store_true", help="Sync every film with a tmdbId.")
    parser.add_argument("--review-pages", type=int, default=1, help="Pages of reviews per film (20 per page).")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch films already synced from TMDB.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between films.")
    args = parser.parse_args()

    client = TmdbClient.from_env()
    if client is None:
        sys.exit("Add TMDB_READ_TOKEN=... (or TMDB_API_KEY=...) to .env first. Get one at https://www.themoviedb.org/settings/api")
    store = FilmStore()
    links = pd.read_csv(ROOT / "data" / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    if args.movie_ids:
        links = links[links["movieId"].isin(args.movie_ids)]
    elif args.popular:
        counts = pd.read_csv(ROOT / "data" / "ratings.csv", usecols=["movieId"])["movieId"].value_counts()
        links = links.assign(n=links["movieId"].map(counts).fillna(0)).sort_values("n", ascending=False).head(args.popular)

    with store.connect() as db:
        done = set() if args.refresh else {int(r[0]) for r in db.execute("SELECT movie_id FROM films WHERE source = 'tmdb'")}
    todo = [(int(r.movieId), int(r.tmdbId)) for r in links.itertuples() if int(r.movieId) not in done]
    print(f"Syncing {len(todo)} films from TMDB ({len(links) - len(todo)} already synced).")
    for index, (movie_id, tmdb_id) in enumerate(todo, 1):
        try:
            data = sync_movie(client, store, movie_id, tmdb_id, args.review_pages)
            label = data.get("title") if data else "not found on TMDB"
        except requests.RequestException as exc:
            label = f"failed: {exc}"
        if index % 25 == 0 or index == len(todo) or label.startswith("failed"):
            print(f"  {index}/{len(todo)}  movieId {movie_id}: {label}")
        time.sleep(args.sleep)
    counts = store.counts()
    print(f"Done. {counts['films']} films, {counts['with_poster']} with posters, {counts['reviews']} reviews cached.")


if __name__ == "__main__":
    main()
