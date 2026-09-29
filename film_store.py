"""SQLite cache of film details, images and reviews, keyed by MovieLens movieId.

Filled by wikipedia_source.py (no key needed) and optionally tmdb.py.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "instance" / "films.sqlite3"
SOURCE_NAMES = {"wikipedia": "Wikipedia", "tmdb": "TMDB"}
REVIEW_SORTS = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    "highest": "rating IS NULL, rating DESC, created_at DESC",
    "lowest": "rating IS NULL, rating ASC, created_at DESC",
}
FILM_FIELDS = (
    "source", "source_id", "title", "tagline", "overview", "runtime", "release_date", "language",
    "certification", "director", "rating", "rating_count", "critic_scores", "poster_url",
    "poster_url_large", "backdrop_url", "page_url", "review_total",
)


class FilmStore:
    def __init__(self, db_path: Path = DB_PATH, fetcher: Callable[[int], None] | None = None):
        self.db_path = Path(db_path)
        self.fetcher = fetcher
        self._lock = threading.Lock()
        self._init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS films (
                    movie_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,                 -- 'ok' or 'missing'
                    source TEXT, source_id TEXT,
                    title TEXT, tagline TEXT, overview TEXT,
                    runtime INTEGER, release_date TEXT, language TEXT, certification TEXT, director TEXT,
                    rating REAL, rating_count INTEGER,
                    critic_scores TEXT,                   -- JSON {"Rotten Tomatoes": "87%", ...}
                    poster_url TEXT, poster_url_large TEXT, backdrop_url TEXT, page_url TEXT,
                    review_total INTEGER NOT NULL DEFAULT 0,
                    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    movie_id INTEGER NOT NULL REFERENCES films(movie_id),
                    source TEXT NOT NULL,
                    author TEXT NOT NULL, rating REAL,
                    content TEXT NOT NULL, url TEXT,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS reviews_by_movie ON reviews(movie_id, created_at);
                CREATE INDEX IF NOT EXISTS reviews_by_date ON reviews(created_at);
                """
            )

    # ---------- writes ----------
    def save_film(self, movie_id: int, film: dict | None, reviews: list[dict] | None = None) -> None:
        """film: keys from FILM_FIELDS (critic_scores as a dict). None marks the film as not found."""
        with self._lock, self.connect() as db:
            if film is None:
                db.execute("INSERT OR REPLACE INTO films(movie_id, status) VALUES (?, 'missing')", (int(movie_id),))
                return
            existing = db.execute("SELECT * FROM films WHERE movie_id = ? AND status = 'ok'", (int(movie_id),)).fetchone()
            values = {field: film.get(field) for field in FILM_FIELDS}
            scores = dict(film.get("critic_scores") or {})
            if existing is not None:
                # A second source fills gaps; it never blanks what the first one found.
                for field in FILM_FIELDS:
                    if values[field] in (None, "") and existing[field] not in (None, ""):
                        values[field] = existing[field]
                scores = {**json.loads(existing["critic_scores"] or "{}"), **scores}
            values["critic_scores"] = json.dumps(scores)
            values["review_total"] = int(film.get("review_total") or len(reviews or []))
            columns = ", ".join(("movie_id", "status", *FILM_FIELDS))
            placeholders = ", ".join("?" * (len(FILM_FIELDS) + 2))
            db.execute(
                f"INSERT OR REPLACE INTO films({columns}) VALUES ({placeholders})",
                (int(movie_id), "ok", *[values[field] for field in FILM_FIELDS]),
            )
            db.execute("DELETE FROM reviews WHERE movie_id = ? AND source = ?", (int(movie_id), film.get("source")))
            for review in reviews or []:
                db.execute(
                    """INSERT OR REPLACE INTO reviews(review_id, movie_id, source, author, rating, content, url, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (review["id"], int(movie_id), film.get("source"), review.get("author") or "Anonymous",
                     review.get("rating"), review.get("content") or "", review.get("url"), review.get("created_at")),
                )

    # ---------- reads ----------
    @staticmethod
    def _film_dict(row: sqlite3.Row) -> dict:
        poster = row["poster_url"] or ""
        return {
            "source": row["source"],
            "source_name": SOURCE_NAMES.get(row["source"], row["source"]),
            "title": row["title"],
            "tagline": row["tagline"],
            "overview": row["overview"],
            "runtime": row["runtime"],
            "release_date": row["release_date"],
            "language": row["language"],
            "certification": row["certification"],
            "director": row["director"],
            "rating": round(row["rating"], 1) if row["rating"] is not None else None,
            "rating_count": row["rating_count"],
            "critic_scores": json.loads(row["critic_scores"] or "{}"),
            "poster": poster,
            "poster_large": row["poster_url_large"] or poster,
            "backdrop": row["backdrop_url"] or "",
            "url": row["page_url"],
            "review_total": row["review_total"],
        }

    def films(self, movie_ids: list[int]) -> dict[int, dict]:
        ids = [int(i) for i in movie_ids]
        if not ids:
            return {}
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM films WHERE status = 'ok' AND movie_id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
        return {int(row["movie_id"]): self._film_dict(row) for row in rows}

    def film(self, movie_id: int, fetch: bool = False) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM films WHERE movie_id = ?", (int(movie_id),)).fetchone()
        if row is None and fetch and self.fetcher:
            try:
                self.fetcher(int(movie_id))
            except Exception:  # a failed lookup should never break the page
                return None
            return self.film(movie_id)
        return self._film_dict(row) if row is not None and row["status"] == "ok" else None

    @staticmethod
    def _review_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["review_id"],
            "movieId": int(row["movie_id"]),
            "source": row["source"],
            "source_name": SOURCE_NAMES.get(row["source"], row["source"]),
            "author": row["author"],
            "rating": row["rating"],
            "content": row["content"],
            "url": row["url"],
            "created_at": row["created_at"],
        }

    def reviews(self, movie_id: int, sort: str = "newest", limit: int = 20, offset: int = 0) -> list[dict]:
        order = REVIEW_SORTS.get(sort, REVIEW_SORTS["newest"])
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM reviews WHERE movie_id = ? ORDER BY {order}, review_id LIMIT ? OFFSET ?",
                (int(movie_id), int(limit), int(offset)),
            ).fetchall()
        return [self._review_dict(row) for row in rows]

    def review_count(self, movie_id: int) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM reviews WHERE movie_id = ?", (int(movie_id),)).fetchone()[0])

    def latest_reviews(self, limit: int = 6, min_length: int = 200) -> list[dict]:
        """One substantial review per film, most recently added first."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM (
                       SELECT r.*, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY created_at DESC, review_id) AS n
                       FROM reviews r WHERE LENGTH(content) >= ?)
                   WHERE n = 1 ORDER BY created_at DESC, movie_id LIMIT ?""",
                (int(min_length), int(limit)),
            ).fetchall()
        return [self._review_dict(row) for row in rows]

    def counts(self) -> dict:
        with self.connect() as db:
            films = db.execute("SELECT COUNT(*) FROM films WHERE status = 'ok'").fetchone()[0]
            with_poster = db.execute("SELECT COUNT(*) FROM films WHERE poster_url IS NOT NULL AND poster_url != ''").fetchone()[0]
            reviews = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            sources = [row[0] for row in db.execute("SELECT DISTINCT source FROM films WHERE status = 'ok'").fetchall()]
        return {"films": int(films), "with_poster": int(with_poster), "reviews": int(reviews), "sources": sources}

    def synced_ids(self, source: str | None = None) -> set[int]:
        with self.connect() as db:
            if source:
                rows = db.execute("SELECT movie_id FROM films WHERE source = ? OR status = 'missing'", (source,)).fetchall()
            else:
                rows = db.execute("SELECT movie_id FROM films").fetchall()
        return {int(row[0]) for row in rows}
