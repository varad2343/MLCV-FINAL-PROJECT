"""Local Flask dashboard for MovieLens and Moctale recommendations."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import tmdb
import wikipedia_source
from film_store import REVIEW_SORTS, FilmStore
from recommend import MOCTALE_CATEGORIES, MovieRecommender

ROOT = Path(__file__).resolve().parent
POSTER_DIR = ROOT / "licensed_posters"
VOTES_DB = ROOT / "instance" / "moctale_votes.sqlite3"
VOTE_VALUES = set(MOCTALE_CATEGORIES)

app = Flask(__name__, static_folder="static", template_folder="templates")
recommender = MovieRecommender(data_dir=ROOT / "data", cache_dir=ROOT / "outputs", poster_dir=POSTER_DIR)
tmdb_client = tmdb.TmdbClient.from_env()


def _fetch_missing_film(movie_id: int) -> None:
    """First visit to an uncached film: look it up on Wikipedia, and on TMDB when a key is set."""
    imdb_id = IMDB_IDS.get(movie_id)
    if imdb_id:
        wikipedia_source.sync([(movie_id, imdb_id)], film_store, sleep=0, log=lambda _message: None)
    if tmdb_client and movie_id in TMDB_IDS:
        tmdb.sync_movie(tmdb_client, film_store, movie_id, TMDB_IDS[movie_id])


film_store = FilmStore(fetcher=_fetch_missing_film)


def _movielens_stats():
    """Per-film rating count, mean, Bayesian score and rank (films with 50+ ratings are ranked)."""
    ratings = recommender.ratings_df
    global_mean = float(ratings["rating"].mean())
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats["bayesian"] = (stats["count"] * stats["mean"] + 50 * global_mean) / (stats["count"] + 50)
    ranked = stats[stats["count"] >= 50].sort_values("bayesian", ascending=False)
    stats["rank"] = None
    stats.loc[ranked.index, "rank"] = range(1, len(ranked) + 1)
    return stats


ML_STATS = _movielens_stats()
TMDB_IDS = {
    int(row.movieId): int(row.tmdbId)
    for row in recommender.movies_df[["movieId", "tmdbId"]].dropna().itertuples(index=False)
}
IMDB_IDS = {int(row.movieId): f"tt{row.imdbId}" for row in recommender.movies_df.itertuples(index=False) if row.imdbId}


def _viewer_tags() -> dict[int, list[dict]]:
    """Free-text tags MovieLens viewers attached to each film (data/tags.csv), most used first."""
    tags = pd.read_csv(ROOT / "data" / "tags.csv", usecols=["userId", "movieId", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.strip()
    tags = tags[tags["tag"].str.len().between(2, 40)]
    tags["key"] = tags["tag"].str.casefold()
    grouped = (
        tags.groupby(["movieId", "key"])
        .agg(tag=("tag", "first"), viewers=("userId", "nunique"))
        .reset_index()
        .sort_values(["movieId", "viewers", "tag"], ascending=[True, False, True])
    )
    return {
        int(movie_id): [{"tag": row.tag, "viewers": int(row.viewers)} for row in group.head(20).itertuples()]
        for movie_id, group in grouped.groupby("movieId")
    }


VIEWER_TAGS = _viewer_tags()


@contextmanager
def _connect_votes() -> Iterator[sqlite3.Connection]:
    VOTES_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(VOTES_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _init_vote_store() -> None:
    with _connect_votes() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS movie_votes (
                visitor_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                vote TEXT NOT NULL CHECK (vote IN ('Perfection', 'Go for it', 'Timepass', 'Skip')),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (visitor_id, movie_id)
            )"""
        )


_init_vote_store()


def _visitor_id(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return None


def _get_visitor_votes(visitor_id: str | None) -> list[dict]:
    if not visitor_id:
        return []
    with _connect_votes() as connection:
        rows = connection.execute(
            "SELECT movie_id, vote FROM movie_votes WHERE visitor_id = ?",
            (visitor_id,),
        ).fetchall()
    return [{"movieId": int(row["movie_id"]), "vote": row["vote"]} for row in rows]


def _visitor_profile(visitor_id: str | None) -> dict:
    counts = {category: 0 for category in MOCTALE_CATEGORIES}
    votes = _get_visitor_votes(visitor_id)
    for vote in votes:
        counts[vote["vote"]] += 1
    total = len(votes)
    percentages = {
        category: round(counts[category] * 100 / total, 1) if total else 0.0
        for category in MOCTALE_CATEGORIES
    }
    return {"total_votes": total, "category_counts": counts, "category_percentages": percentages}


def _live_moctale(movie_id: int, visitor_id: str | None = None, predicted_rating: float | None = None) -> dict:
    info = recommender.get_movie_moctale_info(movie_id, predicted_rating)
    counts = recommender.get_moctale_counts(movie_id)
    recorded_total = sum(counts.values())
    votes = _get_visitor_votes(visitor_id)
    my_vote = None
    for vote in votes:
        if vote["movieId"] == movie_id:
            my_vote = vote["vote"]
            break

    with _connect_votes() as connection:
        local_counts = connection.execute(
            "SELECT vote, COUNT(*) AS total FROM movie_votes WHERE movie_id = ? GROUP BY vote",
            (movie_id,),
        ).fetchall()
    for row in local_counts:
        counts[row["vote"]] += int(row["total"])

    total = sum(counts.values())
    if total:
        for key, category in zip(("perfection_pct", "goforit_pct", "timepass_pct", "skip_pct"), MOCTALE_CATEGORIES):
            info[key] = round(counts[category] * 100 / total, 1)
        info["verdict"] = max(MOCTALE_CATEGORIES, key=lambda category: counts[category])
        info["vote_count"] = total
        info["estimated"] = False
        info["meter_source"] = "MovieLens ratings and local votes" if recorded_total else "Local community votes"
        info["verdict_display"] = {
            "Perfection": "🌟 Perfection",
            "Go for it": "👍 Go for it",
            "Timepass": "⏳ Timepass",
            "Skip": "⏭️ Skip",
        }[info["verdict"]]
    info["category_counts"] = counts
    info["my_vote"] = my_vote
    return info


def _movielens_summary(movie_id: int) -> dict | None:
    if movie_id not in ML_STATS.index:
        return None
    row = ML_STATS.loc[movie_id]
    return {
        "count": int(row["count"]),
        "mean": round(float(row["mean"]), 2),
        "bayesian": round(float(row["bayesian"]), 2),
        "rank": int(row["rank"]) if row["rank"] is not None else None,
    }


POSTER_DIRS = [POSTER_DIR, ROOT / "data" / "posters"]  # open-licensed first, then offline downloads
POSTER_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif")


def _local_poster_file(movie_id: int) -> Path | None:
    for folder in POSTER_DIRS:
        for extension in POSTER_EXTENSIONS:
            path = folder / f"{int(movie_id)}.{extension}"
            if path.is_file():
                return path
    return None


def _local_poster(movie_id: int) -> str:
    """URL of a poster saved on disk (licensed_posters/ or download_data.py --posters), or ''."""
    return f"/posters/{int(movie_id)}" if _local_poster_file(movie_id) else ""


def _with_local_poster(movie_id: int, film: dict | None) -> dict | None:
    """Prefer the offline copy of a poster so pages work without a connection."""
    local = _local_poster(movie_id)
    if film and local:
        film = {**film, "poster": local, "poster_large": local}
    return film


def _attach_details(movies: list[dict]) -> list[dict]:
    """Add cached film details; their poster replaces the open-licensed one when both exist."""
    films = film_store.films([int(movie["movieId"]) for movie in movies])
    for movie in movies:
        movie_id = int(movie["movieId"])
        film = _with_local_poster(movie_id, films.get(movie_id))
        movie["details"] = film
        path = _local_poster_file(movie_id)
        if path is not None:
            movie["poster_url"] = f"/posters/{movie_id}"
            if path.parent != POSTER_DIR:
                movie["poster_attribution"] = None  # an offline copy of the Wikipedia poster, not an open-licensed file
        elif film and film["poster"]:
            movie["poster_url"] = film["poster"]
            movie["poster_attribution"] = None
    return movies


def _attach_live_data(movies: list[dict], visitor_id: str | None) -> list[dict]:
    for movie in movies:
        movie_id = int(movie["movieId"])
        movie["moctale"] = _live_moctale(movie_id, visitor_id, movie.get("predicted_rating"))
        movie["my_vote"] = movie["moctale"]["my_vote"]
        movie["movielens"] = _movielens_summary(movie_id)
        movie.setdefault("poster_attribution", recommender.get_poster_attribution(movie_id))
    return _attach_details(movies)


def _movie_exists(movie_id: int) -> bool:
    return bool((recommender.movies_df["movieId"] == movie_id).any())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    ratings = recommender.ratings_df
    movies = recommender.movies_df
    return jsonify({
        "total_ratings": len(ratings),
        "total_users": int(ratings["userId"].nunique()),
        "total_movies": len(movies),
        "global_mean": round(float(ratings["rating"].mean()), 3),
        "algorithms": [
            {"name": "Hybrid recommendation", "type": "SVD + genre + poster features + MMR diversity", "weights": {"svd": .5, "genre": .3, "visual": .2}},
            {"name": "Moctale Rating System", "type": "Community votes", "categories": list(MOCTALE_CATEGORIES)},
            {"name": "Cold-start meter", "type": "Extra Trees over genres and available poster features"},
        ],
        "visual_embeddings_loaded": len(recommender._poster_embeddings),
        "films": {**film_store.counts(), "tmdb_configured": tmdb_client is not None},
    })


@app.route("/api/movies/search")
def search_movies():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    visitor_id = _visitor_id(request.args.get("visitor_id"))
    matches = recommender.movies_df[
        recommender.movies_df["title"].str.contains(query, case=False, na=False, regex=False)
    ].head(15)
    results = []
    for _, row in matches.iterrows():
        movie_id = int(row["movieId"])
        results.append({
            "movieId": movie_id,
            "title": row["title"],
            "genres": row["genres"],
            "moctale": _live_moctale(movie_id, visitor_id),
            "poster_url": recommender.get_poster_url(movie_id),
            "poster_attribution": recommender.get_poster_attribution(movie_id),
            "imdb_url": f"https://www.imdb.com/title/tt{row.get('imdbId', '')}/" if row.get("imdbId") else "",
        })
    return jsonify(_attach_details(results))


@app.route("/api/movies/<int:movie_id>")
def movie_detail(movie_id: int):
    """Everything the film page needs: details, MovieLens stats and tags, verdicts, reviews, similar films."""
    match = recommender.movies_df[recommender.movies_df["movieId"] == movie_id]
    if match.empty:
        return jsonify({"success": False, "error": "Movie not found"}), 404
    row = match.iloc[0]
    visitor_id = _visitor_id(request.args.get("visitor_id"))
    user_id = request.args.get("user_id", type=int) or 1
    film = _with_local_poster(movie_id, film_store.film(movie_id, fetch=True))
    movie = {
        "movieId": movie_id,
        "title": row["title"],
        "genres": row["genres"],
        "imdb_url": f"https://www.imdb.com/title/tt{row['imdbId']}/" if row.get("imdbId") else "",
        "poster_url": recommender.get_poster_url(movie_id),
        "poster_attribution": recommender.get_poster_attribution(movie_id),
    }
    _attach_live_data([movie], visitor_id)
    similar = []
    try:
        _target, similar = recommender.recommend_by_movie(movie_id, top_n=6, user_id=user_id)
        _attach_live_data(similar, visitor_id)
    except ValueError:
        pass
    return jsonify({
        "success": True,
        "movie": movie,
        "details": film,
        "reviews": film_store.reviews(movie_id, "newest", limit=20),
        "review_count": film_store.review_count(movie_id),
        "tags": VIEWER_TAGS.get(movie_id, []),
        "similar": similar,
        "visitor_profile": _visitor_profile(visitor_id),
    })


@app.route("/api/films/details", methods=["POST"])
def film_details_batch():
    """Details for up to 24 films, looking up uncached ones on Wikipedia in one batch."""
    payload = request.get_json(silent=True) or {}
    try:
        ids = [int(i) for i in payload.get("ids", [])][:24]
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "ids must be a list of MovieLens movie IDs"}), 400
    cached = film_store.films(ids)
    known = film_store.synced_ids()
    local = {i: _local_poster(i) for i in ids}
    missing = [(i, IMDB_IDS[i]) for i in ids if i not in known and i in IMDB_IDS and not local[i]]
    if missing:
        try:
            wikipedia_source.sync(missing, film_store, sleep=0, log=lambda _message: None)
        except Exception as exc:  # network trouble shouldn't break the page
            app.logger.warning("Wikipedia lookup failed: %s", exc)
        cached = film_store.films(ids)
    films = {str(k): _with_local_poster(k, v) for k, v in cached.items()}
    for movie_id, url in local.items():
        if url and str(movie_id) not in films:
            films[str(movie_id)] = {"poster": url, "poster_large": url}
    return jsonify({"success": True, "films": films})


@app.route("/api/movies/<int:movie_id>/reviews")
def movie_reviews(movie_id: int):
    sort = request.args.get("sort", "newest")
    if sort not in REVIEW_SORTS:
        return jsonify({"success": False, "error": f"sort must be one of {', '.join(REVIEW_SORTS)}"}), 400
    limit = max(1, min(request.args.get("limit", 20, type=int), 50))
    offset = max(0, request.args.get("offset", 0, type=int))
    return jsonify({
        "success": True,
        "reviews": film_store.reviews(movie_id, sort, limit, offset),
        "review_count": film_store.review_count(movie_id),
    })


@app.route("/api/reviews/latest")
def latest_reviews():
    limit = max(1, min(request.args.get("limit", 6, type=int), 30))
    reviews = film_store.latest_reviews(limit)
    titles = recommender.movies_df.set_index("movieId")["title"]
    films = film_store.films([review["movieId"] for review in reviews])
    for review in reviews:
        film = films.get(review["movieId"]) or {}
        review["title"] = str(titles.get(review["movieId"], ""))
        review["poster"] = _local_poster(review["movieId"]) or film.get("poster", "")
        review["critic_scores"] = film.get("critic_scores", {})
    return jsonify({"success": True, "reviews": reviews})


# ---------- Project report: evaluation results and live model internals ----------
REPORT_PATH = ROOT / "outputs" / "report" / "metrics.json"
_latent_cache: dict = {}


def _latent():
    """Unit-normalised SVD item factors for films with 20+ ratings (fewer ratings = noisy vectors)."""
    if not _latent_cache:
        svd = recommender.get_svd_model()
        counts = recommender.ratings_df["movieId"].value_counts()
        ids = [int(m) for m in counts[counts >= 20].index if int(m) in svd.trainset._raw2inner_id_items]
        inner = [svd.trainset._raw2inner_id_items[m] for m in ids]
        vectors = svd.qi[inner]
        _latent_cache.update(ids=ids, position={m: n for n, m in enumerate(ids)},
                             unit=vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return _latent_cache


def _title(movie_id: int) -> str:
    match = recommender.movies_df.loc[recommender.movies_df["movieId"] == movie_id, "title"]
    return str(match.iloc[0]) if not match.empty else ""


def _movie_arg(name: str) -> int:
    movie_id = request.args.get(name, type=int)
    if movie_id is None or not _movie_exists(movie_id):
        response = jsonify({"success": False, "error": f"{name} must be a MovieLens movie ID"})
        response.status_code = 400
        abort(response)
    return movie_id


@app.route("/api/report")
def report_metrics():
    try:
        return jsonify({"success": True, "metrics": json.loads(REPORT_PATH.read_text(encoding="utf-8"))})
    except (OSError, json.JSONDecodeError):
        return jsonify({"success": False, "error": "No evaluation results yet. Run: python evaluate.py"}), 404


@app.route("/api/report/explain")
def report_explain():
    """One SVD prediction broken into  mu + b_u + b_i + p_u . q_i."""
    user_id = request.args.get("user_id", type=int) or 1
    movie_id = _movie_arg("movie_id")
    svd = recommender.get_svd_model()
    trainset = svd.trainset
    if not trainset.knows_user(trainset._raw2inner_id_users.get(user_id, -1)):
        return jsonify({"success": False, "error": f"Viewer {user_id} is not in the training data"}), 400
    u = trainset._raw2inner_id_users[user_id]
    i = trainset._raw2inner_id_items.get(movie_id)
    actual = recommender.ratings_df.query("userId == @user_id and movieId == @movie_id")["rating"]
    base = {"user_id": user_id, "movie_id": movie_id, "title": _title(movie_id),
            "global_mean": round(float(trainset.global_mean), 3), "user_bias": round(float(svd.bu[u]), 3),
            "user_ratings": len(trainset.ur[u]), "actual": float(actual.iloc[0]) if not actual.empty else None}
    if i is None:
        # Cold start: no ratings means no item vector; SVD falls back to mu + b_u.
        estimate = float(np.clip(trainset.global_mean + svd.bu[u], 0.5, 5.0))
        return jsonify({"success": True, **base, "cold_start": True, "prediction": round(estimate, 2)})
    contributions = svd.pu[u] * svd.qi[i]
    top = np.argsort(-np.abs(contributions))[:6]
    raw = trainset.global_mean + svd.bu[u] + svd.bi[i] + contributions.sum()
    return jsonify({"success": True, **base, "cold_start": False,
                    "item_bias": round(float(svd.bi[i]), 3), "interaction": round(float(contributions.sum()), 3),
                    "raw": round(float(raw), 3), "prediction": round(float(np.clip(raw, 0.5, 5.0)), 2),
                    "item_ratings": len(trainset.ir[i]), "factors": len(contributions),
                    "top_factors": [{"factor": int(f), "user": round(float(svd.pu[u][f]), 3),
                                     "item": round(float(svd.qi[i][f]), 3),
                                     "product": round(float(contributions[f]), 3)} for f in top]})


@app.route("/api/report/similarity")
def report_similarity():
    """Three ways to call two films similar: shared genres, learned factors, co-ratings."""
    a, b = _movie_arg("a"), _movie_arg("b")
    tfidf, index = recommender.tfidf_matrix, recommender._movie_id_to_idx
    genre = float(cosine_similarity(tfidf[index[a]], tfidf[index[b]])[0, 0])
    latent = _latent()
    factor = None
    if a in latent["position"] and b in latent["position"]:
        factor = float(latent["unit"][latent["position"][a]] @ latent["unit"][latent["position"][b]])
    ratings = recommender.ratings_df
    both = ratings[ratings["movieId"].isin([a, b])].pivot_table(index="userId", columns="movieId", values="rating").dropna()
    pearson = float(both[a].corr(both[b])) if len(both) >= 5 and both[a].std() > 0 and both[b].std() > 0 else None
    genres = lambda m: recommender.movies_df.loc[recommender.movies_df["movieId"] == m, "genres"].iloc[0].split("|")
    return jsonify({"success": True,
                    "a": {"movieId": a, "title": _title(a), "genres": genres(a)},
                    "b": {"movieId": b, "title": _title(b), "genres": genres(b)},
                    "genre_cosine": round(genre, 3), "factor_cosine": round(factor, 3) if factor is not None else None,
                    "pearson": round(pearson, 3) if pearson is not None else None, "co_raters": int(len(both))})


@app.route("/api/report/neighbors")
def report_neighbors():
    """Nearest films by learned SVD factors versus by genre TF-IDF."""
    movie_id = _movie_arg("movie_id")
    counts = recommender.ratings_df["movieId"].value_counts()
    latent = _latent()
    by_factor = []
    if movie_id in latent["position"]:
        sims = latent["unit"] @ latent["unit"][latent["position"][movie_id]]
        for n in np.argsort(-sims)[1:9]:
            by_factor.append({"movieId": latent["ids"][n], "title": _title(latent["ids"][n]), "score": round(float(sims[n]), 3)})
    tfidf, index = recommender.tfidf_matrix, recommender._movie_id_to_idx
    sims = cosine_similarity(tfidf[index[movie_id]], tfidf).ravel()
    ids = recommender.movies_df["movieId"].astype(int).to_numpy()
    popularity = np.array([counts.get(m, 0) for m in ids])
    order = np.lexsort((-popularity, -sims))  # ties in genre broken by popularity
    by_genre = [{"movieId": int(ids[n]), "title": _title(int(ids[n])), "score": round(float(sims[n]), 3)}
                for n in order if int(ids[n]) != movie_id][:8]
    return jsonify({"success": True, "movieId": movie_id, "title": _title(movie_id),
                    "rating_count": int(counts.get(movie_id, 0)), "by_factor": by_factor, "by_genre": by_genre})


# ---------- Analyze page: artifacts written by analysis.py ----------
ANALYSIS_DIR = ROOT / "outputs" / "analysis"
ANALYSIS_DATASETS = {"ml-latest-small": "MovieLens Small", "ml-25m": "MovieLens 25M", "ml-32m": "MovieLens 32M"}
_analysis_cache: dict = {}


def _analysis_path(dataset: str, name: str) -> Path:
    if dataset not in ANALYSIS_DATASETS and dataset != "posters":
        abort(404)
    path = ANALYSIS_DIR / dataset / name
    if not path.is_file():
        response = jsonify({"success": False, "error": f"Not built yet. Run: python analysis.py"
                            + (f" --dataset {dataset}" if dataset != "posters" else " --posters-only")})
        response.status_code = 404
        abort(response)
    return path


def _cached(dataset: str, name: str, loader):
    """Load an analysis file once, and again only if analysis.py rewrote it."""
    path = _analysis_path(dataset, name)
    key, stamp = (dataset, name), path.stat().st_mtime
    if key not in _analysis_cache or _analysis_cache[key][0] != stamp:
        _analysis_cache[key] = (stamp, loader(path))
    return _analysis_cache[key][1]


def _analysis_json(dataset: str, name: str) -> dict:
    return _cached(dataset, name, lambda p: json.loads(p.read_text(encoding="utf-8")))


def _analysis_movies(dataset: str) -> pd.DataFrame:
    def load(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame["title_key"] = frame["title"].str.casefold()
        return frame.set_index("movieId", drop=False)
    return _cached(dataset, "movies.csv", load)


def _analysis_npz(dataset: str, name: str) -> dict:
    def load(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if "vectors" in arrays:
            arrays["position"] = {int(m): i for i, m in enumerate(arrays["movie_ids"])}
        if "relevance" in arrays:
            rel = arrays["relevance"].astype(np.float32)
            centred = rel - rel.mean(axis=0)  # centre per tag so shared "generic" tags don't dominate
            arrays["unit"] = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-9)
            arrays["relevance"] = rel
            arrays["position"] = {int(m): i for i, m in enumerate(arrays["movie_ids"])}
        return arrays
    return _cached(dataset, name, load)


def _film_brief(dataset: str, movie_id: int, **extra) -> dict:
    movies = _analysis_movies(dataset)
    row = movies.loc[movie_id] if movie_id in movies.index else None
    return {"movieId": int(movie_id), "title": str(row["title"]) if row is not None else str(movie_id),
            "year": None if row is None or pd.isna(row["year"]) else int(row["year"]),
            "n": 0 if row is None else int(row["n"]), "mean": None if row is None or pd.isna(row["mean"]) else round(float(row["mean"]), 2),
            "poster": _local_poster(movie_id), **extra}


@app.route("/api/analysis")
def analysis_index():
    datasets = []
    for key, label in ANALYSIS_DATASETS.items():
        if (ANALYSIS_DIR / key / "summary.json").is_file():
            summary = _analysis_json(key, "summary.json")
            datasets.append({"id": key, "label": label, "size": summary["size"], "generated_at": summary["generated_at"],
                             "genome": "genome" in summary, "latent": (ANALYSIS_DIR / key / "latent.json").is_file()})
    posters = None
    if (ANALYSIS_DIR / "posters" / "posters.json").is_file():
        data = _analysis_json("posters", "posters.json")
        posters = {"posters": data["posters"], "generated_at": data["generated_at"]}
    poster_ids = sorted({int(p.stem) for folder in POSTER_DIRS if folder.is_dir() for p in folder.iterdir()
                         if p.stem.isdigit() and p.suffix.lower().lstrip(".") in POSTER_EXTENSIONS})
    return jsonify({"success": True, "datasets": datasets, "posters": posters,
                    "posters_on_disk": len(poster_ids), "poster_ids": poster_ids})


@app.route("/api/analysis/<dataset>/summary")
def analysis_summary(dataset: str):
    return jsonify({"success": True, **_analysis_json(dataset, "summary.json")})


@app.route("/api/analysis/<dataset>/latent")
def analysis_latent(dataset: str):
    return jsonify({"success": True, **_analysis_json(dataset, "latent.json")})


@app.route("/api/analysis/<dataset>/movies")
def analysis_movies(dataset: str):
    """Filter, sort and page the per-film table; also returns a scatter sample of the filtered set."""
    movies = _analysis_movies(dataset)
    q = request.args.get("q", "").strip().casefold()
    genre = request.args.get("genre", "").strip()
    year_from = request.args.get("year_from", type=int)
    year_to = request.args.get("year_to", type=int)
    min_ratings = max(0, request.args.get("min_ratings", 0, type=int))
    sort = request.args.get("sort", "bayes")
    ascending = request.args.get("order", "desc") == "asc"
    limit = max(1, min(request.args.get("limit", 50, type=int), 200))
    offset = max(0, request.args.get("offset", 0, type=int))
    if sort not in {"bayes", "n", "mean", "std", "year", "title"}:
        return jsonify({"success": False, "error": "sort must be bayes, n, mean, std, year or title"}), 400
    view = movies[movies["n"] >= max(min_ratings, 1)]
    if q:
        view = view[view["title_key"].str.contains(q, regex=False)]
    if genre:
        view = view[view["genres"].fillna("").str.split("|").map(lambda gs: genre in gs)]
    if year_from is not None:
        view = view[view["year"] >= year_from]
    if year_to is not None:
        view = view[view["year"] <= year_to]
    ordered = view.sort_values(sort, ascending=ascending, na_position="last")
    page = ordered.iloc[offset:offset + limit]
    sample = view.nlargest(1500, "n")
    rows = [{"movieId": int(r.movieId), "title": r.title, "year": None if pd.isna(r.year) else int(r.year),
             "genres": r.genres, "n": int(r.n), "mean": round(float(r.mean), 3), "std": round(float(r.std), 3),
             "bayes": round(float(r.bayes), 3), "first": None if pd.isna(r.first) else int(r.first),
             "last": None if pd.isna(r.last) else int(r.last)} for r in page.itertuples()]
    return jsonify({"success": True, "total": int(len(view)), "rows": rows,
                    "scatter": [[int(r.movieId), int(r.n), round(float(r.mean), 3), r.title] for r in sample.itertuples()],
                    "stats": {"mean_of_means": round(float(view["mean"].mean()), 3) if len(view) else None,
                              "median_ratings": float(view["n"].median()) if len(view) else None}})


@app.route("/api/analysis/<dataset>/search")
def analysis_search(dataset: str):
    movies = _analysis_movies(dataset)
    q = request.args.get("q", "").strip().casefold()
    if len(q) < 2:
        return jsonify({"success": True, "results": []})
    hits = movies[movies["title_key"].str.contains(q, regex=False)].nlargest(10, "n")
    return jsonify({"success": True, "results": [_film_brief(dataset, int(m)) for m in hits["movieId"]]})


@app.route("/api/analysis/<dataset>/similar")
def analysis_similar(dataset: str):
    """Nearest films in the PureSVD factor space (analysis.py), by cosine."""
    movie_id = request.args.get("movie_id", type=int)
    # few ratings = a noisy factor vector that can match anything by chance, so neighbours need support
    min_ratings = request.args.get("min_ratings", 20 if dataset == "ml-latest-small" else 300, type=int)
    latent_data = _analysis_npz(dataset, "latent.npz")
    if movie_id not in latent_data["position"]:
        return jsonify({"success": False, "error": "This film has too few ratings for a stable factor vector."}), 404
    vectors = latent_data["vectors"]
    counts = _analysis_movies(dataset)["n"].reindex(latent_data["movie_ids"]).fillna(0).to_numpy()
    sims = np.where(counts >= min_ratings, vectors @ vectors[latent_data["position"][movie_id]], -np.inf)
    order = [i for i in np.argsort(-sims)[:13] if int(latent_data["movie_ids"][i]) != movie_id and np.isfinite(sims[i])][:12]
    return jsonify({"success": True, "film": _film_brief(dataset, movie_id),
                    "similar": [_film_brief(dataset, int(latent_data["movie_ids"][i]), score=round(float(sims[i]), 3)) for i in order]})


@app.route("/api/analysis/<dataset>/genome/tags")
def genome_tags(dataset: str):
    genome = _analysis_npz(dataset, "genome.npz")
    mean = genome["relevance"].mean(axis=0)
    return jsonify({"success": True, "tags": [{"id": int(t), "tag": str(name), "mean": round(float(m), 3)}
                                              for t, name, m in zip(genome["tag_ids"], genome["tags"], mean)]})


@app.route("/api/analysis/<dataset>/genome/tag")
def genome_tag(dataset: str):
    """Films the genome scores highest for one tag (optionally only well-rated or popular ones)."""
    genome = _analysis_npz(dataset, "genome.npz")
    tag_id = request.args.get("tag_id", type=int)
    min_ratings = max(0, request.args.get("min_ratings", 100, type=int))
    column = np.searchsorted(genome["tag_ids"], tag_id) if tag_id is not None else -1
    if column < 0 or column >= len(genome["tag_ids"]) or genome["tag_ids"][column] != tag_id:
        return jsonify({"success": False, "error": "Unknown tag"}), 404
    movies = _analysis_movies(dataset)
    scores = genome["relevance"][:, column]
    counts = movies["n"].reindex(genome["movie_ids"]).fillna(0).to_numpy()
    scores = np.where(counts >= min_ratings, scores, -1)
    order = np.argsort(-scores)[:24]
    return jsonify({"success": True, "tag": str(genome["tags"][column]),
                    "films": [_film_brief(dataset, int(genome["movie_ids"][i]), relevance=round(float(scores[i]), 3))
                              for i in order if scores[i] >= 0]})


@app.route("/api/analysis/<dataset>/genome/film")
def genome_film(dataset: str):
    """A film's tag profile, and its nearest films by tag genome vs. by shared genres."""
    genome = _analysis_npz(dataset, "genome.npz")
    movie_id = request.args.get("movie_id", type=int)
    if movie_id not in genome["position"]:
        return jsonify({"success": False, "error": "The tag genome covers 13,816 films; this isn't one of them."}), 404
    row = genome["position"][movie_id]
    rel = genome["relevance"][row]
    distinct = rel - genome["relevance"].mean(axis=0)
    top = np.argsort(-rel)[:15]
    signature = np.argsort(-distinct)[:15]
    sims = genome["unit"] @ genome["unit"][row]
    movies = _analysis_movies(dataset)
    counts = movies["n"].reindex(genome["movie_ids"]).fillna(0).to_numpy()
    sims_pop = np.where(counts >= 50, sims, -np.inf)
    by_genome = [i for i in np.argsort(-sims_pop)[:11] if i != row][:10]
    film_genres = set(str(movies.at[movie_id, "genres"]).split("|")) if movie_id in movies.index else set()
    pool = movies[movies["n"] >= 50]
    jaccard = pool["genres"].fillna("").str.split("|").map(lambda gs: len(film_genres & set(gs)) / max(len(film_genres | set(gs)), 1))
    by_genre = jaccard.drop(movie_id, errors="ignore").sort_values(ascending=False).head(10)
    return jsonify({"success": True, "film": _film_brief(dataset, movie_id),
                    "top_tags": [{"tag": str(genome["tags"][j]), "relevance": round(float(rel[j]), 3)} for j in top],
                    "signature_tags": [{"tag": str(genome["tags"][j]), "relevance": round(float(rel[j]), 3),
                                        "above_average": round(float(distinct[j]), 3)} for j in signature],
                    "by_genome": [_film_brief(dataset, int(genome["movie_ids"][i]), score=round(float(sims[i]), 3)) for i in by_genome],
                    "by_genre": [_film_brief(dataset, int(m), score=round(float(s), 3)) for m, s in by_genre.items()]})


@app.route("/api/analysis/posters")
def analysis_posters():
    return jsonify({"success": True, **_analysis_json("posters", "posters.json")})


@app.route("/api/analysis/posters/similar")
def analysis_posters_similar():
    """Posters that look alike: cosine over standardized colour, composition and HSV-histogram features."""
    posters = _analysis_npz("posters", "posters.npz")
    movie_id = request.args.get("movie_id", type=int)
    if movie_id not in posters["position"]:
        return jsonify({"success": False, "error": "No analysed poster for this film"}), 404
    sims = posters["vectors"] @ posters["vectors"][posters["position"][movie_id]]
    order = [i for i in np.argsort(-sims)[:13] if int(posters["movie_ids"][i]) != movie_id][:12]
    dataset = next((d for d in ("ml-25m", "ml-32m", "ml-latest-small") if (ANALYSIS_DIR / d / "movies.csv").is_file()), "ml-latest-small")
    return jsonify({"success": True, "film": _film_brief(dataset, movie_id),
                    "similar": [_film_brief(dataset, int(posters["movie_ids"][i]), score=round(float(sims[i]), 3)) for i in order]})


@app.route("/api/cold-start", methods=["POST"])
def predict_cold_start():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    genres = payload.get("genres", [])
    if isinstance(genres, list):
        genres = "|".join(str(genre) for genre in genres if str(genre).strip())
    elif not isinstance(genres, str):
        genres = ""
    if not title or not genres.strip():
        return jsonify({"success": False, "error": "A title and at least one genre are required."}), 400
    try:
        poster_movie_id = payload.get("poster_movie_id")
        poster_movie_id = int(poster_movie_id) if poster_movie_id not in (None, "") else None
        prediction = recommender.predict_cold_start_for_metadata(genres, poster_movie_id)
        return jsonify({"success": True, "title": title, "genres": genres.split("|"), **prediction})
    except (TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/user-profile/<int:user_id>")
def get_user_profile(user_id: int):
    try:
        return jsonify({"success": True, "profile": recommender.get_user_profile(user_id)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/visitor-profile")
def get_visitor_profile():
    visitor_id = _visitor_id(request.args.get("visitor_id"))
    return jsonify({"success": True, "profile": _visitor_profile(visitor_id)})


@app.route("/api/movies/<int:movie_id>/votes")
def get_movie_votes(movie_id: int):
    if not _movie_exists(movie_id):
        return jsonify({"success": False, "error": "Movie not found"}), 404
    visitor_id = _visitor_id(request.args.get("visitor_id"))
    return jsonify({
        "success": True,
        "movieId": movie_id,
        "moctale": _live_moctale(movie_id, visitor_id),
        "visitor_profile": _visitor_profile(visitor_id),
    })


@app.route("/api/movies/<int:movie_id>/vote", methods=["POST"])
def cast_movie_vote(movie_id: int):
    if not _movie_exists(movie_id):
        return jsonify({"success": False, "error": "Movie not found"}), 404
    payload = request.get_json(silent=True) or {}
    visitor_id = _visitor_id(payload.get("visitor_id"))
    vote = payload.get("vote")
    if not visitor_id:
        return jsonify({"success": False, "error": "A valid visitor ID is required"}), 400
    if not isinstance(vote, str) or vote not in VOTE_VALUES:
        return jsonify({"success": False, "error": "Vote must be Perfection, Go for it, Timepass, or Skip"}), 400

    with _connect_votes() as connection:
        connection.execute(
            """INSERT INTO movie_votes(visitor_id, movie_id, vote, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(visitor_id, movie_id) DO UPDATE SET
                 vote = excluded.vote, updated_at = CURRENT_TIMESTAMP""",
            (visitor_id, movie_id, vote),
        )
    return jsonify({
        "success": True,
        "movieId": movie_id,
        "moctale": _live_moctale(movie_id, visitor_id),
        "visitor_profile": _visitor_profile(visitor_id),
    })


@app.route("/api/recommend/user")
def recommend_user():
    try:
        user_id = int(request.args.get("user_id", 1))
        top_n = max(1, min(int(request.args.get("top_n", 5)), 20))
        visitor_id = _visitor_id(request.args.get("visitor_id"))
        visitor_votes = _get_visitor_votes(visitor_id)
        recs = recommender.recommend_for_user(user_id=user_id, top_n=top_n, visitor_votes=visitor_votes)
        return jsonify({
            "success": True,
            "user_id": user_id,
            "profile": recommender.get_user_profile(user_id=user_id),
            "visitor_profile": _visitor_profile(visitor_id),
            "recommendations": _attach_live_data(recs, visitor_id),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/recommend/movie")
def recommend_movie():
    try:
        query = request.args.get("query", "").strip()
        top_n = max(1, min(int(request.args.get("top_n", 5)), 20))
        user_id = int(request.args.get("user_id", 1))
        visitor_id = _visitor_id(request.args.get("visitor_id"))
        if not query:
            return jsonify({"success": False, "error": "Query movie parameter is required"}), 400
        target, recs = recommender.recommend_by_movie(
            query, top_n=top_n, user_id=user_id, visitor_votes=_get_visitor_votes(visitor_id)
        )
        target["moctale"] = _live_moctale(int(target["movieId"]), visitor_id)
        target["my_vote"] = target["moctale"]["my_vote"]
        target["movielens"] = _movielens_summary(int(target["movieId"]))
        _attach_details([target])
        return jsonify({
            "success": True,
            "target": target,
            "visitor_profile": _visitor_profile(visitor_id),
            "recommendations": _attach_live_data(recs, visitor_id),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/top-rated")
def top_rated():
    try:
        top_n = max(1, min(int(request.args.get("top_n", 12)), 30))
        visitor_id = _visitor_id(request.args.get("visitor_id"))
        recs = recommender.get_top_rated_movies(top_n=top_n)
        return jsonify({
            "success": True,
            "visitor_profile": _visitor_profile(visitor_id),
            "movies": _attach_live_data(recs, visitor_id),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/posters/<int:movie_id>")
def serve_poster(movie_id: int):
    path = _local_poster_file(movie_id)
    if path is None:
        abort(404)
    response = send_from_directory(path.parent, path.name, conditional=True)
    response.cache_control.max_age = 86400
    return response


THUMB_DIR = ROOT / "data" / "posters" / "thumbs"


@app.route("/posters/<int:movie_id>/thumb")
def serve_poster_thumb(movie_id: int):
    """80 x 120 JPEG made once with Pillow and cached, for lists and the poster map."""
    path = _local_poster_file(movie_id)
    if path is None:
        abort(404)
    thumb = THUMB_DIR / f"{movie_id}.jpg"
    if not thumb.is_file() or thumb.stat().st_mtime < path.stat().st_mtime:
        from PIL import Image, ImageOps

        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            ImageOps.fit(image.convert("RGB"), (80, 120), Image.LANCZOS).save(thumb, "JPEG", quality=82)
    response = send_from_directory(THUMB_DIR, thumb.name, conditional=True)
    response.cache_control.max_age = 86400
    return response


@app.route("/outputs/moctale/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(ROOT / "outputs" / "moctale", filename)


@app.route("/images/<path:filename>")
def serve_images(filename: str):
    return send_from_directory(ROOT / "images", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🎬 MovieLens Recommender App launching on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
