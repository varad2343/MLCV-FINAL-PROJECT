"""Offline data analysis behind the Analyze page (machine learning + computer vision).

  python analysis.py                          # every dataset found in data/, then posters
  python analysis.py --dataset ml-25m         # one dataset
  python analysis.py --posters-only           # just the poster (computer-vision) analysis

Writes outputs/analysis/<dataset>/:
  summary.json  size, sparsity, rating scale, activity over time, long tail, viewers,
                genres and genre co-occurrence, release decades, ranked film lists
  movies.csv    one row per film: year, genres, ratings, mean, spread, Bayesian score
  latent.json   PureSVD on the bias-corrected rating matrix: singular values, a 2-D
                map of films (PCA and t-SNE), viewer segments (k-means), and how well
                genre / tag-genome similarity agrees with rating-based similarity
  latent.npz    unit-length film factor vectors for nearest-neighbour queries
  genome.npz    (ml-25m) tag-genome relevance matrix, 13,816 films x 1,128 tags
and outputs/analysis/posters/: per-poster colour and composition features, genre
profiles and palettes, decade trends, correlations, a poster map, and a
cross-validated experiment predicting genre from poster features.

Ratings are read in chunks with compact types, so MovieLens 25M needs ~2 GB of RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "analysis"
SEED = 24
NO_GENRE = "(no genres listed)"
DATASETS = {
    # prior: Bayesian-average pseudo-votes; list_min: ratings needed for ranked lists;
    # map_*: which films go on the latent map; gems: rating-count window for "hidden gems".
    "ml-latest-small": dict(dir=DATA, label="MovieLens Small", prior=50, list_min=50, gems=(15, 50),
                            map_min=20, map_size=1500, neighbour_min=20, factors=32, segments=5),
    "ml-25m": dict(dir=DATA / "ml-25m", label="MovieLens 25M", prior=250, list_min=1000, gems=(100, 800),
                   map_min=500, map_size=3000, neighbour_min=50, factors=48, segments=6),
    "ml-32m": dict(dir=DATA / "ml-32m", label="MovieLens 32M", prior=250, list_min=1000, gems=(100, 800),
                   map_min=500, map_size=3000, neighbour_min=50, factors=48, segments=6),
}
POSTER_DIRS = [DATA / "posters", ROOT / "licensed_posters"]
POSTER_FEATURES = ["brightness", "contrast", "saturation", "colorfulness", "edge_density", "warmth",
                   "dark_share", "entropy", "aspect_ratio"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def to_json(value):
    """json.dump default for numpy types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=to_json, separators=(",", ":")), encoding="utf-8")


def r(value, digits: int = 3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


# ---------------------------------------------------------------- loading
def load_ratings(folder: Path) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {"user": [], "movie": [], "stars2": [], "ts": []}
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    for chunk in pd.read_csv(folder / "ratings.csv", dtype=dtypes, chunksize=4_000_000):
        parts["user"].append(chunk["userId"].to_numpy())
        parts["movie"].append(chunk["movieId"].to_numpy())
        parts["stars2"].append(np.rint(chunk["rating"].to_numpy() * 2).astype(np.int8))  # half-stars as 1..10
        parts["ts"].append(chunk["timestamp"].to_numpy().astype(np.int32))
    return {key: np.concatenate(value) for key, value in parts.items()}


def load_movies(folder: Path) -> pd.DataFrame:
    movies = pd.read_csv(folder / "movies.csv")
    movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    movies["genre_list"] = movies["genres"].fillna("").map(lambda g: [x for x in g.split("|") if x and x != NO_GENRE])
    return movies


def years_of(ts: np.ndarray) -> np.ndarray:
    return (ts.astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970).astype(np.int16)


def log_bins(values: np.ndarray, edges: list[int]) -> list[dict]:
    counts, _ = np.histogram(values, bins=edges + [np.inf])
    return [{"from": lo, "to": (edges[i + 1] - 1) if i + 1 < len(edges) else None, "count": int(c)}
            for i, (lo, c) in enumerate(zip(edges, counts))]


def film_rows(frame: pd.DataFrame) -> list[dict]:
    return [{"movieId": int(row.movieId), "title": row.title, "year": None if pd.isna(row.year) else int(row.year),
             "genres": row.genres, "n": int(row.n), "mean": r(row.mean), "std": r(row.std), "bayes": r(row.bayes)}
            for row in frame.itertuples()]


# ---------------------------------------------------------------- overview
def overview(name: str, cfg: dict, ratings: dict, movies: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    stars = ratings["stars2"].astype(np.float32) / 2
    users, films, ts = ratings["user"], ratings["movie"], ratings["ts"]
    mu = float(stars.mean())

    # per-film aggregates with bincount (no 25M-row groupby needed)
    size = int(max(films.max(), movies["movieId"].max())) + 1
    n = np.bincount(films, minlength=size)
    total = np.bincount(films, weights=stars, minlength=size)
    squares = np.bincount(films, weights=stars.astype(np.float64) ** 2, minlength=size)
    stats = movies[["movieId", "title", "year", "genres"]].copy()
    ids = stats["movieId"].to_numpy()
    stats["n"] = n[ids]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats["mean"] = total[ids] / n[ids]
        stats["std"] = np.sqrt(np.maximum(squares[ids] / n[ids] - stats["mean"] ** 2, 0))
    stats["bayes"] = (total[ids] + cfg["prior"] * mu) / (n[ids] + cfg["prior"])
    span = pd.DataFrame({"m": films, "y": years_of(ts)}).groupby("m")["y"].agg(["min", "max"])
    stats["first"] = stats["movieId"].map(span["min"])
    stats["last"] = stats["movieId"].map(span["max"])

    # viewers
    u_n = np.bincount(users)
    u_total = np.bincount(users, weights=stars)
    u_sq = np.bincount(users, weights=stars.astype(np.float64) ** 2)
    active = u_n > 0
    u_mean = u_total[active] / u_n[active]
    u_std = np.sqrt(np.maximum(u_sq[active] / u_n[active] - u_mean ** 2, 0))
    u_count = u_n[active]

    # time
    years = years_of(ts)
    y0, y1 = int(years.min()), int(years.max())
    y_n = np.bincount(years - y0)
    y_sum = np.bincount(years - y0, weights=stars)
    y_half = np.bincount(years - y0, weights=(ratings["stars2"] % 2 == 1).astype(np.float32))
    key = (years.astype(np.int64) - y0) * (int(users.max()) + 1) + users
    y_users = np.bincount((np.unique(key) // (int(users.max()) + 1)).astype(np.int64), minlength=y1 - y0 + 1)
    first_year = pd.Series(years).groupby(users).min()
    new_users = first_year.value_counts().reindex(range(y0, y1 + 1), fill_value=0)
    per_year = [{"year": y0 + i, "ratings": int(y_n[i]), "mean": r(y_sum[i] / y_n[i]) if y_n[i] else None,
                 "half_star_share": r(y_half[i] / y_n[i]) if y_n[i] else None, "active_users": int(y_users[i]),
                 "new_users": int(new_users.get(y0 + i, 0))} for i in range(len(y_n))]
    del years, key
    gc.collect()

    # long tail (Lorenz curve over films that have ratings)
    rated = np.sort(n[n > 0])[::-1]
    cumulative = np.cumsum(rated) / rated.sum()
    points = [{"films_pct": p, "ratings_pct": r(cumulative[max(int(np.ceil(p / 100 * len(rated))) - 1, 0)] * 100, 2)}
              for p in [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    ascending = rated[::-1].astype(np.float64)
    k = np.arange(1, len(ascending) + 1)
    gini = float((2 * (k * ascending).sum()) / (len(ascending) * ascending.sum()) - (len(ascending) + 1) / len(ascending))

    # genres (film-level sums, so no 25M-row explode)
    exploded = stats.assign(genre=movies["genre_list"]).explode("genre").dropna(subset=["genre"])
    exploded["total"] = exploded["n"] * exploded["mean"].fillna(0)
    genre = exploded.groupby("genre").agg(films=("movieId", "size"), ratings=("n", "sum"), total=("total", "sum"),
                                          median_year=("year", "median"))
    genre["mean"] = genre["total"] / genre["ratings"].replace(0, np.nan)
    genre = genre.sort_values("ratings", ascending=False)
    names = genre.index.tolist()
    index = {g: i for i, g in enumerate(names)}
    co = np.zeros((len(names), len(names)), dtype=np.int64)
    for genres in movies["genre_list"]:
        idx = [index[g] for g in genres if g in index]
        for a in idx:
            for b in idx:
                co[a, b] += 1
    films_total = len(movies)
    share = np.diag(co) / films_total
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (co / films_total) / np.outer(share, share)
    decade_rows = []
    by_decade = exploded[exploded["year"] >= 1920].assign(decade=lambda d: (d["year"] // 10 * 10).astype(int))
    decade_films = by_decade.drop_duplicates("movieId").groupby("decade").size()
    for decade, part in by_decade.groupby("decade"):
        counts = part.groupby("genre").size()
        decade_rows.append({"decade": int(decade), "films": int(decade_films.get(decade, 0)),
                            "genre_share": {g: r(counts.get(g, 0) / decade_films.get(decade, 1), 4) for g in names}})
    rated_films = stats[stats["n"] > 0]
    release = rated_films[rated_films["year"] >= 1920].assign(decade=lambda d: (d["year"] // 10 * 10).astype(int))
    release["total"] = release["n"] * release["mean"]
    release = release.groupby("decade").agg(films=("movieId", "size"), ratings=("n", "sum"), total=("total", "sum"))
    release_rows = [{"decade": int(d), "films": int(row.films), "ratings": int(row.ratings),
                     "mean": r(row.total / row.ratings)} for d, row in release.iterrows()]

    # ranked lists
    eligible = stats[stats["n"] >= cfg["list_min"]]
    lo, hi = cfg["gems"]
    lists = {
        "top_rated": film_rows(eligible.nlargest(15, "bayes")),
        "most_rated": film_rows(stats.nlargest(15, "n")),
        "most_divisive": film_rows(eligible.nlargest(15, "std")),
        "lowest_rated": film_rows(eligible.nsmallest(15, "bayes")),
        "hidden_gems": film_rows(stats[(stats["n"] >= lo) & (stats["n"] <= hi)].nlargest(15, "mean")),
    }

    stars_hist = np.bincount(ratings["stars2"], minlength=11)[1:]
    catalog_rated = int((n > 0).sum())
    summary = {
        "dataset": name, "label": cfg["label"], "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {"bayes_prior": cfg["prior"], "list_min": cfg["list_min"], "gems": list(cfg["gems"])},
        "size": {
            "ratings": int(len(stars)), "users": int(active.sum()), "films_rated": catalog_rated,
            "films_catalog": int(len(movies)), "first_year": y0, "last_year": y1,
            "sparsity_pct": r((1 - len(stars) / (active.sum() * catalog_rated)) * 100, 3),
            "mean_rating": r(mu), "median_rating": float(np.median(stars)),
            "tags": int(sum(1 for _ in open(cfg["dir"] / "tags.csv", encoding="utf-8")) - 1) if (cfg["dir"] / "tags.csv").is_file() else None,
        },
        "rating_scale": [{"stars": (i + 1) / 2, "count": int(c)} for i, c in enumerate(stars_hist)],
        "per_year": per_year,
        "long_tail": {"gini": r(gini), "curve": points,
                      "films_with_1_rating_pct": r((rated == 1).mean() * 100, 1),
                      "films_with_10_or_fewer_pct": r((rated <= 10).mean() * 100, 1),
                      "top_1pct_share": r(cumulative[max(int(len(rated) * 0.01) - 1, 0)] * 100, 1)},
        "films_per_count": log_bins(n[n > 0], [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]),
        "viewers": {
            "per_count": log_bins(u_count, [1, 20, 35, 50, 100, 200, 500, 1000, 2000, 5000]),
            "median_ratings": float(np.median(u_count)), "max_ratings": int(u_count.max()),
            "mean_hist": [{"from": float(lo_), "count": int(c)} for lo_, c in
                          zip(np.arange(0.5, 5.0, 0.25), np.histogram(u_mean, bins=np.arange(0.5, 5.25, 0.25))[0])],
            "std_median": r(np.median(u_std)),
            "activity_vs_mean_r": r(np.corrcoef(np.log10(u_count), u_mean)[0, 1]),
            "harsh_pct": r((u_mean < 3.0).mean() * 100, 1), "generous_pct": r((u_mean >= 4.0).mean() * 100, 1),
        },
        "genres": {
            "names": names,
            "stats": [{"genre": g, "films": int(row.films), "ratings": int(row.ratings), "mean": r(row["mean"]),
                       "median_year": None if pd.isna(row.median_year) else int(row.median_year)}
                      for g, row in genre.iterrows()],
            "cooccurrence": co.tolist(),
            "lift": [[r(v, 2) for v in row] for row in lift],
            "by_decade": decade_rows,
        },
        "release_decades": release_rows,
        "lists": lists,
    }
    return summary, stats


# ---------------------------------------------------------------- latent structure
def latent(name: str, cfg: dict, ratings: dict, movies: pd.DataFrame, stats: pd.DataFrame,
           genome: tuple[np.ndarray, np.ndarray] | None) -> tuple[dict, dict]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.utils.extmath import randomized_svd

    stars = ratings["stars2"].astype(np.float32) / 2
    user_ids, u_idx = np.unique(ratings["user"], return_inverse=True)
    item_ids, i_idx = np.unique(ratings["movie"], return_inverse=True)
    nu, ni = len(user_ids), len(item_ids)
    mu = float(stars.mean())
    n_i = np.bincount(i_idx, minlength=ni)
    b_i = np.bincount(i_idx, weights=stars - mu, minlength=ni) / (n_i + 10)       # damped item bias
    residual = stars - mu - b_i[i_idx]
    n_u = np.bincount(u_idx, minlength=nu)
    b_u = np.bincount(u_idx, weights=residual, minlength=nu) / (n_u + 10)          # damped user bias
    residual = (residual - b_u[u_idx]).astype(np.float32)
    frob = float((residual.astype(np.float64) ** 2).sum())
    matrix = sparse.csr_matrix((residual, (u_idx, i_idx)), shape=(nu, ni))
    del residual
    gc.collect()
    k = cfg["factors"]
    log(f"  randomized SVD, {k} factors, {nu:,} x {ni:,} matrix, {matrix.nnz:,} ratings")
    U, S, Vt = randomized_svd(matrix, n_components=k, n_iter=5, random_state=SEED)
    del matrix
    gc.collect()
    item_vectors = (Vt.T * S).astype(np.float32)
    item_unit = item_vectors / np.maximum(np.linalg.norm(item_vectors, axis=1, keepdims=True), 1e-9)

    meta = stats.set_index("movieId")
    genre_freq = pd.Series([g for gs in movies["genre_list"] for g in gs if g != "IMAX"]).value_counts()  # IMAX is a format

    # 2-D map of the most-rated films
    order = np.argsort(-n_i)
    pool = [i for i in order if n_i[i] >= cfg["map_min"]][: cfg["map_size"]]
    vectors = item_unit[pool]
    centred = vectors - vectors.mean(axis=0)
    _, _, pcs = np.linalg.svd(centred, full_matrices=False)
    pca = centred @ pcs[:2].T
    log(f"  t-SNE on {len(pool):,} films")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, learning_rate="auto").fit_transform(vectors)
    points = []
    for n_, i in enumerate(pool):
        movie_id = int(item_ids[i])
        points.append({"id": movie_id, "t": meta.at[movie_id, "title"], "g": str(meta.at[movie_id, "genres"]),
                       "n": int(n_i[i]), "m": r(meta.at[movie_id, "mean"], 2),
                       "p": [r(pca[n_, 0], 4), r(pca[n_, 1], 4)], "s": [r(tsne[n_, 0], 2), r(tsne[n_, 1], 2)]})

    # viewer segments: k-means on unit-length user factor vectors
    user_unit = (U * S).astype(np.float32)
    user_unit /= np.maximum(np.linalg.norm(user_unit, axis=1, keepdims=True), 1e-9)
    kmeans = KMeans(n_clusters=cfg["segments"], n_init=4, random_state=SEED).fit(user_unit)
    labels = kmeans.labels_
    one_hot = sparse.csr_matrix((np.ones(nu, dtype=np.float32), (labels, np.arange(nu))), shape=(cfg["segments"], nu))
    liked = sparse.csr_matrix(((stars >= 4).astype(np.float32), (u_idx, i_idx)), shape=(nu, ni))
    seen = sparse.csr_matrix((np.ones(len(stars), dtype=np.float32), (u_idx, i_idx)), shape=(nu, ni))
    raw = sparse.csr_matrix((stars, (u_idx, i_idx)), shape=(nu, ni))
    genre_names = genre_freq.index.tolist()
    g_index = {g: j for j, g in enumerate(genre_names)}
    rows, cols = [], []
    for i, movie_id in enumerate(item_ids):
        for g in str(meta.at[int(movie_id), "genres"]).split("|"):
            if g in g_index:
                rows.append(i)
                cols.append(g_index[g])
    item_genre = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(ni, len(genre_names)))
    seg_genre = np.asarray((one_hot @ liked @ item_genre).todense())
    all_genre = seg_genre.sum(axis=0)
    seg_sum = np.asarray((one_hot @ raw).todense())
    seg_cnt = np.asarray((one_hot @ seen).todense())
    item_mean = np.bincount(i_idx, weights=stars, minlength=ni) / np.maximum(n_i, 1)
    seg_users = np.bincount(labels, minlength=cfg["segments"])
    seg_ratings = np.bincount(labels, weights=n_u, minlength=cfg["segments"])
    rating_years = pd.Series(years_of(ratings["ts"])).groupby(u_idx)  # when each viewer was active
    first_year, last_year = rating_years.min().to_numpy(), rating_years.max().to_numpy()
    seg_stars = np.asarray(one_hot @ np.bincount(u_idx, weights=stars, minlength=nu)).ravel()
    del liked, seen, raw
    gc.collect()
    segments = []
    for c in range(cfg["segments"]):
        genre_share = seg_genre[c] / max(seg_genre[c].sum(), 1)
        overall_share = all_genre / max(all_genre.sum(), 1)
        # lift = how over-represented a genre is in this segment's liked films; only count
        # genres that make up a real part (3%+) of what the segment likes, or tiny genres dominate
        lift = np.where(genre_share >= 0.03, genre_share / np.maximum(overall_share, 1e-9), 0)
        top_genres = [genre_names[j] for j in np.argsort(-lift)[:6] if lift[j] > 0]
        min_support = max(10, int(0.02 * seg_users[c]))
        with np.errstate(invalid="ignore", divide="ignore"):
            seg_mean = seg_sum[c] / seg_cnt[c]
            watch_lift = (seg_cnt[c] / seg_users[c]) / (n_i / nu)
        support = (seg_cnt[c] >= min_support) & (n_i >= cfg["neighbour_min"])
        loved = np.where(support, seg_mean - item_mean, -np.inf)
        watched = np.where(support, watch_lift, -np.inf)
        film = lambda i: {"movieId": int(item_ids[i]), "title": meta.at[int(item_ids[i]), "title"]}
        segments.append({
            "id": c, "users": int(seg_users[c]), "share_pct": r(seg_users[c] / nu * 100, 1),
            "median_ratings": float(np.median(n_u[labels == c])),
            "mean_given": r(seg_stars[c] / seg_ratings[c]),
            "active_from": int(np.median(first_year[labels == c])), "active_to": int(np.median(last_year[labels == c])),
            "name": " · ".join(top_genres[:2]),
            "genres": [{"genre": genre_names[j], "lift": r(lift[j], 2), "share_pct": r(genre_share[j] * 100, 1)}
                       for j in np.argsort(-lift)[:6] if lift[j] > 0],
            "loves": [{**film(i), "delta": r(loved[i], 2), "segment_mean": r(seg_mean[i], 2), "all_mean": r(item_mean[i], 2)}
                      for i in np.argsort(-loved)[:6] if np.isfinite(loved[i])],
            "watches": [{**film(i), "lift": r(watched[i], 2)} for i in np.argsort(-watched)[:6] if np.isfinite(watched[i])],
        })

    # which content representation agrees with rating-based similarity?
    rng = np.random.default_rng(SEED)
    pool_ids = np.array([int(item_ids[i]) for i in pool])
    comparison = {"pool": len(pool), "sample": 0, "overlap_at_10": {}}
    genre_text = [str(meta.at[m, "genres"]).replace("|", " ") for m in pool_ids]
    genre_matrix = TfidfVectorizer(token_pattern=r"[^ ]+").fit_transform(genre_text)
    genre_unit = genre_matrix.toarray().astype(np.float32)
    content = {"genre": genre_unit}
    if genome is not None:
        g_ids, g_rel = genome
        position = {int(m): j for j, m in enumerate(g_ids)}
        keep = np.array([m in position for m in pool_ids])
        if keep.sum() > 200:
            rel = g_rel[[position[m] for m in pool_ids[keep]]].astype(np.float32)
            rel -= rel.mean(axis=0)
            rel /= np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
            content = {"genre": genre_unit[keep], "tag genome": rel}
            vectors, pool_ids = vectors[keep], pool_ids[keep]
    sample = rng.choice(len(pool_ids), size=min(500, len(pool_ids)), replace=False)
    comparison.update(pool=int(len(pool_ids)), sample=int(len(sample)))

    def top10(matrix: np.ndarray, row: int) -> set[int]:
        sims = matrix @ matrix[row] + rng.random(len(matrix)) * 1e-6  # tiny noise breaks genre ties at random
        sims[row] = -np.inf
        return set(np.argpartition(-sims, 10)[:10].tolist())

    collab = [top10(vectors, s) for s in sample]
    for label, matrix in content.items():
        overlaps = [len(top10(matrix, s) & collab[j]) / 10 for j, s in enumerate(sample)]
        comparison["overlap_at_10"][label] = r(float(np.mean(overlaps)), 4)
    comparison["overlap_at_10"]["random"] = r(10 / (len(pool_ids) - 1), 4)

    eligible = n_i >= cfg["neighbour_min"]
    arrays = {"movie_ids": item_ids[eligible].astype(np.int32), "vectors": item_unit[eligible].astype(np.float32)}
    result = {
        "dataset": name, "method": "PureSVD: randomized truncated SVD of the rating matrix after removing global, "
                                   "user and film biases (damping 10)",
        "factors": k, "users": nu, "films": ni,
        "singular_values": [r(v, 2) for v in S],
        "energy_pct": [r(v, 3) for v in (S ** 2 / frob * 100)],
        "map": points, "map_min_ratings": cfg["map_min"],
        "segments": segments,
        "content_vs_collab": comparison,
    }
    return result, arrays


# ---------------------------------------------------------------- tag genome
def load_genome(folder: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    if not (folder / "genome-scores.csv").is_file():
        return None
    tags = pd.read_csv(folder / "genome-tags.csv")
    movie_parts, tag_parts, rel_parts = [], [], []
    for chunk in pd.read_csv(folder / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
                             chunksize=4_000_000):
        movie_parts.append(chunk["movieId"].to_numpy())
        tag_parts.append(chunk["tagId"].to_numpy())
        rel_parts.append(chunk["relevance"].to_numpy())
    movie = np.concatenate(movie_parts)
    tag = np.concatenate(tag_parts)
    rel = np.concatenate(rel_parts)
    movie_ids, m_idx = np.unique(movie, return_inverse=True)
    tag_ids = tags["tagId"].to_numpy()
    t_idx = np.searchsorted(tag_ids, tag)
    matrix = np.zeros((len(movie_ids), len(tag_ids)), dtype=np.float32)
    matrix[m_idx, t_idx] = rel
    return movie_ids.astype(np.int32), matrix, tags


# ---------------------------------------------------------------- posters (computer vision)
def poster_features(path: Path) -> tuple[dict, np.ndarray, list] | None:
    from PIL import Image, ImageFilter
    from sklearn.cluster import KMeans

    try:
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            small = rgb.resize((96, 144), Image.BILINEAR)
    except Exception:
        return None
    px = np.asarray(small, dtype=np.float32)
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255
    red_green = px[..., 0] - px[..., 1]
    yellow_blue = 0.5 * (px[..., 0] + px[..., 1]) - px[..., 2]
    colorfulness = np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2) + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    edges = np.asarray(small.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255
    luminance_hist = np.histogram(gray, bins=32, range=(0, 1))[0] / gray.size
    nonzero = luminance_hist[luminance_hist > 0]
    features = {
        "brightness": float(gray.mean()), "contrast": float(gray.std()), "saturation": float(hsv[..., 1].mean()),
        "colorfulness": float(colorfulness), "edge_density": float((edges > 0.25).mean()),
        "warmth": float(((px[..., 0] - px[..., 2]) / 255).mean()), "dark_share": float((gray < 0.2).mean()),
        "entropy": float(-(nonzero * np.log2(nonzero)).sum()), "aspect_ratio": float(width / height) if height else 0.0,
    }
    h = np.minimum((hsv[..., 0] * 12).astype(int), 11)
    s = np.minimum((hsv[..., 1] * 3).astype(int), 2)
    v = np.minimum((hsv[..., 2] * 3).astype(int), 2)
    histogram = np.bincount((h * 9 + s * 3 + v).ravel(), minlength=108).astype(np.float32) / (96 * 144)
    pixels = px[::2, ::2].reshape(-1, 3)
    km = KMeans(n_clusters=5, n_init=2, random_state=SEED).fit(pixels)
    shares = np.bincount(km.labels_, minlength=5) / len(pixels)
    palette = [("#%02x%02x%02x" % tuple(int(c) for c in km.cluster_centers_[j]), float(shares[j])) for j in np.argsort(-shares)]
    return features, histogram, palette


def analyze_posters() -> None:
    from scipy.stats import pearsonr
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    files: dict[int, Path] = {}
    for folder in POSTER_DIRS:
        if folder.is_dir():
            for path in folder.iterdir():
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"} and path.stem.isdigit():
                    files.setdefault(int(path.stem), path)
    if len(files) < 30:
        log(f"posters: only {len(files)} images found; run download_data.py --posters 1000 first")
        return
    source = next((OUT / d / "movies.csv" for d in ("ml-25m", "ml-32m", "ml-latest-small") if (OUT / d / "movies.csv").is_file()), None)
    if source is None:
        log("posters: run the dataset analysis first (it provides titles, genres and ratings)")
        return
    meta = pd.read_csv(source).set_index("movieId")
    log(f"posters: extracting features from {len(files):,} images (ratings from {source.parent.name})")
    rows, hists, palettes = [], [], {}
    for movie_id, path in sorted(files.items()):
        if movie_id not in meta.index:
            continue
        extracted = poster_features(path)
        if extracted is None:
            continue
        features, histogram, palette = extracted
        film = meta.loc[movie_id]
        rows.append({"movieId": movie_id, "title": film["title"], "year": film["year"], "genres": film["genres"],
                     "n": int(film["n"]), "mean": film["mean"], "file": path.name, **features,
                     "palette": " ".join(f"{c}:{s:.3f}" for c, s in palette)})
        hists.append(histogram)
        palettes[movie_id] = palette
    table = pd.DataFrame(rows)
    hist = np.vstack(hists)
    scalars = table[POSTER_FEATURES].to_numpy(dtype=np.float64)
    full = np.hstack([scalars, np.sqrt(hist)])  # sqrt = Hellinger geometry for histograms
    log(f"posters: {len(table):,} analysed")

    # genre profiles and palettes
    genre_lists = table["genres"].fillna("").str.split("|")
    genre_counts = pd.Series([g for gs in genre_lists for g in gs if g and g not in (NO_GENRE, "IMAX")]).value_counts()  # IMAX is a format
    genres = [g for g, c in genre_counts.items() if c >= 15]
    profiles = []
    for g in genres:
        mask = genre_lists.map(lambda gs: g in gs).to_numpy()
        part = table.loc[mask, POSTER_FEATURES]
        colours, weights = [], []
        for movie_id in table.loc[mask, "movieId"]:
            for colour, share in palettes[int(movie_id)]:
                colours.append([int(colour[i:i + 2], 16) for i in (1, 3, 5)])
                weights.append(share)
        km = KMeans(n_clusters=6, n_init=3, random_state=SEED).fit(np.array(colours), sample_weight=np.array(weights))
        w = np.bincount(km.labels_, weights=weights, minlength=6)
        palette = [{"color": "#%02x%02x%02x" % tuple(int(c) for c in km.cluster_centers_[j]), "share": r(w[j] / w.sum(), 3)}
                   for j in np.argsort(-w)]
        profiles.append({"genre": g, "posters": int(mask.sum()), "palette": palette,
                         "features": {f: {"mean": r(part[f].mean(), 4), "ci": r(1.96 * part[f].std() / np.sqrt(len(part)), 4)}
                                      for f in POSTER_FEATURES}})
    overall = {f: {"mean": r(table[f].mean(), 4), "std": r(table[f].std(), 4)} for f in POSTER_FEATURES}

    # decades
    decades = []
    with_year = table.dropna(subset=["year"]).assign(decade=lambda d: (d["year"] // 10 * 10).astype(int))
    for decade, part in with_year.groupby("decade"):
        if len(part) >= 8:
            decades.append({"decade": int(decade), "posters": int(len(part)),
                            **{f: r(part[f].mean(), 4) for f in POSTER_FEATURES}})

    # do visual features relate to ratings or popularity?
    rated = table[table["n"] >= 20]
    correlations = []
    for f in POSTER_FEATURES:
        rm, pm = pearsonr(rated[f], rated["mean"])
        rn, pn = pearsonr(rated[f], np.log10(rated["n"]))
        correlations.append({"feature": f, "r_mean": r(rm), "p_mean": float(pm), "r_popularity": r(rn),
                             "p_popularity": float(pn)})

    # can a model guess genre from the poster?
    experiment = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for g in genres:
        y = genre_lists.map(lambda gs: g in gs).to_numpy().astype(int)
        if y.sum() < 40 or (len(y) - y.sum()) < 40:
            continue
        model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
        auc_scalar = cross_val_score(model(), scalars, y, cv=cv, scoring="roc_auc")
        auc_full = cross_val_score(model(), full, y, cv=cv, scoring="roc_auc")
        fitted = model().fit(scalars, y)
        coefficients = fitted[-1].coef_[0]
        order = np.argsort(-np.abs(coefficients))[:3]
        experiment.append({"genre": g, "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
                           "auc_scalar": r(auc_scalar.mean()), "auc_scalar_sd": r(auc_scalar.std()),
                           "auc_full": r(auc_full.mean()), "auc_full_sd": r(auc_full.std()),
                           "drivers": [{"feature": POSTER_FEATURES[j], "weight": r(coefficients[j], 3)} for j in order]})
    experiment.sort(key=lambda e: -max(e["auc_full"], e["auc_scalar"]))

    # map of posters by visual features
    standard = StandardScaler().fit_transform(full)
    centred = standard - standard.mean(axis=0)
    _, _, pcs = np.linalg.svd(centred, full_matrices=False)
    pca = centred @ pcs[:2].T
    tsne = TSNE(n_components=2, perplexity=min(30, len(table) // 4), init="pca", random_state=SEED,
                learning_rate="auto").fit_transform(standard)
    points = [{"id": int(row.movieId), "t": row.title, "c": palettes[int(row.movieId)][0][0],
               "b": r(row.brightness, 3), "p": [r(pca[i, 0], 3), r(pca[i, 1], 3)], "s": [r(tsne[i, 0], 2), r(tsne[i, 1], 2)]}
              for i, row in enumerate(table.itertuples())]

    folder = OUT / "posters"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "posters.csv", index=False)
    unit = standard / np.maximum(np.linalg.norm(standard, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(folder / "posters.npz", movie_ids=table["movieId"].to_numpy(dtype=np.int32),
                        vectors=unit.astype(np.float32))
    write_json(folder / "posters.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "posters": int(len(table)),
        "ratings_source": source.parent.name, "features": POSTER_FEATURES, "overall": overall,
        "genres": profiles, "decades": decades, "correlations": correlations, "genre_prediction": experiment,
        "map": points,
        "examples": {f: {"low": table.nsmallest(6, f)[["movieId", "title"]].to_dict("records"),
                         "high": table.nlargest(6, f)[["movieId", "title"]].to_dict("records")} for f in POSTER_FEATURES},
    })
    log(f"posters: wrote {folder.relative_to(ROOT)}")


# ---------------------------------------------------------------- driver
def analyze_dataset(name: str) -> None:
    cfg = DATASETS[name]
    folder = cfg["dir"]
    if not (folder / "ratings.csv").is_file():
        log(f"{name}: not downloaded (python download_data.py --dataset {name})")
        return
    started = time.time()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{name}: loading ratings")
    ratings = load_ratings(folder)
    movies = load_movies(folder)
    log(f"{name}: {len(ratings['user']):,} ratings loaded; computing overview")
    summary, stats = overview(name, cfg, ratings, movies)
    stats.to_csv(out / "movies.csv", index=False, float_format="%.4f")
    genome = load_genome(folder)
    genome_pair = None
    if genome is not None:
        movie_ids, matrix, tags = genome
        np.savez_compressed(out / "genome.npz", movie_ids=movie_ids, tag_ids=tags["tagId"].to_numpy(np.int32),
                            tags=tags["tag"].to_numpy(dtype=str), relevance=matrix.astype(np.float16))
        summary["genome"] = {"films": int(len(movie_ids)), "tags": int(len(tags)),
                             "mean_relevance": r(float(matrix.mean()))}
        genome_pair = (movie_ids, matrix)
        log(f"{name}: tag genome {matrix.shape[0]:,} films x {matrix.shape[1]:,} tags")
    write_json(out / "summary.json", summary)
    log(f"{name}: latent structure")
    result, arrays = latent(name, cfg, ratings, movies, stats, genome_pair)
    np.savez_compressed(out / "latent.npz", **arrays)
    write_json(out / "latent.json", result)
    del ratings, genome, genome_pair
    gc.collect()
    log(f"{name}: done in {time.time() - started:.0f}s -> {out.relative_to(ROOT)}")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Analyse one dataset (default: all found).")
    parser.add_argument("--posters-only", action="store_true", help="Only run the poster analysis.")
    parser.add_argument("--skip-posters", action="store_true")
    args = parser.parse_args()
    if not args.posters_only:
        for name in ([args.dataset] if args.dataset else list(DATASETS)):
            analyze_dataset(name)
    if not args.skip_posters:
        analyze_posters()


if __name__ == "__main__":
    main()
