"""Offline evaluation behind the project report (Data -> Report in the app).

Writes outputs/report/metrics.json:
  1. rating prediction: 5-fold cross-validated RMSE / MAE for baselines, user-based
     KNN under four similarity metrics, and three matrix-factorization models
  2. latent-factor sweep: SVD test RMSE as the number of factors grows
  3. neighbourhood-size sweep for KNN (pearson_baseline)
  4. top-10 ranking quality: precision@10 / recall@10 ("liked" = rated >= 4.0)
  5. what the hybrid score and the MMR re-rank each change versus plain SVD
     top-10 lists: catalog coverage, genre diversity, popularity

  python evaluate.py            # ~5 minutes on a laptop
  python evaluate.py --quick    # 3 folds, fewer users; for a smoke test

Item-based KNN is left out: a 9,742 x 9,742 similarity matrix needs ~750 MB.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import NMF, SVD, BaselineOnly, Dataset, KNNBasic, KNNWithMeans, NormalPredictor, Reader, SVDpp, accuracy
from surprise import __version__ as surprise_version
from surprise.model_selection import KFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "report" / "metrics.json"
SEED = 24
# The app's production SVD settings (recommend.py get_svd_model).
APP_SVD = dict(n_factors=100, n_epochs=50, lr_all=0.01, reg_all=0.1, random_state=SEED)
LIKED = 4.0
TOP_N = 10


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def rating_prediction(data, folds: int, include_svdpp: bool) -> list[dict]:
    user_knn = lambda name, cls=KNNBasic: cls(k=40, sim_options={"name": name, "user_based": True}, verbose=False)
    models = [
        ("Random (NormalPredictor)", "baseline", "—", NormalPredictor()),
        ("Global + user + item bias (BaselineOnly)", "baseline", "—", BaselineOnly(verbose=False)),
        ("User KNN · cosine", "neighbourhood", "cosine", user_knn("cosine")),
        ("User KNN · mean squared difference", "neighbourhood", "msd", user_knn("msd")),
        ("User KNN · Pearson", "neighbourhood", "pearson", user_knn("pearson")),
        ("User KNN with means · Pearson-baseline", "neighbourhood", "pearson_baseline", user_knn("pearson_baseline", KNNWithMeans)),
        ("NMF (non-negative MF)", "matrix factorization", "—", NMF(random_state=SEED)),
        ("SVD · library defaults", "matrix factorization", "—", SVD(random_state=SEED)),
        ("SVD · tuned (used in the app)", "matrix factorization", "—", SVD(**APP_SVD)),
    ]
    if include_svdpp:
        models.append(("SVD++ (implicit feedback)", "matrix factorization", "—", SVDpp(n_factors=20, n_epochs=20, random_state=SEED)))
    results = []
    kfold = KFold(n_splits=folds, random_state=SEED, shuffle=True)
    for name, family, similarity, algo in models:
        log(f"cross-validating {name}")
        scores = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=kfold, verbose=False)
        results.append({
            "model": name,
            "family": family,
            "similarity": similarity,
            "rmse": round(float(np.mean(scores["test_rmse"])), 4),
            "rmse_std": round(float(np.std(scores["test_rmse"])), 4),
            "mae": round(float(np.mean(scores["test_mae"])), 4),
            "fit_seconds": round(float(np.mean(scores["fit_time"])), 2),
        })
    return results


def factor_sweep(trainset, testset) -> list[dict]:
    rows = []
    for factors in (2, 5, 10, 25, 50, 100, 200):
        algo = SVD(**{**APP_SVD, "n_factors": factors})
        started = time.time()
        algo.fit(trainset)
        rmse = accuracy.rmse(algo.test(testset), verbose=False)
        train_rmse = accuracy.rmse(algo.test(trainset.build_testset()), verbose=False)
        rows.append({"factors": factors, "test_rmse": round(rmse, 4), "train_rmse": round(train_rmse, 4),
                     "fit_seconds": round(time.time() - started, 2)})
        log(f"  SVD {factors:>3} factors: test RMSE {rmse:.4f}, train RMSE {train_rmse:.4f}")
    return rows


def k_sweep(trainset, testset) -> list[dict]:
    rows = []
    for k in (5, 10, 20, 40, 80):
        algo = KNNWithMeans(k=k, sim_options={"name": "pearson_baseline", "user_based": True}, verbose=False)
        algo.fit(trainset)
        rows.append({"k": k, "test_rmse": round(accuracy.rmse(algo.test(testset), verbose=False), 4)})
    return rows


def precision_recall_at_k(predictions, k: int = TOP_N, threshold: float = LIKED) -> tuple[float, float]:
    """Surprise FAQ recipe: rank each user's test items by prediction; relevant = true rating >= threshold."""
    by_user = defaultdict(list)
    for p in predictions:
        by_user[p.uid].append((p.est, p.r_ui))
    precisions, recalls = [], []
    for ratings in by_user.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        relevant = sum(true >= threshold for _, true in ratings)
        top = ratings[:k]
        recommended_relevant = sum(true >= threshold and est >= threshold for est, true in top)
        recommended = sum(est >= threshold for est, _ in top)
        precisions.append(recommended_relevant / recommended if recommended else 0.0)
        recalls.append(recommended_relevant / relevant if relevant else 0.0)
    return float(np.mean(precisions)), float(np.mean(recalls))


def ranking(trainset, testset) -> list[dict]:
    models = [
        ("Random (NormalPredictor)", NormalPredictor()),
        ("BaselineOnly", BaselineOnly(verbose=False)),
        ("User KNN with means · Pearson-baseline", KNNWithMeans(k=40, sim_options={"name": "pearson_baseline"}, verbose=False)),
        ("SVD · tuned (used in the app)", SVD(**APP_SVD)),
    ]
    rows = []
    for name, algo in models:
        algo.fit(trainset)
        precision, recall = precision_recall_at_k(algo.test(testset))
        rows.append({"model": name, "precision_at_10": round(precision, 4), "recall_at_10": round(recall, 4)})
        log(f"  {name}: P@10 {precision:.3f}, R@10 {recall:.3f}")
    return rows


def hybrid_vs_svd(user_count: int) -> dict:
    """Compare plain SVD top-10 lists with the app's hybrid (SVD + genre) + MMR lists."""
    sys.path.insert(0, str(ROOT))
    from recommend import MovieRecommender

    recommender = MovieRecommender(data_dir=ROOT / "data", cache_dir=ROOT / "outputs", poster_dir=ROOT / "licensed_posters")
    svd = recommender.get_svd_model()
    trainset = svd.trainset
    movies = recommender.movies_df
    counts = recommender.ratings_df["movieId"].value_counts()
    tfidf = recommender.tfidf_matrix
    index_of = recommender._movie_id_to_idx

    rng = np.random.default_rng(SEED)
    users = sorted(rng.choice(recommender.ratings_df["userId"].unique(), size=user_count, replace=False).tolist())
    all_ids = movies["movieId"].astype(int).to_numpy()
    raw_to_inner = {int(raw): inner for raw, inner in trainset._raw2inner_id_items.items()}

    def list_stats(lists: list[list[int]]) -> dict:
        diversity, popularity = [], []
        for ids in lists:
            rows = tfidf[[index_of[i] for i in ids]]
            sim = cosine_similarity(rows)
            diversity.append(1 - float(sim[np.triu_indices(len(ids), 1)].mean()))
            popularity.append(float(np.median([counts.get(i, 0) for i in ids])))
        unique = len({i for ids in lists for i in ids})
        return {
            "coverage_pct": round(unique / len(all_ids) * 100, 2),
            "unique_titles": unique,
            "genre_diversity": round(float(np.mean(diversity)), 3),
            "median_ratings_per_title": round(float(np.median(popularity)), 1),
        }

    mmr = recommender._mmr_order
    no_rerank = lambda ids, relevance, top_n, weight=0.82: mmr(ids, relevance, top_n, weight=1.0)  # weight 1 = relevance only

    svd_lists, hybrid_lists, mmr_lists = [], [], []
    for user in users:
        inner_user = trainset.to_inner_uid(user)
        seen = {int(trainset.to_raw_iid(i)) for i, _ in trainset.ur[inner_user]}
        candidates = [m for m in all_ids if m not in seen and m in raw_to_inner]
        inner_items = np.array([raw_to_inner[m] for m in candidates])
        scores = svd.trainset.global_mean + svd.bu[inner_user] + svd.bi[inner_items] + svd.qi[inner_items] @ svd.pu[inner_user]
        svd_lists.append([int(candidates[i]) for i in np.argsort(-scores)[:TOP_N]])
        recommender._mmr_order = no_rerank
        hybrid_lists.append([int(m["movieId"]) for m in recommender.recommend_for_user(int(user), top_n=TOP_N)])
        recommender._mmr_order = mmr
        mmr_lists.append([int(m["movieId"]) for m in recommender.recommend_for_user(int(user), top_n=TOP_N)])
    log(f"  compared {len(users)} users")
    return {"users": len(users), "svd": list_stats(svd_lists), "hybrid": list_stats(hybrid_lists),
            "hybrid_mmr": list_stats(mmr_lists)}


def dataset_summary(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    users, items = ratings["userId"].nunique(), ratings["movieId"].nunique()
    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()
    genres = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(movies["genres"].str.replace("|", " ", regex=False))
    return {
        "ratings": int(len(ratings)),
        "users": int(users),
        "rated_movies": int(items),
        "catalog_movies": int(len(movies)),
        "sparsity_pct": round((1 - len(ratings) / (users * items)) * 100, 2),
        "ratings_per_user_median": int(per_user.median()),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_movie_median": int(per_item.median()),
        "movies_with_one_rating_pct": round(float((per_item == 1).mean() * 100), 1),
        "mean_rating": round(float(ratings["rating"].mean()), 3),
        "liked_share_pct": round(float((ratings["rating"] >= LIKED).mean() * 100), 1),
        "genre_terms": len(genres.vocabulary_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="3 folds and 20 users: a fast smoke test.")
    parser.add_argument("--svdpp", action="store_true", help="Also cross-validate SVD++ (slow: several minutes).")
    args = parser.parse_args()

    started = time.time()
    ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
    movies = pd.read_csv(ROOT / "data" / "movies.csv")
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5.0)))
    trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "environment": {"python": platform.python_version(), "surprise": surprise_version},
              "settings": {"folds": 3 if args.quick else 5, "seed": SEED, "liked_threshold": LIKED, "top_n": TOP_N,
                           "app_svd": {k: v for k, v in APP_SVD.items() if k != "random_state"}},
              "dataset": dataset_summary(ratings, movies)}
    log("1/5 rating prediction (cross-validation)")
    report["rating_prediction"] = rating_prediction(data, report["settings"]["folds"], args.svdpp)
    log("2/5 latent-factor sweep (80/20 split)")
    report["factor_sweep"] = factor_sweep(trainset, testset)
    log("3/5 neighbourhood-size sweep")
    report["k_sweep"] = k_sweep(trainset, testset)
    log("4/5 top-10 ranking")
    report["ranking"] = ranking(trainset, testset)
    log("5/5 hybrid + MMR vs plain SVD")
    report["hybrid_vs_svd"] = hybrid_vs_svd(20 if args.quick else 100)
    report["runtime_seconds"] = round(time.time() - started, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {OUT.relative_to(ROOT)} in {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
