"""Rating-pattern and optional poster-image analysis for the MovieLens project.

Ratings are measured from the included MovieLens data. Poster analysis runs only
when a local poster directory is supplied (one image per movie, named
<movieId>.<extension>).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visual_embeddings import MODEL_ID, extract_resnet18_embeddings, load_embedding_archive, poster_fingerprint


RATING_LABELS = {
    0.5: "Hated it",
    1.0: "Hated it",
    1.5: "Didn't like it",
    2.0: "Didn't like it",
    2.5: "It's OK",
    3.0: "It's OK",
    3.5: "Liked it",
    4.0: "Liked it",
    4.5: "Loved it",
    5.0: "Loved it",
}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings, movies


def rating_summary(ratings: pd.DataFrame, movies: pd.DataFrame, prior_votes: int) -> dict[str, object]:
    global_mean = float(ratings["rating"].mean())
    distribution = (
        ratings["rating"].value_counts(normalize=True).sort_index().rename("share").rename_axis("rating")
    )
    sentiment = ratings.assign(sentiment=ratings["rating"].map(RATING_LABELS)).groupby("sentiment").size()
    sentiment = sentiment.reindex(["Loved it", "Liked it", "It's OK", "Didn't like it", "Hated it"], fill_value=0)
    by_movie = ratings.groupby("movieId")["rating"].agg(["count", "mean", "std"])
    by_movie["bayesian_rating"] = (
        by_movie["count"] * by_movie["mean"] + prior_votes * global_mean
    ) / (by_movie["count"] + prior_votes)
    by_movie["standard_error"] = by_movie["std"] / np.sqrt(by_movie["count"])
    by_movie["confidence"] = pd.cut(
        by_movie["count"], bins=[-1, 49, 499, np.inf], labels=["Low", "Medium", "High"]
    ).astype(str)
    movie_summary = by_movie.join(movies.set_index("movieId")["title"]).reset_index()
    movie_summary = movie_summary.sort_values(["bayesian_rating", "count"], ascending=False)

    genre_rows = movies.assign(genre=movies["genres"].str.split("|")).explode("genre")
    genre_ratings = ratings.merge(genre_rows[["movieId", "genre"]], on="movieId")
    genre_summary = genre_ratings.groupby("genre")["rating"].agg(["count", "mean"]).sort_values("mean", ascending=False)

    user_summary = ratings.groupby("userId")["rating"].agg(["count", "mean"])
    extreme_share = ratings.assign(extreme=ratings["rating"].isin([0.5, 1.0, 4.5, 5.0])).groupby("userId")["extreme"].mean()
    user_summary["extreme_rating_share"] = extreme_share
    user_summary["rating_style"] = np.select(
        [user_summary["extreme_rating_share"] >= 0.7, user_summary["extreme_rating_share"] <= 0.3],
        ["Mostly extreme", "Mostly moderate"],
        default="Mixed",
    )
    monthly = ratings.set_index("rated_at").resample("MS")["rating"].agg(["count", "mean"]).dropna()
    monthly.index = monthly.index.strftime("%Y-%m")

    return {
        "global_mean": global_mean,
        "distribution": distribution,
        "sentiment": sentiment.rename("rating_count"),
        "movies": movie_summary,
        "genres": genre_summary,
        "users": user_summary,
        "monthly": monthly,
    }


def save_rating_outputs(ratings: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["movies"].to_csv(output_dir / "movie_rating_summary.csv", index=False)
    summary["genres"].to_csv(output_dir / "genre_rating_summary.csv")
    summary["users"].to_csv(output_dir / "user_rating_patterns.csv")
    summary["monthly"].to_csv(output_dir / "monthly_rating_patterns.csv")
    summary["sentiment"].to_csv(output_dir / "sentiment_tag_summary.csv")

    ax = summary["distribution"].mul(100).plot(kind="bar", color="#5470c6", figsize=(8, 4))
    ax.set(title="MovieLens rating distribution", xlabel="Star rating", ylabel="Share of ratings (%)")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "rating_distribution.png", dpi=160)
    plt.close(ax.figure)

    genre_plot = summary["genres"].sort_values("mean").tail(15)
    ax = genre_plot["mean"].plot(kind="barh", color="#91cc75", figsize=(8, 5))
    ax.set(title="Average rating by genre (genres can overlap)", xlabel="Average rating", ylabel="Genre")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "genre_average_rating.png", dpi=160)
    plt.close(ax.figure)

    monthly = summary["monthly"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["mean"], color="#5470c6")
    ax.set(title="Monthly average rating by rating timestamp", xlabel="Month (UTC)", ylabel="Average rating")
    ax.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_average_rating.png", dpi=160)
    plt.close(fig)

    global_mean = summary["global_mean"]
    lines = [
        "# MovieLens rating-pattern results",
        "",
        f"- Ratings analyzed: {len(ratings):,}",
        f"- Users: {ratings['userId'].nunique():,}",
        f"- Movies with ratings: {ratings['movieId'].nunique():,}",
        f"- Overall mean rating: {global_mean:.3f} / 5",
        f"- Timestamp range (UTC): {ratings['rated_at'].min().date()} to {ratings['rated_at'].max().date()}",
        "",
        "The supplied data are MovieLens community ratings, not Moctale data. Rating timestamps",
        "record when MovieLens ratings were submitted; they do not identify movie release dates",
        "or prove when a movie was watched. See the CSV files for the full rating, genre, and",
        "user summaries. Bayesian movie means shrink low-vote titles toward the dataset mean;",
        "the prior strength is configurable with `--prior-votes` (default: 50).",
    ]
    (output_dir / "rating_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def poster_features(image_path: Path) -> dict[str, float | int]:
    """Extract lightweight poster quality/color features without pretrained weights."""
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        small = image.resize((128, 192))
        gray = small.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_variance = float(ImageStat.Stat(edges).var[0])
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
        hist, _ = np.histogramdd(
            hsv.reshape(-1, 3), bins=(8, 4, 4), range=((0, 256), (0, 256), (0, 256))
        )
        hist = hist.ravel()
        hist /= max(float(np.linalg.norm(hist)), 1e-12)
        luminance = np.asarray(gray, dtype=np.float32)
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else 0.0,
            "brightness": float(luminance.mean() / 255),
            "colorfulness": float(np.std(hsv[:, :, 1]) / 255),
            "edge_variance": edge_variance,
            "histogram": hist,
        }


def analyze_posters(
    movies: pd.DataFrame,
    poster_dir: Path,
    output_dir: Path,
    query_movie_id: int | None,
    embedding_batch_size: int = 24,
) -> None:
    poster_paths = {}
    for path in poster_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                movie_id = int(path.stem)
            except ValueError:
                continue
            poster_paths[movie_id] = path

    rows, histograms = [], {}
    for movie_id, path in poster_paths.items():
        try:
            features = poster_features(path)
            histograms[movie_id] = features.pop("histogram")
            rows.append({"movieId": movie_id, "poster_file": str(path), **features})
        except Exception as exc:
            rows.append({"movieId": movie_id, "poster_file": str(path), "error": str(exc)})

    poster_frame = pd.DataFrame(rows)
    if poster_frame.empty:
        print(f"No supported poster images found in {poster_dir}")
        return
    poster_frame = poster_frame.merge(movies[["movieId", "title"]], on="movieId", how="left")
    if not {"brightness", "colorfulness", "edge_variance"}.issubset(poster_frame.columns):
        poster_frame.to_csv(output_dir / "poster_quality_features.csv", index=False)
        print("Poster files were found, but none could be read as supported images.")
        return
    valid = poster_frame.dropna(subset=["brightness", "colorfulness", "edge_variance"])
    if valid.empty:
        poster_frame.to_csv(output_dir / "poster_quality_features.csv", index=False)
        return

    # Relative indicators only: image resolution/edge activity are not a licensing,
    # safety, authenticity, or perceptual-quality verdict.
    valid = valid.copy()
    valid["resolution_ok"] = (valid["width"] >= 300) & (valid["height"] >= 450)
    valid["low_detail_flag"] = valid["edge_variance"] < valid["edge_variance"].quantile(0.05)
    valid["quality_review_flag"] = ~valid["resolution_ok"] | valid["low_detail_flag"]
    poster_frame = poster_frame.merge(
        valid[["movieId", "resolution_ok", "low_detail_flag", "quality_review_flag"]],
        on="movieId", how="left",
    )
    poster_frame.to_csv(output_dir / "poster_quality_features.csv", index=False)

    if histograms:
        ids = list(histograms)
        vectors = np.vstack([histograms[i] for i in ids])
        movie_index = movies.set_index("movieId")
        query_id = query_movie_id if query_movie_id is not None else ids[0]
        if query_id not in histograms:
            print(f"No readable poster found for query movieId={query_id}; visual matches were skipped.")
            return
        similarities = vectors @ vectors[ids.index(query_id)]
        nearest = [i for i in np.argsort(similarities)[::-1] if ids[i] != query_id][:10]
        matched = [
            {"query_movieId": query_id, "movieId": ids[i], "similarity": float(similarities[i]),
             "title": movie_index.at[ids[i], "title"] if ids[i] in movie_index.index else "Unknown"}
            for i in nearest
        ]
        pd.DataFrame(matched).to_csv(output_dir / "visually_similar_movies.csv", index=False)
        print(f"Visual similarity results saved for movieId={query_id}.")
    print(f"Poster feature analysis covered {len(valid):,} readable poster(s).")

    # The CNN uses the pooled 512-D representation from a pretrained ResNet-18.
    # It is intentionally separate from the lightweight color/edge report above.
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_file = output_dir / "poster_visual_embeddings.npz"
    poster_items = sorted(poster_paths.items())
    fingerprint = poster_fingerprint(poster_items)
    cached_vectors, cached_model, cached_fingerprint = load_embedding_archive(embedding_file)

    if cached_fingerprint == fingerprint and cached_model == MODEL_ID and cached_vectors:
        embedding_ids = list(cached_vectors)
        embedding_matrix = np.vstack([cached_vectors[movie_id] for movie_id in embedding_ids])
        print(f"Loaded cached {MODEL_ID} embeddings for {len(embedding_ids):,} poster(s).")
        embedding_failures: list[tuple[int, str]] = []
    else:
        try:
            embedding_ids, embedding_matrix, embedding_failures = extract_resnet18_embeddings(
                poster_items, batch_size=embedding_batch_size
            )
        except RuntimeError as exc:
            print(f"Deep visual embeddings skipped: {exc}")
            return
        if embedding_ids:
            np.savez_compressed(
                embedding_file,
                movie_ids=np.asarray(embedding_ids, dtype=np.int64),
                embeddings=embedding_matrix.astype(np.float32),
                model_id=np.asarray(MODEL_ID),
                fingerprint=np.asarray(fingerprint),
            )

    if embedding_failures:
        pd.DataFrame(embedding_failures, columns=["movieId", "error"]).to_csv(
            output_dir / "poster_embedding_errors.csv", index=False
        )
    if not embedding_ids:
        print("No poster embeddings could be extracted.")
        return

    movie_index = movies.set_index("movieId")
    query_id = query_movie_id if query_movie_id is not None else embedding_ids[0]
    if query_id not in set(embedding_ids):
        print(f"No deep embedding found for query movieId={query_id}; visual neighbors were skipped.")
        return
    query_vector = embedding_matrix[embedding_ids.index(query_id)]
    similarities = embedding_matrix @ query_vector
    nearest = [i for i in np.argsort(similarities)[::-1] if embedding_ids[i] != query_id][:10]
    matched = [
        {
            "query_movieId": query_id,
            "movieId": embedding_ids[i],
            "similarity": float(similarities[i]),
            "title": movie_index.at[embedding_ids[i], "title"] if embedding_ids[i] in movie_index.index else "Unknown",
        }
        for i in nearest
    ]
    pd.DataFrame(matched).to_csv(output_dir / "visually_similar_movies_resnet18.csv", index=False)
    print(f"ResNet-18 visual similarity results saved for movieId={query_id}.")


def calculate_weighted_ratings(ratings: pd.DataFrame, movies: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    user_counts = ratings.groupby("userId").size()
    max_count = user_counts.max()
    activity_weight = np.clip(np.log2(user_counts) / np.log2(max_count), 0.0, 1.0)
    
    user_stds = ratings.groupby("userId")["rating"].std().fillna(0.0)
    max_std = user_stds.max()
    if max_std > 0:
        consistency_weight = 1.0 - (user_stds / max_std) * 0.3
    else:
        consistency_weight = pd.Series(1.0, index=user_stds.index)
        
    min_ts = ratings["timestamp"].min()
    max_ts = ratings["timestamp"].max()
    if max_ts > min_ts:
        recency_weight = 0.7 + 0.3 * ((ratings["timestamp"] - min_ts) / (max_ts - min_ts))
    else:
        recency_weight = pd.Series(1.0, index=ratings.index)
        
    act_w = ratings["userId"].map(activity_weight)
    cons_w = ratings["userId"].map(consistency_weight)
    ratings_weight = act_w * cons_w * recency_weight
    
    weighted_ratings = ratings.assign(weight=ratings_weight)
    
    def w_avg(df):
        w_sum = df["weight"].sum()
        if w_sum > 0:
            return (df["rating"] * df["weight"]).sum() / w_sum
        return df["rating"].mean()
        
    weighted_mean = weighted_ratings.groupby("movieId").apply(w_avg).rename("weighted_mean")
    
    movie_summary = summary["movies"].set_index("movieId")
    res = movie_summary.join(weighted_mean).reset_index()
    res = res.rename(columns={"mean": "raw_mean", "count": "vote_count"})
    cols = ["movieId", "title", "raw_mean", "weighted_mean", "bayesian_rating", "vote_count", "confidence"]
    res[cols].to_csv(output_dir / "weighted_movie_ratings.csv", index=False)
    
    top20 = res.sort_values("vote_count", ascending=False).head(20).sort_values("raw_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top20))
    width = 0.35
    ax.barh(y - width/2, top20["raw_mean"], width, label="Raw Mean", color="#5470c6")
    ax.barh(y + width/2, top20["weighted_mean"], width, label="Weighted Mean", color="#91cc75")
    ax.set_yticks(y)
    ax.set_yticklabels(top20["title"])
    ax.legend()
    ax.set_title("Top 20 Most Voted Movies: Raw vs Weighted Rating")
    fig.tight_layout()
    fig.savefig(output_dir / "weighted_vs_raw_comparison.png", dpi=160)
    plt.close(fig)


def detect_rating_anomalies(ratings: pd.DataFrame, movies: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    users = summary["users"]
    global_user_mean = users["mean"].mean()
    global_user_std = users["mean"].std()
    suspicious_users = users[
        (np.abs(users["mean"] - global_user_mean) > 2 * global_user_std) &
        (users["extreme_rating_share"] > 0.8)
    ]
    
    global_dist = summary["distribution"].values + 1e-9
    global_dist /= global_dist.sum()
    
    movies_counts = ratings.groupby(["movieId", "rating"]).size().unstack(fill_value=0)
    movies_counts = movies_counts.reindex(columns=summary["distribution"].index, fill_value=0)
    
    suspicious_movies_ids = []
    for mid, row in movies_counts.iterrows():
        count = row.sum()
        if count >= 10:
            dist = row.values + 1e-9
            dist /= dist.sum()
            kl = np.sum(dist * np.log(dist / global_dist))
            if kl > 1.0:
                suspicious_movies_ids.append(mid)
    
    monthly_counts = ratings.assign(month=ratings["rated_at"].dt.tz_localize(None).dt.to_period("M")).groupby(["movieId", "month"]).size()
    avg_monthly = monthly_counts.groupby("movieId").mean()
    max_monthly = monthly_counts.groupby("movieId").max()
    
    spike_movies_ids = max_monthly[(max_monthly > 3 * avg_monthly) & (max_monthly >= 5)].index.tolist()
    
    anomaly_records = []
    for uid in suspicious_users.index:
        anomaly_records.append({"entity_type": "user", "entity_id": uid, "reason": "High dev & extreme ratings"})
    for mid in suspicious_movies_ids:
        anomaly_records.append({"entity_type": "movie", "entity_id": mid, "reason": "Anomalous rating distribution"})
    for mid in spike_movies_ids:
        anomaly_records.append({"entity_type": "movie", "entity_id": mid, "reason": "Temporal spike detected"})
        
    anomaly_df = pd.DataFrame(anomaly_records)
    if anomaly_df.empty:
        anomaly_df = pd.DataFrame(columns=["entity_type", "entity_id", "reason"])
    anomaly_df.to_csv(output_dir / "anomaly_report.csv", index=False)
    
    md = [
        "# Rating Anomaly Summary",
        f"- Suspicious users detected: {len(suspicious_users)}",
        f"- Movies with anomalous rating distributions: {len(suspicious_movies_ids)}",
        f"- Movies with temporal rating spikes: {len(spike_movies_ids)}"
    ]
    (output_dir / "anomaly_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def generate_confidence_display(ratings: pd.DataFrame, movies: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    movies_df = summary["movies"].copy()
    movies_df["ci_95_lower"] = movies_df["mean"] - 1.96 * movies_df["standard_error"]
    movies_df["ci_95_upper"] = movies_df["mean"] + 1.96 * movies_df["standard_error"]
    
    def get_tier(count):
        if count < 10: return "⚠️ Few reviews"
        if count < 50: return "Limited data"
        if count < 500: return "Community rated"
        return "Highly rated"
        
    movies_df["display_tier"] = movies_df["count"].apply(get_tier)
    movies_df.to_csv(output_dir / "confidence_display.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing ratings.csv and movies.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moctale"), help="Directory for reports, tables, and charts")
    parser.add_argument("--prior-votes", type=int, default=50, help="Prior vote strength used to shrink movie averages")
    parser.add_argument("--posters", type=Path, help="Optional folder of local posters named <movieId>.jpg/.png/.webp")
    parser.add_argument("--poster-query", type=int, help="MovieLens movieId to use as the poster similarity query")
    parser.add_argument("--embedding-batch-size", type=int, default=24, help="Images per ResNet-18 inference batch")
    args = parser.parse_args()
    if args.prior_votes < 0:
        parser.error("--prior-votes must be non-negative")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")

    ratings, movies = load_data(args.data_dir)
    summary = rating_summary(ratings, movies, args.prior_votes)
    save_rating_outputs(ratings, summary, args.output_dir)
    
    calculate_weighted_ratings(ratings, movies, summary, args.output_dir)
    detect_rating_anomalies(ratings, movies, summary, args.output_dir)
    generate_confidence_display(ratings, movies, summary, args.output_dir)
    
    print(f"Overall mean rating: {summary['global_mean']:.3f} / 5")
    print(f"Rating analysis saved to {args.output_dir}")
    if args.posters:
        analyze_posters(movies, args.posters, args.output_dir, args.poster_query, args.embedding_batch_size)


if __name__ == "__main__":
    main()
