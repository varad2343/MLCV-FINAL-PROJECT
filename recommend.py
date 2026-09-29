"""MovieLens Recommendation System Runner with Moctale Rating Meter.

Provides:
- Moctale Rating System: Categorical consensus (Perfection, Go for it, Timepass, Skip)
  along with Moctale Meter Percentages and Vibe Chart distributions.
- Collaborative Filtering (SVD Matrix Factorization)
- Hybrid SVD, genre TF-IDF, and optional ResNet-18 poster embeddings
- Bayesian Mean Shrinkage
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from visual_embeddings import extract_poster_visual_features, load_embedding_archive

# Moctale Meter Categories and Thresholds
MOCTALE_CATEGORIES = ("Perfection", "Go for it", "Timepass", "Skip")
MOCTALE_EMOJIS = {
    "Perfection": "🌟 Perfection",
    "Go for it": "👍 Go for it",
    "Timepass": "⏳ Timepass",
    "Skip": "⏭️ Skip",
}


def rating_to_moctale(r: float) -> str:
    """Map numerical star rating to Moctale category."""
    if r >= 4.5:
        return "Perfection"
    if r >= 3.5:
        return "Go for it"
    if r >= 2.5:
        return "Timepass"
    return "Skip"


class MovieRecommender:
    def __init__(
        self,
        data_dir: Path = Path("data"),
        cache_dir: Path = Path("outputs"),
        poster_dir: Path = Path("licensed_posters"),
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.poster_dir = Path(poster_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._poster_sources = self._load_poster_sources()
        
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.links_df: Optional[pd.DataFrame] = None
        self.moctale_stats: Optional[pd.DataFrame] = None
        self.svd_model: Optional[SVD] = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self._movie_id_to_idx = {}
        self._movie_idx_to_id = {}
        self._poster_embeddings: dict[int, np.ndarray] = {}
        self._poster_visual_features: dict[int, dict[str, float]] = {}
        self._moctale_snapshot: dict = {}
        self._cold_start_model: ExtraTreesRegressor | None = None

        self.load_data()
        self.init_moctale_meter()
        self.init_content_based()
        self.init_visual_embeddings()
        self._poster_visual_features = self._load_local_poster_visual_features()

    def _load_poster_sources(self) -> dict[str, dict]:
        source_path = self.data_dir / "open_posters.json"
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_local_poster_visual_features(self) -> dict[int, dict[str, float]]:
        features = {}
        if not self.poster_dir.is_dir():
            return features
        for path in self.poster_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                movie_id = int(path.stem)
                features[movie_id] = extract_poster_visual_features(path)
            except (ValueError, OSError, RuntimeError):
                continue
        return features

    def load_data(self) -> None:
        ratings_path = self.data_dir / "ratings.csv"
        movies_path = self.data_dir / "movies.csv"
        links_path = self.data_dir / "links.csv"

        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError(f"Missing data files in {self.data_dir}")

        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        
        if links_path.exists():
            self.links_df = pd.read_csv(links_path)
            self.movies_df = self.movies_df.merge(self.links_df, on="movieId", how="left")
            self.movies_df["imdbId"] = self.movies_df["imdbId"].apply(
                lambda x: f"{int(x):07d}" if pd.notnull(x) else ""
            )
        else:
            self.movies_df["imdbId"] = ""
            self.movies_df["tmdbId"] = ""

        # Precompute mappings for fast lookup
        for idx, row in self.movies_df.iterrows():
            self._movie_id_to_idx[row["movieId"]] = idx
            self._movie_idx_to_id[idx] = row["movieId"]

    def init_moctale_meter(self) -> None:
        """Compute Moctale meter percentages and dominant verdict for each movie."""
        df = self.ratings_df.copy()
        df["moctale_cat"] = df["rating"].apply(rating_to_moctale)
        
        counts = df.groupby(["movieId", "moctale_cat"]).size().unstack(fill_value=0)
        for col in MOCTALE_CATEGORIES:
            if col not in counts.columns:
                counts[col] = 0

        totals = counts.sum(axis=1)
        pcts = counts.div(totals, axis=0) * 100
        pcts = pcts.round(1)
        pcts["total_votes"] = totals
        pcts["dominant_verdict"] = pcts[list(MOCTALE_CATEGORIES)].idxmax(axis=1)
        self.moctale_stats = pcts
        self.moctale_counts = counts[list(MOCTALE_CATEGORIES)].astype(int)

    def init_visual_embeddings(self) -> None:
        """Load the optional ResNet-18 cache produced by moctale_analysis.py."""
        cache_file = self.cache_dir / "moctale" / "poster_visual_embeddings.npz"
        self._poster_embeddings, _model_id, _fingerprint = load_embedding_archive(cache_file)

    @staticmethod
    def _normalise_title(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(title).casefold())

    def get_poster_url(self, movie_id: int) -> str:
        """Return local art first, then a curated openly licensed Commons image."""
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            if (self.poster_dir / f"{int(movie_id)}{extension}").is_file():
                return f"/posters/{int(movie_id)}"
        source = self._poster_sources.get(str(int(movie_id)), {})
        return str(source.get("image_url", ""))

    def get_poster_attribution(self, movie_id: int) -> dict | None:
        source = self._poster_sources.get(str(int(movie_id)))
        return dict(source) if source else None

    def get_moctale_counts(self, movie_id: int) -> dict[str, int]:
        if movie_id not in self.moctale_counts.index:
            return {category: 0 for category in MOCTALE_CATEGORIES}
        row = self.moctale_counts.loc[movie_id]
        return {category: int(row[category]) for category in MOCTALE_CATEGORIES}

    def _cold_start_feature_rows(
        self,
        movie_ids: list[int],
        genres: list[str] | None = None,
        poster_movie_ids: list[int | None] | None = None,
    ) -> np.ndarray:
        if genres is None:
            indices = [self._movie_id_to_idx[movie_id] for movie_id in movie_ids]
            genre_features = self.tfidf_matrix[indices].toarray().astype(np.float32)
        else:
            genre_features = self.tfidf_vectorizer.transform(genres).toarray().astype(np.float32)

        visual_vectors = np.zeros((len(movie_ids), 512), dtype=np.float32)
        visual_properties = np.zeros((len(movie_ids), 7), dtype=np.float32)
        feature_keys = ("brightness", "contrast", "colorfulness", "edge_complexity", "saturation", "aspect_ratio")
        for index, movie_id in enumerate(movie_ids):
            poster_id = poster_movie_ids[index] if poster_movie_ids else movie_id
            if poster_id is None:
                continue
            poster_id = int(poster_id)
            embedding = self._poster_embeddings.get(poster_id)
            if embedding is not None and len(embedding) == visual_vectors.shape[1]:
                visual_vectors[index] = embedding
                visual_properties[index, 6] = 1.0
            properties = self._poster_visual_features.get(poster_id, {})
            for column, key in enumerate(feature_keys):
                visual_properties[index, column] = float(properties.get(key, 0.0))
            if properties:
                visual_properties[index, 6] = 1.0

        return np.concatenate((genre_features, visual_vectors, visual_properties), axis=1)

    def _ensure_cold_start_model(self) -> None:
        if self._cold_start_model is not None:
            return
        ratings = self.ratings_df[["movieId", "rating"]].copy()
        ratings["category"] = ratings["rating"].map(rating_to_moctale)
        counts = ratings.groupby(["movieId", "category"]).size().unstack(fill_value=0)
        for category in MOCTALE_CATEGORIES:
            if category not in counts.columns:
                counts[category] = 0
        counts = counts[list(MOCTALE_CATEGORIES)]
        proportions = counts.div(counts.sum(axis=1), axis=0).mul(100.0)
        train_ids = [int(movie_id) for movie_id in proportions.index if int(movie_id) in self._movie_id_to_idx]
        if len(train_ids) < 20:
            raise RuntimeError("Not enough rated movies to train the cold-start meter.")

        train_x = self._cold_start_feature_rows(train_ids)
        train_y = proportions.loc[train_ids, list(MOCTALE_CATEGORIES)].to_numpy(dtype=np.float32)
        model = ExtraTreesRegressor(
            n_estimators=48,
            max_depth=18,
            min_samples_leaf=4,
            max_features=0.75,
            n_jobs=-1,
            random_state=24,
        )
        model.fit(train_x, train_y)
        self._cold_start_model = model

    def predict_cold_start_for_metadata(
        self,
        genres: str,
        poster_movie_id: int | None = None,
    ) -> dict:
        """Estimate the Moctale meter from metadata before a film has ratings."""
        self._ensure_cold_start_model()
        feature = self._cold_start_feature_rows(
            [0],
            genres=[str(genres or "")],
            poster_movie_ids=[poster_movie_id],
        )
        raw = np.clip(self._cold_start_model.predict(feature)[0], 0.0, 100.0)
        total = float(raw.sum())
        if total <= 0:
            raw = np.array([25.0, 45.0, 20.0, 10.0], dtype=np.float32)
            total = float(raw.sum())
        percentages = (raw / total * 100.0).round(1)
        percentages[-1] = round(float(percentages[-1] + 100.0 - percentages.sum()), 1)
        return {
            "category_percentages": {
                category: float(value) for category, value in zip(MOCTALE_CATEGORIES, percentages)
            },
            "verdict": MOCTALE_CATEGORIES[int(np.argmax(percentages))],
            "poster_features_used": bool(
                poster_movie_id and int(poster_movie_id) in self._poster_embeddings
            ),
            "source": "Cold-start model · MovieLens historical ratings",
        }

    def _cold_start_for_movie(self, movie_id: int) -> dict | None:
        row = self.movies_df[self.movies_df["movieId"].astype(int) == int(movie_id)]
        if row.empty:
            return None
        movie = row.iloc[0]
        return self.predict_cold_start_for_metadata(
            str(movie.get("genres", "")), poster_movie_id=int(movie_id)
        )

    def init_content_based(self) -> None:
        clean_genres = self.movies_df["genres"].fillna("").str.replace("|", " ", regex=False)
        self.tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_genres)
        snapshot_path = self.data_dir / "moctale_public_snapshot.json"
        if snapshot_path.is_file():
            try:
                self._moctale_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._moctale_snapshot = {}

    def get_svd_model(self, force_retrain: bool = False) -> SVD:
        if self.svd_model is not None and not force_retrain:
            return self.svd_model

        model_cache = self.cache_dir / "svd_model.pkl"
        if model_cache.exists() and not force_retrain:
            try:
                with open(model_cache, "rb") as f:
                    self.svd_model = pickle.load(f)
                return self.svd_model
            except Exception:
                pass

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings_df[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()

        model = SVD(n_factors=100, n_epochs=50, lr_all=0.01, reg_all=0.1, random_state=24)
        model.fit(trainset)
        self.svd_model = model

        try:
            with open(model_cache, "wb") as f:
                pickle.dump(model, f)
        except Exception:
            pass

        return self.svd_model

    def get_movie_moctale_info(self, movie_id: int, predicted_rating: Optional[float] = None) -> dict:
        """Extract Moctale meter breakdown and vibe chart for any movie."""
        if self.moctale_stats is not None and movie_id in self.moctale_stats.index:
            row = self.moctale_stats.loc[movie_id]
            verdict = row["dominant_verdict"]
            perfection = float(row["Perfection"])
            goforit = float(row["Go for it"])
            timepass = float(row["Timepass"])
            skip = float(row["Skip"])
            votes = int(row["total_votes"])
            estimated = False
            meter_source = "MovieLens ratings"
        else:
            prediction = None
            try:
                prediction = self._cold_start_for_movie(movie_id)
            except Exception:
                prediction = None
            if prediction:
                proportions = prediction["category_percentages"]
                perfection = float(proportions["Perfection"])
                goforit = float(proportions["Go for it"])
                timepass = float(proportions["Timepass"])
                skip = float(proportions["Skip"])
                verdict = prediction["verdict"]
                meter_source = prediction["source"]
            else:
                r = predicted_rating if predicted_rating is not None else 3.5
                verdict = rating_to_moctale(r)
                if verdict == "Perfection":
                    perfection, goforit, timepass, skip = 85.0, 15.0, 0.0, 0.0
                elif verdict == "Go for it":
                    perfection, goforit, timepass, skip = 25.0, 60.0, 15.0, 0.0
                elif verdict == "Timepass":
                    perfection, goforit, timepass, skip = 5.0, 25.0, 55.0, 15.0
                else:
                    perfection, goforit, timepass, skip = 0.0, 10.0, 30.0, 60.0
                meter_source = "Predicted rating estimate"
            votes = 0
            estimated = True

        # Use the attributed public snapshot where it matches; otherwise represent
        # the MovieLens genres as an evenly weighted local vibe profile.
        movie_row = self.movies_df[self.movies_df["movieId"] == movie_id]
        vibe_chart = {}
        if not movie_row.empty:
            movie = movie_row.iloc[0]
            genres_str = str(movie["genres"])
            if genres_str and genres_str != "(no genres listed)":
                genres = [g.strip() for g in genres_str.split("|") if g.strip()]
                if genres:
                    share = round(100.0 / len(genres), 1)
                    vibe_chart = {g: share for g in genres}
            snapshot_title = self._normalise_title(self._moctale_snapshot.get("title", ""))
            if snapshot_title and self._normalise_title(movie["title"]) == snapshot_title:
                snapshot_vibe = self._moctale_snapshot.get("vibe_chart_percentages", {})
                if isinstance(snapshot_vibe, dict):
                    vibe_chart = {str(genre): float(pct) for genre, pct in snapshot_vibe.items()}

        return {
            "verdict": verdict,
            "verdict_display": MOCTALE_EMOJIS.get(verdict, verdict),
            "perfection_pct": perfection,
            "goforit_pct": goforit,
            "timepass_pct": timepass,
            "skip_pct": skip,
            "vote_count": votes,
            "vibe_chart": vibe_chart,
            "estimated": estimated,
            "meter_source": meter_source,
        }

    def _taste_seeds(self, user_id: int, visitor_votes: list[dict] | None = None):
        """Build positive and negative movie seeds from MovieLens plus this visitor's votes."""
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        positive: dict[int, float] = {}
        negative: dict[int, float] = {}
        visited = set(int(mid) for mid in user_ratings["movieId"])

        for row in user_ratings.itertuples(index=False):
            movie_id = int(row.movieId)
            rating = float(row.rating)
            if rating >= 4.0:
                positive[movie_id] = (rating - 3.0) / 2.0
            elif rating <= 2.5:
                negative[movie_id] = min(1.0, (3.0 - rating) / 2.5)

        vote_weights = {"Perfection": (1.0, 0.0), "Go for it": (0.75, 0.0), "Timepass": (0.0, 0.3), "Skip": (0.0, 1.0)}
        for vote in visitor_votes or []:
            try:
                movie_id = int(vote["movieId"])
                category = str(vote["vote"])
            except (KeyError, TypeError, ValueError):
                continue
            if category not in vote_weights:
                continue
            visited.add(movie_id)
            positive.pop(movie_id, None)
            negative.pop(movie_id, None)
            positive_weight, negative_weight = vote_weights[category]
            if positive_weight:
                positive[movie_id] = positive_weight
            if negative_weight:
                negative[movie_id] = negative_weight
        return positive, negative, visited

    def _genre_profile_similarity(self, candidate_ids: list[int], positive: dict[int, float], negative: dict[int, float]):
        candidate_indices = [self._movie_id_to_idx.get(movie_id) for movie_id in candidate_ids]
        if any(index is None for index in candidate_indices):
            return None
        candidate_matrix = self.tfidf_matrix[candidate_indices]

        def profile_vector(seeds: dict[int, float]):
            valid = [(self._movie_id_to_idx.get(movie_id), weight) for movie_id, weight in seeds.items()]
            valid = [(index, weight) for index, weight in valid if index is not None and weight > 0]
            if not valid:
                return None
            rows = self.tfidf_matrix[[index for index, _weight in valid]]
            weights = np.asarray([weight for _index, weight in valid], dtype=np.float32)
            return np.asarray(rows.multiply(weights[:, None]).sum(axis=0), dtype=np.float32)

        positive_vector = profile_vector(positive)
        negative_vector = profile_vector(negative)
        if positive_vector is None and negative_vector is None:
            return None
        if positive_vector is None:
            return 1.0 - cosine_similarity(negative_vector, candidate_matrix).ravel()
        scores = cosine_similarity(positive_vector, candidate_matrix).ravel()
        if negative_vector is not None:
            scores -= 0.35 * cosine_similarity(negative_vector, candidate_matrix).ravel()
        return np.clip(scores, 0.0, 1.0)

    def _visual_profile_similarity(self, candidate_ids: list[int], positive: dict[int, float], negative: dict[int, float]):
        def profile_vector(seeds: dict[int, float]):
            vectors = [(self._poster_embeddings.get(movie_id), weight) for movie_id, weight in seeds.items()]
            vectors = [(vector, weight) for vector, weight in vectors if vector is not None and weight > 0]
            if not vectors:
                return None
            matrix = np.vstack([vector for vector, _weight in vectors])
            weights = np.asarray([weight for _vector, weight in vectors], dtype=np.float32)
            profile = np.average(matrix, axis=0, weights=weights)
            norm = float(np.linalg.norm(profile))
            return profile / norm if norm else None

        positive_vector = profile_vector(positive)
        negative_vector = profile_vector(negative)
        if positive_vector is None and negative_vector is None:
            return None
        scores = []
        for movie_id in candidate_ids:
            vector = self._poster_embeddings.get(movie_id)
            if vector is None:
                scores.append(None)
                continue
            score = float(np.dot(positive_vector, vector)) if positive_vector is not None else 0.0
            if negative_vector is not None:
                negative_similarity = float(np.dot(negative_vector, vector))
                score = score - 0.35 * negative_similarity if positive_vector is not None else 1.0 - negative_similarity
            scores.append(float(np.clip(score, 0.0, 1.0)))
        return scores

    @staticmethod
    def _combine_scores(svd_ratings: list[float], genre_scores, visual_scores):
        weights = {"svd": 0.50, "genre": 0.30, "visual": 0.20}
        combined = []
        for index, rating in enumerate(svd_ratings):
            parts = {"svd": float(np.clip((rating - 0.5) / 4.5, 0.0, 1.0)), "genre": None, "visual": None}
            if genre_scores is not None:
                parts["genre"] = float(genre_scores[index])
            if visual_scores is not None:
                parts["visual"] = visual_scores[index]
            available_weight = sum(weights[name] for name, value in parts.items() if value is not None)
            score = sum(weights[name] * value for name, value in parts.items() if value is not None) / available_weight
            combined.append((float(score), parts))
        return combined

    def _mmr_order(self, candidate_ids: list[int], relevance: list[float], top_n: int, weight: float = 0.82):
        """Maximal marginal relevance: trade relevance against similarity to titles already picked.

        Redundancy is poster-embedding cosine when both titles have one, otherwise genre
        TF-IDF cosine, so the re-rank still diversifies when no poster vectors are loaded.
        """
        pool_size = min(len(candidate_ids), max(top_n, min(20, top_n * 4)))
        remaining = sorted(range(len(candidate_ids)), key=lambda index: relevance[index], reverse=True)[:pool_size]
        pool_rows = [self._movie_id_to_idx.get(candidate_ids[index]) for index in remaining]
        genre_rows = {
            index: self.tfidf_matrix[row] for index, row in zip(remaining, pool_rows) if row is not None
        }

        def similarity(a: int, b: int) -> float:
            vector_a = self._poster_embeddings.get(candidate_ids[a])
            vector_b = self._poster_embeddings.get(candidate_ids[b])
            if vector_a is not None and vector_b is not None:
                return float(np.dot(vector_a, vector_b))
            if a in genre_rows and b in genre_rows:  # TF-IDF rows are L2-normalised, so the dot is the cosine
                return float(genre_rows[a].multiply(genre_rows[b]).sum())
            return 0.0

        selected: list[tuple[int, float]] = []
        while remaining and len(selected) < top_n:
            best_index = remaining[0]
            best_score = -float("inf")
            for index in remaining:
                max_similarity = max((similarity(index, chosen) for chosen, _score in selected), default=0.0)
                mmr_score = weight * relevance[index] - (1.0 - weight) * max_similarity
                if mmr_score > best_score:
                    best_index, best_score = index, mmr_score
            selected.append((best_index, float(best_score)))
            remaining.remove(best_index)
        return selected

    def _movie_links(self, row) -> tuple[str, str]:
        imdb_id = str(row.get("imdbId", "") or "")
        imdb_url = f"https://www.imdb.com/title/tt{imdb_id}/" if imdb_id else ""
        return imdb_url, self.get_poster_url(int(row["movieId"]))

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 5,
        visitor_votes: list[dict] | None = None,
    ) -> list[dict]:
        """Rank unseen titles with SVD, genre taste, and optional poster features."""
        if user_id not in self.ratings_df["userId"].values:
            raise ValueError(f"User ID {user_id} not found. Valid IDs: 1 to {self.ratings_df['userId'].max()}")

        svd = self.get_svd_model()
        positive, negative, seen_movie_ids = self._taste_seeds(user_id, visitor_votes)
        candidates = self.movies_df[~self.movies_df["movieId"].astype(int).isin(seen_movie_ids)]
        rows = [row for _, row in candidates.iterrows()]
        candidate_ids = [int(row["movieId"]) for row in rows]
        predicted = [float(svd.predict(user_id, movie_id).est) for movie_id in candidate_ids]
        genre_scores = self._genre_profile_similarity(candidate_ids, positive, negative)
        visual_scores = self._visual_profile_similarity(candidate_ids, positive, negative)
        ranked = self._combine_scores(predicted, genre_scores, visual_scores)
        order = self._mmr_order(candidate_ids, [item[0] for item in ranked], top_n)

        results = []
        for index, mmr_score in order:
            row = rows[index]
            movie_id = candidate_ids[index]
            hybrid_score, breakdown = ranked[index]
            imdb_url, poster_url = self._movie_links(row)
            results.append({
                "movieId": movie_id,
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": round(predicted[index], 2),
                "predicted_moctale": rating_to_moctale(predicted[index]),
                "similarity_score": breakdown["genre"] or 0.0,
                "hybrid_score": round(hybrid_score, 4),
                "mmr_score": round(mmr_score, 4),
                "score_breakdown": breakdown,
                "moctale": self.get_movie_moctale_info(movie_id, predicted_rating=predicted[index]),
                "imdb_url": imdb_url,
                "poster_url": poster_url,
                "poster_attribution": self.get_poster_attribution(movie_id),
            })
        return results

    def find_movie_by_title(self, query: str) -> Optional[dict]:
        query = query.strip()
        exact = self.movies_df[self.movies_df["title"].str.lower() == query.lower()]
        if not exact.empty:
            return exact.iloc[0].to_dict()

        contains = self.movies_df[self.movies_df["title"].str.contains(query, case=False, na=False)]
        if not contains.empty:
            return contains.iloc[0].to_dict()

        return None

    def recommend_by_movie(
        self,
        title_or_id: str | int,
        top_n: int = 5,
        user_id: int = 1,
        visitor_votes: list[dict] | None = None,
    ) -> tuple[dict, list[dict]]:
        """Rank similar titles with a hybrid of SVD, genres, and local poster features."""
        if isinstance(title_or_id, int) or (isinstance(title_or_id, str) and title_or_id.isdigit()):
            movie_id = int(title_or_id)
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if match.empty:
                raise ValueError(f"Movie ID {movie_id} not found.")
            target_movie = match.iloc[0]
        else:
            found = self.find_movie_by_title(str(title_or_id))
            if not found:
                raise ValueError(f"Movie matching '{title_or_id}' was not found in catalog.")
            target_movie = pd.Series(found)

        if user_id not in self.ratings_df["userId"].values:
            raise ValueError(f"User ID {user_id} not found. Valid IDs: 1 to {self.ratings_df['userId'].max()}")

        target_mid = int(target_movie["movieId"])
        positive, negative, _seen = self._taste_seeds(user_id, visitor_votes)
        positive[target_mid] = max(positive.get(target_mid, 0.0), 1.0)
        candidates = self.movies_df[self.movies_df["movieId"].astype(int) != target_mid]
        rows = [row for _, row in candidates.iterrows()]
        candidate_ids = [int(row["movieId"]) for row in rows]
        svd = self.get_svd_model()
        predicted = [float(svd.predict(user_id, mid).est) for mid in candidate_ids]
        genre_scores = self._genre_profile_similarity(candidate_ids, positive, negative)
        visual_scores = self._visual_profile_similarity(candidate_ids, positive, negative)
        ranked = self._combine_scores(predicted, genre_scores, visual_scores)
        order = self._mmr_order(candidate_ids, [item[0] for item in ranked], top_n)

        results = []
        for index, mmr_score in order:
            row = rows[index]
            mid = candidate_ids[index]
            hybrid_score, breakdown = ranked[index]
            imdb_url, poster_url = self._movie_links(row)
            results.append({
                "movieId": mid,
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": round(predicted[index], 2),
                "similarity_score": float(breakdown["genre"] or 0.0),
                "hybrid_score": round(hybrid_score, 4),
                "mmr_score": round(mmr_score, 4),
                "score_breakdown": breakdown,
                "moctale": self.get_movie_moctale_info(mid, predicted[index]),
                "imdb_url": imdb_url,
                "poster_url": poster_url,
                "poster_attribution": self.get_poster_attribution(mid),
            })

        target_imdb, target_poster = self._movie_links(target_movie)
        target_dict = {
            "movieId": target_mid,
            "title": target_movie["title"],
            "genres": target_movie["genres"],
            "moctale": self.get_movie_moctale_info(target_mid),
            "imdb_url": target_imdb,
            "poster_url": target_poster,
            "poster_attribution": self.get_poster_attribution(target_mid),
        }
        return target_dict, results

    def get_top_rated_movies(self, top_n: int = 10, min_ratings: int = 50) -> list[dict]:
        """Bayesian-shrunk top rated movies with Moctale meter distribution."""
        global_mean = float(self.ratings_df["rating"].mean())
        stats = self.ratings_df.groupby("movieId")["rating"].agg(["count", "mean"])
        
        prior = 50
        stats["bayesian_score"] = (stats["count"] * stats["mean"] + prior * global_mean) / (stats["count"] + prior)
        stats = stats[stats["count"] >= min_ratings].sort_values("bayesian_score", ascending=False).head(top_n)

        results = []
        for mid, row in stats.iterrows():
            movie_row = self.movies_df[self.movies_df["movieId"] == mid]
            if not movie_row.empty:
                m = movie_row.iloc[0]
                results.append({
                    "movieId": int(mid),
                    "title": m["title"],
                    "genres": m["genres"],
                    "rating_count": int(row["count"]),
                    "raw_average": round(float(row["mean"]), 2),
                    "bayesian_score": round(float(row["bayesian_score"]), 2),
                    "moctale": self.get_movie_moctale_info(int(mid)),
                    "imdb_url": f"https://www.imdb.com/title/tt{m.get('imdbId', '')}/" if m.get("imdbId") else "",
                    "poster_url": self.get_poster_url(int(mid)),
                })
        return results

    def get_user_profile(self, user_id: int) -> dict:
        """Behavioral, genre, temporal, and available poster signals for one viewer."""
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id].copy()
        if user_ratings.empty:
            raise ValueError(f"User {user_id} not found.")

        mean_rating = float(user_ratings["rating"].mean())
        rating_stddev = float(user_ratings["rating"].std(ddof=0))
        endpoint_share = float(user_ratings["rating"].isin([0.5, 5.0]).mean() * 100.0)
        user_ratings["moctale"] = user_ratings["rating"].apply(rating_to_moctale)
        style_counts = user_ratings["moctale"].value_counts(normalize=True) * 100

        merged = user_ratings.merge(self.movies_df, on="movieId")
        top_favorites = merged.sort_values("rating", ascending=False).head(5)
        genre_counts = (
            merged["genres"].fillna("").str.split("|").explode().loc[lambda values: values.ne("")].value_counts()
        )
        genre_total = max(int(genre_counts.sum()), 1)
        genre_share = [
            {"genre": str(genre), "share": round(float(count) * 100.0 / genre_total, 1)}
            for genre, count in genre_counts.head(6).items()
        ]

        rated_at = pd.to_datetime(user_ratings["timestamp"], unit="s", utc=True, errors="coerce")
        first_rating = rated_at.min()
        last_rating = rated_at.max()
        active_months = max(float((last_rating - first_rating).total_seconds() / (86400 * 30.4375)), 1.0 / 30.4375)
        velocity = float(len(user_ratings) / active_months)
        merged["rated_at"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True, errors="coerce")
        release_year = pd.to_numeric(merged["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
        release_date = pd.to_datetime(release_year.astype("Int64").astype(str) + "-01-01", utc=True, errors="coerce")
        rating_age = ((merged["rated_at"] - release_date).dt.total_seconds() / (86400 * 365.25)).dropna()
        rating_age = rating_age[rating_age >= 0]

        visual_rows = [
            (int(row.movieId), float(row.rating), self._poster_visual_features.get(int(row.movieId)))
            for row in merged.itertuples(index=False)
            if self._poster_visual_features.get(int(row.movieId))
        ]
        liked_visuals = [item[2] for item in visual_rows if item[1] >= 4.0]
        visual_sample = liked_visuals or [item[2] for item in visual_rows]
        if visual_sample:
            brightness = float(np.mean([item["brightness"] for item in visual_sample]))
            saturation = float(np.mean([item["saturation"] for item in visual_sample]))
            brightness_label = "Low-light imagery" if brightness < 0.42 else "Bright imagery" if brightness > 0.62 else "Balanced light"
            color_label = "Muted palettes" if saturation < 0.28 else "Vivid palettes" if saturation > 0.56 else "Moderate color"
            visual_persona = {
                "available": True,
                "poster_count": len(visual_rows),
                "brightness": round(brightness, 3),
                "saturation": round(saturation, 3),
                "brightness_label": brightness_label,
                "color_label": color_label,
                "summary": f"{brightness_label.lower()} with {color_label.lower()} across the posters you rate highest.",
            }
        else:
            visual_persona = {
                "available": False,
                "poster_count": 0,
                "summary": "Add openly licensed poster artwork to measure visual preferences.",
            }

        if endpoint_share >= 30:
            rating_style = "Strong opinions"
        elif endpoint_share <= 10:
            rating_style = "Measured scale"
        else:
            rating_style = "Mixed scale"

        return {
            "userId": user_id,
            "total_ratings": len(user_ratings),
            "average_rating": round(mean_rating, 2),
            "rating_stddev": round(rating_stddev, 3),
            "endpoint_rating_share": round(endpoint_share, 1),
            "rating_style_label": rating_style,
            "submission_velocity_per_month": round(velocity, 1),
            "median_film_age_at_rating_years": round(float(rating_age.median()), 1) if not rating_age.empty else None,
            "genre_share": genre_share,
            "visual_persona": visual_persona,
            "svd_latent_factors": 100,
            "moctale_style": {
                "perfection_pct": round(float(style_counts.get("Perfection", 0.0)), 1),
                "goforit_pct": round(float(style_counts.get("Go for it", 0.0)), 1),
                "timepass_pct": round(float(style_counts.get("Timepass", 0.0)), 1),
                "skip_pct": round(float(style_counts.get("Skip", 0.0)), 1),
            },
            "top_favorites": [
                {
                    "title": r["title"],
                    "rating": r["rating"],
                    "moctale_verdict": MOCTALE_EMOJIS.get(rating_to_moctale(r["rating"]), ""),
                    "genres": r["genres"]
                }
                for _, r in top_favorites.iterrows()
            ]
        }


def print_user_recommendations(recommender: MovieRecommender, user_id: int, top_n: int = 5) -> None:
    profile = recommender.get_user_profile(user_id)
    style = profile["moctale_style"]
    print("=" * 76)
    print(f"🎬 Collaborative Filtering Recommendations for User #{user_id}")
    print("=" * 76)
    print(f"User Profile: {profile['total_ratings']} movies rated | Average Rating: {profile['average_rating']:.2f} / 5.0")
    print(f"Moctale Rating Style: 🌟 {style['perfection_pct']}% Perfection | 👍 {style['goforit_pct']}% Go for it | ⏳ {style['timepass_pct']}% Timepass | ⏭️ {style['skip_pct']}% Skip")
    print("\nTop User Favorites:")
    for m in profile["top_favorites"]:
        print(f"  ★ {m['rating']:.1f}/5.0 [{m['moctale_verdict']}] - {m['title']} ({m['genres']})")
    print("-" * 76)
    print(f"Top {top_n} Unseen Recommendations (SVD Matrix Factorization + Moctale Meter):")
    recs = recommender.recommend_for_user(user_id, top_n=top_n)
    for i, r in enumerate(recs, 1):
        m_info = r["moctale"]
        print(f"  {i}. {r['title']}")
        print(f"     Consensus: {m_info['verdict_display']} | Predicted Rating: {r['predicted_rating']:.2f}/5.0")
        print(f"     Moctale Meter: 🌟 {m_info['perfection_pct']}% Perfection | 👍 {m_info['goforit_pct']}% Go for it | ⏳ {m_info['timepass_pct']}% Timepass | ⏭️ {m_info['skip_pct']}% Skip ({m_info['vote_count']} votes)")
        print(f"     Genres: {r['genres']}")
        if r['imdb_url']:
            print(f"     IMDb: {r['imdb_url']}")
    print("=" * 76 + "\n")


def print_content_recommendations(recommender: MovieRecommender, query: str, top_n: int = 5) -> None:
    target, recs = recommender.recommend_by_movie(query, top_n=top_n)
    t_moctale = target["moctale"]
    print("=" * 76)
    print(f"🎯 Hybrid Movie Recommendations (SVD + Genre + Optional Poster Features)")
    print("=" * 76)
    print(f"Target Movie: {target['title']}")
    print(f"Moctale Consensus: {t_moctale['verdict_display']} (🌟 {t_moctale['perfection_pct']}% Perfection | 👍 {t_moctale['goforit_pct']}% Go for it)")
    if t_moctale["vibe_chart"]:
        vibes = ", ".join([f"{k}: {v}%" for k, v in t_moctale["vibe_chart"].items()])
        print(f"Moctale Vibe Chart: {vibes}")
    if target["imdb_url"]:
        print(f"IMDb: {target['imdb_url']}")
    print("-" * 76)
    print(f"Top {top_n} Similar Movies (hybrid score):")
    for i, r in enumerate(recs, 1):
        m_info = r["moctale"]
        print(f"  {i}. {r['title']} [Similarity: {r['similarity_score']*100:.1f}%]")
        print(f"     Moctale Consensus: {m_info['verdict_display']}")
        print(f"     Meter Breakdown: 🌟 {m_info['perfection_pct']}% Perfection | 👍 {m_info['goforit_pct']}% Go for it | ⏳ {m_info['timepass_pct']}% Timepass | ⏭️ {m_info['skip_pct']}% Skip")
        print(f"     Genres: {r['genres']}")
        if r['imdb_url']:
            print(f"     IMDb: {r['imdb_url']}")
    print("=" * 76 + "\n")


def run_interactive(recommender: MovieRecommender) -> None:
    print("\n" + "=" * 76)
    print("🍿 Welcome to the MovieLens + Moctale Recommendation System")
    print("=" * 76)
    while True:
        print("\nChoose an option:")
        print("  1. Personalized Recommendations for a User ID (Collaborative Filtering + Moctale)")
        print("  2. Similar Movie Recommendations by Title (Content-Based + Moctale Vibe)")
        print("  3. Top Rated Movies in Dataset (Bayesian Weighted + Moctale Consensus)")
        print("  4. Exit")
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            user_input = input("Enter User ID (1 to 610): ").strip()
            if not user_input.isdigit():
                print("⚠️ Invalid User ID.")
                continue
            user_id = int(user_input)
            try:
                print_user_recommendations(recommender, user_id)
            except Exception as e:
                print(f"⚠️ Error: {e}")

        elif choice == "2":
            query = input("Enter movie title (e.g. 'Toy Story', 'Inception', 'Matrix'): ").strip()
            if not query:
                continue
            try:
                print_content_recommendations(recommender, query)
            except Exception as e:
                print(f"⚠️ Error: {e}")

        elif choice == "3":
            print("\n" + "=" * 76)
            print("⭐ Top 10 Movies by Bayesian-Weighted Rating & Moctale Meter:")
            print("=" * 76)
            top_movies = recommender.get_top_rated_movies(10)
            for i, m in enumerate(top_movies, 1):
                m_info = m["moctale"]
                print(f"  {i}. {m['title']} - Score: {m['bayesian_score']:.2f} ({m['rating_count']} reviews)")
                print(f"     Consensus: {m_info['verdict_display']} (🌟 {m_info['perfection_pct']}% Perfection | 👍 {m_info['goforit_pct']}% Go for it)")
                print(f"     Genres: {m['genres']}")
            print("=" * 76)

        elif choice in ("4", "q", "quit", "exit"):
            print("Exiting MovieLens Recommendation System. Enjoy your movies!")
            break
        else:
            print("⚠️ Invalid choice. Please select 1, 2, 3, or 4.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens Recommendation System CLI with Moctale Meter")
    parser.add_argument("--user", type=int, help="User ID (1-610) for collaborative filtering")
    parser.add_argument("--movie", type=str, help="Movie title for content-based recommendations")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to return (default: 5)")
    parser.add_argument("--top-rated", action="store_true", help="Display top rated movies in the dataset")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI mode")
    args = parser.parse_args()

    recommender = MovieRecommender()

    if args.interactive:
        run_interactive(recommender)
    elif args.user:
        print_user_recommendations(recommender, args.user, top_n=args.top)
    elif args.movie:
        print_content_recommendations(recommender, args.movie, top_n=args.top)
    elif args.top_rated:
        top_movies = recommender.get_top_rated_movies(args.top)
        print("\n" + "=" * 76)
        print(f"⭐ Top {args.top} Movies by Bayesian-Weighted Rating & Moctale Meter:")
        print("=" * 76)
        for i, m in enumerate(top_movies, 1):
            m_info = m["moctale"]
            print(f"  {i}. {m['title']} - Score: {m['bayesian_score']:.2f} ({m['rating_count']} reviews)")
            print(f"     Consensus: {m_info['verdict_display']} (🌟 {m_info['perfection_pct']}% Perfection | 👍 {m_info['goforit_pct']}% Go for it)")
            print(f"     Genres: {m['genres']}")
        print("=" * 76 + "\n")
    else:
        print("\n" + "=" * 76)
        print("🎬 MovieLens Recommendation System + Moctale Meter Demo")
        print("=" * 76)
        print("\n1. Testing Collaborative Filtering for User #1:")
        print_user_recommendations(recommender, 1, top_n=args.top)

        print("\n2. Testing Content-Based Filtering with Moctale Vibe for 'Toy Story (1995)':")
        print_content_recommendations(recommender, "Toy Story (1995)", top_n=args.top)

        print("💡 Tip: Run with `--interactive` for interactive mode, or `--help` for options.\n")


if __name__ == "__main__":
    main()
