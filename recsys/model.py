import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .config import (
    ARTIFACTS,
    METHODS,
    POSITIVE_RATING,
    NEIGHBORS,
    SHRINKAGE,
    MIN_COMMON,
    PRIOR_STRENGTH,
    DATA_SHA256,
)
from .utils import save_json


def genre_tokens(value):
    return [token for token in str(value).split("|") if token != "(no genres listed)"]


class MovieModel:

    def _index(self):
        self.movie_ids = self.movies.movieId.to_numpy(dtype=np.int64)
        self.movie_index = {int(value): index for index, value in enumerate(self.movie_ids)}
        self.user_index = {int(value): index for index, value in enumerate(self.user_ids)}
        self.genres = sorted(
            {genre for value in self.movies.genres for genre in genre_tokens(value)}
        )

    @classmethod
    def fit(
        cls, movies, train, cutoff, neighbors=NEIGHBORS, shrinkage=SHRINKAGE, min_common=MIN_COMMON
    ):
        if neighbors < 1 or shrinkage < 0 or min_common < 1:
            raise ValueError("Invalid item-neighborhood parameters.")
        started = time.perf_counter()
        self = cls()
        self.movies = (
            movies[movies.movieId.isin(train.movieId)].sort_values("movieId").reset_index(drop=True)
        )
        self.user_ids = np.sort(train.userId.unique()).astype(np.int64)
        self._index()
        rows = train.userId.map(self.user_index).to_numpy()
        columns = train.movieId.map(self.movie_index).to_numpy()
        shape = (len(self.user_ids), len(self.movies))
        self.seen = sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32), (rows, columns)), shape=shape
        )
        positive = train.rating.to_numpy() >= POSITIVE_RATING
        weights = (train.rating.to_numpy()[positive] - 3.5).astype(np.float32)
        self.preferences = sparse.csr_matrix(
            (weights, (rows[positive], columns[positive])), shape=shape
        )
        likes = self.preferences.copy()
        likes.data[:] = 1.0
        support = np.asarray(likes.sum(axis=0)).ravel()
        common = (likes.T @ likes).tocsr()
        common.setdiag(0)
        common.eliminate_zeros()
        graph_rows, graph_columns, graph_scores = ([], [], [])
        for item in range(len(self.movies)):
            start, end = common.indptr[item : item + 2]
            candidates = common.indices[start:end]
            counts = common.data[start:end]
            enough = counts >= min_common
            candidates, counts = (candidates[enough], counts[enough])
            if not len(candidates) or support[item] == 0:
                continue
            scores = counts / np.sqrt(support[item] * support[candidates])
            scores *= counts / (counts + shrinkage)
            order = np.lexsort((self.movie_ids[candidates], -scores))[:neighbors]
            graph_rows.extend([item] * len(order))
            graph_columns.extend(candidates[order].tolist())
            graph_scores.extend(scores[order].tolist())
        self.graph = sparse.csr_matrix(
            (np.asarray(graph_scores, dtype=np.float32), (graph_rows, graph_columns)),
            shape=(len(self.movies), len(self.movies)),
        )
        vectorizer = TfidfVectorizer(analyzer=genre_tokens, dtype=np.float32, norm="l2")
        self.features = vectorizer.fit_transform(self.movies.genres).tocsr()
        self.feature_names = vectorizer.get_feature_names_out().tolist()
        grouped = train.groupby("movieId").rating.agg(["count", "mean"])
        self.counts = self.movies.movieId.map(grouped["count"]).to_numpy(dtype=np.int64)
        mean_ratings = self.movies.movieId.map(grouped["mean"]).to_numpy(dtype=np.float64)
        global_mean = float(train.rating.mean())
        self.bayesian_scores = (self.counts * mean_ratings + PRIOR_STRENGTH * global_mean) / (
            self.counts + PRIOR_STRENGTH
        )
        self.metadata = {
            "model_version": f"movielens-{DATA_SHA256[:8]}-cutoff-{cutoff}-k{neighbors}-s{shrinkage}-c{min_common}",
            "cutoff_timestamp": int(cutoff),
            "fit_seconds": round(time.perf_counter() - started, 4),
            "train_ratings": len(train),
            "train_users": len(self.user_ids),
            "catalog_movies": len(self.movies),
            "neighbors": neighbors,
            "shrinkage": shrinkage,
            "min_common_users": min_common,
            "positive_rating_threshold": POSITIVE_RATING,
            "bayesian_prior_strength": PRIOR_STRENGTH,
            "global_mean_rating": global_mean,
            "graph_edges": self.graph.nnz,
            "feature_names": self.feature_names,
            "catalog_policy": "Only films with at least one training rating; titles/tags are not recommendation features.",
            "preference_policy": "Ratings >=4.0 are likes, weighted by rating-3.5. All rated films are excluded from recommendations.",
        }
        return self

    def save(self, directory=ARTIFACTS):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.movies.to_csv(directory / "catalog.csv", index=False)
        for name in ("seen", "preferences", "features", "graph"):
            sparse.save_npz(directory / (name + ".npz"), getattr(self, name))
        np.savez_compressed(
            directory / "statistics.npz",
            user_ids=self.user_ids,
            counts=self.counts,
            bayesian_scores=self.bayesian_scores,
        )
        save_json(self.metadata, directory / "metadata.json")

    @classmethod
    def load(cls, directory=ARTIFACTS):
        directory = Path(directory)
        if not (directory / "metadata.json").exists():
            raise FileNotFoundError("Fitted model missing. Run: python fit.py")
        self = cls()
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.movies = pd.read_csv(directory / "catalog.csv")
        for name in ("seen", "preferences", "features", "graph"):
            setattr(self, name, sparse.load_npz(directory / (name + ".npz")).tocsr())
        with np.load(directory / "statistics.npz", allow_pickle=False) as values:
            self.user_ids = values["user_ids"]
            self.counts = values["counts"]
            self.bayesian_scores = values["bayesian_scores"]
        self.feature_names = self.metadata["feature_names"]
        self._index()
        return self

    def user_vector(self, user_id, liked_movie_ids=()):
        position = self.user_index.get(int(user_id)) if user_id is not None else None
        if position is None:
            preferences = sparse.csr_matrix((1, len(self.movies)), dtype=np.float32)
            seen = set()
        else:
            preferences = self.preferences.getrow(position).copy()
            seen = set(self.seen.getrow(position).indices.tolist())
        extras = sorted(set((int(value) for value in liked_movie_ids)))
        unknown = [value for value in extras if value not in self.movie_index]
        if unknown:
            raise ValueError(f"Liked movie IDs are outside the fitted catalog: {unknown}")
        if extras:
            indices = [self.movie_index[value] for value in extras]
            preferences += sparse.csr_matrix(
                (np.ones(len(indices), dtype=np.float32), ([0] * len(indices), indices)),
                shape=preferences.shape,
            )
            seen.update(indices)
        return (preferences, seen, position is not None)

    def infer_genre(self, preferences):
        totals = {genre: 0.0 for genre in self.genres}
        for index, weight in zip(preferences.indices, preferences.data):
            for genre in genre_tokens(self.movies.iloc[index].genres):
                totals[genre] += float(weight)
        return (
            sorted(totals, key=lambda genre: (-totals[genre], genre))[0]
            if any(totals.values())
            else None
        )

    def recommend(
        self,
        method,
        user_id=None,
        top_k=10,
        liked_movie_ids=(),
        preferred_genre=None,
        blocked_movie_ids=(),
    ):
        if method not in METHODS:
            raise ValueError(f"Method must be one of {METHODS}.")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100.")
        preferences, seen, known_user = self.user_vector(user_id, liked_movie_ids)
        preferred = None
        if preferred_genre:
            lookup = {genre.casefold(): genre for genre in self.genres}
            preferred = lookup.get(preferred_genre.casefold())
            if preferred is None:
                raise ValueError(f"Unknown genre. Available genres: {', '.join(self.genres)}")
        selected_genre = preferred or self.infer_genre(preferences)
        scores = np.zeros(len(self.movies), dtype=np.float64)
        signal = np.zeros(len(self.movies), dtype=bool)
        if method == "content":
            profile = preferences @ self.features
            if profile.nnz == 0 and preferred:
                feature_index = self.feature_names.index(preferred)
                profile = sparse.csr_matrix(([1.0], ([0], [feature_index])), shape=profile.shape)
            if profile.nnz:
                scores = (self.features @ normalize(profile).T).toarray().ravel()
                signal = scores > 0
        elif method == "item_cf" and preferences.nnz:
            scores = (preferences @ self.graph).toarray().ravel()
            signal = scores > 0
        elif method == "genre" and selected_genre:
            signal = np.array(
                [selected_genre in genre_tokens(value) for value in self.movies.genres]
            )
            signal &= self.counts >= 5
            scores = self.bayesian_scores.copy()
        elif method == "popular":
            scores = self.counts.astype(float)
            signal[:] = True
        banned = seen | {
            self.movie_index[int(value)]
            for value in blocked_movie_ids
            if int(value) in self.movie_index
        }
        eligible = np.array(
            [index for index in range(len(self.movies)) if index not in banned], dtype=int
        )
        personalized = eligible[signal[eligible]]
        fallback = eligible[~signal[eligible]]
        personalized_order = np.lexsort(
            (
                self.movie_ids[personalized],
                -self.bayesian_scores[personalized],
                -self.counts[personalized],
                -scores[personalized],
            )
        )
        fallback_order = np.lexsort(
            (self.movie_ids[fallback], -self.bayesian_scores[fallback], -self.counts[fallback])
        )
        ranked = np.concatenate([personalized[personalized_order], fallback[fallback_order]])[
            :top_k
        ]
        recommendations = []
        for index in ranked:
            movie = self.movies.iloc[index]
            is_fallback = not signal[index]
            source = "popularity_fallback" if is_fallback else method
            if source in ("popular", "popularity_fallback"):
                reason = f"Rated by {self.counts[index]} distinct training users."
                score_type = "training_rating_count"
                score = float(self.counts[index])
            elif source == "content":
                reason = "Genre similarity to your liked-film profile or selected genre."
                score_type, score = ("genre_cosine_similarity", float(scores[index]))
            elif source == "genre":
                reason = f"High shrinkage-adjusted mean rating in {selected_genre}."
                score_type, score = ("bayesian_global_rating", float(scores[index]))
            else:
                reason = "Similarity in the co-like patterns of historical training users."
                score_type, score = ("weighted_neighbor_similarity_sum", float(scores[index]))
            recommendations.append(
                {
                    "movie_id": int(movie.movieId),
                    "title": movie.title,
                    "genres": genre_tokens(movie.genres),
                    "score": round(score, 6),
                    "score_type": score_type,
                    "source": source,
                    "reason": reason,
                    "training_rating_count": int(self.counts[index]),
                }
            )
        return {
            "method": method,
            "user_id": user_id,
            "known_training_user": known_user,
            "selected_genre": selected_genre,
            "model_version": self.metadata["model_version"],
            "requested_k": top_k,
            "returned_k": len(recommendations),
            "fallback_item_count": sum(
                (item["source"] == "popularity_fallback" for item in recommendations)
            ),
            "recommendations": recommendations,
        }
