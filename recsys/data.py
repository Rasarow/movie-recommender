import numpy as np
import pandas as pd
from .config import RAW


def load_data():
    if not (RAW / "ratings.csv").exists():
        raise FileNotFoundError("Dataset missing. Run: python prepare.py")
    movies = pd.read_csv(RAW / "movies.csv").sort_values("movieId").reset_index(drop=True)
    ratings = pd.read_csv(RAW / "ratings.csv")
    for frame, columns in (
        (movies, ("movieId", "title", "genres")),
        (ratings, ("userId", "movieId", "rating", "timestamp")),
    ):
        if not set(columns).issubset(frame.columns) or frame[list(columns)].isna().any().any():
            raise ValueError("Dataset schema has missing columns or values.")
    if movies.movieId.duplicated().any() or ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate movie IDs or user/movie ratings require explicit cleaning.")
    if not ratings.rating.between(0.5, 5.0).all():
        raise ValueError("Ratings must lie between 0.5 and 5.0.")
    if not np.isfinite(ratings[["rating", "timestamp"]].to_numpy()).all():
        raise ValueError("Invalid numeric values in ratings.")
    if not ratings.movieId.isin(movies.movieId).all():
        raise ValueError("A rating refers to a missing movie.")
    return (movies, ratings)


def chronological_split(ratings, train_fraction=0.8):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between zero and one.")
    ordered = ratings.sort_values(["timestamp", "userId", "movieId"], kind="stable")
    cutoff_index = max(0, min(len(ordered) - 2, int(len(ordered) * train_fraction) - 1))
    cutoff = int(ordered.iloc[cutoff_index].timestamp)
    train = ordered[ordered.timestamp <= cutoff].copy()
    test = ordered[ordered.timestamp > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Tied timestamps prevent a nonempty chronological split.")
    assert int(train.timestamp.max()) < int(test.timestamp.min())
    return (train, test, cutoff)
