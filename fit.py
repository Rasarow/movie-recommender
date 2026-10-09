import argparse
import time
import psutil
from recsys.data import load_data, chronological_split
from recsys.model import MovieModel
from recsys.config import ARTIFACTS, ROOT
from recsys.utils import save_json, block_network


def fit():
    block_network()
    started = time.perf_counter()
    movies, ratings = load_data()
    train, test, cutoff = chronological_split(ratings)
    model = MovieModel.fit(movies, train, cutoff)
    model.save()
    info = {
        "raw_movies": len(movies),
        "raw_users": int(ratings.userId.nunique()),
        "raw_ratings": len(ratings),
        "train_ratings": len(train),
        "test_ratings": len(test),
        "train_max_timestamp": int(train.timestamp.max()),
        "test_min_timestamp": int(test.timestamp.min()),
        "strict_time_separation": bool(train.timestamp.max() < test.timestamp.min()),
        "wall_seconds": round(time.perf_counter() - started, 4),
        "process_rss_after_fit_mb": round(psutil.Process().memory_info().rss / 1024**2, 2),
        "artifact_bytes": sum(
            (path.stat().st_size for path in ARTIFACTS.iterdir() if path.is_file())
        ),
        **model.metadata,
    }
    save_json(info, ROOT / "results/fit_summary.json")
    print(
        f"Fitted {len(model.movies)} films and {len(model.user_ids)} users. Strict chronological split: {info['strict_time_separation']}"
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    fit()
