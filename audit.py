import argparse
import numpy as np
from recsys.config import ROOT
from recsys.data import load_data, chronological_split
from recsys.model import MovieModel
from recsys.utils import block_network, save_json


def rank_of(scores, ids, target_index):
    order = np.lexsort((ids, -scores))
    return int(np.where(order == target_index)[0][0] + 1)


def audit(model=None):
    block_network()
    model = model or MovieModel.load()
    unknown_user = int(model.user_ids.max() + 100000)
    no_history = model.recommend("item_cf", user_id=unknown_user)
    with_genre = model.recommend("genre", user_id=unknown_user, preferred_genre="Sci-Fi")
    with_seeds = model.recommend("content", user_id=unknown_user, liked_movie_ids=[1, 260])
    _, ratings = load_data()
    train, _, _ = chronological_split(ratings)
    target_index = int(np.lexsort((model.movie_ids, model.counts))[0])
    attack_size = int(model.counts.max() + 50)
    injected = model.counts.copy()
    injected[target_index] += attack_size
    history_counts = train.groupby("userId").size()
    trusted_events = train[train.userId.isin(history_counts[history_counts >= 5].index)]
    filtered = (
        model.movies.movieId.map(trusted_events.groupby("movieId").size())
        .fillna(0)
        .to_numpy(dtype=int)
    )
    result = {
        "cold_start": {
            "unknown_user_id": unknown_user,
            "without_preferences": {
                "method": "item_cf",
                "fallback_items": no_history["fallback_item_count"],
                "top_movies": [item["title"] for item in no_history["recommendations"]],
            },
            "with_explicit_genre": {
                "genre": "Sci-Fi",
                "fallback_items": with_genre["fallback_item_count"],
                "top_movies": [item["title"] for item in with_genre["recommendations"]],
            },
            "with_two_liked_films": {
                "seed_movie_ids": [1, 260],
                "fallback_items": with_seeds["fallback_item_count"],
                "top_movies": [item["title"] for item in with_seeds["recommendations"]],
            },
        },
        "popularity_manipulation": {
            "scope": "Synthetic single-film accounts added only to a separate count-vector experiment; real models/test data unchanged.",
            "target_movie_id": int(model.movie_ids[target_index]),
            "target_title": model.movies.iloc[target_index].title,
            "target_initial_rating_count": int(model.counts[target_index]),
            "synthetic_accounts": attack_size,
            "rank_before": rank_of(model.counts, model.movie_ids, target_index),
            "rank_after": rank_of(injected, model.movie_ids, target_index),
            "simple_history_filter": {
                "minimum_rated_films": 5,
                "rank_before": rank_of(filtered, model.movie_ids, target_index),
                "rank_after": rank_of(filtered, model.movie_ids, target_index),
                "excluded_legitimate_training_users": int((history_counts < 5).sum()),
                "excluded_legitimate_training_ratings": int(
                    train.userId.isin(history_counts[history_counts < 5].index).sum()
                ),
                "tradeoff": "Single-rating fake accounts are removed, but legitimate new/sparse accounts are also excluded. Coordinated accounts can build history.",
            },
        },
    }
    save_json(result, ROOT / "results/audit.json")
    print(
        f"Cold-start checks completed. Synthetic target popularity rank: {result['popularity_manipulation']['rank_before']} -> {result['popularity_manipulation']['rank_after']}"
    )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
