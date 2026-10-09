import argparse
import importlib.metadata
import time
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import psutil
from recsys.config import ROOT, METHODS, POSITIVE_RATING
from recsys.data import load_data, chronological_split
from recsys.model import MovieModel
from recsys.metrics import ranking_metrics, genre_diversity, paired_bootstrap_difference
from recsys.utils import save_json, block_network


def evaluate():
    block_network()
    started = time.perf_counter()
    model = MovieModel.load()
    _, ratings = load_data()
    train, test, cutoff = chronological_split(ratings)
    if model.metadata["cutoff_timestamp"] != cutoff or model.metadata["train_ratings"] != len(
        train
    ):
        raise RuntimeError("Fitted model and evaluation split disagree. Run fit.py first.")
    positive_test = test[test.rating >= POSITIVE_RATING]
    history_sizes = train.groupby("userId").size().to_dict()
    rows = []
    item_exposure = {method: set() for method in METHODS}
    unreachable_positives = 0
    eligible_positives = 0
    excluded_users = 0
    exposure_probability = (model.counts + 1) / (model.counts.sum() + len(model.movies))
    for user_id, events in positive_test.groupby("userId", sort=True):
        _, seen, _ = model.user_vector(int(user_id))
        unseen_relevant = {
            int(value) for value in events.movieId if model.movie_index.get(int(value)) not in seen
        }
        relevant = unseen_relevant.intersection(model.movie_index)
        unreachable_positives += len(unseen_relevant - relevant)
        eligible_positives += len(relevant)
        if not relevant:
            excluded_users += 1
            continue
        history_size = int(history_sizes.get(user_id, 0))
        cohort = "cold" if history_size == 0 else "sparse" if history_size < 5 else "warm"
        for method in METHODS:
            query_start = time.perf_counter()
            result = model.recommend(method, user_id=int(user_id), top_k=10)
            latency_ms = (time.perf_counter() - query_start) * 1000
            recommendations = [item["movie_id"] for item in result["recommendations"]]
            assert not set(recommendations) & {int(model.movie_ids[index]) for index in seen}
            values = ranking_metrics(recommendations, relevant, 10)
            indices = [model.movie_index[value] for value in recommendations]
            item_exposure[method].update(recommendations)
            rows.append(
                {
                    "user_id": int(user_id),
                    "cohort": cohort,
                    "history_ratings": history_size,
                    "method": method,
                    "relevant_count": len(relevant),
                    "recommended_movie_ids": recommendations,
                    **values,
                    "genre_diversity": genre_diversity(model.features, indices),
                    "historical_novelty_bits": float(
                        np.mean(-np.log2(exposure_probability[indices]))
                    ),
                    "query_latency_ms": latency_ms,
                    "fallback_item_count": result["fallback_item_count"],
                }
            )
    if not rows:
        raise RuntimeError("No eligible users in chronological holdout.")
    summaries = {}
    cohorts = {}
    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "hit_rate_at_k",
        "genre_diversity",
        "historical_novelty_bits",
    )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {
            metric: round(float(np.mean([row[metric] for row in method_rows])), 6)
            for metric in metric_names
        }
        summaries[method].update(
            users=len(method_rows),
            catalog_coverage=round(len(item_exposure[method]) / len(model.movies), 6),
            unique_recommended_movies=len(item_exposure[method]),
            latency_p50_ms=round(
                float(np.quantile([row["query_latency_ms"] for row in method_rows], 0.5)), 4
            ),
            latency_p95_ms=round(
                float(np.quantile([row["query_latency_ms"] for row in method_rows], 0.95)), 4
            ),
            fallback_slot_rate=round(
                sum((row["fallback_item_count"] for row in method_rows)) / (10 * len(method_rows)),
                6,
            ),
        )
        cohorts[method] = {}
        for cohort in ("warm", "sparse", "cold"):
            selected = [row for row in method_rows if row["cohort"] == cohort]
            cohorts[method][cohort] = {
                "users": len(selected),
                **{
                    metric: (
                        round(float(np.mean([row[metric] for row in selected])), 6)
                        if selected
                        else None
                    )
                    for metric in metric_names[:4]
                },
            }
    comparisons = {}
    for cohort in ("all", "warm"):
        reference_rows = [
            row
            for row in rows
            if row["method"] == "popular" and (cohort == "all" or row["cohort"] == cohort)
        ]
        if not reference_rows:
            continue
        comparisons[cohort] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["method"] == method and (cohort == "all" or row["cohort"] == cohort)
            ]
            assert [row["user_id"] for row in selected] == [
                row["user_id"] for row in reference_rows
            ]
            comparisons[cohort][method] = paired_bootstrap_difference(
                [row["ndcg_at_k"] for row in selected], [row["ndcg_at_k"] for row in reference_rows]
            )
    evaluation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "offline_network_blocked": True,
        "model_version": model.metadata["model_version"],
        "k": 10,
        "protocol": "Global 80/20 chronological split; identical train-observed catalog; all unseen candidates, no sampled negatives.",
        "relevance": "Held-out observed ratings >=4.0; targets outside the training catalog are excluded and counted separately.",
        "target_users_with_test_likes": int(positive_test.userId.nunique()),
        "evaluated_users": len([row for row in rows if row["method"] == "popular"]),
        "excluded_users_without_reachable_targets": excluded_users,
        "reachable_positive_targets": eligible_positives,
        "unreachable_positive_targets": unreachable_positives,
        "reachable_target_fraction": round(
            eligible_positives / (eligible_positives + unreachable_positives), 6
        ),
        "summary": summaries,
        "cohort_metrics": cohorts,
        "paired_ndcg_differences_vs_popular": comparisons,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "process_rss_after_evaluation_mb": round(psutil.Process().memory_info().rss / 1024**2, 2),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn")
        },
        "limitations": [
            "Unobserved films are not confirmed dislikes; exposure is not randomized.",
            "MovieLens is a historical educational sample, not current streaming-service traffic.",
            "No CTR, conversion, revenue, watch time or retention has been measured.",
            "Scores are ranking signals, not calibrated probabilities or personalized rating predictions.",
            "Cold users receive popularity fallback without externally supplied preferences.",
            "The bootstrap interval does not remove exposure bias or temporal nonstationarity.",
        ],
    }
    save_json(evaluation, ROOT / "results/evaluation.json")
    print(
        f"Evaluated {evaluation['evaluated_users']} users. Reachable targets: {evaluation['reachable_target_fraction']:.3f}",
        flush=True,
    )
    for method, metrics in summaries.items():
        print(
            f"{method:8s} Precision@10={metrics['precision_at_k']:.4f} Recall@10={metrics['recall_at_k']:.4f} NDCG@10={metrics['ndcg_at_k']:.4f} p95={metrics['latency_p95_ms']:.2f}ms",
            flush=True,
        )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    evaluate()
