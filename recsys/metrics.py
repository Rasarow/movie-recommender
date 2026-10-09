import math
import numpy as np


def ranking_metrics(recommendations, relevant, k):
    if k < 1 or not relevant:
        raise ValueError("Ranking evaluation needs positive k and nonempty relevant items.")
    if len(recommendations) != len(set(recommendations)):
        raise ValueError("Recommendations contain duplicate items.")
    ranked = recommendations[:k]
    hits = [1 if item in relevant else 0 for item in ranked]
    dcg = sum((hit / math.log2(rank + 2) for rank, hit in enumerate(hits)))
    ideal = sum((1 / math.log2(rank + 2) for rank in range(min(k, len(relevant)))))
    return {
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(hits) / len(relevant),
        "ndcg_at_k": dcg / ideal,
        "hit_rate_at_k": float(any(hits)),
    }


def genre_diversity(features, indices):
    selected = features[indices]
    selected = selected[np.asarray(selected.getnnz(axis=1)).ravel() > 0]
    if selected.shape[0] < 2:
        return 0.0
    similarity = (selected @ selected.T).toarray()
    upper = np.triu_indices(selected.shape[0], 1)
    return float(np.mean(1 - np.clip(similarity[upper], 0, 1)))


def paired_bootstrap_difference(values, baseline, seed=42, repetitions=1000):
    if len(values) != len(baseline) or not len(values):
        raise ValueError("Paired bootstrap requires matching nonempty user samples.")
    difference = np.asarray(values, dtype=float) - np.asarray(baseline, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(difference, size=(repetitions, len(difference)), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": round(float(difference.mean()), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }
