"""Strict query-level metrics and paired bootstrap uncertainty."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import AbstractSet, Iterable, Mapping, Sequence

import numpy as np


def deduplicate(ranking: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ranking))


def recall_at_k(ranking: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    if not gold:
        raise ValueError("Recall is undefined for an empty gold set")
    unique = deduplicate(ranking)[:k]
    return len(set(unique) & set(gold)) / len(gold)


def ceiling_normalized_recall_at_k(
    ranking: Sequence[str], gold: AbstractSet[str], k: int
) -> float:
    """Hits divided by the maximum possible hits at cutoff k."""

    if not gold:
        raise ValueError("Recall is undefined for an empty gold set")
    unique = deduplicate(ranking)[:k]
    hits = len(set(unique) & set(gold))
    return hits / min(k, len(gold))


def ndcg_at_k(ranking: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    if not gold:
        raise ValueError("nDCG is undefined for an empty gold set")
    unique = deduplicate(ranking)[:k]
    dcg = sum(
        1.0 / log2(rank + 2.0)
        for rank, doc_id in enumerate(unique)
        if doc_id in gold
    )
    ideal_count = min(k, len(gold))
    idcg = sum(1.0 / log2(rank + 2.0) for rank in range(ideal_count))
    return dcg / idcg


def set_metrics(candidates: AbstractSet[str], gold: AbstractSet[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("Set metrics require a non-empty gold set")
    candidate_set = set(candidates)
    gold_set = set(gold)
    intersection = len(candidate_set & gold_set)
    precision = intersection / len(candidate_set) if candidate_set else 0.0
    recall = intersection / len(gold_set)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    union = len(candidate_set | gold_set)
    jaccard = intersection / union if union else 1.0
    return {
        "candidate_precision": precision,
        "candidate_recall": recall,
        "candidate_f1": f1,
        "candidate_jaccard": jaccard,
    }


def evaluate_ranking(
    ranking: Sequence[str],
    gold: AbstractSet[str],
    *,
    candidates: AbstractSet[str] | None = None,
) -> dict[str, float | int]:
    unique = deduplicate(ranking)
    candidate_set = set(unique if candidates is None else candidates)
    values: dict[str, float | int] = {
        "retrieved_size": len(unique),
        "candidate_size": len(candidate_set),
        "recall_at_5": recall_at_k(unique, gold, 5),
        "recall_at_20": recall_at_k(unique, gold, 20),
        "recall_at_100": recall_at_k(unique, gold, 100),
        "ceiling_normalized_recall_at_100": ceiling_normalized_recall_at_k(
            unique, gold, 100
        ),
        "ndcg_at_5": ndcg_at_k(unique, gold, 5),
        "ndcg_at_20": ndcg_at_k(unique, gold, 20),
        "ndcg_at_100": ndcg_at_k(unique, gold, 100),
    }
    values.update(set_metrics(candidate_set, gold))
    return values


def gold_size_bin(size: int) -> str:
    if size <= 2:
        return "01-02"
    if size <= 5:
        return "03-05"
    if size <= 20:
        return "06-20"
    if size <= 50:
        return "21-50"
    if size <= 100:
        return "51-100"
    return "101-200"


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    probability_positive: float
    samples: int
    seed: int


def paired_bootstrap_delta(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    strata: Sequence[str] | None = None,
    samples: int = 10_000,
    seed: int = 20260804,
) -> BootstrapResult:
    """Paired query bootstrap for treatment minus baseline.

    If strata are supplied, each bootstrap draw resamples within every stratum and
    preserves the observed stratum sizes.
    """

    baseline_array = np.asarray(baseline, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    if baseline_array.shape != treatment_array.shape or baseline_array.ndim != 1:
        raise ValueError("baseline and treatment must be equal-length 1-D arrays")
    if baseline_array.size == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    if samples <= 0:
        raise ValueError("samples must be positive")
    delta = treatment_array - baseline_array
    estimate = float(np.mean(delta))
    rng = np.random.default_rng(seed)

    if strata is None:
        groups = [np.arange(delta.size)]
    else:
        if len(strata) != delta.size:
            raise ValueError("strata must match the score arrays")
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(strata):
            grouped[str(label)].append(index)
        groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]

    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        total = 0.0
        count = 0
        for indices in groups:
            sampled = rng.choice(indices, size=indices.size, replace=True)
            total += float(np.sum(delta[sampled]))
            count += indices.size
        draws[draw] = total / count

    low, high = np.quantile(draws, (0.025, 0.975))
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        probability_positive=float(np.mean(draws > 0.0)),
        samples=samples,
        seed=seed,
    )
