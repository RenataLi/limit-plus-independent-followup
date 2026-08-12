"""End-to-end audit, retrieval, composition, and uncertainty analysis."""

from __future__ import annotations

import csv
import gzip
import json
import platform
import time
from collections import Counter, defaultdict
from importlib.metadata import version as package_version
from pathlib import Path
from typing import AbstractSet, Iterable, Mapping, Sequence

import numpy as np

from .bm25 import PAPER_QUICKSORT, TIE_BREAK_MODES, BM25Index, TOKENIZERS
from .data import DatasetBundle, LimitPlusQuery, load_and_audit
from .logic import (
    CompositionResult,
    QueryPlan,
    compose_rankings,
    execute_exact,
    positive_atomic_condition,
)
from .metrics import evaluate_ranking, gold_size_bin, paired_bootstrap_delta


METRIC_COLUMNS = (
    "recall_at_5",
    "recall_at_20",
    "recall_at_100",
    "ceiling_normalized_recall_at_100",
    "ndcg_at_5",
    "ndcg_at_20",
    "ndcg_at_100",
    "candidate_precision",
    "candidate_recall",
    "candidate_f1",
    "candidate_jaccard",
    "retrieved_size",
    "candidate_size",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def _base_row(query: LimitPlusQuery, method: str, depth: int | None) -> dict:
    return {
        "method": method,
        "qid": query.qid,
        "template": query.template,
        "n_atoms": query.plan.n_atoms,
        "gold_size": len(query.gold),
        "gold_size_bin": gold_size_bin(len(query.gold)),
        "atomic_depth": "" if depth is None else depth,
        "final_k": 100,
    }


def _summary_rows(per_query: Sequence[Mapping[str, object]]) -> list[dict]:
    methods = sorted({str(row["method"]) for row in per_query})
    rows: list[dict] = []
    for method in methods:
        method_rows = [row for row in per_query if row["method"] == method]
        group_specs: list[tuple[str, str, list[Mapping[str, object]]]] = [
            ("overall", "all", method_rows)
        ]
        for column, group_type in (("template", "template"), ("gold_size_bin", "gold_size_bin")):
            values = sorted({str(row[column]) for row in method_rows})
            group_specs.extend(
                (group_type, value, [row for row in method_rows if str(row[column]) == value])
                for value in values
            )
        for group_type, group_value, group in group_specs:
            summary: dict[str, object] = {
                "method": method,
                "group_type": group_type,
                "group_value": group_value,
                "n_queries": len(group),
            }
            for metric in METRIC_COLUMNS:
                summary[metric] = float(np.mean([float(row[metric]) for row in group]))
            rows.append(summary)
    return rows


def _ranking_error_counts(
    gold: AbstractSet[str],
    candidates: AbstractSet[str],
    ranking: Sequence[str],
    *,
    final_k: int,
) -> dict[str, int]:
    """Partition misses into candidate, avoidable-ranking, and cutoff loss."""

    if final_k <= 0:
        raise ValueError("final_k must be positive")
    ranked = tuple(ranking)
    if len(ranked) != len(set(ranked)):
        raise AssertionError("Error decomposition requires a duplicate-free ranking")
    if len(ranked) > final_k:
        raise AssertionError("Error decomposition ranking exceeds final_k")

    gold_set = set(gold)
    candidate_set = set(candidates)
    ranked_set = set(ranked)
    if not ranked_set <= candidate_set:
        raise AssertionError("Every ranked document must belong to the candidate set")

    candidate_gold = gold_set & candidate_set
    observed_hits = len(gold_set & ranked_set)
    attainable_hits = min(final_k, len(candidate_gold))
    counts = {
        "candidate_miss": len(gold_set - candidate_set),
        "avoidable_ranking_miss": attainable_hits - observed_hits,
        "structural_cutoff_miss": len(candidate_gold) - attainable_hits,
    }
    total_misses = len(gold_set - ranked_set)
    if sum(counts.values()) != total_misses:
        raise AssertionError("Error buckets do not sum to the total number of misses")
    return {"total_misses": total_misses, **counts}


def _error_counts(
    query: LimitPlusQuery,
    result: CompositionResult,
    *,
    candidate_mode: str,
) -> dict[str, int]:
    ranked_counts = _ranking_error_counts(
        query.gold,
        result.candidates,
        result.ranking,
        final_k=100,
    )
    counts = {
        "parse_or_oracle_mismatch": 0,
        "atomic_miss": 0,
        "composition_or_exclusion_miss": 0,
        **ranked_counts,
    }
    for doc_id in query.gold - result.candidates:
        if candidate_mode == "union_verify":
            survives_atomic = any(
                doc_id in result.positive_hits[attr]
                for attr in query.plan.positive_attrs
            )
        else:
            survives_atomic = positive_atomic_condition(
                query.plan, doc_id, result.positive_hits
            )
        if not survives_atomic:
            counts["atomic_miss"] += 1
        else:
            counts["composition_or_exclusion_miss"] += 1
    if (
        counts["atomic_miss"] + counts["composition_or_exclusion_miss"]
        != counts["candidate_miss"]
    ):
        raise AssertionError("Candidate-loss sub-buckets do not sum to candidate_miss")
    return counts


def _run_oracle(bundle: DatasetBundle) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    run_rows: list[dict] = []
    for query in bundle.queries:
        exact = execute_exact(query.plan, bundle.attr_to_docs)
        ranking = tuple(sorted(exact, key=bundle.doc_ordinal.__getitem__)[:100])
        row = _base_row(query, "exact_boolean_oracle", None)
        row.update(evaluate_ranking(ranking, query.gold, candidates=exact))
        metric_rows.append(row)
        run_rows.append({"qid": query.qid, "ranking": ranking})
    return metric_rows, run_rows


def _run_full_bm25(
    bundle: DatasetBundle,
    index: BM25Index,
    *,
    pool_k: int,
) -> tuple[list[dict], list[dict], dict[str, tuple[str, ...]]]:
    metric_rows: list[dict] = []
    run_rows: list[dict] = []
    rankings: dict[str, tuple[str, ...]] = {}
    for query in bundle.queries:
        result = index.search(query.text, top_k=pool_k)
        if len(result.doc_ids) != len(set(result.doc_ids)):
            raise AssertionError(f"Duplicate ids in BM25 ranking for query {query.qid}")
        rankings[query.qid] = result.doc_ids
        row = _base_row(query, "full_query_bm25", None)
        row["pool_k"] = pool_k
        row.update(
            evaluate_ranking(
                result.doc_ids[:100],
                query.gold,
                candidates=frozenset(result.doc_ids),
            )
        )
        metric_rows.append(row)
        run_rows.append(
            {
                "qid": query.qid,
                "query": query.text,
                "ranking": result.doc_ids,
                "scores": result.scores,
            }
        )
    if set(rankings) != {query.qid for query in bundle.queries}:
        raise AssertionError("Full BM25 did not return all 700 query ids")
    return metric_rows, run_rows, rankings


def _run_full_verify(
    bundle: DatasetBundle,
    full_rankings: Mapping[str, Sequence[str]],
    *,
    pool_k: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Exact symbolic verification over an unmodified full-query BM25 pool."""

    method = f"oracle_full_verify_F{pool_k}"
    metric_rows: list[dict] = []
    run_rows: list[dict] = []
    error_rows: list[dict] = []
    for query in bundle.queries:
        pool = tuple(full_rankings[query.qid][:pool_k])
        valid = execute_exact(query.plan, bundle.attr_to_docs)
        ranking = tuple(doc_id for doc_id in pool if doc_id in valid)[:100]
        candidates = frozenset(doc_id for doc_id in pool if doc_id in valid)
        row = _base_row(query, method, None)
        row["full_query_pool"] = pool_k
        row["nominal_candidate_budget"] = pool_k
        row.update(evaluate_ranking(ranking, query.gold, candidates=candidates))
        metric_rows.append(row)
        errors = _ranking_error_counts(
            query.gold,
            candidates,
            ranking,
            final_k=100,
        )
        error_rows.append(
            {
                "method": method,
                "qid": query.qid,
                "template": query.template,
                "gold_size": len(query.gold),
                "parse_or_oracle_mismatch": 0,
                **errors,
            }
        )
        run_rows.append(
            {
                "method": method,
                "qid": query.qid,
                "ranking": ranking,
                "candidate_size": len(candidates),
            }
        )
    return metric_rows, run_rows, error_rows


def _run_hybrid_verify(
    bundle: DatasetBundle,
    full_rankings: Mapping[str, Sequence[str]],
    atomic_rankings: Mapping[str, Sequence[str]],
    *,
    full_pool_k: int,
    depths: Sequence[int],
    rrf_constant: int = 60,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Union full-query and atomic pools, then verify the exact predicate."""

    metric_rows: list[dict] = []
    run_rows: list[dict] = []
    error_rows: list[dict] = []
    for depth in depths:
        method = f"oracle_hybrid_verify_F{full_pool_k}_L{depth}"
        for query in bundle.queries:
            full_pool = tuple(full_rankings[query.qid][:full_pool_k])
            atomic_lists = {
                attr: tuple(atomic_rankings[attr][:depth])
                for attr in query.plan.positive_attrs
            }
            retrieved = set(full_pool)
            retrieved.update(*(set(values) for values in atomic_lists.values()))
            valid = execute_exact(query.plan, bundle.attr_to_docs)
            candidates = frozenset(retrieved & valid)

            score = {doc_id: 0.0 for doc_id in candidates}
            for rank, doc_id in enumerate(full_pool, start=1):
                if doc_id in score:
                    score[doc_id] += 1.0 / (rrf_constant + rank)
            for values in atomic_lists.values():
                for rank, doc_id in enumerate(values, start=1):
                    if doc_id in score:
                        score[doc_id] += 1.0 / (rrf_constant + rank)
            ranking = tuple(
                sorted(
                    candidates,
                    key=lambda doc_id: (-score[doc_id], bundle.doc_ordinal[doc_id]),
                )[:100]
            )

            row = _base_row(query, method, depth)
            row["full_query_pool"] = full_pool_k
            row["atomic_lists"] = len(query.plan.positive_attrs)
            row["nominal_candidate_budget"] = (
                full_pool_k + len(query.plan.positive_attrs) * depth
            )
            row.update(evaluate_ranking(ranking, query.gold, candidates=candidates))
            metric_rows.append(row)
            errors = _ranking_error_counts(
                query.gold,
                candidates,
                ranking,
                final_k=100,
            )
            error_rows.append(
                {
                    "method": method,
                    "qid": query.qid,
                    "template": query.template,
                    "gold_size": len(query.gold),
                    "parse_or_oracle_mismatch": 0,
                    **errors,
                }
            )
            run_rows.append(
                {
                    "method": method,
                    "qid": query.qid,
                    "ranking": ranking,
                    "candidate_size": len(candidates),
                }
            )
    return metric_rows, run_rows, error_rows


def _atomic_rankings(
    bundle: DatasetBundle,
    index: BM25Index,
    max_depth: int,
) -> tuple[dict[str, tuple[str, ...]], list[dict]]:
    attrs = tuple(dict.fromkeys(attr for query in bundle.queries for attr in query.plan.attrs))
    rankings: dict[str, tuple[str, ...]] = {}
    artifact: list[dict] = []
    for attr in attrs:
        atomic_query = f"Who likes {attr}?"
        result = index.search(atomic_query, top_k=max_depth)
        rankings[attr] = result.doc_ids
        artifact.append(
            {
                "attribute": attr,
                "query": atomic_query,
                "ranking": result.doc_ids,
                "scores": result.scores,
            }
        )
    return rankings, artifact


def _atomic_diagnostic_rows(
    bundle: DatasetBundle,
    atomic_rankings: Mapping[str, Sequence[str]],
    depths: Sequence[int],
) -> tuple[list[dict], list[dict]]:
    """Measure atomic BM25 coverage against exact attribute postings.

    The per-attribute rows separate unavoidable top-L truncation from avoidable
    retrieval misses.  The summary rows report both macro and posting-weighted
    coverage so large attributes cannot be hidden by a mean over attributes.
    """

    normalized_depths = tuple(sorted(set(int(depth) for depth in depths)))
    if not normalized_depths or normalized_depths[0] <= 0:
        raise ValueError("depths must contain positive integers")

    attrs = tuple(dict.fromkeys(attr for query in bundle.queries for attr in query.plan.attrs))
    missing = [attr for attr in attrs if attr not in atomic_rankings]
    if missing:
        raise KeyError(f"Missing atomic rankings for attributes: {missing}")

    total_usage: Counter[str] = Counter()
    positive_usage: Counter[str] = Counter()
    negative_usage: Counter[str] = Counter()
    for query in bundle.queries:
        total_usage.update(query.plan.attrs)
        positive_usage.update(query.plan.positive_attrs)
        negative_usage.update(query.plan.negative_attrs)

    corpus_ids = set(bundle.doc_ordinal)
    rows: list[dict] = []
    for attr in attrs:
        ranking = tuple(atomic_rankings[attr])
        if len(ranking) != len(set(ranking)):
            raise AssertionError(f"Atomic ranking for {attr!r} contains duplicate ids")
        unknown = set(ranking) - corpus_ids
        if unknown:
            raise AssertionError(
                f"Atomic ranking for {attr!r} contains unknown ids: {sorted(unknown)[:3]}"
            )
        posting = set(bundle.attr_to_docs[attr])
        posting_size = len(posting)
        if posting_size == 0:
            raise AssertionError(f"Exact posting for {attr!r} is empty")

        relevant_ranks = tuple(
            rank for rank, doc_id in enumerate(ranking, start=1) if doc_id in posting
        )
        for depth in normalized_depths:
            retrieved_size = min(depth, len(ranking))
            hit_ranks = tuple(rank for rank in relevant_ranks if rank <= retrieved_size)
            hits = len(hit_ranks)
            maximum_possible_hits = min(retrieved_size, posting_size)
            structural_misses = max(posting_size - retrieved_size, 0)
            avoidable_misses = maximum_possible_hits - hits
            if structural_misses + avoidable_misses != posting_size - hits:
                raise AssertionError("Atomic miss decomposition is inconsistent")

            rows.append(
                {
                    "attribute": attr,
                    "atomic_query": f"Who likes {attr}?",
                    "depth": depth,
                    "posting_size": posting_size,
                    "query_occurrences": total_usage[attr],
                    "positive_occurrences": positive_usage[attr],
                    "negative_occurrences": negative_usage[attr],
                    "retrieved_size": retrieved_size,
                    "relevant_retrieved": hits,
                    "nonrelevant_retrieved": retrieved_size - hits,
                    "atomic_recall": hits / posting_size,
                    "atomic_precision": hits / retrieved_size if retrieved_size else 0.0,
                    "structural_recall_ceiling": maximum_possible_hits / posting_size,
                    "ceiling_normalized_atomic_recall": (
                        hits / maximum_possible_hits if maximum_possible_hits else 0.0
                    ),
                    "structural_truncation_misses": structural_misses,
                    "avoidable_retrieval_misses": avoidable_misses,
                    "first_relevant_rank": hit_ranks[0] if hit_ranks else "",
                    "last_relevant_rank": hit_ranks[-1] if hit_ranks else "",
                    "mean_relevant_rank": float(np.mean(hit_ranks)) if hit_ranks else "",
                    "reciprocal_rank": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
                    "complete_posting_covered": int(hits == posting_size),
                }
            )

    summaries: list[dict] = []
    for depth in normalized_depths:
        group = [row for row in rows if row["depth"] == depth]
        posting_total = sum(int(row["posting_size"]) for row in group)
        hits_total = sum(int(row["relevant_retrieved"]) for row in group)
        usage_total = sum(int(row["query_occurrences"]) for row in group)
        positive_total = sum(int(row["positive_occurrences"]) for row in group)
        negative_total = sum(int(row["negative_occurrences"]) for row in group)
        first_ranks = [
            float(row["first_relevant_rank"])
            for row in group
            if row["first_relevant_rank"] != ""
        ]

        def weighted_metric(
            metric_column: str, weight_column: str, denominator: int
        ) -> float:
            if denominator == 0:
                return 0.0
            return sum(
                float(row[metric_column]) * int(row[weight_column]) for row in group
            ) / denominator

        summaries.append(
            {
                "depth": depth,
                "unique_attributes": len(group),
                "posting_size_mean": float(
                    np.mean([int(row["posting_size"]) for row in group])
                ),
                "posting_size_median": float(
                    np.median([int(row["posting_size"]) for row in group])
                ),
                "posting_size_p90": float(
                    np.quantile([int(row["posting_size"]) for row in group], 0.9)
                ),
                "relevant_retrieved_total": hits_total,
                "posting_weighted_atomic_recall": hits_total / posting_total,
                "macro_atomic_recall": float(
                    np.mean([float(row["atomic_recall"]) for row in group])
                ),
                "macro_ceiling_normalized_atomic_recall": float(
                    np.mean(
                        [float(row["ceiling_normalized_atomic_recall"]) for row in group]
                    )
                ),
                "posting_weighted_ceiling_normalized_atomic_recall": weighted_metric(
                    "ceiling_normalized_atomic_recall", "posting_size", posting_total
                ),
                "query_occurrence_weighted_atomic_recall": weighted_metric(
                    "atomic_recall", "query_occurrences", usage_total
                ),
                "positive_occurrence_weighted_atomic_recall": weighted_metric(
                    "atomic_recall", "positive_occurrences", positive_total
                ),
                "positive_occurrence_weighted_ceiling_normalized_atomic_recall": weighted_metric(
                    "ceiling_normalized_atomic_recall",
                    "positive_occurrences",
                    positive_total,
                ),
                "negative_occurrence_weighted_atomic_recall": weighted_metric(
                    "atomic_recall", "negative_occurrences", negative_total
                ),
                "attributes_with_zero_hits": sum(
                    int(row["relevant_retrieved"]) == 0 for row in group
                ),
                "attributes_with_full_coverage": sum(
                    int(row["complete_posting_covered"]) for row in group
                ),
                "mean_first_relevant_rank_hits_only": (
                    float(np.mean(first_ranks)) if first_ranks else ""
                ),
                "median_first_relevant_rank_hits_only": (
                    float(np.median(first_ranks)) if first_ranks else ""
                ),
            }
        )
    return rows, summaries


def _run_composition(
    bundle: DatasetBundle,
    atomic_rankings: Mapping[str, Sequence[str]],
    *,
    depths: Sequence[int],
    exact_negation: bool,
    candidate_mode: str = "hard_compose",
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows: list[dict] = []
    run_rows: list[dict] = []
    error_rows: list[dict] = []
    if candidate_mode == "union_verify":
        method_stem = "oracle_union_verify"
    else:
        not_label = "exact_not" if exact_negation else "truncated_not"
        method_stem = f"oracle_compose_{not_label}"
    for depth in depths:
        method = f"{method_stem}_L{depth}"
        for query in bundle.queries:
            result = compose_rankings(
                query.plan,
                atomic_rankings,
                depth=depth,
                final_k=100,
                doc_ordinal=bundle.doc_ordinal,
                exact_attr_to_docs=bundle.attr_to_docs,
                exact_negation=exact_negation,
                candidate_mode=candidate_mode,
            )
            row = _base_row(query, method, depth)
            row["atomic_lists"] = query.plan.n_atoms
            row["nominal_atomic_budget"] = query.plan.n_atoms * depth
            row.update(evaluate_ranking(result.ranking, query.gold, candidates=result.candidates))
            metric_rows.append(row)
            errors = _error_counts(query, result, candidate_mode=candidate_mode)
            error_rows.append(
                {
                    "method": method,
                    "qid": query.qid,
                    "template": query.template,
                    "gold_size": len(query.gold),
                    **errors,
                }
            )
            run_rows.append(
                {
                    "method": method,
                    "qid": query.qid,
                    "ranking": result.ranking,
                    "candidate_size": len(result.candidates),
                }
            )
    return metric_rows, run_rows, error_rows


def _bootstrap_rows(per_query: Sequence[Mapping[str, object]], samples: int) -> list[dict]:
    baseline_rows = {
        str(row["qid"]): row for row in per_query if row["method"] == "full_query_bm25"
    }
    treatment_methods = sorted(
        {
            str(row["method"])
            for row in per_query
            if str(row["method"]).startswith("oracle_compose_exact_not_")
            or str(row["method"]).startswith("oracle_union_verify_")
            or str(row["method"]).startswith("oracle_full_verify_")
            or str(row["method"]).startswith("oracle_hybrid_verify_")
        }
    )
    qids = sorted(baseline_rows, key=int)
    if len(qids) != 700:
        raise AssertionError("Bootstrap baseline must contain all 700 queries")
    strata = [str(baseline_rows[qid]["template"]) for qid in qids]
    rows: list[dict] = []
    for method in treatment_methods:
        treatment = {
            str(row["qid"]): row for row in per_query if row["method"] == method
        }
        if set(treatment) != set(qids):
            raise AssertionError(f"Treatment {method} does not contain all 700 queries")
        for metric in (
            "recall_at_100",
            "ceiling_normalized_recall_at_100",
            "ndcg_at_20",
        ):
            result = paired_bootstrap_delta(
                [float(baseline_rows[qid][metric]) for qid in qids],
                [float(treatment[qid][metric]) for qid in qids],
                strata=strata,
                samples=samples,
            )
            rows.append(
                {
                    "baseline": "full_query_bm25",
                    "treatment": method,
                    "metric": metric,
                    "delta": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "probability_positive": result.probability_positive,
                    "bootstrap_samples": result.samples,
                    "seed": result.seed,
                    "stratified_by": "template",
                }
            )
    return rows


def run_experiment(
    *,
    data_dir: Path,
    output_dir: Path,
    depths: Sequence[int] = (10, 25, 50, 100, 250, 500, 1000),
    tokenizer_name: str = "nltk",
    tie_break: str = PAPER_QUICKSORT,
    bootstrap_samples: int = 10_000,
    verify_hashes: bool = True,
) -> dict:
    depths = tuple(sorted(set(int(depth) for depth in depths)))
    if not depths or depths[0] <= 0:
        raise ValueError("depths must contain positive integers")
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer {tokenizer_name!r}; choose from {sorted(TOKENIZERS)}")
    if tie_break not in TIE_BREAK_MODES:
        raise ValueError(
            f"Unknown tie-break mode {tie_break!r}; choose from "
            f"{sorted(TIE_BREAK_MODES)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bundle, audit = load_and_audit(data_dir, verify_hashes=verify_hashes)
    _write_json(output_dir / "data_audit.json", audit)

    index_started = time.perf_counter()
    index = BM25Index(
        [doc.doc_id for doc in bundle.documents],
        [doc.text for doc in bundle.documents],
        tokenizer=TOKENIZERS[tokenizer_name],
        tie_break=tie_break,
    )
    index_seconds = time.perf_counter() - index_started

    oracle_metrics, oracle_run = _run_oracle(bundle)
    bm25_started = time.perf_counter()
    full_pool_k = 1000
    bm25_metrics, bm25_run, full_rankings = _run_full_bm25(
        bundle, index, pool_k=full_pool_k
    )
    bm25_seconds = time.perf_counter() - bm25_started

    atomic_started = time.perf_counter()
    atomic_rankings, atomic_artifact = _atomic_rankings(bundle, index, max(depths))
    atomic_metrics, atomic_summary = _atomic_diagnostic_rows(
        bundle,
        atomic_rankings,
        depths,
    )
    atomic_seconds = time.perf_counter() - atomic_started

    exact_metrics, exact_runs, exact_errors = _run_composition(
        bundle, atomic_rankings, depths=depths, exact_negation=True
    )
    truncated_metrics, truncated_runs, truncated_errors = _run_composition(
        bundle, atomic_rankings, depths=depths, exact_negation=False
    )
    verified_metrics, verified_runs, verified_errors = _run_composition(
        bundle,
        atomic_rankings,
        depths=depths,
        exact_negation=True,
        candidate_mode="union_verify",
    )
    full_verify_metrics, full_verify_runs, full_verify_errors = _run_full_verify(
        bundle, full_rankings, pool_k=full_pool_k
    )
    hybrid_metrics, hybrid_runs, hybrid_errors = _run_hybrid_verify(
        bundle,
        full_rankings,
        atomic_rankings,
        full_pool_k=full_pool_k,
        depths=depths,
    )
    per_query = (
        oracle_metrics
        + bm25_metrics
        + exact_metrics
        + truncated_metrics
        + verified_metrics
        + full_verify_metrics
        + hybrid_metrics
    )
    summaries = _summary_rows(per_query)
    bootstrap = _bootstrap_rows(per_query, bootstrap_samples)

    _write_csv(output_dir / "per_query_metrics.csv", per_query)
    _write_csv(output_dir / "summary_metrics.csv", summaries)
    _write_csv(output_dir / "bootstrap_deltas.csv", bootstrap)
    _write_csv(output_dir / "atomic_metrics.csv", atomic_metrics)
    _write_csv(output_dir / "atomic_summary.csv", atomic_summary)
    _write_csv(
        output_dir / "error_decomposition.csv",
        exact_errors
        + truncated_errors
        + verified_errors
        + full_verify_errors
        + hybrid_errors,
    )
    _write_run(output_dir / "runs" / "exact_boolean_oracle.jsonl.gz", oracle_run)
    _write_run(output_dir / "runs" / "full_query_bm25.jsonl.gz", bm25_run)
    _write_run(output_dir / "runs" / "atomic_top_max_depth.jsonl.gz", atomic_artifact)
    _write_run(
        output_dir / "runs" / "composed.jsonl.gz",
        exact_runs
        + truncated_runs
        + verified_runs
        + full_verify_runs
        + hybrid_runs,
    )

    manifest = {
        "experiment": "Constraint-aware retrieve-then-compose on LIMIT+",
        "dataset_revisions": {
            "limit_huggingface": "215834026c13176e520b3bc9d0a055099537ef99",
            "limit_plus_github": "0a4105a328474d4a4c58b8e4fc613ec05c59fc22",
        },
        "tokenizer": tokenizer_name,
        "bm25": {
            "implementation": "independent inverted-index BM25Okapi",
            "k1": index.k1,
            "b": index.b,
            "epsilon": index.epsilon,
            "corpus_size": index.corpus_size,
            "vocabulary_size": index.vocabulary_size,
            "average_document_length": index.avgdl,
            "tie_break": index.tie_break,
            "ranking_expression": (
                "np.argsort(-scores)[:k]"
                if tie_break == PAPER_QUICKSORT
                else "score descending; equal-score ties by corpus ordinal"
            ),
            "deterministic_ablation": "corpus_ordinal",
            "indexed_field": "text",
            "canonical_doc_id_field": "_id",
        },
        "composition": {
            "decomposition": "oracle template metadata (not automatic language parsing)",
            "atomic_query": "Who likes {attribute}?",
            "depths": depths,
            "full_query_pool": full_pool_k,
            "final_k": 100,
            "fusion": "RRF, constant=60",
            "primary_negation": "exact lexical attribute-membership filter",
            "ablation_negation": "truncated negative top-L list",
            "symbolic_hybrid": (
                "union positive atomic top-L lists, then verify the complete predicate "
                "against exact parsed corpus attributes"
            ),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": 20260804,
            "paired": True,
            "stratified_by": "template",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "nltk": package_version("nltk"),
            "platform": platform.platform(),
            "index_seconds": index_seconds,
            "full_bm25_seconds": bm25_seconds,
            "atomic_retrieval_seconds": atomic_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "audit": "data_audit.json",
            "per_query": "per_query_metrics.csv",
            "summary": "summary_metrics.csv",
            "bootstrap": "bootstrap_deltas.csv",
            "atomic_metrics": "atomic_metrics.csv",
            "atomic_summary": "atomic_summary.csv",
            "error_decomposition": "error_decomposition.csv",
            "runs": "runs/*.jsonl.gz",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
