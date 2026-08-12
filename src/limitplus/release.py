"""Load and cross-check the compact metrics used by release artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping

from .metrics import paired_bootstrap_delta


BM25 = "BM25"
QWEN = "Qwen/Qwen3-Reranker-4B"
EXACT = "exact_constraint_verifier"

TEMPLATE_LABELS = (
    ("_", "A"),
    ("_ or _", "A OR B"),
    ("_ or _ or _", "A OR B OR C"),
    ("_ that are also _", "A AND B"),
    ("_ that are also both _ and _", "A AND B AND C"),
    ("_ that are also _ but not _", "(A AND B) NOT C"),
    ("_ that are not _", "A NOT B"),
)

SUMMARY_METHODS = {
    "bm25": "full_query_bm25",
    "hard_composition": "oracle_compose_exact_not_L1000",
    "exact": "oracle_full_verify_F1000",
    "hybrid": "oracle_hybrid_verify_F1000_L1000",
    "oracle": "exact_boolean_oracle",
}


@dataclass(frozen=True)
class MethodMetrics:
    recall_at_100: float
    normalized_recall_at_100: float
    ndcg_at_20: float
    ndcg_at_100: float
    candidate_recall: float


@dataclass(frozen=True)
class DeltaInterval:
    delta: float
    ci_low: float
    ci_high: float
    probability_positive: float


@dataclass(frozen=True)
class TemplateMetrics:
    template: str
    label: str
    bm25_ndcg_at_20: float
    qwen_ndcg_at_20: float
    exact_ndcg_at_20: float
    candidate_recall: float


@dataclass(frozen=True)
class ReleaseMetrics:
    bm25: MethodMetrics
    qwen: MethodMetrics
    exact: MethodMetrics
    hard_composition: MethodMetrics
    hybrid: MethodMetrics
    oracle: MethodMetrics
    templates: tuple[TemplateMetrics, ...]
    qwen_vs_bm25: Mapping[str, DeltaInterval]
    exact_vs_qwen: Mapping[str, DeltaInterval]
    atomic_positive_recall_at_1000: float
    micro_pool_coverage: float
    recall_gap_closed: float
    ndcg20_gap_closed: float
    negation_share_of_net_ndcg20_gain: float
    non_negation_ndcg20_delta: float
    non_negation_ndcg20_ci_low: float
    non_negation_ndcg20_ci_high: float
    bootstrap_draws: int
    pairs_scored: int
    queries: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Expected at least one data row in {path}")
    return rows


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _float(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}: {row[field]}")
    return value


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Cross-file mismatch for {label}: {actual} != {expected}")


def _method_from_row(row: Mapping[str, str]) -> MethodMetrics:
    return MethodMetrics(
        recall_at_100=_float(row, "recall_at_100"),
        normalized_recall_at_100=_float(
            row, "ceiling_normalized_recall_at_100"
        ),
        ndcg_at_20=_float(row, "ndcg_at_20"),
        ndcg_at_100=_float(row, "ndcg_at_100"),
        candidate_recall=_float(row, "candidate_recall"),
    )


def _method_from_json(value: Mapping[str, object]) -> MethodMetrics:
    return MethodMetrics(
        recall_at_100=float(value["recall_at_100"]),
        normalized_recall_at_100=float(
            value["ceiling_normalized_recall_at_100"]
        ),
        ndcg_at_20=float(value["ndcg_at_20"]),
        ndcg_at_100=float(value["ndcg_at_100"]),
        candidate_recall=float(value["candidate_recall"]),
    )


def _delta_map(value: Mapping[str, object]) -> dict[str, DeltaInterval]:
    result: dict[str, DeltaInterval] = {}
    for metric in ("recall_at_100", "ndcg_at_20"):
        item = value[metric]
        if not isinstance(item, dict):
            raise ValueError(f"Expected delta object for {metric}")
        result[metric] = DeltaInterval(
            delta=float(item["delta"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            probability_positive=float(item["probability_positive"]),
        )
    return result


def _check_delta(
    stored: DeltaInterval,
    baseline: list[float],
    treatment: list[float],
    strata: list[str],
    *,
    samples: int,
    seed: int,
    label: str,
) -> None:
    computed = paired_bootstrap_delta(
        baseline,
        treatment,
        strata=strata,
        samples=samples,
        seed=seed,
    )
    _close(stored.delta, computed.estimate, f"{label}.delta")
    for field in ("ci_low", "ci_high", "probability_positive"):
        _close(getattr(stored, field), getattr(computed, field), f"{label}.{field}")


def _check_method(actual: MethodMetrics, expected: MethodMetrics, label: str) -> None:
    for field in MethodMetrics.__dataclass_fields__:
        _close(getattr(actual, field), getattr(expected, field), f"{label}.{field}")


def load_release_metrics(repository_root: Path) -> ReleaseMetrics:
    """Load committed results and fail if their overlapping evidence disagrees."""

    root = repository_root.resolve()
    main_dir = root / "results" / "main"
    qwen_dir = root / "results" / "qwen4b_paper_pool1000_full700"

    summary_rows = _read_csv(main_dir / "summary_metrics.csv")
    overall_rows = {
        row["method"]: row
        for row in summary_rows
        if row["group_type"] == "overall" and row["group_value"] == "all"
    }
    missing = set(SUMMARY_METHODS.values()) - set(overall_rows)
    if missing:
        raise ValueError(f"Missing overall CPU summary rows: {sorted(missing)}")
    cpu = {
        label: _method_from_row(overall_rows[method])
        for label, method in SUMMARY_METHODS.items()
    }

    paper_pdf_values = {
        "recall_at_5": 0.632,
        "recall_at_20": 0.733,
        "recall_at_100": 0.837,
        "ndcg_at_5": 0.758,
        "ndcg_at_20": 0.785,
        "ndcg_at_100": 0.801,
    }
    bm25_summary_row = overall_rows[SUMMARY_METHODS["bm25"]]
    for field, paper_value in paper_pdf_values.items():
        actual = _float(bm25_summary_row, field)
        if f"{actual:.3f}" != f"{paper_value:.3f}":
            raise ValueError(
                f"BM25 {field} no longer matches the typeset Table 2 value at 3 decimals"
            )

    qwen_summary = _read_json(qwen_dir / "summary.json")
    comparison = qwen_summary["comparison_mean"]
    if not isinstance(comparison, dict):
        raise ValueError("Qwen comparison_mean must be an object")
    qwen_methods = {
        name: _method_from_json(comparison[name])
        for name in (BM25, QWEN, EXACT)
    }
    _check_method(cpu["bm25"], qwen_methods[BM25], "CPU/Qwen BM25")
    _check_method(cpu["exact"], qwen_methods[EXACT], "CPU/Qwen exact verifier")
    qwen_top_level_mean = qwen_summary["mean"]
    if not isinstance(qwen_top_level_mean, dict):
        raise ValueError("Qwen mean must be an object")
    _check_method(
        qwen_methods[QWEN],
        _method_from_json(qwen_top_level_mean),
        "Qwen comparison/top-level mean",
    )

    per_query = _read_csv(qwen_dir / "per_query_metrics.csv")
    expected_queries = int(qwen_summary["queries"])
    expected_models = {BM25, QWEN, EXACT}
    rows_by_qid: dict[int, list[dict[str, str]]] = defaultdict(list)
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in per_query:
        qid = int(row["qid"])
        rows_by_qid[qid].append(row)
        rows_by_model[row["model"]].append(row)
        if int(row["pool_k"]) != 1000 or int(row["candidate_size"]) != 1000:
            raise ValueError(f"Unexpected candidate pool for qid {qid}")

    if set(rows_by_qid) != set(range(expected_queries)):
        raise ValueError("Qwen per-query IDs are not exactly 0..queries-1")
    for qid, rows in rows_by_qid.items():
        if len(rows) != 3 or {row["model"] for row in rows} != expected_models:
            raise ValueError(f"Expected exactly three methods for qid {qid}")
        templates = {row["template"] for row in rows}
        gold_sizes = {row["gold_size"] for row in rows}
        recalls = {row["candidate_recall"] for row in rows}
        if len(templates) != 1 or len(gold_sizes) != 1 or len(recalls) != 1:
            raise ValueError(f"Candidate metadata differs across methods for qid {qid}")

    metric_fields = {
        "recall_at_100": "recall_at_100",
        "normalized_recall_at_100": "ceiling_normalized_recall_at_100",
        "ndcg_at_20": "ndcg_at_20",
        "ndcg_at_100": "ndcg_at_100",
        "candidate_recall": "candidate_recall",
    }
    for model, rows in rows_by_model.items():
        if len(rows) != expected_queries:
            raise ValueError(f"Expected {expected_queries} rows for {model}")
        for field, csv_field in metric_fields.items():
            actual = fmean(_float(row, csv_field) for row in rows)
            _close(actual, getattr(qwen_methods[model], field), f"per-query mean {model}.{field}")

    templates: list[TemplateMetrics] = []
    delta_by_template: dict[str, float] = {}
    label_by_template = dict(TEMPLATE_LABELS)
    observed_templates = {row["template"] for row in rows_by_model[BM25]}
    if observed_templates != set(label_by_template):
        raise ValueError(f"Unexpected template set: {sorted(observed_templates)}")
    for template, label in TEMPLATE_LABELS:
        grouped = {
            model: [row for row in rows_by_model[model] if row["template"] == template]
            for model in expected_models
        }
        if any(len(rows) != 100 for rows in grouped.values()):
            raise ValueError(f"Expected 100 rows per method for template {template}")
        means = {
            model: fmean(_float(row, "ndcg_at_20") for row in rows)
            for model, rows in grouped.items()
        }
        candidate_recall = fmean(
            _float(row, "candidate_recall") for row in grouped[BM25]
        )
        templates.append(
            TemplateMetrics(
                template=template,
                label=label,
                bm25_ndcg_at_20=means[BM25],
                qwen_ndcg_at_20=means[QWEN],
                exact_ndcg_at_20=means[EXACT],
                candidate_recall=candidate_recall,
            )
        )
        delta_by_template[template] = means[QWEN] - means[BM25]

    qwen_vs_bm25 = _delta_map(qwen_summary["paired_delta_vs_bm25"])
    exact_vs_qwen = _delta_map(
        qwen_summary["paired_delta_exact_verifier_vs_qwen"]
    )
    bootstrap = qwen_summary["bootstrap"]
    runtime = qwen_summary["runtime"]
    if not isinstance(bootstrap, dict) or not isinstance(runtime, dict):
        raise ValueError("Qwen bootstrap and runtime must be objects")
    bootstrap_draws = int(bootstrap["draws"])
    bootstrap_seed = int(bootstrap["seed"])
    if (
        bootstrap_draws != 10_000
        or bootstrap_seed != 20260804
        or bootstrap.get("stratification") != "template"
    ):
        raise ValueError(f"Unexpected Qwen bootstrap contract: {bootstrap}")

    per_query_by_qid = {
        model: {
            int(row["qid"]): row
            for row in rows_by_model[model]
        }
        for model in expected_models
    }
    ordered_qids = sorted(rows_by_qid)
    strata = [per_query_by_qid[BM25][qid]["template"] for qid in ordered_qids]
    for metric in ("recall_at_100", "ndcg_at_20"):
        bm25_values = [
            _float(per_query_by_qid[BM25][qid], metric) for qid in ordered_qids
        ]
        qwen_values = [
            _float(per_query_by_qid[QWEN][qid], metric) for qid in ordered_qids
        ]
        exact_values = [
            _float(per_query_by_qid[EXACT][qid], metric) for qid in ordered_qids
        ]
        _check_delta(
            qwen_vs_bm25[metric],
            bm25_values,
            qwen_values,
            strata,
            samples=bootstrap_draws,
            seed=bootstrap_seed,
            label=f"Qwen-BM25 {metric}",
        )
        _check_delta(
            exact_vs_qwen[metric],
            qwen_values,
            exact_values,
            strata,
            samples=bootstrap_draws,
            seed=bootstrap_seed,
            label=f"exact-Qwen {metric}",
        )
    _close(
        qwen_methods[QWEN].recall_at_100 - qwen_methods[BM25].recall_at_100,
        qwen_vs_bm25["recall_at_100"].delta,
        "Qwen-BM25 recall delta",
    )
    _close(
        qwen_methods[QWEN].ndcg_at_20 - qwen_methods[BM25].ndcg_at_20,
        qwen_vs_bm25["ndcg_at_20"].delta,
        "Qwen-BM25 nDCG delta",
    )
    _close(
        qwen_methods[EXACT].recall_at_100 - qwen_methods[QWEN].recall_at_100,
        exact_vs_qwen["recall_at_100"].delta,
        "exact-Qwen recall delta",
    )
    _close(
        qwen_methods[EXACT].ndcg_at_20 - qwen_methods[QWEN].ndcg_at_20,
        exact_vs_qwen["ndcg_at_20"].delta,
        "exact-Qwen nDCG delta",
    )

    cpu_bootstrap_rows = _read_csv(main_dir / "bootstrap_deltas.csv")
    cpu_exact_deltas = {
        row["metric"]: row
        for row in cpu_bootstrap_rows
        if row["baseline"] == SUMMARY_METHODS["bm25"]
        and row["treatment"] == SUMMARY_METHODS["exact"]
    }
    for metric, field in (
        ("recall_at_100", "recall_at_100"),
        ("ceiling_normalized_recall_at_100", "normalized_recall_at_100"),
        ("ndcg_at_20", "ndcg_at_20"),
    ):
        if metric not in cpu_exact_deltas:
            raise ValueError(f"Missing CPU bootstrap row for {metric}")
        _close(
            getattr(cpu["exact"], field) - getattr(cpu["bm25"], field),
            _float(cpu_exact_deltas[metric], "delta"),
            f"CPU exact-BM25 {metric} delta",
        )

    bm25_rows = rows_by_model[BM25]
    gold_total = sum(int(row["gold_size"]) for row in bm25_rows)
    candidate_gold_total = sum(
        round(int(row["gold_size"]) * _float(row, "candidate_recall"))
        for row in bm25_rows
    )
    micro_pool_coverage = candidate_gold_total / gold_total

    negative_templates = {
        "_ that are also _ but not _",
        "_ that are not _",
    }
    total_template_delta = sum(delta_by_template.values())
    negation_share = (
        sum(delta_by_template[template] for template in negative_templates)
        / total_template_delta
    )
    non_negative_rows = [
        row
        for row in per_query
        if row["template"] not in negative_templates and row["model"] in {BM25, QWEN}
    ]
    non_negative_by_qid: dict[int, dict[str, str | float]] = defaultdict(dict)
    for row in non_negative_rows:
        values = non_negative_by_qid[int(row["qid"])]
        values["template"] = row["template"]
        values[row["model"]] = _float(row, "ndcg_at_20")

    non_negative_qids = sorted(non_negative_by_qid)
    non_negation_bootstrap = paired_bootstrap_delta(
        [float(non_negative_by_qid[qid][BM25]) for qid in non_negative_qids],
        [float(non_negative_by_qid[qid][QWEN]) for qid in non_negative_qids],
        strata=[str(non_negative_by_qid[qid]["template"]) for qid in non_negative_qids],
        samples=bootstrap_draws,
        seed=bootstrap_seed,
    )

    atomic_rows = _read_csv(main_dir / "atomic_summary.csv")
    atomic_1000 = [row for row in atomic_rows if int(row["depth"]) == 1000]
    if len(atomic_1000) != 1:
        raise ValueError("Expected one atomic-summary row at depth 1000")

    recall_gap = (
        qwen_methods[QWEN].recall_at_100 - qwen_methods[BM25].recall_at_100
    ) / (qwen_methods[EXACT].recall_at_100 - qwen_methods[BM25].recall_at_100)
    ndcg_gap = (
        qwen_methods[QWEN].ndcg_at_20 - qwen_methods[BM25].ndcg_at_20
    ) / (qwen_methods[EXACT].ndcg_at_20 - qwen_methods[BM25].ndcg_at_20)

    if int(qwen_summary["data_audit_matches"]) != expected_queries:
        raise ValueError("Qwen data audit does not cover every query")
    if int(runtime["pairs_scored_all_checkpoints"]) != expected_queries * 1000:
        raise ValueError("Qwen scored-pair count does not equal queries * pool_k")

    return ReleaseMetrics(
        bm25=cpu["bm25"],
        qwen=qwen_methods[QWEN],
        exact=cpu["exact"],
        hard_composition=cpu["hard_composition"],
        hybrid=cpu["hybrid"],
        oracle=cpu["oracle"],
        templates=tuple(templates),
        qwen_vs_bm25=qwen_vs_bm25,
        exact_vs_qwen=exact_vs_qwen,
        atomic_positive_recall_at_1000=_float(
            atomic_1000[0], "positive_occurrence_weighted_atomic_recall"
        ),
        micro_pool_coverage=micro_pool_coverage,
        recall_gap_closed=recall_gap,
        ndcg20_gap_closed=ndcg_gap,
        negation_share_of_net_ndcg20_gain=negation_share,
        non_negation_ndcg20_delta=non_negation_bootstrap.estimate,
        non_negation_ndcg20_ci_low=non_negation_bootstrap.ci_low,
        non_negation_ndcg20_ci_high=non_negation_bootstrap.ci_high,
        bootstrap_draws=bootstrap_draws,
        pairs_scored=int(runtime["pairs_scored_all_checkpoints"]),
        queries=expected_queries,
    )
