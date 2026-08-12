"""Rerank an unmodified BM25 pool with pinned Qwen3-Reranker-4B.

The script never injects gold documents.  It checkpoints one JSON file per query,
so a long GPU run can be resumed safely after interruption.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import time
from importlib.metadata import version as package_version
from pathlib import Path

from limitplus.data import load_and_audit
from limitplus.logic import execute_exact
from limitplus.metrics import evaluate_ranking, paired_bootstrap_delta


MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
INSTRUCTION = (
    "Given a set-compositional entity-search query, judge whether the document "
    "satisfies the complete Boolean information need. Treat conjunction, "
    "disjunction, and exclusion literally; a document is relevant only when "
    "the entire query is satisfied."
)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based '
    'on the Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PAIR_FORMAT_TEMPLATE = (
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
)
SCORE_CONTRACT = "final-token logit(yes) - logit(no)"
PROTOCOL_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {
            "instruction": INSTRUCTION,
            "prefix": PREFIX,
            "suffix": SUFFIX,
            "pair_format": PAIR_FORMAT_TEMPLATE,
            "score": SCORE_CONTRACT,
            "answer_tokens": ["no", "yes"],
            "padding_side": "left",
            "truncation": "longest_first",
            "logits_to_keep": 1,
            "use_cache": False,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
DEFAULT_BM25_RUN = Path(
    "results/reproduction_cpu_paper/runs/full_query_bm25.jsonl.gz"
)
RECORDED_POOL_FINGERPRINT = (
    "11eaac068169f7355dfc080620ed138a5866cc9e951513c7c5650eaec8993748"
)


def read_bm25_run(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in rows:
                raise ValueError(f"Duplicate qid in BM25 run: {qid}")
            rows[qid] = row
    if set(rows) != set(map(str, range(700))):
        raise ValueError("The BM25 pool must contain all query ids 0..699")
    return rows


def validate_bm25_pools(
    rows: dict[str, dict], *, pool_k: int, known_doc_ids: set[str]
) -> None:
    for qid, row in rows.items():
        ranking = row.get("ranking")
        if not isinstance(ranking, list) or len(ranking) < pool_k:
            raise ValueError(f"BM25 qid={qid} has fewer than pool-k={pool_k} candidates")
        pool = ranking[:pool_k]
        if len(set(pool)) != len(pool):
            raise ValueError(f"BM25 qid={qid} contains duplicate IDs in its candidate pool")
        unknown = set(pool) - known_doc_ids
        if unknown:
            raise ValueError(
                f"BM25 qid={qid} contains {len(unknown)} unknown document IDs"
            )


def bm25_pool_fingerprint(rows: dict[str, dict], *, pool_k: int) -> str:
    """Hash the ordered qid/document-id pools independently of gzip metadata."""

    digest = hashlib.sha256()
    for qid in sorted(rows, key=int):
        canonical = json.dumps(
            {"qid": qid, "ranking": rows[qid]["ranking"][:pool_k]},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def format_pair(query: str, document: str) -> str:
    return PAIR_FORMAT_TEMPLATE.format(
        instruction=INSTRUCTION,
        query=query,
        document=document,
    )


def select_queries(queries, per_template: int | None):
    if per_template is None:
        return list(queries)
    selected = []
    counts: dict[str, int] = {}
    for query in queries:
        count = counts.get(query.template, 0)
        if count < per_template:
            selected.append(query)
            counts[query.template] = count + 1
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_is_compatible(path: Path, query, bm25_row: dict, args) -> bool:
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "qid": query.qid,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "pool_k": args.pool_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "instruction": INSTRUCTION,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    expected_pool = list(bm25_row["ranking"][: args.pool_k])
    observed_ranking = record.get("ranking")
    observed_scores = record.get("scores")
    source_ranks = record.get("source_bm25_rank")
    structurally_valid = (
        isinstance(observed_ranking, list)
        and isinstance(observed_scores, list)
        and isinstance(source_ranks, list)
        and len(observed_ranking) == len(expected_pool)
        and len(observed_scores) == len(expected_pool)
        and len(source_ranks) == len(expected_pool)
        and len(set(observed_ranking)) == len(observed_ranking)
        and all(type(rank) is int for rank in source_ranks)
        and sorted(source_ranks) == list(range(1, len(expected_pool) + 1))
        and all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            for score in observed_scores
        )
    )
    reconstructed_pool = (
        [
            doc_id
            for _, doc_id in sorted(
                zip(source_ranks, observed_ranking, strict=True),
                key=lambda pair: pair[0],
            )
        ]
        if structurally_valid
        else None
    )
    if reconstructed_pool != expected_pool:
        mismatches["candidate_pool"] = ("checkpoint differs", "current BM25 pool")
    if mismatches:
        raise ValueError(
            f"Checkpoint {path} is incompatible with this run: {mismatches}. "
            "Use a new output directory or remove only that checkpoint."
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--bm25-run",
        type=Path,
        default=DEFAULT_BM25_RUN,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen4b"))
    parser.add_argument("--pool-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--per-template",
        type=int,
        help="Run the first N released queries per template (deterministic smoke subset).",
    )
    args = parser.parse_args()
    if args.pool_k <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("pool-k, batch-size, and max-length must be positive")
    if args.per_template is not None and args.per_template <= 0:
        raise ValueError("per-template must be positive when supplied")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-Reranker-4B experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    bundle, audit = load_and_audit(args.data_dir)
    queries = select_queries(bundle.queries, args.per_template)
    bm25 = read_bm25_run(args.bm25_run)
    doc_text = {document.doc_id: document.text for document in bundle.documents}
    validate_bm25_pools(bm25, pool_k=args.pool_k, known_doc_ids=set(doc_text))
    pool_fingerprint = bm25_pool_fingerprint(bm25, pool_k=args.pool_k)
    if (
        args.bm25_run.resolve() == DEFAULT_BM25_RUN.resolve()
        and args.pool_k == 1000
        and pool_fingerprint != RECORDED_POOL_FINGERPRINT
    ):
        raise ValueError(
            "The reproduced BM25 pool does not match the recorded full-run fingerprint; "
            "do not start neural scoring with a different candidate ordering"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "scores"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        query
        for query in queries
        if not checkpoint_is_compatible(
            checkpoint_dir / f"{query.qid}.json",
            query,
            bm25[query.qid],
            args,
        )
    ]

    load_seconds = 0.0
    score_seconds = 0.0
    if pending:
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        load_seconds = time.perf_counter() - load_started
        false_id = tokenizer.convert_tokens_to_ids("no")
        true_id = tokenizer.convert_tokens_to_ids("yes")
        prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
        content_limit = args.max_length - len(prefix_tokens) - len(suffix_tokens)
        if content_limit <= 0:
            raise ValueError("max-length is too small for the fixed prompt")

        for query_index, query in enumerate(pending, start=1):
            pool = tuple(bm25[query.qid]["ranking"][: args.pool_k])
            all_scores: list[float] = []
            started = time.perf_counter()
            for offset in range(0, len(pool), args.batch_size):
                batch_ids = pool[offset : offset + args.batch_size]
                pairs = [format_pair(query.text, doc_text[doc_id]) for doc_id in batch_ids]
                encoded = tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    max_length=content_limit,
                    return_attention_mask=False,
                )
                encoded["input_ids"] = [
                    prefix_tokens + ids + suffix_tokens for ids in encoded["input_ids"]
                ]
                inputs = tokenizer.pad(
                    encoded,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
                with torch.inference_mode():
                    # Qwen3 supports computing only the final vocabulary projection.
                    # Without this, CausalLM materializes logits for every input token.
                    logits = model(
                        **inputs,
                        logits_to_keep=1,
                        use_cache=False,
                    ).logits[:, -1, :]
                    scores = logits[:, true_id] - logits[:, false_id]
                all_scores.extend(float(value) for value in scores.float().cpu())
            elapsed = time.perf_counter() - started
            score_seconds += elapsed
            order = sorted(range(len(pool)), key=lambda index: (-all_scores[index], index))
            record = {
                "qid": query.qid,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "pool_k": args.pool_k,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "instruction": INSTRUCTION,
                "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                "ranking": [pool[index] for index in order],
                "scores": [all_scores[index] for index in order],
                "source_bm25_rank": [order_index + 1 for order_index in order],
                "seconds": elapsed,
            }
            target = checkpoint_dir / f"{query.qid}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(target)
            print(
                f"[{query_index}/{len(pending)}] qid={query.qid} "
                f"pairs={len(pool)} seconds={elapsed:.1f}",
                flush=True,
            )

    metric_rows: list[dict] = []
    qwen_metric_rows: list[dict] = []
    baseline_values: dict[str, list[float]] = {
        "recall_at_100": [],
        "ceiling_normalized_recall_at_100": [],
        "ndcg_at_20": [],
        "ndcg_at_100": [],
    }
    qwen_values = {key: [] for key in baseline_values}
    verifier_values = {key: [] for key in baseline_values}
    strata: list[str] = []
    run_rows: list[dict] = []
    for query in queries:
        record = json.loads((checkpoint_dir / f"{query.qid}.json").read_text(encoding="utf-8"))
        qwen_ranking = tuple(record["ranking"])
        pool = tuple(bm25[query.qid]["ranking"][: args.pool_k])
        qwen_metrics = evaluate_ranking(qwen_ranking[:100], query.gold, candidates=set(pool))
        baseline_metrics = evaluate_ranking(pool[:100], query.gold, candidates=set(pool))
        valid = execute_exact(query.plan, bundle.attr_to_docs)
        verifier_ranking = tuple(doc_id for doc_id in pool if doc_id in valid)[:100]
        verifier_metrics = evaluate_ranking(
            verifier_ranking,
            query.gold,
            candidates=set(pool),
        )
        common = {
            "qid": query.qid,
            "template": query.template,
            "gold_size": len(query.gold),
            "pool_k": args.pool_k,
        }
        rows_for_query = [
            {**common, "model": "BM25", **baseline_metrics},
            {**common, "model": MODEL_ID, **qwen_metrics},
            {
                **common,
                "model": "exact_constraint_verifier",
                **verifier_metrics,
            },
        ]
        metric_rows.extend(rows_for_query)
        qwen_metric_rows.append(
            {
                "qid": query.qid,
                "template": query.template,
                "gold_size": len(query.gold),
                "model": MODEL_ID,
                "pool_k": args.pool_k,
                **qwen_metrics,
            }
        )
        for metric in baseline_values:
            baseline_values[metric].append(float(baseline_metrics[metric]))
            qwen_values[metric].append(float(qwen_metrics[metric]))
            verifier_values[metric].append(float(verifier_metrics[metric]))
        strata.append(query.template)
        run_rows.append(record)

    summary = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "instruction": INSTRUCTION,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "pool": f"unmodified BM25 top-{args.pool_k}; no gold injection",
        "pool_fingerprint": pool_fingerprint,
        "queries": len(queries),
        "per_template": args.per_template,
        "selection": (
            "all released queries"
            if args.per_template is None
            else "deterministic first N queries per template in released order; "
            "convenience subset, not a random sample"
        ),
        "mean": {
            key: sum(float(row[key]) for row in qwen_metric_rows)
            / len(qwen_metric_rows)
            for key in (
                "recall_at_100",
                "ceiling_normalized_recall_at_100",
                "ndcg_at_20",
                "ndcg_at_100",
                "candidate_recall",
            )
        },
        "comparison_mean": {
            label: {
                key: sum(float(row[key]) for row in metric_rows if row["model"] == label)
                / sum(row["model"] == label for row in metric_rows)
                for key in (
                    "recall_at_100",
                    "ceiling_normalized_recall_at_100",
                    "ndcg_at_20",
                    "ndcg_at_100",
                    "candidate_recall",
                )
            }
            for label in ("BM25", MODEL_ID, "exact_constraint_verifier")
        },
        "paired_delta_vs_bm25": {},
        "paired_delta_exact_verifier_vs_qwen": {},
        "runtime": {
            "model_load_seconds_this_run": load_seconds,
            "scoring_seconds_this_run": score_seconds,
            "scoring_seconds_all_checkpoints": sum(
                float(record["seconds"]) for record in run_rows
            ),
            "pairs_scored_all_checkpoints": sum(
                len(record["ranking"]) for record in run_rows
            ),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "os": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface-hub"),
            "safetensors": package_version("safetensors"),
            "cuda_runtime": torch.version.cuda,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "score": SCORE_CONTRACT,
        },
        "bootstrap": {
            "draws": 10_000,
            "seed": 20260804,
            "stratification": "template",
            "scope_note": (
                "Exploratory interval conditional on the selected convenience subset"
                if args.per_template is not None
                else "Paired query interval over the full released benchmark"
            ),
        },
        "data_audit_matches": audit["exact_boolean_oracle_matches"],
    }
    all_seconds = summary["runtime"]["scoring_seconds_all_checkpoints"]
    all_pairs = summary["runtime"]["pairs_scored_all_checkpoints"]
    pairs_per_second = all_pairs / all_seconds if all_seconds else None
    summary["runtime"]["pairs_per_second_all_checkpoints"] = pairs_per_second
    summary["runtime"]["projected_700_query_hours_at_observed_throughput"] = (
        700 * args.pool_k / pairs_per_second / 3600.0
        if pairs_per_second
        else None
    )
    for metric in baseline_values:
        bootstrap = paired_bootstrap_delta(
            baseline_values[metric],
            qwen_values[metric],
            strata=strata,
            samples=10_000,
        )
        summary["paired_delta_vs_bm25"][metric] = {
            "delta": bootstrap.estimate,
            "ci_low": bootstrap.ci_low,
            "ci_high": bootstrap.ci_high,
            "probability_positive": bootstrap.probability_positive,
        }
        verifier_bootstrap = paired_bootstrap_delta(
            qwen_values[metric],
            verifier_values[metric],
            strata=strata,
            samples=10_000,
        )
        summary["paired_delta_exact_verifier_vs_qwen"][metric] = {
            "delta": verifier_bootstrap.estimate,
            "ci_low": verifier_bootstrap.ci_low,
            "ci_high": verifier_bootstrap.ci_high,
            "probability_positive": verifier_bootstrap.probability_positive,
        }

    write_csv(args.output_dir / "per_query_metrics.csv", metric_rows)
    with gzip.open(args.output_dir / "run.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in run_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
