"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bm25 import PAPER_QUICKSORT, TIE_BREAK_MODES
from .data import load_and_audit
from .pipeline import run_experiment


def _depths(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("depths must contain positive integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limitplus",
        description="Audit and diagnose candidate generation on LIMIT+.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Verify files and all 700 Boolean qrels")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit.add_argument("--no-checksums", action="store_true")

    run = subparsers.add_parser("run", help="Run BM25 and retrieve-then-compose")
    run.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    run.add_argument("--output-dir", type=Path, default=Path("results/main"))
    run.add_argument(
        "--depths",
        type=_depths,
        default=(10, 25, 50, 100, 250, 500, 1000),
        help="Atomic top-L sweep (default: 10,25,50,100,250,500,1000)",
    )
    run.add_argument("--tokenizer", choices=("nltk", "regex"), default="nltk")
    run.add_argument(
        "--tie-break",
        choices=tuple(sorted(TIE_BREAK_MODES)),
        default=PAPER_QUICKSORT,
        help=(
            "BM25 equal-score ordering: paper_quicksort preserves the released "
            "ranking expression; corpus_ordinal is the cross-platform profile"
        ),
    )
    run.add_argument("--bootstrap-samples", type=int, default=10_000)
    run.add_argument("--no-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        _, audit = load_and_audit(args.data_dir, verify_hashes=not args.no_checksums)
        print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "run":
        manifest = run_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            depths=args.depths,
            tokenizer_name=args.tokenizer,
            tie_break=args.tie_break,
            bootstrap_samples=args.bootstrap_samples,
            verify_hashes=not args.no_checksums,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
