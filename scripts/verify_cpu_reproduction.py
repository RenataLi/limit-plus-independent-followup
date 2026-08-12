"""Compare a fresh CPU reproduction with the committed compact reference."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


CSV_KEYS = {
    "summary_metrics.csv": ("method", "group_type", "group_value"),
    "bootstrap_deltas.csv": ("baseline", "treatment", "metric"),
    "atomic_summary.csv": ("depth",),
}

RUNTIME_VARIATION = {
    "platform",
    "index_seconds",
    "full_bm25_seconds",
    "atomic_retrieval_seconds",
    "total_seconds",
}

GENERATED_CSVS = (
    "per_query_metrics.csv",
    "atomic_metrics.csv",
    "error_decomposition.csv",
)

GENERATED_RUNS = (
    "exact_boolean_oracle.jsonl.gz",
    "full_query_bm25.jsonl.gz",
    "atomic_top_max_depth.jsonl.gz",
    "composed.jsonl.gz",
)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _compare_values(reference: object, reproduced: object, path: str) -> None:
    if isinstance(reference, bool) or isinstance(reproduced, bool):
        if reference is not reproduced:
            raise AssertionError(f"{path}: {reproduced!r} != {reference!r}")
        return
    if isinstance(reference, (int, float)) and isinstance(reproduced, (int, float)):
        if not math.isclose(
            float(reference), float(reproduced), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AssertionError(f"{path}: {reproduced!r} != {reference!r}")
        return
    if isinstance(reference, dict) and isinstance(reproduced, dict):
        if set(reference) != set(reproduced):
            missing = sorted(set(reference) - set(reproduced))
            extra = sorted(set(reproduced) - set(reference))
            raise AssertionError(f"{path}: key mismatch; missing={missing}, extra={extra}")
        for key in reference:
            _compare_values(reference[key], reproduced[key], f"{path}.{key}")
        return
    if isinstance(reference, list) and isinstance(reproduced, list):
        if len(reference) != len(reproduced):
            raise AssertionError(
                f"{path}: list length {len(reproduced)} != {len(reference)}"
            )
        for index, (expected, actual) in enumerate(zip(reference, reproduced, strict=True)):
            _compare_values(expected, actual, f"{path}[{index}]")
        return
    if reference != reproduced:
        raise AssertionError(f"{path}: {reproduced!r} != {reference!r}")


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames or not rows:
        raise AssertionError(f"{path}: expected a header and at least one data row")
    return fieldnames, rows


def _index_rows(
    rows: list[dict[str, str]], keys: tuple[str, ...], path: Path
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in indexed:
            raise AssertionError(f"{path}: duplicate key {key}")
        indexed[key] = row
    return indexed


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        raise AssertionError(f"Non-finite numeric value: {value!r}")
    return result


def _compare_csv(reference_path: Path, reproduced_path: Path, keys: tuple[str, ...]) -> None:
    reference_fields, reference_rows = _read_csv(reference_path)
    reproduced_fields, reproduced_rows = _read_csv(reproduced_path)
    if reference_fields != reproduced_fields:
        raise AssertionError(
            f"{reproduced_path}: columns {reproduced_fields} != {reference_fields}"
        )
    reference = _index_rows(reference_rows, keys, reference_path)
    reproduced = _index_rows(reproduced_rows, keys, reproduced_path)
    if set(reference) != set(reproduced):
        missing = sorted(set(reference) - set(reproduced))
        extra = sorted(set(reproduced) - set(reference))
        raise AssertionError(
            f"{reproduced_path}: row-key mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, expected_row in reference.items():
        actual_row = reproduced[key]
        for field in reference_fields:
            expected_number = _number(expected_row[field])
            actual_number = _number(actual_row[field])
            if expected_number is not None and actual_number is not None:
                if not math.isclose(
                    expected_number, actual_number, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError(
                        f"{reproduced_path}:{key}.{field}: "
                        f"{actual_row[field]} != {expected_row[field]}"
                    )
            elif expected_row[field] != actual_row[field]:
                raise AssertionError(
                    f"{reproduced_path}:{key}.{field}: "
                    f"{actual_row[field]!r} != {expected_row[field]!r}"
                )


def _manifest_without_expected_variation(value: object) -> object:
    if not isinstance(value, dict) or not isinstance(value.get("runtime"), dict):
        raise AssertionError("manifest.json must contain a runtime object")
    result = dict(value)
    runtime = dict(result["runtime"])
    expected_runtime_fields = RUNTIME_VARIATION | {"python"}
    if not expected_runtime_fields.issubset(runtime):
        missing = sorted(expected_runtime_fields - set(runtime))
        raise AssertionError(f"manifest runtime is missing expected keys: {missing}")
    for field in RUNTIME_VARIATION:
        runtime.pop(field)
    python_parts = str(runtime["python"]).split(".")
    if len(python_parts) < 2:
        raise AssertionError(f"Unrecognized Python version: {runtime['python']!r}")
    runtime["python"] = ".".join(python_parts[:2])
    result["runtime"] = runtime
    return result


def _check_generated_outputs(reproduction_dir: Path) -> None:
    for filename in GENERATED_CSVS:
        _read_csv(reproduction_dir / filename)
    for filename in GENERATED_RUNS:
        path = reproduction_dir / "runs" / filename
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = 0
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise AssertionError(f"{path}:{line_number}: blank JSONL record")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise AssertionError(
                        f"{path}:{line_number}: JSONL record is not an object"
                    )
                records += 1
        if not records:
            raise AssertionError(f"{path}: empty run artifact")


def verify(reference_dir: Path, reproduction_dir: Path) -> None:
    reference_dir = reference_dir.resolve()
    reproduction_dir = reproduction_dir.resolve()
    for filename, keys in CSV_KEYS.items():
        _compare_csv(reference_dir / filename, reproduction_dir / filename, keys)

    _compare_values(
        _load_json(reference_dir / "data_audit.json"),
        _load_json(reproduction_dir / "data_audit.json"),
        "data_audit",
    )
    _compare_values(
        _manifest_without_expected_variation(_load_json(reference_dir / "manifest.json")),
        _manifest_without_expected_variation(
            _load_json(reproduction_dir / "manifest.json")
        ),
        "manifest",
    )
    _check_generated_outputs(reproduction_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a full CPU reproduction against committed reference metrics."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("results/corpus_ordinal_reference"),
    )
    parser.add_argument(
        "--reproduction-dir",
        type=Path,
        default=Path("results/reproduction_cpu_portable"),
    )
    args = parser.parse_args()
    verify(args.reference_dir, args.reproduction_dir)
    print(
        "PASS: CPU audit, aggregate metrics, bootstrap intervals, atomic summary, "
        "protocol manifest, and generated outputs match the committed reference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
