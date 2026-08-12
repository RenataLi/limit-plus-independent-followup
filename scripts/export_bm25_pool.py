"""Export a deterministic BM25 pool without query or document text."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


def _read_rankings(path: Path, *, queries: int, pool_k: int) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            qid = str(row.get("qid"))
            ranking = row.get("ranking")
            if qid in rankings:
                raise ValueError(f"{path}:{line_number}: duplicate qid {qid}")
            if not isinstance(ranking, list) or len(ranking) < pool_k:
                raise ValueError(
                    f"{path}:{line_number}: qid={qid} has fewer than {pool_k} candidates"
                )
            pool = ranking[:pool_k]
            if any(not isinstance(doc_id, str) or not doc_id for doc_id in pool):
                raise ValueError(f"{path}:{line_number}: invalid document id")
            if len(pool) != len(set(pool)):
                raise ValueError(f"{path}:{line_number}: duplicate candidate id")
            rankings[qid] = pool

    expected = set(map(str, range(queries)))
    if set(rankings) != expected:
        missing = sorted(expected - set(rankings), key=int)
        extra = sorted(set(rankings) - expected)
        raise ValueError(f"Unexpected qids; missing={missing[:5]}, extra={extra[:5]}")
    return rankings


def export_pool(
    source: Path,
    output: Path,
    *,
    queries: int = 700,
    pool_k: int = 1000,
) -> dict[str, object]:
    rankings = _read_rankings(source, queries=queries, pool_k=pool_k)
    canonical_lines = [
        json.dumps(
            {"qid": qid, "ranking": rankings[qid]},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for qid in sorted(rankings, key=int)
    ]
    canonical = "".join(canonical_lines).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline="\n", write_through=True
            ) as stream:
                stream.write(canonical.decode("utf-8"))

    return {
        "artifact": output.as_posix(),
        "queries": queries,
        "pool_k": pool_k,
        "candidate_ids": queries * pool_k,
        "uncompressed_sha256": hashlib.sha256(canonical).hexdigest(),
        "compressed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export qid and ordered document IDs from a full BM25 JSONL.gz run; "
            "query text, scores, and qrels are omitted."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=700)
    parser.add_argument("--pool-k", type=int, default=1000)
    args = parser.parse_args()
    if args.queries <= 0 or args.pool_k <= 0:
        raise ValueError("queries and pool-k must be positive")
    summary = export_pool(
        args.source,
        args.output,
        queries=args.queries,
        pool_k=args.pool_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
