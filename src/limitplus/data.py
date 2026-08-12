"""Pinned LIMIT/LIMIT+ loading and fail-fast integrity checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .logic import QueryPlan, TEMPLATES, execute_exact


EXPECTED_SHA256 = {
    "limit_corpus.jsonl": "10209a6916e029c199676caec3349cb925e2a74451161899c30c36e1b9032f82",
    "limit_queries.jsonl": "1ead4c54487728173aa1433778a2ab0f4cf1cf8aeeedb886c05238b32d818594",
    "limit_qrels.jsonl": "a4f9b25b694623c240c6499fb8d8a4896355db9c840f29bd14580d0d4100ea89",
    "limit_plus_queries.jsonl": "c412df625e1530e81012e31e95fe6f339f5e7027f14acdffc8fe0e348eef55cd",
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    attrs: frozenset[str]


@dataclass(frozen=True)
class LimitPlusQuery:
    qid: str
    text: str
    gold: frozenset[str]
    plan: QueryPlan
    original_query: str

    @property
    def template(self) -> str:
        return self.plan.template


@dataclass(frozen=True)
class DatasetBundle:
    documents: tuple[Document, ...]
    queries: tuple[LimitPlusQuery, ...]
    attr_to_docs: Mapping[str, frozenset[str]]
    doc_ordinal: Mapping[str, int]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(data_dir: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/download_data.py before the experiment."
            )
        actual = sha256_file(path)
        observed[filename] = actual
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
    return observed


def parse_likes_attributes(text: str) -> frozenset[str]:
    """Match the released LIMIT+ generator's corpus parser exactly.

    This is intentionally a benchmark-compatibility parser, including its handling
    of the final comma-separated item.  Exact agreement with released qrels is
    checked separately, so a parser change cannot silently redefine relevance.
    """

    if "likes" not in text:
        return frozenset()
    _, after = text.split("likes", 1)
    raw_items = after.strip().split(",")
    attrs: set[str] = set()
    n_items = len(raw_items)
    for index, item in enumerate(raw_items):
        token = item.strip().rstrip(".")
        if n_items > 1 and index == n_items - 1 and " and " in token:
            for part in (part.strip() for part in token.split(" and ")):
                if part.lower().startswith("and "):
                    part = part[4:].strip()
                if part:
                    attrs.add(part)
            continue
        if token.lower().startswith("and "):
            token = token[4:].strip()
        if token:
            attrs.add(token)
    return frozenset(attrs)


def load_corpus(path: Path) -> tuple[tuple[Document, ...], dict[str, frozenset[str]]]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    postings: defaultdict[str, set[str]] = defaultdict(set)
    for raw in iter_jsonl(path):
        doc_id = raw.get("_id")
        text = raw.get("text")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Every LIMIT document must have a non-empty string _id")
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate corpus id: {doc_id!r}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Document {doc_id!r} has empty text")
        attrs = parse_likes_attributes(text)
        if not attrs:
            raise ValueError(f"Document {doc_id!r} yielded no attributes")
        seen_ids.add(doc_id)
        documents.append(Document(doc_id=doc_id, text=text, attrs=attrs))
        for attr in attrs:
            postings[attr].add(doc_id)
    frozen_postings = {attr: frozenset(ids) for attr, ids in postings.items()}
    return tuple(documents), frozen_postings


def load_limit_plus(path: Path) -> tuple[LimitPlusQuery, ...]:
    queries: list[LimitPlusQuery] = []
    seen_qids: set[str] = set()
    for raw in iter_jsonl(path):
        qid = str(raw.get("id"))
        text = raw.get("query")
        docs = raw.get("docs")
        metadata = raw.get("metadata")
        if qid in seen_qids:
            raise ValueError(f"Duplicate LIMIT+ query id: {qid}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Query {qid} has no text")
        if not isinstance(docs, list) or not all(isinstance(x, str) for x in docs):
            raise ValueError(f"Query {qid} has invalid docs")
        if len(docs) != len(set(docs)):
            raise ValueError(f"Query {qid} has duplicate gold document ids")
        if raw.get("num_docs") != len(docs):
            raise ValueError(f"Query {qid} num_docs does not match docs length")
        if not docs:
            raise ValueError(f"Query {qid} has an empty gold set")
        if not isinstance(metadata, dict):
            raise ValueError(f"Query {qid} has no metadata")
        template = metadata.get("template")
        attrs = metadata.get("attrs")
        if not isinstance(template, str) or not isinstance(attrs, list):
            raise ValueError(f"Query {qid} has malformed template metadata")
        if not all(isinstance(attr, str) and attr for attr in attrs):
            raise ValueError(f"Query {qid} has invalid attributes")
        plan = QueryPlan(template=template, attrs=tuple(attrs))
        if plan.render() != text:
            raise ValueError(
                f"Query {qid} text does not match its released template plan:\n"
                f"observed={text!r}\nrendered={plan.render()!r}"
            )
        queries.append(
            LimitPlusQuery(
                qid=qid,
                text=text,
                gold=frozenset(docs),
                plan=plan,
                original_query=str(raw.get("original_query", "")),
            )
        )
        seen_qids.add(qid)
    return tuple(queries)


def load_and_audit(data_dir: Path, *, verify_hashes: bool = True) -> tuple[DatasetBundle, dict]:
    checksums = verify_checksums(data_dir) if verify_hashes else {
        name: sha256_file(data_dir / name) for name in EXPECTED_SHA256
    }
    documents, attr_to_docs = load_corpus(data_dir / "limit_corpus.jsonl")
    queries = load_limit_plus(data_dir / "limit_plus_queries.jsonl")
    doc_ordinal = {doc.doc_id: index for index, doc in enumerate(documents)}

    if len(documents) != 50_000:
        raise ValueError(f"Expected 50,000 LIMIT documents, got {len(documents)}")
    if len(queries) != 700:
        raise ValueError(f"Expected 700 LIMIT+ queries, got {len(queries)}")
    if set(query.qid for query in queries) != set(map(str, range(700))):
        raise ValueError("LIMIT+ query ids must be exactly 0..699")

    template_counts = Counter(query.template for query in queries)
    if set(template_counts) != set(TEMPLATES) or any(
        template_counts[template] != 100 for template in TEMPLATES
    ):
        raise ValueError(f"Unexpected template distribution: {template_counts}")

    unknown_gold = {
        doc_id
        for query in queries
        for doc_id in query.gold
        if doc_id not in doc_ordinal
    }
    if unknown_gold:
        raise ValueError(f"Gold references unknown corpus ids: {sorted(unknown_gold)[:5]}")

    unknown_attrs = {
        attr for query in queries for attr in query.plan.attrs if attr not in attr_to_docs
    }
    if unknown_attrs:
        raise ValueError(f"Query metadata references unknown attributes: {unknown_attrs}")

    oracle_mismatches: list[dict] = []
    for query in queries:
        predicted = execute_exact(query.plan, attr_to_docs)
        if predicted != query.gold:
            oracle_mismatches.append(
                {
                    "qid": query.qid,
                    "false_positive": sorted(predicted - query.gold),
                    "false_negative": sorted(query.gold - predicted),
                }
            )
    if oracle_mismatches:
        first = oracle_mismatches[0]
        raise ValueError(
            f"Exact Boolean oracle disagrees with released qrels on "
            f"{len(oracle_mismatches)} queries; first mismatch: {first}"
        )

    gold_sizes = [len(query.gold) for query in queries]
    audit = {
        "checksums": checksums,
        "corpus_documents": len(documents),
        "unique_corpus_ids": len(doc_ordinal),
        "parsed_attributes": len(attr_to_docs),
        "limit_plus_queries": len(queries),
        "template_counts": dict(template_counts),
        "gold_documents_total": sum(gold_sizes),
        "gold_size_min": min(gold_sizes),
        "gold_size_max": max(gold_sizes),
        "gold_size_mean": sum(gold_sizes) / len(gold_sizes),
        "referenced_unique_documents": len(set().union(*(q.gold for q in queries))),
        "unknown_gold_ids": 0,
        "unknown_attributes": 0,
        "canonical_plan_text_matches": len(queries),
        "exact_boolean_oracle_matches": len(queries),
        "exact_boolean_oracle_mismatches": 0,
    }
    return (
        DatasetBundle(
            documents=documents,
            queries=queries,
            attr_to_docs=attr_to_docs,
            doc_ordinal=doc_ordinal,
        ),
        audit,
    )
