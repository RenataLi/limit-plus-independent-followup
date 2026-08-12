"""Logical plans and symbolic composition for the seven released LIMIT+ templates.

The benchmark includes gold-producing template metadata.  Using that metadata is an
*oracle decomposition*: it isolates retrieval and composition from language parsing.
It must not be presented as an automatic semantic parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping, Sequence


ATOM = "_"
OR_2 = "_ or _"
OR_3 = "_ or _ or _"
AND_2 = "_ that are also _"
AND_3 = "_ that are also both _ and _"
AND_NOT_3 = "_ that are also _ but not _"
NOT_2 = "_ that are not _"

TEMPLATES = (ATOM, OR_2, OR_3, AND_2, AND_3, AND_NOT_3, NOT_2)
OR_TEMPLATES = frozenset((OR_2, OR_3))
AND_TEMPLATES = frozenset((AND_2, AND_3))
NEGATION_TEMPLATES = frozenset((AND_NOT_3, NOT_2))


@dataclass(frozen=True)
class QueryPlan:
    """A template-grounded logical plan for one LIMIT+ query."""

    template: str
    attrs: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = {
            ATOM: 1,
            OR_2: 2,
            OR_3: 3,
            AND_2: 2,
            AND_3: 3,
            AND_NOT_3: 3,
            NOT_2: 2,
        }
        if self.template not in expected:
            raise ValueError(f"Unknown LIMIT+ template: {self.template!r}")
        if len(self.attrs) != expected[self.template]:
            raise ValueError(
                f"Template {self.template!r} expects {expected[self.template]} "
                f"attributes, got {len(self.attrs)}"
            )
        if any(not attr for attr in self.attrs):
            raise ValueError("Attributes must be non-empty strings")

    @property
    def n_atoms(self) -> int:
        return len(self.attrs)

    @property
    def positive_attrs(self) -> tuple[str, ...]:
        if self.template in NEGATION_TEMPLATES:
            return self.attrs[:-1]
        return self.attrs

    @property
    def negative_attrs(self) -> tuple[str, ...]:
        if self.template in NEGATION_TEMPLATES:
            return self.attrs[-1:]
        return ()

    @property
    def positive_operator(self) -> str:
        if self.template in OR_TEMPLATES:
            return "or"
        if self.template == ATOM or self.template == NOT_2:
            return "atom"
        return "and"

    def render(self) -> str:
        """Render the exact canonical wording used by the released generator."""

        a = self.attrs
        if self.template == ATOM:
            return f"Who likes {a[0]}?"
        if self.template == OR_2:
            return f"Who likes {a[0]} or {a[1]}?"
        if self.template == OR_3:
            return f"Who likes {a[0]} or {a[1]} or {a[2]}?"
        if self.template == AND_2:
            return f"Who likes {a[0]} and also likes {a[1]}?"
        if self.template == AND_3:
            return f"Who likes {a[0]} and also likes both {a[1]} and {a[2]}?"
        if self.template == AND_NOT_3:
            return f"Who likes {a[0]} and also likes {a[1]} but not {a[2]}?"
        if self.template == NOT_2:
            return f"Who likes {a[0]} but not {a[1]}?"
        raise AssertionError("unreachable")


def execute_exact(
    plan: QueryPlan,
    attr_to_docs: Mapping[str, AbstractSet[str]],
) -> set[str]:
    """Execute a plan over exact corpus postings."""

    postings = [set(attr_to_docs.get(attr, ())) for attr in plan.attrs]
    if plan.template == ATOM:
        return postings[0]
    if plan.template in OR_TEMPLATES:
        return set().union(*postings)
    if plan.template in AND_TEMPLATES:
        return set.intersection(*postings)
    if plan.template == AND_NOT_3:
        return (postings[0] & postings[1]) - postings[2]
    if plan.template == NOT_2:
        return postings[0] - postings[1]
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class CompositionResult:
    ranking: tuple[str, ...]
    candidates: frozenset[str]
    positive_hits: Mapping[str, frozenset[str]]


def _positive_candidates(
    plan: QueryPlan,
    positive_hits: Mapping[str, AbstractSet[str]],
) -> set[str]:
    sets = [set(positive_hits[attr]) for attr in plan.positive_attrs]
    if plan.positive_operator == "or":
        return set().union(*sets)
    if plan.positive_operator == "and":
        return set.intersection(*sets)
    return sets[0]


def compose_rankings(
    plan: QueryPlan,
    atomic_rankings: Mapping[str, Sequence[str]],
    *,
    depth: int,
    final_k: int,
    doc_ordinal: Mapping[str, int],
    exact_attr_to_docs: Mapping[str, AbstractSet[str]] | None = None,
    exact_negation: bool = True,
    candidate_mode: str = "hard_compose",
    rrf_constant: int = 60,
) -> CompositionResult:
    """Retrieve atomic lists, compose them, and rank a fixed-size final output.

    ``hard_compose`` applies AND/OR directly to top-``depth`` lists.  The alternative
    ``union_verify`` first unions positive lists, then verifies the complete Boolean
    predicate against exact corpus facts.  The latter is a disclosed symbolic hybrid,
    not a pure retrieval model.

    For negation in ``hard_compose``, the primary variant uses an exact lexical
    membership filter; treating absence from a truncated negative list as evidence
    of NOT is available only as an explicitly labelled approximation.
    """

    if depth <= 0 or final_k <= 0:
        raise ValueError("depth and final_k must be positive")
    missing = [attr for attr in plan.attrs if attr not in atomic_rankings]
    if missing:
        raise KeyError(f"Missing atomic rankings for attributes: {missing}")

    truncated = {
        attr: tuple(dict.fromkeys(atomic_rankings[attr][:depth]))
        for attr in plan.attrs
    }
    positive_hits = {
        attr: frozenset(truncated[attr]) for attr in plan.positive_attrs
    }
    if candidate_mode == "hard_compose":
        candidates = _positive_candidates(plan, positive_hits)
        if plan.negative_attrs:
            negative_attr = plan.negative_attrs[0]
            if exact_negation:
                if exact_attr_to_docs is None:
                    raise ValueError("exact_attr_to_docs is required for exact negation")
                excluded = set(exact_attr_to_docs.get(negative_attr, ()))
            else:
                excluded = set(truncated[negative_attr])
            candidates.difference_update(excluded)
    elif candidate_mode == "union_verify":
        if exact_attr_to_docs is None:
            raise ValueError("exact_attr_to_docs is required for union_verify")
        retrieved_union = set().union(*(positive_hits.values()))
        candidates = retrieved_union & execute_exact(plan, exact_attr_to_docs)
    else:
        raise ValueError(
            f"Unknown candidate_mode {candidate_mode!r}; expected hard_compose or union_verify"
        )

    # Rank-based fusion avoids comparing raw BM25 scores across different queries.
    score: dict[str, float] = {doc_id: 0.0 for doc_id in candidates}
    for attr in plan.positive_attrs:
        for rank, doc_id in enumerate(truncated[attr], start=1):
            if doc_id in score:
                score[doc_id] += 1.0 / (rrf_constant + rank)

    ranked = sorted(
        candidates,
        key=lambda doc_id: (-score[doc_id], doc_ordinal[doc_id]),
    )[:final_k]
    return CompositionResult(
        ranking=tuple(ranked),
        candidates=frozenset(candidates),
        positive_hits={k: frozenset(v) for k, v in positive_hits.items()},
    )


def positive_atomic_condition(
    plan: QueryPlan,
    doc_id: str,
    positive_hits: Mapping[str, AbstractSet[str]],
) -> bool:
    """Whether a document survives the positive atomic retrieval stage."""

    membership = [doc_id in positive_hits[attr] for attr in plan.positive_attrs]
    if plan.positive_operator == "or":
        return any(membership)
    if plan.positive_operator == "and":
        return all(membership)
    return membership[0]
