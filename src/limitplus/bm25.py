"""A small BM25Okapi implementation for the LIMIT corpus."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


Tokenizer = Callable[[str], list[str]]


PAPER_QUICKSORT = "paper_quicksort"
CORPUS_ORDINAL = "corpus_ordinal"
TIE_BREAK_MODES = frozenset((PAPER_QUICKSORT, CORPUS_ORDINAL))


_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", flags=re.UNICODE)


def regex_tokenize(text: str) -> list[str]:
    """Dependency-free, case-preserving tokenizer used only as an ablation."""

    return _TOKEN_RE.findall(text)


def nltk_tokenize(text: str) -> list[str]:
    """Match the authors' case-preserving NLTK word tokenizer.

    ``preserve_line=True`` skips Punkt sentence segmentation.  Every LIMIT document
    and query is one sentence, so tokenization is the same without a runtime data
    download.
    """

    try:
        from nltk.tokenize import word_tokenize
    except ImportError as exc:  # pragma: no cover - exercised in installation errors
        raise RuntimeError(
            "The paper-compatible tokenizer requires NLTK. Install the project "
            "with `pip install -e .`."
        ) from exc
    return word_tokenize(text, preserve_line=True)


TOKENIZERS: dict[str, Tokenizer] = {
    "nltk": nltk_tokenize,
    "regex": regex_tokenize,
}


@dataclass(frozen=True)
class RankedResult:
    doc_ids: tuple[str, ...]
    scores: tuple[float, ...]


class BM25Index:
    """BM25Okapi with Gensim 3/rank-bm25-compatible defaults.

    The implementation uses an inverted index, avoiding the original repository's
    approximately 35 million per-document log messages and obsolete Gensim import.
    The default ranking path deliberately uses the authors' released
    ``np.argsort(-scores)`` expression. NumPy's default quicksort is not stable;
    preserving this detail follows the released ranking path.
    Corpus-ordinal tie breaking remains available as an explicit deterministic
    ablation.
    """

    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        tokenizer: Tokenizer = nltk_tokenize,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tie_break: str = PAPER_QUICKSORT,
    ) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if not doc_ids:
            raise ValueError("Cannot build BM25 over an empty corpus")
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("BM25 document ids must be unique")
        if tie_break not in TIE_BREAK_MODES:
            raise ValueError(
                f"Unknown tie-break mode {tie_break!r}; choose from "
                f"{sorted(TIE_BREAK_MODES)}"
            )
        self.doc_ids = tuple(doc_ids)
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.tie_break = tie_break

        doc_lengths = np.empty(len(texts), dtype=np.float64)
        mutable_postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, text in enumerate(texts):
            counts = Counter(tokenizer(text))
            doc_lengths[doc_index] = sum(counts.values())
            for term, frequency in counts.items():
                mutable_postings[term].append((doc_index, frequency))

        self.doc_lengths = doc_lengths
        self.avgdl = float(np.mean(doc_lengths))
        self._length_norm = self.k1 * (
            1.0 - self.b + self.b * self.doc_lengths / self.avgdl
        )
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_idf: dict[str, float] = {}
        n_docs = len(self.doc_ids)
        for term, values in mutable_postings.items():
            indices = np.fromiter((pair[0] for pair in values), dtype=np.int32)
            frequencies = np.fromiter((pair[1] for pair in values), dtype=np.float64)
            self.postings[term] = (indices, frequencies)
            document_frequency = len(values)
            raw_idf[term] = float(
                np.log(n_docs - document_frequency + 0.5)
                - np.log(document_frequency + 0.5)
            )

        self.average_idf = float(np.mean(tuple(raw_idf.values())))
        floor = self.epsilon * self.average_idf
        self.idf = {
            term: (value if value >= 0.0 else floor)
            for term, value in raw_idf.items()
        }

    @property
    def corpus_size(self) -> int:
        return len(self.doc_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        query_counts = Counter(self.tokenizer(query))
        for term, query_frequency in query_counts.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, term_frequencies = posting
            numerator = term_frequencies * (self.k1 + 1.0)
            denominator = term_frequencies + self._length_norm[indices]
            scores[indices] += (
                self.idf[term]
                * query_frequency
                * numerator
                / denominator
            )
        return scores

    @staticmethod
    def _top_indices(
        scores: np.ndarray,
        top_k: int,
        *,
        tie_break: str = PAPER_QUICKSORT,
    ) -> np.ndarray:
        """Return top-k using either the paper or deterministic tie protocol."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if tie_break not in TIE_BREAK_MODES:
            raise ValueError(
                f"Unknown tie-break mode {tie_break!r}; choose from "
                f"{sorted(TIE_BREAK_MODES)}"
            )
        n_items = scores.shape[0]
        k = min(top_k, n_items)
        if tie_break == PAPER_QUICKSORT:
            # Keep this expression identical to the released baseline. Supplying
            # kind="stable" changes the reproduced Table 2 row because LIMIT has
            # many cutoff ties.
            return np.argsort(-scores)[:k]
        if k == n_items:
            return np.argsort(-scores, kind="stable")

        provisional = np.argpartition(-scores, k - 1)[:k]
        cutoff_score = float(np.min(scores[provisional]))
        better = np.flatnonzero(scores > cutoff_score)
        needed = k - better.size
        tied = np.flatnonzero(scores == cutoff_score)[:needed]
        selected = np.concatenate((better, tied))
        # selected contains at most k items; stable order provides ordinal tie-break.
        return selected[np.argsort(-scores[selected], kind="stable")]

    def search(
        self,
        query: str,
        *,
        top_k: int = 100,
        tie_break: str | None = None,
    ) -> RankedResult:
        scores = self.scores(query)
        indices = self._top_indices(
            scores,
            top_k,
            tie_break=self.tie_break if tie_break is None else tie_break,
        )
        return RankedResult(
            doc_ids=tuple(self.doc_ids[int(index)] for index in indices),
            scores=tuple(float(scores[int(index)]) for index in indices),
        )
