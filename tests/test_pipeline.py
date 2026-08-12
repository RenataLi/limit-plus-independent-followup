import unittest
from pathlib import Path

from limitplus.bm25 import CORPUS_ORDINAL, PAPER_QUICKSORT
from limitplus.cli import build_parser
from limitplus.data import DatasetBundle, Document, LimitPlusQuery
from limitplus.logic import ATOM, CompositionResult, NOT_2, QueryPlan
from limitplus.pipeline import (
    _atomic_diagnostic_rows,
    _error_counts,
    _ranking_error_counts,
)


class ErrorDecompositionTest(unittest.TestCase):
    def test_separates_candidate_ranking_and_structural_misses(self):
        counts = _ranking_error_counts(
            gold={"a", "b", "c", "d", "e"},
            candidates={"a", "b", "c", "d", "x"},
            ranking=("a", "x"),
            final_k=2,
        )

        self.assertEqual(
            counts,
            {
                "total_misses": 4,
                "candidate_miss": 1,
                "avoidable_ranking_miss": 1,
                "structural_cutoff_miss": 2,
            },
        )
        self.assertEqual(
            counts["candidate_miss"]
            + counts["avoidable_ranking_miss"]
            + counts["structural_cutoff_miss"],
            counts["total_misses"],
        )

    def test_perfect_oracle_ranking_has_only_structural_cutoff_loss(self):
        gold = {f"d{i}" for i in range(171)}
        ranking = tuple(f"d{i}" for i in range(100))
        counts = _ranking_error_counts(
            gold=gold,
            candidates=gold,
            ranking=ranking,
            final_k=100,
        )

        self.assertEqual(counts["candidate_miss"], 0)
        self.assertEqual(counts["avoidable_ranking_miss"], 0)
        self.assertEqual(counts["structural_cutoff_miss"], 71)
        self.assertEqual(counts["total_misses"], 71)

    def test_rejects_ranked_documents_outside_candidate_set(self):
        with self.assertRaises(AssertionError):
            _ranking_error_counts(
                gold={"a"},
                candidates={"a"},
                ranking=("outside",),
                final_k=1,
            )

    def test_composition_candidate_sub_buckets_sum_to_candidate_miss(self):
        plan = QueryPlan(NOT_2, ("A", "B"))
        query = LimitPlusQuery(
            qid="q",
            text=plan.render(),
            gold=frozenset({"g1", "g2", "g3"}),
            plan=plan,
            original_query="",
        )
        result = CompositionResult(
            ranking=("g1",),
            candidates=frozenset({"g1", "g3"}),
            positive_hits={"A": frozenset({"g1", "g2", "g3"})},
        )

        counts = _error_counts(query, result, candidate_mode="hard_compose")

        self.assertEqual(counts["candidate_miss"], 1)
        self.assertEqual(counts["atomic_miss"], 0)
        self.assertEqual(counts["composition_or_exclusion_miss"], 1)
        self.assertEqual(counts["avoidable_ranking_miss"], 1)
        self.assertEqual(counts["structural_cutoff_miss"], 0)
        self.assertEqual(counts["total_misses"], 2)


class AtomicDiagnosticsTest(unittest.TestCase):
    def test_separates_structural_and_avoidable_atomic_misses(self):
        atom_plan = QueryPlan(ATOM, ("A",))
        not_plan = QueryPlan(NOT_2, ("A", "B"))
        documents = (
            Document("d0", "d0 likes A.", frozenset({"A"})),
            Document("d1", "d1 likes B.", frozenset({"B"})),
            Document("d2", "d2 likes A.", frozenset({"A"})),
        )
        bundle = DatasetBundle(
            documents=documents,
            queries=(
                LimitPlusQuery(
                    qid="0",
                    text=atom_plan.render(),
                    gold=frozenset({"d0", "d2"}),
                    plan=atom_plan,
                    original_query="",
                ),
                LimitPlusQuery(
                    qid="1",
                    text=not_plan.render(),
                    gold=frozenset({"d0", "d2"}),
                    plan=not_plan,
                    original_query="",
                ),
            ),
            attr_to_docs={"A": frozenset({"d0", "d2"}), "B": frozenset({"d1"})},
            doc_ordinal={"d0": 0, "d1": 1, "d2": 2},
        )

        rows, summaries = _atomic_diagnostic_rows(
            bundle,
            {"A": ("d1", "d0", "d2"), "B": ("d0", "d1", "d2")},
            (1, 2, 3),
        )
        a_at_2 = next(
            row for row in rows if row["attribute"] == "A" and row["depth"] == 2
        )
        self.assertEqual(a_at_2["posting_size"], 2)
        self.assertEqual(a_at_2["relevant_retrieved"], 1)
        self.assertEqual(a_at_2["structural_truncation_misses"], 0)
        self.assertEqual(a_at_2["avoidable_retrieval_misses"], 1)
        self.assertEqual(a_at_2["first_relevant_rank"], 2)
        self.assertAlmostEqual(a_at_2["atomic_recall"], 0.5)
        self.assertEqual(a_at_2["positive_occurrences"], 2)

        at_2 = next(row for row in summaries if row["depth"] == 2)
        self.assertEqual(at_2["unique_attributes"], 2)
        self.assertAlmostEqual(at_2["posting_weighted_atomic_recall"], 2 / 3)
        self.assertAlmostEqual(
            at_2["posting_weighted_ceiling_normalized_atomic_recall"],
            2 / 3,
        )
        self.assertAlmostEqual(at_2["macro_atomic_recall"], 0.75)
        self.assertAlmostEqual(
            at_2["positive_occurrence_weighted_atomic_recall"],
            0.5,
        )
        self.assertAlmostEqual(
            at_2[
                "positive_occurrence_weighted_ceiling_normalized_atomic_recall"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            at_2["negative_occurrence_weighted_atomic_recall"],
            1.0,
        )


class CliProtocolTest(unittest.TestCase):
    def test_run_defaults_to_released_quicksort_profile(self):
        args = build_parser().parse_args(["run"])
        self.assertEqual(args.tie_break, PAPER_QUICKSORT)

    def test_run_accepts_portable_corpus_ordinal_profile(self):
        args = build_parser().parse_args(
            ["run", "--tie-break", CORPUS_ORDINAL, "--output-dir", "portable"]
        )
        self.assertEqual(args.tie_break, CORPUS_ORDINAL)
        self.assertEqual(args.output_dir, Path("portable"))


if __name__ == "__main__":
    unittest.main()
