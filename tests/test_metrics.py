import unittest

from limitplus.metrics import (
    ceiling_normalized_recall_at_k,
    evaluate_ranking,
    ndcg_at_k,
    paired_bootstrap_delta,
    recall_at_k,
)


class MetricsTest(unittest.TestCase):
    def test_duplicate_run_ids_do_not_inflate_recall(self):
        ranking = ("a", "a", "x")
        gold = {"a", "b"}
        self.assertEqual(recall_at_k(ranking, gold, 3), 0.5)

    def test_large_gold_set_has_structural_recall_ceiling(self):
        gold = {f"d{i}" for i in range(171)}
        ranking = tuple(f"d{i}" for i in range(100))
        self.assertAlmostEqual(recall_at_k(ranking, gold, 100), 100 / 171)
        self.assertEqual(ceiling_normalized_recall_at_k(ranking, gold, 100), 1.0)
        self.assertEqual(ndcg_at_k(ranking, gold, 100), 1.0)

    def test_evaluate_ranking_reports_all_paper_cutoffs(self):
        values = evaluate_ranking(("a", "b", "x"), {"a", "b"})
        for metric in (
            "recall_at_5",
            "recall_at_20",
            "recall_at_100",
            "ndcg_at_5",
            "ndcg_at_20",
            "ndcg_at_100",
        ):
            self.assertIn(metric, values)

    def test_paired_bootstrap_detects_consistent_gain(self):
        result = paired_bootstrap_delta(
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            strata=["a", "a", "b", "b"],
            samples=500,
            seed=7,
        )
        self.assertAlmostEqual(result.estimate, 0.1)
        self.assertGreater(result.ci_low, 0.0)


if __name__ == "__main__":
    unittest.main()
