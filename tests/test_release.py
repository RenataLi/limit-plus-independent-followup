import csv
import json
import unittest
from pathlib import Path

from limitplus.release import load_release_metrics


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_release_metrics(ROOT)

    def test_headline_metrics_are_frozen(self):
        metrics = self.metrics
        self.assertAlmostEqual(metrics.bm25.recall_at_100, 0.8369301694523535, places=12)
        self.assertAlmostEqual(metrics.bm25.ndcg_at_20, 0.7853256160343247, places=12)
        self.assertAlmostEqual(metrics.qwen.recall_at_100, 0.9036073758347879, places=12)
        self.assertAlmostEqual(metrics.qwen.ndcg_at_20, 0.8862284647703489, places=12)
        self.assertAlmostEqual(metrics.exact.recall_at_100, 0.9274954079426914, places=12)
        self.assertAlmostEqual(metrics.exact.ndcg_at_20, 0.9914261260246955, places=12)
        self.assertAlmostEqual(
            metrics.hard_composition.recall_at_100,
            0.6392075846639946,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.hybrid.recall_at_100, 0.9356129928170202, places=12
        )
        self.assertAlmostEqual(
            metrics.oracle.recall_at_100, 0.9419801582231779, places=12
        )

    def test_neural_deltas_and_diagnostics_are_frozen(self):
        metrics = self.metrics
        self.assertAlmostEqual(
            metrics.qwen_vs_bm25["recall_at_100"].delta,
            0.06667720638243442,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.qwen_vs_bm25["ndcg_at_20"].delta,
            0.10090284873602429,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.exact_vs_qwen["recall_at_100"].delta,
            0.0238880321079035,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.exact_vs_qwen["ndcg_at_20"].delta,
            0.10519766125434643,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.recall_gap_closed, 0.736233984406147, places=12
        )
        self.assertAlmostEqual(
            metrics.ndcg20_gap_closed, 0.4895807814388157, places=12
        )
        self.assertAlmostEqual(
            metrics.negation_share_of_net_ndcg20_gain,
            0.982189640318077,
            places=12,
        )
        self.assertAlmostEqual(
            metrics.atomic_positive_recall_at_1000,
            0.7041441578083519,
            places=12,
        )
        self.assertAlmostEqual(metrics.micro_pool_coverage, 24693 / 25860, places=12)
        self.assertEqual(metrics.queries, 700)
        self.assertEqual(metrics.pairs_scored, 700_000)
        self.assertEqual(metrics.bootstrap_draws, 10_000)

        expected_intervals = {
            ("qwen", "recall_at_100"): (0.056138660913407495, 0.07743922179268915),
            ("qwen", "ndcg_at_20"): (0.08674213213391474, 0.11521457281743767),
            ("exact", "recall_at_100"): (0.01948673553454095, 0.028816840212080096),
            ("exact", "ndcg_at_20"): (0.0948990168792711, 0.11575024142967663),
        }
        for (comparison, metric), (ci_low, ci_high) in expected_intervals.items():
            interval = (
                metrics.qwen_vs_bm25[metric]
                if comparison == "qwen"
                else metrics.exact_vs_qwen[metric]
            )
            self.assertAlmostEqual(interval.ci_low, ci_low, places=12)
            self.assertAlmostEqual(interval.ci_high, ci_high, places=12)
            self.assertEqual(interval.probability_positive, 1.0)

    def test_three_way_conjunction_crossover_is_frozen(self):
        row = next(
            item for item in self.metrics.templates if item.label == "A AND B AND C"
        )
        self.assertAlmostEqual(row.bm25_ndcg_at_20, 0.9941846410367658, places=12)
        self.assertAlmostEqual(row.qwen_ndcg_at_20, 0.8453011343335229, places=12)
        self.assertAlmostEqual(row.exact_ndcg_at_20, 1.0, places=12)
        self.assertAlmostEqual(row.candidate_recall, 1.0, places=12)

    def test_portable_reference_is_separate_from_reported_profile(self):
        main_manifest = json.loads(
            (ROOT / "results" / "main" / "manifest.json").read_text(encoding="utf-8")
        )
        portable_dir = ROOT / "results" / "corpus_ordinal_reference"
        portable_manifest = json.loads(
            (portable_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(main_manifest["bm25"]["tie_break"], "paper_quicksort")
        self.assertEqual(portable_manifest["bm25"]["tie_break"], "corpus_ordinal")
        with (portable_dir / "summary_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        bm25 = next(
            row
            for row in rows
            if row["method"] == "full_query_bm25"
            and row["group_type"] == "overall"
        )
        self.assertAlmostEqual(
            float(bm25["recall_at_100"]), 0.83826435453972, places=12
        )
        self.assertAlmostEqual(
            float(bm25["ndcg_at_20"]), 0.7856536046711865, places=12
        )


if __name__ == "__main__":
    unittest.main()
