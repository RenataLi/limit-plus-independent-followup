import unittest

import numpy as np

from limitplus.bm25 import (
    CORPUS_ORDINAL,
    PAPER_QUICKSORT,
    BM25Index,
    regex_tokenize,
)


class BM25Test(unittest.TestCase):
    def test_case_is_preserved(self):
        self.assertNotEqual(regex_tokenize("Apple"), regex_tokenize("apple"))

    def test_search_is_reproducible_and_unique(self):
        index = BM25Index(
            ("d0", "d1", "d2"),
            ("Ada likes Apples.", "Bea likes Apples.", "Cal likes Pears."),
            tokenizer=regex_tokenize,
        )
        first = index.search("Who likes Apples?", top_k=3)
        second = index.search("Who likes Apples?", top_k=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first.doc_ids), len(set(first.doc_ids)))

    def test_default_top_k_is_exact_paper_numpy_expression(self):
        scores = np.array([0.0, 2.0, 2.0, 1.0, 2.0, 0.0], dtype=np.float64)
        expected = np.argsort(-scores)[:4]
        observed = BM25Index._top_indices(scores, 4)
        np.testing.assert_array_equal(observed, expected)

    def test_corpus_ordinal_tie_break_is_explicit_ablation(self):
        scores = np.array([0.0, 2.0, 2.0, 1.0, 2.0, 0.0], dtype=np.float64)
        observed = BM25Index._top_indices(
            scores,
            4,
            tie_break=CORPUS_ORDINAL,
        )
        np.testing.assert_array_equal(observed, np.array([1, 2, 4, 3]))

    def test_search_can_override_tie_break(self):
        index = BM25Index(
            tuple(f"d{i}" for i in range(20)),
            tuple("alpha" for _ in range(20)),
            tokenizer=regex_tokenize,
            tie_break=PAPER_QUICKSORT,
        )
        stable = index.search("alpha", top_k=10, tie_break=CORPUS_ORDINAL)
        self.assertEqual(stable.doc_ids, tuple(f"d{i}" for i in range(10)))

    def test_unknown_tie_break_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tie-break mode"):
            BM25Index(
                ("d0",),
                ("alpha",),
                tokenizer=regex_tokenize,
                tie_break="not-a-mode",
            )


if __name__ == "__main__":
    unittest.main()
