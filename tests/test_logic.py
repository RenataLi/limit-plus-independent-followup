import unittest

from limitplus.logic import (
    AND_2,
    AND_3,
    AND_NOT_3,
    ATOM,
    NOT_2,
    OR_2,
    OR_3,
    QueryPlan,
    compose_rankings,
    execute_exact,
)


class LogicTest(unittest.TestCase):
    def setUp(self):
        self.postings = {
            "Apple": {"d1", "d2", "d4"},
            "Pineapple": {"d3"},
            "Blue": {"d2", "d3", "d4"},
            "Round": {"d1", "d2", "d3"},
        }

    def test_all_released_templates(self):
        cases = [
            (QueryPlan(ATOM, ("Apple",)), {"d1", "d2", "d4"}),
            (QueryPlan(OR_2, ("Apple", "Pineapple")), {"d1", "d2", "d3", "d4"}),
            (QueryPlan(OR_3, ("Apple", "Pineapple", "Blue")), {"d1", "d2", "d3", "d4"}),
            (QueryPlan(AND_2, ("Apple", "Blue")), {"d2", "d4"}),
            (QueryPlan(AND_3, ("Apple", "Blue", "Round")), {"d2"}),
            (QueryPlan(AND_NOT_3, ("Apple", "Blue", "Round")), {"d4"}),
            (QueryPlan(NOT_2, ("Apple", "Blue")), {"d1"}),
        ]
        for plan, expected in cases:
            with self.subTest(plan=plan):
                self.assertEqual(execute_exact(plan, self.postings), expected)

    def test_exact_attribute_matching_avoids_substrings(self):
        self.assertEqual(execute_exact(QueryPlan(ATOM, ("Apple",)), self.postings), {"d1", "d2", "d4"})
        self.assertNotIn("d3", execute_exact(QueryPlan(ATOM, ("Apple",)), self.postings))

    def test_rendered_canonical_queries(self):
        self.assertEqual(
            QueryPlan(AND_NOT_3, ("A", "B", "C")).render(),
            "Who likes A and also likes B but not C?",
        )
        self.assertEqual(
            QueryPlan(AND_3, ("A", "B", "C")).render(),
            "Who likes A and also likes both B and C?",
        )

    def test_exact_negation_does_not_treat_truncation_as_not(self):
        plan = QueryPlan(NOT_2, ("Apple", "Blue"))
        rankings = {
            "Apple": ("d4", "d1", "d2"),
            # d4 is truly Blue, but falls below depth=1 in this approximate list.
            "Blue": ("d3", "d4", "d2"),
        }
        ordinal = {f"d{i}": i for i in range(1, 5)}
        exact = compose_rankings(
            plan,
            rankings,
            depth=1,
            final_k=10,
            doc_ordinal=ordinal,
            exact_attr_to_docs=self.postings,
            exact_negation=True,
        )
        approximate = compose_rankings(
            plan,
            rankings,
            depth=1,
            final_k=10,
            doc_ordinal=ordinal,
            exact_negation=False,
        )
        self.assertNotIn("d4", exact.candidates)
        self.assertIn("d4", approximate.candidates)

    def test_empty_intersection(self):
        plan = QueryPlan(AND_2, ("Apple", "Pineapple"))
        result = compose_rankings(
            plan,
            {"Apple": ("d1",), "Pineapple": ("d3",)},
            depth=1,
            final_k=100,
            doc_ordinal={"d1": 0, "d3": 1},
        )
        self.assertEqual(result.ranking, ())
        self.assertEqual(result.candidates, frozenset())

    def test_union_then_verify_can_rescue_and_from_one_atomic_list(self):
        plan = QueryPlan(AND_2, ("Apple", "Blue"))
        # d4 is missed by Blue top-1 but retrieved by Apple top-1.
        rankings = {"Apple": ("d4",), "Blue": ("d3",)}
        hard = compose_rankings(
            plan,
            rankings,
            depth=1,
            final_k=100,
            doc_ordinal={"d1": 0, "d2": 1, "d3": 2, "d4": 3},
            exact_attr_to_docs=self.postings,
            candidate_mode="hard_compose",
        )
        verified = compose_rankings(
            plan,
            rankings,
            depth=1,
            final_k=100,
            doc_ordinal={"d1": 0, "d2": 1, "d3": 2, "d4": 3},
            exact_attr_to_docs=self.postings,
            candidate_mode="union_verify",
        )
        self.assertEqual(hard.candidates, frozenset())
        self.assertEqual(verified.candidates, frozenset({"d4"}))


if __name__ == "__main__":
    unittest.main()
