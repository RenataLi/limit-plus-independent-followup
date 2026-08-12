import unittest

from limitplus.data import parse_likes_attributes


class CorpusParserTest(unittest.TestCase):
    def test_matches_generator_comma_and_final_and_rules(self):
        text = "Ada Example likes Apples, Candy Canes, Research and Jazz."
        self.assertEqual(
            parse_likes_attributes(text),
            frozenset({"Apples", "Candy Canes", "Research", "Jazz"}),
        )

    def test_word_and_inside_nonfinal_attribute_is_preserved(self):
        text = "Ada Example likes Rock and Roll, Apples, Jazz and Tea."
        self.assertIn("Rock and Roll", parse_likes_attributes(text))

    def test_missing_likes_is_empty(self):
        self.assertEqual(parse_likes_attributes("No predicate here."), frozenset())


if __name__ == "__main__":
    unittest.main()
