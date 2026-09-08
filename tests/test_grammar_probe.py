import unittest

from emergent_hunt.grammar_probe import (mutual_information, normalized_mi, probe,
                                          shuffle_tokens, stratified_probe)


class GrammarProbeTests(unittest.TestCase):
    def test_information_bounds(self):
        self.assertEqual(mutual_information([0, 1, 0, 1], [0, 1, 0, 1]), 1.0)
        self.assertAlmostEqual(normalized_mi([0, 0, 1, 1], [1, 1, 1, 1]), 0.0)

    def test_compositional_positions_are_detected(self):
        records = []
        for typ in (0, 1):
            for direction in (0, 1):
                records.append({"tokens": (typ, direction),
                                "factors": {"direction": direction, "type": typ}})
        result = probe(records, vocab=4)
        self.assertGreater(result["factors"]["type"]["positions"]["0"]["normalized_mi"], .9)
        self.assertGreater(result["factors"]["direction"]["positions"]["1"]["normalized_mi"], .9)
        self.assertTrue(result["candidate_factorized_structure"])

    def test_holistic_code_has_collisions_different_from_slots(self):
        records = [
            {"tokens": (0, 0), "factors": {"type": 0, "direction": 0}},
            {"tokens": (1, 1), "factors": {"type": 0, "direction": 1}},
            {"tokens": (2, 2), "factors": {"type": 1, "direction": 0}},
            {"tokens": (3, 3), "factors": {"type": 1, "direction": 1}},
        ]
        result = probe(records, vocab=4)
        self.assertEqual(result["unique_token_sequences"], 4)
        self.assertEqual(result["token_sequence_collision_rate"], 0.0)
        self.assertFalse(result["candidate_factorized_structure"])

    def test_shuffle_preserves_factors_but_breaks_alignment(self):
        records = [{"tokens": (x % 2,), "factors": {"type": x % 2}}
                    for x in range(8)]
        shuffled = shuffle_tokens(records, seed=3)
        self.assertEqual([r["factors"] for r in shuffled], [r["factors"] for r in records])
        self.assertLess(probe(shuffled, 4)["factors"]["type"]["sequence_normalized_mi"], 1.0)

    def test_stratified_probe_reports_coverage(self):
        records = [{"tokens": (0,), "split": "train",
                    "factors": {"type": 0, "direction": 1}},
                   {"tokens": (1,), "split": "test",
                    "factors": {"type": 1, "direction": 0}}]
        report = stratified_probe(records, 2)
        self.assertEqual(set(report), {"train", "test"})
        self.assertEqual(report["test"]["factor_combinations"], [(0, 1)])

    def test_probe_warns_on_tiny_near_unique_sample(self):
        records = [{"tokens": (i,), "factors": {"type": i}} for i in range(4)]
        result = probe(records, 4)
        self.assertTrue(any("small_sample" in w for w in result["warnings"]))
        self.assertTrue(any("near_unique" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
