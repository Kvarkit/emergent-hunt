import unittest

from emergent_hunt.grammar_probe import mutual_information, normalized_mi, probe


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


if __name__ == "__main__":
    unittest.main()
