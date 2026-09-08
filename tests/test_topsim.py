import json
import unittest
from pathlib import Path

from emergent_hunt.environment import states
from emergent_hunt.intervention import holistic_policy
from emergent_hunt.topsim import topographic_similarity

EH_INT_JSON = Path(__file__).parent.parent / 'experiments' / 'receiver-slots-0-eh-int.json'


class TopsimTests(unittest.TestCase):
    def _learned_and_holistic(self, split):
        target_states = states(3, split, 'pair')
        _, sender, _ = holistic_policy(3)
        holistic = [((s.prey, s.direction, s.trap), sender(s.prey, s.direction))
                    for s in target_states]

        report = json.loads(EH_INT_JSON.read_text(encoding='utf-8'))
        target = {(s.prey, s.direction, s.trap) for s in target_states}
        learned = {}
        for row in report['rows']:
            base = (row['base']['prey'], row['base']['direction'], row['base']['trap'])
            if base in target:
                learned[base] = tuple(row['sent_before'])
        learned = list(learned.items())
        return learned, holistic

    def test_held_out_only_is_degenerate(self):
        # board #25694: naive held-out-only computation can't separate the
        # policies -- by='pair' held-out states span only 3 distinct
        # (prey, direction) meaning-classes.
        learned, holistic = self._learned_and_holistic('test')
        distinct_pairs = {s[:2] for s, _ in learned}
        self.assertEqual(len(distinct_pairs), 3)

        rho_learned, n = topographic_similarity(learned, meaning_factors=(0, 1, 2))
        rho_holistic, _ = topographic_similarity(holistic, meaning_factors=(0, 1, 2))
        self.assertEqual(n, 36)
        # both come back high and close together -- not a usable separation
        self.assertGreater(rho_learned, 0.8)
        self.assertGreater(rho_holistic, 0.6)
        self.assertLess(rho_learned - rho_holistic, 0.3)

    def test_full_corpus_pd_only_separates_learned_from_holistic(self):
        # board #25714: full 27-state corpus + meaning-distance over
        # (prey, direction) only (trap is never carried by the message).
        learned, holistic = self._learned_and_holistic('all')
        rho_learned, n = topographic_similarity(learned, meaning_factors=(0, 1))
        rho_holistic, _ = topographic_similarity(holistic, meaning_factors=(0, 1))
        self.assertEqual(n, len(learned) * (len(learned) - 1) // 2)
        self.assertGreater(rho_learned, 0.6)
        self.assertLess(rho_holistic, 0.3)
        self.assertGreater(rho_learned - rho_holistic, 0.4)

    def test_undefined_when_no_meaning_variance(self):
        rho, n = topographic_similarity([((0, 0, 0), (1,)), ((0, 0, 1), (2,))],
                                         meaning_factors=(0, 1))
        self.assertIsNone(rho)
        self.assertEqual(n, 1)


if __name__ == '__main__':
    unittest.main()
