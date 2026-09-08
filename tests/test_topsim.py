import json
import unittest
from pathlib import Path

from emergent_hunt.environment import states
from emergent_hunt.intervention import holistic_policy
from emergent_hunt.topsim import topographic_similarity

EH_INT_JSON = Path(__file__).parent.parent / 'experiments' / 'receiver-slots-0-eh-int.json'
# Pinned at the time the topsim numbers below were measured (board #25694,
# #25714). nadir-codex #25734, gate 3: if this checkpoint is ever
# regenerated -- even with an identical training recipe, since REINFORCE is
# stochastic -- its sha256 will differ and this test fails with a message
# that says so, instead of the topsim regression tests below failing with a
# bare "0.62 != 0.76" that looks like acceptable numeric drift from a
# provenance change.
_EXPECTED_CHECKPOINT_SHA256 = '0a7b677d0983f65373073e931d0f0d63db00307e94612e93ee956fc478831c33'


class TopsimTests(unittest.TestCase):

    def test_eh_int_json_checkpoint_provenance_is_pinned(self):
        report = json.loads(EH_INT_JSON.read_text(encoding='utf-8'))
        actual = report['checkpoint_sha256']
        self.assertEqual(
            actual, _EXPECTED_CHECKPOINT_SHA256,
            f'experiments/receiver-slots-0-eh-int.json now reports checkpoint '
            f'{actual}, not the {_EXPECTED_CHECKPOINT_SHA256} the topsim '
            f'numbers in this file were measured against -- the checkpoint was '
            f'regenerated (even a re-run with the same recipe changes it, '
            f'REINFORCE is stochastic) or replaced. The topsim regression '
            f'tests below will need re-measuring against the new checkpoint, '
            f'not just this constant bumped to match.')
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
