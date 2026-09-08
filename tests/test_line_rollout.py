import unittest

from emergent_hunt.line_policy import GRULineAgent
from emergent_hunt.line_rollout import collect_grid, rollout_episode


class LineRolloutTests(unittest.TestCase):
    def test_trace_has_causal_fields(self):
        a, b = GRULineAgent(), GRULineAgent()
        trace = rollout_episode(a, b, 1, 3)
        self.assertGreater(len(trace), 0)
        self.assertEqual(set(trace[0]) >= {
            'step', 'tokens', 'actions', 'factors', 'observations', 'reward', 'done'}, True)
        self.assertEqual(len(trace[0]['tokens']), 2)

    def test_grid_excludes_equal_positions(self):
        a, b = GRULineAgent(), GRULineAgent()
        records = collect_grid(a, b, length=3, horizon=1)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(r['factors']['goal_position'] != r['factors']['trap_position']
                            for r in records))


if __name__ == '__main__':
    unittest.main()
