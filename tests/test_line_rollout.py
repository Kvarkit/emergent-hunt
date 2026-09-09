import unittest

from emergent_hunt.line_policy import GRULineAgent
from emergent_hunt.line_rollout import (canonical_crossed_rollout, collect_grid,
                                         rollout_episode)


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

    def test_canonical_crossed_policy_solves_all_pairs(self):
        for goal in range(5):
            for trap in range(5):
                if goal == trap:
                    continue
                env, _ = canonical_crossed_rollout(goal, trap)
                self.assertTrue(env.state.success, (goal, trap))


if __name__ == '__main__':
    unittest.main()
