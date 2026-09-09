import unittest
from emergent_hunt.train_line import train, evaluate_dynamic


class DynamicContractTests(unittest.TestCase):
    def test_single_reward_source_and_matched_evaluation(self):
        result = train(seed=0, episodes=10, crossed=True, progress_weight=.1,
                       step_cost=.01, trigger_delay=2)
        self.assertEqual(result['gamma'], 1)
        for row in result['history']:
            self.assertAlmostEqual(row['task_return'],
                                   float(row['success']) - .01*row['elapsed'])
        for mode in ('actual', 'zero'):
            metrics = evaluate_dynamic(result, mode)
            # Uniform 20-state grid has initial total distance exactly four.
            self.assertAlmostEqual(metrics['return']-metrics['task_return'], .4)
            self.assertTrue(0 <= metrics['success'] <= 1)

    def test_rejects_incompatible_discount(self):
        with self.assertRaises(ValueError):
            train(episodes=1, progress_weight=.1, gamma=.97)
