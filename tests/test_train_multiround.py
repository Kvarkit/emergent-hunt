import unittest
from emergent_hunt.train_multiround import run

class MultiRoundLearningTests(unittest.TestCase):
    def test_short_run_finite(self):
        r = run(seed=0, episodes=20)
        self.assertEqual(r['episodes'], 20)
        self.assertTrue(all(abs(x['loss']) < 1e6 for x in r['history']))
        self.assertIn('fixed_grid_eval', r)
        self.assertTrue(0.0 <= r['fixed_grid_eval']['terminal_success'] <= 1.0)

    def test_one_way_small_grid_and_curriculum(self):
        r = run(seed=0, episodes=20, rounds=1, zones=2, type_count=2,
                communication_task='symmetric', curriculum=True)
        self.assertEqual(r['curriculum'], True)
        self.assertEqual(set(r['fixed_grid_eval']),
                         {'zone_score', 'type_score', 'terminal_success'})

if __name__ == '__main__': unittest.main()
