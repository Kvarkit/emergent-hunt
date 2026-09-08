import unittest
from emergent_hunt.train_multiround import run

class MultiRoundLearningTests(unittest.TestCase):
    def test_short_run_finite(self):
        r = run(seed=0, episodes=20)
        self.assertEqual(r['episodes'], 20)
        self.assertTrue(all(abs(x['loss']) < 1e6 for x in r['history']))

if __name__ == '__main__': unittest.main()
