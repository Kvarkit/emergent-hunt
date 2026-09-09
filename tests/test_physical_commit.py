import unittest
import torch

from emergent_hunt.physical_commit import (evaluate, expected_commit_reward,
                                           train_exact)


class PhysicalCommitTests(unittest.TestCase):
    def test_exact_commit_gradient_and_training(self):
        sender = torch.zeros(2, 2, requires_grad=True)
        receiver = torch.zeros(2, 2, requires_grad=True)
        (-expected_commit_reward(sender, receiver)).backward()
        self.assertTrue(torch.isfinite(sender.grad).all())
        s, r, history = train_exact(seed=2, steps=1000)
        self.assertGreater(history[-1]['reward'], .95)
        self.assertGreater(evaluate(s, r), .95)
        self.assertAlmostEqual(evaluate(s, r, 'zero'), .5, places=2)
        self.assertLess(evaluate(s, r, 'shuffled'), .05)


if __name__ == '__main__': unittest.main()
