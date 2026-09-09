import unittest
import torch

from emergent_hunt.signalling_game import evaluate, expected_reward, train_exact


class SignallingGameTests(unittest.TestCase):
    def test_exact_reward_matches_manual_deterministic_code(self):
        sender = torch.tensor([[10., -10.], [-10., 10.]])
        receiver = torch.tensor([[10., -10.], [-10., 10.]])
        self.assertGreater(float(expected_reward(sender, receiver)), .99)
        self.assertGreater(evaluate(sender, receiver, "actual"), .99)
        self.assertLess(evaluate(sender, receiver, "shuffled"), .01)

    def test_exact_gradient_is_finite(self):
        sender = torch.zeros(2, 2, requires_grad=True)
        receiver = torch.zeros(2, 2, requires_grad=True)
        (-expected_reward(sender, receiver)).backward()
        self.assertTrue(torch.isfinite(sender.grad).all())
        self.assertTrue(torch.isfinite(receiver.grad).all())

    def test_exact_training_reaches_communication_solution(self):
        result = train_exact(seed=3, steps=1000)
        self.assertGreater(result['history'][-1]['reward'], .95)
        self.assertGreater(evaluate(result['sender'], result['receiver']), .95)


if __name__ == '__main__':
    unittest.main()
