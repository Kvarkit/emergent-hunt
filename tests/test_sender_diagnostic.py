import unittest
import torch

from emergent_hunt.sender_diagnostic import (evaluate, exact_sender_objective,
                                             train_sender_exact,
                                             train_sender_sampled)


class SenderDiagnosticTests(unittest.TestCase):
    def test_exact_sender_converges(self):
        logits, history = train_sender_exact(seed=1, steps=500)
        self.assertGreater(history[-1]['reward'], .95)
        self.assertEqual(evaluate(logits), 1.0)

    def test_sampled_sender_converges(self):
        logits = train_sender_sampled(seed=2, updates=3000)
        self.assertGreaterEqual(evaluate(logits), .8)

    def test_exact_gradient_is_finite(self):
        logits = torch.zeros(5, 5, requires_grad=True)
        (-exact_sender_objective(logits)).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__': unittest.main()
