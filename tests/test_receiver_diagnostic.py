import unittest
from emergent_hunt.receiver_diagnostic import evaluate_receiver, train_receiver


class ReceiverDiagnosticTests(unittest.TestCase):
    def test_receiver_training_is_finite_and_evaluable(self):
        receiver = train_receiver(seed=0, episodes=10, horizon=4)
        score = evaluate_receiver(receiver, length=5, horizon=4)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == '__main__': unittest.main()
