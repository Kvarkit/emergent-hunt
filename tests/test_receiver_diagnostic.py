import unittest
from emergent_hunt.receiver_diagnostic import (evaluate_receiver, evaluate_staged,
                                                train_receiver,
                                                train_sender_with_frozen_receiver)


class ReceiverDiagnosticTests(unittest.TestCase):
    def test_receiver_training_is_finite_and_evaluable(self):
        receiver = train_receiver(seed=0, episodes=10, horizon=4)
        score = evaluate_receiver(receiver, length=5, horizon=4)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_sender_can_train_against_frozen_receiver(self):
        receiver = train_receiver(seed=0, episodes=5, horizon=4)
        sender = train_sender_with_frozen_receiver(receiver, seed=0, episodes=5, horizon=4)
        self.assertEqual(sender.vocab, receiver.vocab)

    def test_staged_evaluation_is_bounded(self):
        receiver = train_receiver(seed=0, episodes=5, horizon=4)
        sender = train_sender_with_frozen_receiver(receiver, seed=0, episodes=5, horizon=4)
        self.assertTrue(0.0 <= evaluate_staged(sender, receiver, horizon=4) <= 1.0)


if __name__ == '__main__': unittest.main()
