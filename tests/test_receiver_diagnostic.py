import unittest
import torch
from emergent_hunt.line_policy import GRULineAgent
from emergent_hunt.receiver_diagnostic import (evaluate_receiver, evaluate_staged,
                                                train_receiver,
                                                train_sender_with_frozen_receiver)


class ReceiverDiagnosticTests(unittest.TestCase):
    def test_training_and_evaluation_repeat_first_delivered_token(self):
        receiver = GRULineAgent(vocab=5, hidden_dim=32)
        episodes = []

        def record(module, args):
            observation, incoming, _ = args
            if observation[0, 2].item() == 0:
                episodes.append([])
            else:
                episodes[-1].append(incoming.detach().clone())

        hook = receiver.register_forward_pre_hook(record)
        try:
            sender = train_sender_with_frozen_receiver(
                receiver, seed=0, episodes=10, horizon=4)
            training_count = len(episodes)
            evaluate_staged(sender, receiver, horizon=4)
        finally:
            hook.remove()
        self.assertEqual(training_count, 10)
        self.assertEqual(len(episodes), 30)
        for delivered in episodes:
            self.assertGreaterEqual(len(delivered), 1)
            for token in delivered[1:]:
                self.assertTrue(torch.equal(delivered[0], token),
                                'channel replaced the committed token')

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
