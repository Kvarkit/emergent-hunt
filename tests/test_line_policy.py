import unittest
import torch

from emergent_hunt.line_hunt import LineHunt
from emergent_hunt.line_policy import GRULineAgent, line_observation
from emergent_hunt.train_line import train


class LinePolicyTests(unittest.TestCase):
    def test_reset_and_hidden_are_not_aliased(self):
        agent = GRULineAgent()
        h1, h2 = agent.initial_state(), agent.initial_state()
        self.assertFalse(h1.data_ptr() == h2.data_ptr())
        env = LineHunt(); env.reset(1, 3)
        oa, ia = line_observation(env, "a", agent.vocab)
        _, _, n1 = agent(oa, ia, h1)
        self.assertFalse(torch.equal(h1, n1))
        self.assertTrue(torch.equal(h2, torch.zeros_like(h2)))

    def test_local_observation_does_not_leak_other_private_fact(self):
        a = GRULineAgent()
        e1 = LineHunt(); e1.reset(1, 3)
        e2 = LineHunt(); e2.reset(1, 2)
        o1, _ = line_observation(e1, "a", a.vocab)
        o2, _ = line_observation(e2, "a", a.vocab)
        self.assertTrue(torch.equal(o1, o2))

    def test_late_loss_has_gradient_to_early_message_step(self):
        agent = GRULineAgent()
        h = agent.initial_state()
        env = LineHunt(); env.reset(1, 3)
        observations = []
        for _ in range(2):
            obs, incoming = line_observation(env, "a", agent.vocab)
            message, action, h = agent(obs, incoming, h)
            observations.append((message, action))
            env.step(2, 2)
        loss = observations[-1][1].sum()
        loss.backward()
        self.assertIsNotNone(agent.encoder.weight.grad)
        self.assertGreater(float(agent.encoder.weight.grad.abs().sum()), 0.0)

    def test_short_line_training_is_finite(self):
        result = train(seed=0, episodes=5, horizon=3)
        self.assertEqual(result['episodes'], 5)
        self.assertTrue(result['history'])


if __name__ == "__main__":
    unittest.main()
