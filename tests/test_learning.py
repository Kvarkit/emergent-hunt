import unittest
import torch
from torch.distributions import Categorical
from emergent_hunt.train import features, mlp, corpus
from emergent_hunt.environment import SymbolicHunt, states


class LearningTests(unittest.TestCase):
    def test_factorized_policy_gradient_matches_enumerated_joint(self):
        logits = torch.tensor([[.2, -.1, .7], [.1, .4, -.2]], requires_grad=True)
        dist = Categorical(logits=logits)
        exact = torch.autograd.grad(-dist.probs[0, 1]*dist.probs[1, 2], logits, retain_graph=True)[0]
        loss = 0
        for p in range(3):
            for d in range(3):
                prob = (dist.probs[0, p]*dist.probs[1, d]).detach()
                reward = float((p, d) == (1, 2))
                loss = loss - prob*(reward-.4)*(dist.logits[0, p]+dist.logits[1, d])
        torch.testing.assert_close(exact, torch.autograd.grad(loss, logits)[0])

    def test_training_corpus_excludes_heldout_sender_pairs(self):
        train = {tuple(row[:2]) for row in corpus('train', 'pair').tolist()}
        test = {tuple(row[:2]) for row in corpus('test', 'pair').tolist()}
        self.assertEqual((len(train), len(test)), (6, 3))
        self.assertFalse(train & test)

    def test_reinforce_gradient_matches_exact_expected_reward(self):
        logits = torch.tensor([.2, -.1, .7], requires_grad=True)
        rewards = torch.tensor([0., 1., 0.])
        dist = Categorical(logits=logits)
        exact = torch.autograd.grad(-(dist.probs * rewards).sum(), logits, retain_graph=True)[0]
        surrogate = -(dist.probs.detach() * (rewards-.4) * dist.logits).sum()
        estimate = torch.autograd.grad(surrogate, logits)[0]
        torch.testing.assert_close(exact, estimate)

    def test_actor_critic_baseline_is_detached(self):
        value = torch.tensor([.4], requires_grad=True)
        logits = torch.zeros(3, requires_grad=True)
        loss = -((torch.tensor([1.])-value).detach() * Categorical(logits=logits).log_prob(torch.tensor(1))).mean()
        loss.backward()
        self.assertIsNone(value.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_vector_reward_matches_environment(self):
        env = SymbolicHunt()
        for s in states():
            # Exhaustive oracle access here belongs to tests, never policy observations.
            for a in range(9):
                env.reset()
                env._state = s
                env.send((s.prey, s.direction))
                reward, _, _ = env.step((a//3, a%3, s.trap))
                self.assertEqual(reward, float(a == s.prey*3+s.direction))


if __name__ == '__main__':
    unittest.main()
