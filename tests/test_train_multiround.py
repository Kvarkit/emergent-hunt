import unittest
import math
from emergent_hunt.train_multiround import run, _route_messages
import torch

class MultiRoundLearningTests(unittest.TestCase):
    def test_straight_through_wire_keeps_gradient_and_direction(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        wire = torch.nn.functional.gumbel_softmax(logits, hard=True)
        last_a, last_b = _route_messages(wire, None, 4, 'symmetric')
        self.assertIs(last_a, wire)
        self.assertEqual(last_b.sum().item(), 0)
        (last_a * torch.arange(4)).sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_differentiable_training_runs_end_to_end(self):
        result = run(seed=0, episodes=5, rounds=2, zones=2, type_count=2,
                     differentiable_messages=True)
        self.assertTrue(math.isfinite(result['history'][-1]['loss']))

    def test_short_run_finite(self):
        r = run(seed=0, episodes=20)
        self.assertEqual(r['episodes'], 20)
        self.assertTrue(all(abs(x['loss']) < 1e6 for x in r['history']))
        self.assertIn('fixed_grid_eval', r)
        self.assertIn('fixed_grid_nomessage_eval', r)
        self.assertIn('fixed_grid_shift_eval', r)
        self.assertTrue(0.0 <= r['fixed_grid_swap_eval']['terminal_success'] <= 1.0)
        self.assertTrue(0.0 <= r['fixed_grid_eval']['terminal_success'] <= 1.0)
        self.assertTrue(0.0 <= r['fixed_grid_nomessage_eval']['terminal_success'] <= 1.0)

    def test_one_way_small_grid_and_curriculum(self):
        r = run(seed=0, episodes=20, rounds=1, zones=2, type_count=2,
                communication_task='symmetric', curriculum=True)
        self.assertEqual(r['curriculum'], True)
        self.assertEqual(set(r['fixed_grid_eval']),
                         {'zone_score', 'type_score', 'terminal_success'})

    def test_message_routing_regression(self):
        ma, mb = torch.tensor([1]), torch.tensor([2])
        last_a, last_b = _route_messages(ma, mb, 4, 'symmetric', True)
        self.assertEqual(last_a.argmax().item(), 1)  # B receives A
        self.assertEqual(last_b.argmax().item(), 2)  # A receives B
        last_a, last_b = _route_messages(ma, None, 4, 'one_way', True)
        self.assertEqual(last_a.argmax().item(), 1)
        self.assertEqual(float(last_b.sum()), 0.0)

    def test_holdout_split_is_evaluated(self):
        r = run(seed=3, episodes=30, holdout_mod=3, hidden_dim=16,
                communication_task='symmetric', coupled=True)
        self.assertIsNotNone(r['heldout_grid_eval'])
        self.assertEqual(set(r['heldout_grid_eval']),
                         {'zone_score', 'type_score', 'terminal_success'})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in r['heldout_grid_eval'].values()))

    def test_auxiliary_decay_is_finite(self):
        r = run(seed=4, episodes=40, receiver_aux=.1, sender_aux=.05,
                auxiliary_decay=True)
        self.assertTrue(r['auxiliary_decay'])
        self.assertTrue(math.isfinite(r['history'][-1]['loss']))

    def test_action_auxiliary_is_finite(self):
        r = run(seed=5, episodes=30, action_aux=.2, auxiliary_decay=True)
        self.assertEqual(r['action_aux'], .2)
        self.assertTrue(math.isfinite(r['history'][-1]['loss']))

    def test_message_examples_are_exposed(self):
        r = run(seed=6, episodes=30)
        examples = r['protocol_diagnostics']['message_examples']
        self.assertGreater(len(examples), 0)
        self.assertIn('token_a', examples[0])

    def test_soft_curriculum_runs_without_missing_reverse_token(self):
        r = run(seed=7, episodes=25, communication_task='symmetric',
                soft_curriculum=True)
        self.assertTrue(r['soft_curriculum'])
        self.assertTrue(math.isfinite(r['history'][-1]['loss']))

if __name__ == '__main__': unittest.main()
