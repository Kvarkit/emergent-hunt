import unittest

import torch
from torch.nn.functional import one_hot

import emergent_hunt.train_multiround as tm
from emergent_hunt.train_multiround import run, Agent

class MultiRoundLearningTests(unittest.TestCase):
    def test_short_run_finite(self):
        r = run(seed=0, episodes=20)
        self.assertEqual(r['episodes'], 20)
        self.assertTrue(all(abs(x['loss']) < 1e6 for x in r['history']))
        self.assertIn('fixed_grid_eval', r)
        self.assertTrue(0.0 <= r['fixed_grid_eval']['terminal_success'] <= 1.0)

    def test_one_way_small_grid_and_curriculum(self):
        r = run(seed=0, episodes=20, rounds=1, zones=2, type_count=2,
                communication_task='symmetric', curriculum=True)
        self.assertEqual(r['curriculum'], True)
        self.assertEqual(set(r['fixed_grid_eval']),
                         {'zone_score', 'type_score', 'terminal_success'})

    def test_fixed_grid_eval_action_is_wired_to_partners_message_not_own(self):
        # Regression for the wiring bug behind nadir-codex's matched 100k
        # run and the still-flat symmetric results in #25829 (board thread
        # f8356f4a..., commit e6470aa): every call site in both run() and
        # _fixed_grid_eval fed each agent its OWN just-produced message back
        # into itself (`a(pa, last_b)` where last_b held ma, not mb) instead
        # of what it actually received -- present since the first
        # multiround commit and reintroduced verbatim when this file grew
        # curriculum/one_way support.
        #
        # This drives the REAL _fixed_grid_eval (not a reimplementation of
        # its wiring) via forward hooks on the actual Agent instances it
        # uses, and checks the first state/round's action-stage `incoming`
        # tensors against the messages _fixed_grid_eval itself derived that
        # round: A's action call must see B's message (mb, or zero for
        # one_way where B doesn't message back) and B's action call must
        # see A's message (ma), never its own.
        vocab, zones, type_count, rounds = 8, 2, 2, 1
        torch.manual_seed(0)
        a = Agent(type_count + zones, vocab, zones * type_count, rounds)
        b = Agent(type_count + zones, vocab, zones * type_count, rounds)

        calls = []  # (instance, incoming) in call order

        def make_hook(tag):
            def hook(module, args, output):
                calls.append((tag, args[1].clone(), output[0].clone()))  # (tag, incoming, token_logits)
            return hook

        a.register_forward_hook(make_hook('a'))
        b.register_forward_hook(make_hook('b'))

        tm._fixed_grid_eval(a, b, 'one_way', rounds, vocab, zones, type_count)

        # First state (prey_t=0,prey_z=0,trap_t=0,trap_z=0), round 0: calls
        # in source order are [a gen, b gen, a act, b act].
        (tag_a_gen, a_gen_in, a_gen_tok), (tag_b_gen, b_gen_in, b_gen_tok), \
            (tag_a_act, a_act_in, _), (tag_b_act, b_act_in, _) = calls[:4]
        self.assertEqual((tag_a_gen, tag_b_gen, tag_a_act, tag_b_act), ('a', 'b', 'a', 'b'))

        ma = a_gen_tok.argmax(-1)
        ma_onehot = one_hot(ma, vocab).float()

        # one_way: A's action call must NOT see any message (A sends, never
        # receives) -- it must stay zero, not b's own generation output.
        self.assertTrue((a_act_in == 0).all(),
                         "A's action call received a nonzero incoming tensor in the "
                         "one_way task, where A never receives a message")
        # B's action call must see A's message.
        self.assertTrue(torch.equal(b_act_in, ma_onehot),
                         "B's action call was not fed A's message (ma) -- the pre-fix "
                         "bug left B's action structurally unable to depend on any "
                         "message it received")

if __name__ == '__main__': unittest.main()
