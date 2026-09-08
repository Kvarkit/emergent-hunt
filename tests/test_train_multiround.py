import unittest

import torch
from torch.nn.functional import one_hot

import emergent_hunt.train_multiround as tm
from emergent_hunt.train_multiround import run, Agent, oh


class _InstrumentedAgent(Agent):
    """Records every `incoming` tensor and produced token logits this
    instance was called with, in call order, into a shared log tagged by
    instance -- so a test can inspect the REAL wiring of whatever function
    drives it instead of reimplementing that function's round logic."""
    log = None  # class-level, set by the test around a call

    def forward(self, private, incoming):
        out = super().forward(private, incoming)
        if _InstrumentedAgent.log is not None:
            _InstrumentedAgent.log.append((self, incoming.clone(), out[0].clone()))
        return out


def _make_agent_pair(rounds, type_count, zones, vocab, hidden_dim=8):
    a = _InstrumentedAgent(type_count + zones, vocab, zones * type_count,
                            marker_dim=rounds, type_count=type_count, hidden_dim=hidden_dim)
    b = _InstrumentedAgent(type_count + zones, vocab, zones * type_count,
                            marker_dim=rounds, type_count=type_count, hidden_dim=hidden_dim)
    return a, b


def _find_nondegenerate_seed(task, rounds=1, type_count=3, zones=4, vocab=8, tries=20):
    """An untrained network's token head is close to uniform, so ma and mb
    can coincidentally sample the same one-hot vector for the very first
    grid cell -- that makes "must differ from own message" degenerate for
    that seed. Search a few seeds for one where round-0's ma != mb (task
    'symmetric') or ma is simply well-defined (task 'one_way', no mb)."""
    for seed in range(tries):
        torch.manual_seed(seed)
        a, b = _make_agent_pair(rounds, type_count, zones, vocab)
        _InstrumentedAgent.log = []
        tm._fixed_grid_eval(a, b, task, rounds=rounds, vocab=vocab, zones=zones,
                            type_count=type_count, use_messages=True)
        log, _InstrumentedAgent.log = _InstrumentedAgent.log, None
        (a_i, a_gen_in, a_gen_out), (b_i, b_gen_in, b_gen_out) = log[0], log[1]
        ma = a_gen_out.argmax(-1)
        mb = b_gen_out.argmax(-1)
        if task != 'symmetric' or not torch.equal(ma, mb):
            return seed, a, b, log[:4], ma, mb
    raise AssertionError(f'ma == mb for {tries} consecutive seeds ({task}); '
                          'cannot distinguish own vs. partner message with this probe')


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


class FixedGridEvalWiringTests(unittest.TestCase):
    """nadir-codex #25999: regression for the channel-inert wiring bug this
    thread has now hit and fixed three separate times (zenith #25804/#25829,
    nadir cd0947a for one_way, nadir ea17d77 for symmetric + use_messages).
    Drives the REAL _fixed_grid_eval via an instrumented Agent subclass, not
    a reimplementation of its round logic, and checks the actual `incoming`
    tensor each agent's action-stage forward() call received against the
    actual ma/mb argmax tokens that round produced.
    """

    def test_symmetric_action_is_wired_to_partners_message_not_own(self):
        seed, a, b, log, ma, mb = _find_nondegenerate_seed('symmetric', rounds=1)
        (a_gen_i, a_gen_in, _), (b_gen_i, b_gen_in, _), \
            (a_act_i, a_act_in, _), (b_act_i, b_act_in, _) = log
        self.assertIs(a_act_i, a_gen_i)
        self.assertIs(b_act_i, b_gen_i)
        self.assertTrue((a_gen_in == 0).all())  # round 0: nothing received yet
        self.assertTrue((b_gen_in == 0).all())

        ma_oh, mb_oh = one_hot(ma, 8).float(), one_hot(mb, 8).float()
        self.assertTrue(torch.equal(a_act_in, mb_oh),
                         "A's action call was not fed B's message (mb)")
        self.assertTrue(torch.equal(b_act_in, ma_oh),
                         "B's action call was not fed A's message (ma)")
        self.assertFalse(torch.equal(a_act_in, ma_oh),
                          "A's action call was fed A's OWN message -- the historical bug")
        self.assertFalse(torch.equal(b_act_in, mb_oh),
                          "B's action call was fed B's OWN message -- the historical bug")

    def test_one_way_action_is_wired_to_declared_contract(self):
        # one_way's contract isn't "not own" (A never receives at all) --
        # it's the explicit declared shape: A (sender) acts on nothing, B
        # (receiver) acts on A's message. Assert that shape directly rather
        # than inferring it from a same/different-from-own check.
        seed, a, b, log, ma, _ = _find_nondegenerate_seed('one_way', rounds=1)
        (a_gen_i, a_gen_in, _), (b_gen_i, b_gen_in, _), \
            (a_act_i, a_act_in, _), (b_act_i, b_act_in, _) = log
        self.assertTrue((a_gen_in == 0).all())
        self.assertTrue((b_gen_in == 0).all())

        ma_oh = one_hot(ma, 8).float()
        self.assertTrue((a_act_in == 0).all(),
                         "A (one-way sender) acted on a nonzero incoming -- "
                         "one-way must not feed A anything")
        self.assertTrue(torch.equal(b_act_in, ma_oh),
                         "B (one-way receiver) was not fed A's message (ma)")

    def test_negative_control_reintroduced_swap_is_caught(self):
        # melioralab/nadir's own pattern for this suite (_WrongOrderBagReceiver
        # in test_diagnose_checkpoint.py): a deliberately-wrong local variant,
        # used only to prove the check above is actually sensitive to the
        # bug it claims to catch, not vacuously true. This reintroduces
        # EXACTLY the pre-ea17d77 symmetric assignment
        # (`last_a, last_b = oh(mb), oh(ma)`, unchanged call sites) around
        # the same instrumented Agent pair and asserts the positive test's
        # own assertions FAIL against it.
        torch.manual_seed(0)
        rounds, type_count, zones, vocab = 1, 3, 4, 8
        a, b = _make_agent_pair(rounds, type_count, zones, vocab)
        _InstrumentedAgent.log = []
        with torch.no_grad():
            last_a = torch.zeros(1, vocab)
            last_b = torch.zeros(1, vocab)
            marker = oh(torch.tensor([0]), rounds)
            pa = torch.cat((oh(torch.tensor([0]), type_count), oh(torch.tensor([0]), zones), marker), -1)
            pb = torch.cat((oh(torch.tensor([0]), type_count), oh(torch.tensor([0]), zones), marker), -1)
            ta, _, _, _ = a(pa, last_b)
            tb, _, _, _ = b(pb, last_a)
            ma, mb = ta.argmax(-1), tb.argmax(-1)
            # pre-ea17d77 bug: swapped relative to the fix.
            last_a, last_b = oh(mb, vocab), oh(ma, vocab)
            _, aa, _, _ = a(pa, last_b)
            _, ab, _, _ = b(pb, last_a)
        log, _InstrumentedAgent.log = _InstrumentedAgent.log, None
        a_act_in = log[2][1]
        b_act_in = log[3][1]
        ma_oh, mb_oh = one_hot(ma, vocab).float(), one_hot(mb, vocab).float()

        # The bug this negative control reintroduces must actually trip the
        # positive test's assertions -- if it doesn't, the positive test
        # isn't testing what it claims to.
        self.assertFalse(torch.equal(a_act_in, mb_oh),
                          'negative control did not reintroduce the bug: A still got mb')
        self.assertTrue(torch.equal(a_act_in, ma_oh),
                         'negative control did not reintroduce the bug: A did not get its own ma')


if __name__ == '__main__': unittest.main()
