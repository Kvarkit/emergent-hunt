import unittest

import torch

import emergent_hunt.train_multiround as tm
from emergent_hunt.train_multiround import run, Agent


class _InstrumentedAgent(Agent):
    """Records every `incoming` tensor this instance was called with, in
    call order, into a shared log tagged by instance -- so a test can
    inspect run()'s ACTUAL internal wiring instead of reimplementing it."""
    log = None  # class-level, set by the test around a run() call

    def forward(self, private, incoming):
        if _InstrumentedAgent.log is not None:
            _InstrumentedAgent.log.append((self, incoming.clone()))
        return super().forward(private, incoming)


class MultiRoundLearningTests(unittest.TestCase):
    def test_short_run_finite(self):
        r = run(seed=0, episodes=20)
        self.assertEqual(r['episodes'], 20)
        self.assertTrue(all(abs(x['loss']) < 1e6 for x in r['history']))

    def test_action_is_wired_to_partners_message_not_own(self):
        # Regression for the wiring bug behind nadir-codex's 100k-episode
        # matched run (board thread f8356f4a..., zone_score -> 1.0 but
        # type_score/terminal_success stuck at 0.0): every call site fed each
        # agent its OWN just-sent message back into itself (`a(pa, last_b)`
        # where last_b held ma, not mb) instead of what it actually
        # received. The channel was causally inert since the first
        # multiround trainer commit (67f9b56), not something that regressed
        # between "helps" and "doesn't help".
        #
        # This drives run() itself (not a reimplementation of its round
        # logic) and, via an instrumented Agent subclass plus a patched
        # gumbel_softmax, checks the ACTUAL `incoming` tensor each agent's
        # action-stage forward() call received against the ACTUAL ma/mb
        # gumbel_softmax produced that round: A's action call must receive
        # mb (B's message), not ma (its own).
        original_gumbel = tm.gumbel_softmax
        original_agent = tm.Agent

        def instrumented_run(seed):
            messages = []  # (ma, mb) appended by the patched gumbel_softmax, in call order

            def recording_gumbel(logits, **kwargs):
                out = original_gumbel(logits, **kwargs)
                messages.append(out)
                return out

            _InstrumentedAgent.log = []
            try:
                tm.Agent = _InstrumentedAgent
                tm.gumbel_softmax = recording_gumbel
                run(seed=seed, episodes=1, rounds=1)
            finally:
                tm.Agent = original_agent
                tm.gumbel_softmax = original_gumbel
                log, _InstrumentedAgent.log = _InstrumentedAgent.log, None
            return log, messages

        # An untrained network's token head is close to uniform, so ma and
        # mb can coincidentally sample the same one-hot vector (~1/vocab
        # chance) -- that makes "must differ from own message" degenerate
        # for that seed. Search a few seeds for one where they genuinely
        # differ, so the assertions below test the real wiring, not luck.
        for seed in range(10):
            log, messages = instrumented_run(seed)
            ma, mb = messages
            if not torch.equal(ma, mb):
                break
        else:
            self.fail('ma == mb for 10 consecutive seeds; cannot distinguish own '
                      'vs. partner message with this probe')

        self.assertEqual(len(log), 4)  # (a gen, b gen, a act, b act), 1 round
        (a_inst, a_gen_in), (b_inst, b_gen_in), (a_act_inst, a_act_in), (b_act_inst, b_act_in) = log
        self.assertIs(a_act_inst, a_inst)
        self.assertIs(b_act_inst, b_inst)

        # Round 0 has no prior message: both generation calls must see zero.
        self.assertTrue((a_gen_in == 0).all())
        self.assertTrue((b_gen_in == 0).all())

        # The core assertion: each agent's action call must be fed the
        # message it received (the OTHER agent's output), not its own.
        self.assertTrue(torch.equal(a_act_in, mb),
                         "A's action call was not fed B's message (mb)")
        self.assertTrue(torch.equal(b_act_in, ma),
                         "B's action call was not fed A's message (ma)")
        self.assertFalse(torch.equal(a_act_in, ma),
                          "A's action call was fed A's OWN message -- the pre-fix bug")
        self.assertFalse(torch.equal(b_act_in, mb),
                          "B's action call was fed B's OWN message -- the pre-fix bug")


if __name__ == '__main__': unittest.main()
