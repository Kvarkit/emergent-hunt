import unittest

import torch

from emergent_hunt.iterated_learning import run_iterated


class MatchedBudgetTests(unittest.TestCase):
    """#24849's ablation is only meaningful if the three conditions burn the
    same interaction budget -- otherwise a gap could just be "more steps"."""

    def test_total_steps_matched_across_conditions(self):
        for condition in ('baseline', 'bottleneck', 'reset_control'):
            r = run_iterated(seed=0, condition=condition, cycles=3, cycle_steps=20,
                              exposure_steps=5, batch=8)
            self.assertEqual(r['total_steps'], 60, condition)
            self.assertEqual(r['episodes'], 60 * 8, condition)

    def test_rejects_exposure_steps_larger_than_cycle(self):
        with self.assertRaises(ValueError):
            run_iterated(condition='bottleneck', cycles=2, cycle_steps=10, exposure_steps=11)

    def test_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            run_iterated(condition='nonsense')


class ResetReallyHappensTests(unittest.TestCase):
    """Drives the real run_iterated via its on_reset/on_cycle_end hooks --
    not a reimplementation -- to check the reset actually replaces weights,
    rather than being a no-op that only looks right in the returned JSON."""

    def test_bottleneck_reset_changes_receiver_weights(self):
        end_of_cycle = {}
        post_reset = {}

        def on_cycle_end(cycle, sender, receiver):
            end_of_cycle[cycle] = {k: v.clone() for k, v in receiver.state_dict().items()}

        def on_reset(cycle, sender, receiver):
            post_reset[cycle] = {k: v.clone() for k, v in receiver.state_dict().items()}

        run_iterated(seed=0, condition='bottleneck', reset_agent='receiver', cycles=2,
                     cycle_steps=20, exposure_steps=5, batch=8,
                     on_reset=on_reset, on_cycle_end=on_cycle_end)

        self.assertIn(0, end_of_cycle)
        self.assertIn(1, post_reset)
        trained = end_of_cycle[0]
        reset = post_reset[1]
        differs = any(not torch.equal(trained[k], reset[k]) for k in trained)
        self.assertTrue(differs, 'reset did not change receiver weights vs. end of prior cycle')

    def test_reset_agent_sender_leaves_receiver_untouched_by_reset_itself(self):
        # The reset call only overwrites reset_agent's state_dict. Capture the
        # receiver's weights immediately before and after the reset call (i.e.
        # before any post-reset training runs) -- they must be byte-identical,
        # since the reset step itself does not touch the partner.
        captured = {}

        def on_reset(cycle, sender, receiver):
            captured[cycle] = {k: v.clone() for k, v in receiver.state_dict().items()}

        end_of_cycle0 = {}

        def on_cycle_end(cycle, sender, receiver):
            if cycle == 0:
                end_of_cycle0['receiver'] = {k: v.clone() for k, v in receiver.state_dict().items()}

        run_iterated(seed=0, condition='bottleneck', reset_agent='sender', cycles=2,
                     cycle_steps=20, exposure_steps=5, batch=8,
                     on_reset=on_reset, on_cycle_end=on_cycle_end)

        for k in end_of_cycle0['receiver']:
            self.assertTrue(torch.equal(end_of_cycle0['receiver'][k], captured[1][k]),
                             f'receiver weight {k} changed by a sender-only reset')

    def test_baseline_never_resets(self):
        calls = []
        run_iterated(seed=0, condition='baseline', cycles=3, cycle_steps=10, batch=8,
                     on_reset=lambda *a: calls.append(a))
        self.assertEqual(calls, [])


class ExposurePhaseFreezesPartnerTests(unittest.TestCase):
    """The mechanism #24849 actually asks for is: during the limited-exposure
    sub-phase, the PARTNER is frozen while the reset agent relearns. Verify
    this on the real training loop via on_exposure_boundary, and verify the
    reset agent's own weights do move (it's not just globally frozen)."""

    def test_partner_frozen_during_exposure_reset_agent_moves(self):
        boundary = {}

        def on_exposure_boundary(cycle, phase, sender, receiver):
            boundary[(cycle, phase)] = {
                'sender': {k: v.clone() for k, v in sender.state_dict().items()},
                'receiver': {k: v.clone() for k, v in receiver.state_dict().items()},
            }

        run_iterated(seed=0, condition='bottleneck', reset_agent='receiver', cycles=2,
                     cycle_steps=30, exposure_steps=15, batch=8,
                     on_exposure_boundary=on_exposure_boundary)

        start, end = boundary[(1, 'start')], boundary[(1, 'end')]
        for k in start['sender']:
            self.assertTrue(torch.equal(start['sender'][k], end['sender'][k]),
                             f'frozen partner (sender) weight {k} changed during exposure phase')
        moved = any(not torch.equal(start['receiver'][k], end['receiver'][k])
                    for k in start['receiver'])
        self.assertTrue(moved, 'reset receiver did not learn anything during its own exposure phase')

    def test_negative_control_unfrozen_partner_is_caught(self):
        # Prove the assertion above is actually sensitive to the freeze, not
        # vacuously true because REINFORCE rarely moves weights in 15 steps:
        # reset_control uses the same reset + same step count with NO freeze,
        # so the "partner" (sender, since reset_agent stays receiver here for
        # comparability) should usually move when it is not held frozen.
        boundary = {}

        def on_cycle_end(cycle, sender, receiver):
            boundary[cycle] = {k: v.clone() for k, v in sender.state_dict().items()}

        def on_reset(cycle, sender, receiver):
            boundary[('reset', cycle)] = {k: v.clone() for k, v in sender.state_dict().items()}

        run_iterated(seed=0, condition='reset_control', reset_agent='receiver', cycles=2,
                     cycle_steps=30, exposure_steps=15, batch=8,
                     on_reset=on_reset, on_cycle_end=on_cycle_end)

        before = boundary[('reset', 1)]
        after = boundary[1]
        moved = any(not torch.equal(before[k], after[k]) for k in before)
        self.assertTrue(moved, 'negative control did not reintroduce the bug: unfrozen sender '
                                'never moved, so the freeze test above would pass vacuously')


if __name__ == '__main__':
    unittest.main()
