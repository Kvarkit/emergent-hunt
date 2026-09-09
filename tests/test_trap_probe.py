import unittest

from emergent_hunt.trap_prep import (ACTIVATE, PREPARE_BASE, TrapPrepHunt,
                                     Target, TrapTask, mechanism_for,
                                     oracle_rollout, prepare_action,
                                     reference_policies, run_episode, tasks)
from emergent_hunt.trap_probe import (COMPONENTS, decision, holistic_policies,
                                      leakage, mute, observed_values, reverse,
                                      roll, selectivity, signature, substitute)

# Large enough that every token value the reference code can emit is exercised;
# a handful of tasks is not (the entangled code's leakage only shows up once the
# corpus covers several distinct (zone, method) pairs).
CORPUS = tasks(split='all')[:24]


class DecodingTests(unittest.TestCase):
    """The decomposition must be read off executed actions, so that the same
    probe applies unchanged to a learned preparer."""

    def test_decision_splits_an_action_into_target_method_timing(self):
        self.assertEqual(decision({'preparer_pos': 2, 'preparer_action': prepare_action(1)}),
                         {'target': 2, 'method': 1, 'timing': False})
        self.assertEqual(decision({'preparer_pos': 0, 'preparer_action': ACTIVATE}),
                         {'target': 0, 'method': None, 'timing': True})

    def test_signature_tracks_the_reference_protocol(self):
        result = oracle_rollout(TrapTask((Target(2, 1, 1),)))
        summary = signature(result)
        self.assertEqual(summary['target'], (2, 2))
        self.assertEqual(summary['method'], (mechanism_for(1, 3),))
        self.assertEqual(summary['timing_window'], (True,))


class PerturbationTests(unittest.TestCase):
    def test_substitute_touches_one_position_and_optionally_one_step(self):
        self.assertEqual(substitute(1, 7)(3, (0, 1, 2)), (0, 7, 2))
        self.assertEqual(substitute(1, 7, at_step=2)(3, (0, 1, 2)), (0, 1, 2))
        self.assertEqual(substitute(1, 7, at_step=2)(2, (0, 1, 2)), (0, 7, 2))
        self.assertEqual(substitute(5, 7)(1, (0, 1, 2)), (0, 1, 2))

    def test_structural_perturbations(self):
        self.assertEqual(mute(1, (0, 1, 2)), ())
        self.assertEqual(reverse(1, (0, 1, 2)), (2, 1, 0))
        self.assertEqual(roll(1)(1, (0, 1, 2)), (2, 0, 1))
        self.assertEqual(roll(1)(1, ()), ())

    def test_observed_values_reports_only_tokens_that_occur(self):
        env = TrapPrepHunt(targets=2, split='all')
        values = observed_values(reference_policies, env, CORPUS)
        self.assertEqual(len(values), env.driver_message_length)
        for column in values:
            self.assertTrue(set(column) <= set(range(env.vocabulary)))
        self.assertEqual(values[2], [0, 1])  # the cue slot is binary in this code


class BothReferenceCodesSolveTheTaskTests(unittest.TestCase):
    """The compositional and holistic codes must be behaviourally identical when
    intact -- otherwise the probe would be separating success, not structure."""

    def test_intact_behaviour_is_identical(self):
        env = TrapPrepHunt(targets=2, split='all')
        for task in CORPUS:
            compositional = run_episode(env, *reference_policies(), task=task)
            holistic = run_episode(env, *holistic_policies(), task=task)
            self.assertTrue(compositional['success'] and holistic['success'], task)
            self.assertEqual(signature(compositional), signature(holistic), task)


class SelectivityTests(unittest.TestCase):
    """Ground-truth check on the probe itself: it must call the factorized code
    localized and the entangled code not, on identical intact behaviour."""

    def setUp(self):
        self.compositional = selectivity(policy='compositional', corpus=CORPUS)
        self.holistic = selectivity(policy='holistic', corpus=CORPUS)

    def test_compositional_code_binds_each_position_to_one_component(self):
        self.assertEqual(self.compositional['best_component_per_position'],
                         {'0': 'target', '1': 'method', '2': 'timing'})
        self.assertTrue(self.compositional['candidate_localized_binding'])
        self.assertEqual(self.compositional['exclusive_leakage'], 0.0)
        # All the causal mass on "which target" sits in one token position.
        self.assertEqual(self.compositional['column_concentration']['target'], 1.0)

    def test_method_token_never_moves_the_target_in_a_factorized_code(self):
        # The sharp, assumption-free discriminator: perturbing the slot that
        # carries the preparation method must not change WHICH trap is acted on.
        self.assertEqual(self.compositional['positions']['1']['changed']['target'], 0.0)
        self.assertGreater(self.holistic['positions']['1']['changed']['target'], .25)

    def test_holistic_code_is_rejected_despite_identical_intact_behaviour(self):
        self.assertEqual(self.compositional['intact'], self.holistic['intact'])
        self.assertFalse(self.holistic['candidate_localized_binding'])
        self.assertLess(self.holistic['column_concentration']['target'], .8)

    def test_every_position_actually_has_a_causal_effect(self):
        for position in self.compositional['positions'].values():
            self.assertGreater(position['counterfactuals'], 0)
            self.assertLess(position['no_effect'], 1.0)

    def test_controls_collapse_the_protocol(self):
        for name in ('mute', 'reverse', 'roll_1'):
            control = self.compositional['controls'][name]
            self.assertEqual(control['success_rate'], 0.0, name)
            self.assertLess(control['mean_reward'], self.compositional['intact']['mean_reward'])

    def test_leakage_report_matches_the_matrix(self):
        report = leakage(self.compositional)
        for position, row in report.items():
            self.assertEqual(row['best'],
                             self.compositional['best_component_per_position'][position])
            self.assertEqual(row['max_other'], 0.0)

    def test_verdict_is_stable_across_corpus_sizes(self):
        # The statistic must not depend on how many tasks happen to be probed:
        # an earlier exclusivity-based verdict flipped between 12 and 24 tasks.
        for size in (12, 24):
            corpus = tasks(split='all')[:size]
            self.assertTrue(selectivity(policy='compositional',
                                        corpus=corpus)['candidate_localized_binding'], size)
            self.assertFalse(selectivity(policy='holistic',
                                         corpus=corpus)['candidate_localized_binding'], size)

    def test_probe_runs_on_the_held_out_split(self):
        # Novel combinations of familiar conditions: the probe must be runnable
        # on by='pair' test tasks, where no (prey_type, zone) pair was in train.
        held_out = selectivity(policy='compositional', by='pair', split='test',
                               max_tasks=16)
        self.assertTrue(held_out['candidate_localized_binding'])
        self.assertEqual(held_out['intact']['success_rate'], 1.0)


class CustomPolicyInterfaceTests(unittest.TestCase):
    """A learned pair plugs in as a factory of (driver, preparer); nothing in
    the probe assumes the handwritten policies."""

    def test_probe_accepts_an_arbitrary_policy_factory(self):
        class DeafPreparer:
            def reset(self, observation): pass
            def message(self, observation, incoming): return (0,)
            def act(self, observation): return 2  # WAIT, ignores every token

        def factory(n, mechanisms):
            driver, _ = reference_policies(n, mechanisms)
            return driver, DeafPreparer()

        report = selectivity(policy=factory, corpus=CORPUS[:4])
        self.assertEqual(report['policy'], 'custom')
        self.assertEqual(report['intact']['success_rate'], 0.0)
        for position in report['positions'].values():
            if position['counterfactuals']:
                self.assertEqual(position['no_effect'], 1.0)
        self.assertFalse(report['candidate_localized_binding'])


if __name__ == '__main__':
    unittest.main()
