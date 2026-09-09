import unittest

import torch

from emergent_hunt.trap_prep import (ACTIVE, TrapPrepHunt, Target, TrapTask,
                                     blind_reference, reference_policies,
                                     run_episode, tasks)
from emergent_hunt.train_trap import (MODES, TrapDriver, TrapEncoder,
                                      TrapPreparer, evaluate, probe_factory,
                                      rollout, run)
from emergent_hunt.trap_probe import selectivity


def _agents(**kwargs):
    env = TrapPrepHunt(targets=1, split='all', horizon=6, patience=4,
                       stop_when_resolved=False, **kwargs)
    encoder = TrapEncoder(env)
    torch.manual_seed(0)
    return env, encoder, TrapDriver(encoder), TrapPreparer(encoder)


class EncodingTests(unittest.TestCase):
    """The encoder is the privacy boundary in the learned setting: if it leaked
    prey facts into the preparer's features the whole task would be void."""

    def test_preparer_features_are_invariant_to_the_prey(self):
        env, encoder, _, _ = _agents()
        first = env.reset(TrapTask((Target(1, 0, 1),)))[1]
        second = env.reset(TrapTask((Target(1, 2, 3),)))[1]
        self.assertTrue(torch.equal(encoder.preparer(first), encoder.preparer(second)))

    def test_driver_features_change_with_the_prey(self):
        env, encoder, _, _ = _agents()
        first = env.reset(TrapTask((Target(1, 0, 1),)))[0]
        second = env.reset(TrapTask((Target(1, 2, 1),)))[0]
        self.assertFalse(torch.equal(encoder.driver(first), encoder.driver(second)))

    def test_wire_keeps_message_length_as_a_channel(self):
        _, encoder, _, _ = _agents()
        full = encoder.wire((1, 2, 0), 3)
        short = encoder.wire((1,), 3)
        erased = encoder.wire((-1, -1, -1), 3)
        self.assertEqual(full.shape, short.shape)
        self.assertFalse(torch.equal(full, short))
        self.assertEqual(float(short.sum()), 1.0)
        self.assertEqual(float(erased.sum()), 3.0)
        self.assertFalse(torch.equal(erased, torch.zeros_like(erased)))


class RolloutTests(unittest.TestCase):
    def test_rollout_returns_one_entry_per_step(self):
        env, encoder, driver, preparer = _agents()
        out = rollout(env, driver, preparer, encoder,
                      task=TrapTask((Target(1, 0, 1),)))
        for key in ('rewards', 'logps', 'entropies', 'values'):
            self.assertEqual(len(out[key]), len(out['trace']), key)
        self.assertEqual(len(out['trace']), env.horizon)

    def test_muted_channels_are_excluded_from_the_loss(self):
        # A muted channel must not contribute log-probs: otherwise the control
        # would still be paying for tokens nobody can read.
        env, encoder, driver, preparer = _agents()
        task = TrapTask((Target(1, 0, 1),))
        torch.manual_seed(3)
        full = rollout(env, driver, preparer, encoder, task=task, greedy=True)
        torch.manual_seed(3)
        muted = rollout(env, driver, preparer, encoder, task=task,
                        mode='no_message', greedy=True)
        self.assertGreater(float(torch.stack(full["entropies"]).mean().detach()),
                           float(torch.stack(muted["entropies"]).mean().detach()))

    def test_greedy_rollouts_are_deterministic(self):
        env, encoder, driver, preparer = _agents()
        task = TrapTask((Target(2, 1, 2),))
        runs = [rollout(env, driver, preparer, encoder, task=task, greedy=True)
                for _ in range(2)]
        self.assertEqual([i['preparer_action'] for i in runs[0]['trace']],
                         [i['preparer_action'] for i in runs[1]['trace']])


class TrainingLoopTests(unittest.TestCase):
    def test_run_reports_both_splits_and_the_matching_blind_bound(self):
        result = run(seed=0, episodes=40, batch=8, report_every=40)
        row = result['history'][-1]
        for split in ('train', 'test'):
            self.assertIn('catch_rate', row[split]['intact'])
            self.assertIn('catch_rate', row[split]['mute'])
        # The bound reported must be the one for the configuration actually run,
        # since it depends on horizon and patience.
        self.assertEqual(result['blind_bound']['test'],
                         blind_reference(split='test', by=result['by'],
                                         horizon=result['horizon'],
                                         patience=result['patience'])['optimal_blind_success'])

    def test_weights_actually_move(self):
        seen = {}

        def on_report(row, driver, preparer):
            seen[row['episode']] = {k: v.clone() for k, v in driver.state_dict().items()}

        run(seed=0, episodes=40, batch=8, report_every=20, on_report=on_report)
        first, last = seen[20], seen[40]
        self.assertTrue(any(not torch.equal(first[k], last[k]) for k in first))

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            run(mode='nonsense', episodes=1)

    def test_every_mode_runs(self):
        for mode in MODES:
            result = run(seed=0, mode=mode, episodes=8, batch=4, report_every=8)
            self.assertEqual(result['mode'], mode)


class EvaluationTests(unittest.TestCase):
    def test_the_handwritten_pair_scores_one_under_the_same_evaluator(self):
        # Sanity-check the metric itself, not a learned policy: the reference
        # protocol must read as a perfect catch rate through evaluate()'s own
        # accounting, otherwise a low learned number would be uninterpretable.
        env = TrapPrepHunt(targets=1, split='all', horizon=8, patience=4)
        corpus = tasks(3, 1, 'test', 'pair')
        caught = sum(run_episode(env, *reference_policies(), task=task)['caught']
                     for task in corpus)
        self.assertEqual(caught / len(corpus), 1.0)

    def test_evaluate_is_deterministic(self):
        env, encoder, driver, preparer = _agents()
        corpus = tasks(3, 1, 'test', 'pair')
        first = evaluate(driver, preparer, encoder, env, corpus)
        second = evaluate(driver, preparer, encoder, env, corpus)
        self.assertEqual(first, second)


class ProbeAdapterTests(unittest.TestCase):
    """The whole point of the adapter: the same probe that judged the two
    handwritten codes must run unchanged on a learned pair."""

    def test_learned_pair_can_be_probed(self):
        result = run(seed=0, episodes=8, batch=4, report_every=8)
        driver, preparer, encoder, env = result['agents']
        report = selectivity(policy=probe_factory(driver, preparer, encoder),
                             n=env.n, targets=env.targets, by='pair', split='test',
                             max_tasks=6,
                             env_kwargs={'horizon': env.horizon, 'patience': env.patience,
                                         'stop_when_resolved': False})
        self.assertEqual(report['policy'], 'custom')
        self.assertIn('column_concentration', report)
        self.assertIn(report['candidate_localized_binding'], (True, False))

    def test_probe_adapter_is_greedy_and_repeatable(self):
        result = run(seed=0, episodes=8, batch=4, report_every=8)
        driver, preparer, encoder, env = result['agents']
        factory = probe_factory(driver, preparer, encoder)
        env2 = TrapPrepHunt(targets=1, split='all', horizon=env.horizon,
                            patience=env.patience, stop_when_resolved=False)
        task = tasks(3, 1, 'test', 'pair')[0]
        runs = [run_episode(env2, *factory(3, 3), task=task) for _ in range(2)]
        self.assertEqual([i['preparer_action'] for i in runs[0]['trace']],
                         [i['preparer_action'] for i in runs[1]['trace']])


if __name__ == '__main__':
    unittest.main()
