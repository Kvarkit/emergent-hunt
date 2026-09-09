import tempfile
import unittest
from pathlib import Path

import torch

from emergent_hunt.trap_prep import (ACTIVATE, Target, TrapPrepHunt, TrapTask,
                                     mechanism_for, tasks)
from emergent_hunt.trap_diagnose import (STAGES, _mutual_information, diagnose,
                                         load, messages, stages)
from emergent_hunt.train_trap import (TrapDriver, TrapEncoder, TrapPreparer, run)


class MutualInformationTests(unittest.TestCase):
    """The message readout is only worth anything if 'the wire is dead' and
    'the wire carries the fact' come out as 0 and log2(k)."""

    def test_a_constant_message_carries_no_bits(self):
        self.assertAlmostEqual(_mutual_information((('a',), z) for z in range(3)), 0.0)

    def test_a_bijective_code_carries_the_whole_fact(self):
        self.assertAlmostEqual(
            _mutual_information(((z,), z) for z in range(3) for _ in range(4)),
            1.5849625007211559)

    def test_a_two_to_one_code_carries_part_of_it(self):
        bits = _mutual_information(((z // 2,), z) for z in range(4) for _ in range(4))
        self.assertAlmostEqual(bits, 1.0)


class StageTests(unittest.TestCase):
    """Hand-built policies with a known failure point: the histogram must name
    that point and no other."""

    class _Preparer:
        def __init__(self, actions):
            self.actions = actions

    def _run_with(self, preparer_actions, drive_at=None, **env_kwargs):
        """Bypass the networks: patch rollout's decisions via a scripted pair."""
        from emergent_hunt import trap_diagnose

        env = TrapPrepHunt(targets=1, split='all', horizon=4, patience=4,
                           start_pos=1, stop_when_resolved=False, **env_kwargs)
        encoder = TrapEncoder(env)

        def fake_rollout(env, driver, preparer, encoder, task=None, mode=None,
                         greedy=True):
            env.reset(task)
            trace, rewards = [], []
            for step, action in enumerate(preparer_actions, start=1):
                env.driver_send(())
                env.preparer_send(())
                env.driver_act(1 if step == drive_at else 0)
                reward, done, info = env.preparer_act(action)
                trace.append(info)
                rewards.append(reward)
                if done:
                    break
            return {'trace': trace, 'info': trace[-1], 'rewards': rewards}

        original = trap_diagnose.rollout
        trap_diagnose.rollout = fake_rollout
        try:
            corpus = (TrapTask((Target(zone=2, prey_type=2, start_distance=1),)),)
            return stages(None, None, encoder, env, corpus)
        finally:
            trap_diagnose.rollout = original

    def test_never_leaving_the_start_fails_at_reached(self):
        report = self._run_with([0, 0, 0, 0])
        self.assertEqual(report['first_failed_stage'], {'reached': 1})
        self.assertEqual(report['reached_rate']['reached'], 0.0)

    def test_a_wrong_first_mechanism_fails_at_first_guess(self):
        wrong = (mechanism_for(2, 3) + 1) % 3
        report = self._run_with([1, 4 + wrong, 2, 2], drive_at=4)
        self.assertEqual(report['first_failed_stage'], {'first_guess': 1})
        self.assertEqual(report['reached_rate']['reached'], 1.0)

    def test_a_prey_that_never_arrives_fails_at_in_position(self):
        report = self._run_with([1, 4 + mechanism_for(2, 3), 2, 2])
        self.assertEqual(report['first_failed_stage'], {'in_position': 1})
        self.assertEqual(report['reached_rate']['first_guess'], 1.0)

    def test_arriving_but_never_firing_fails_at_fired(self):
        report = self._run_with([1, 4 + mechanism_for(2, 3), 2, 2], drive_at=3)
        self.assertEqual(report['first_failed_stage'], {'fired': 1})
        self.assertEqual(report['reached_rate']['in_position'], 1.0)

    def test_the_whole_chain_reports_no_failure(self):
        report = self._run_with([1, 4 + mechanism_for(2, 3), ACTIVATE, 2],
                                drive_at=3)
        self.assertEqual(report['first_failed_stage'], {'none': 1})
        self.assertEqual([report['reached_rate'][s] for s in STAGES], [1.0] * 5)


class CheckpointTests(unittest.TestCase):
    def test_a_checkpoint_round_trips_into_a_full_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trap.pt'
            run(seed=0, episodes=8, batch=4, report_every=8, horizon=4,
                start_pos=1, approach=.25, checkpoint=path)
            report = diagnose(path)
            self.assertEqual(report['settings']['start_pos'], 1)
            self.assertEqual(report['settings']['horizon'], 4)
            for split in ('train', 'test'):
                self.assertEqual(sum(report[split]['stages']['first_failed_stage'].values()),
                                 report[split]['stages']['prey'])
                self.assertLessEqual(report[split]['messages']['bits']['mechanism'],
                                     report[split]['messages']['max_bits']['mechanism'] + 1e-9)

    def test_older_checkpoints_still_load_with_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            env = TrapPrepHunt(targets=1, split='train', horizon=8)
            encoder = TrapEncoder(env)
            driver, preparer = TrapDriver(encoder), TrapPreparer(encoder)
            torch.save({'driver': driver.state_dict(),
                        'preparer': preparer.state_dict(),
                        'config': {'seed': 0, 'mode': 'communication', 'n': 3,
                                   'targets': 1, 'by': 'pair', 'horizon': 8,
                                   'window': 1}}, path)
            loaded = load(path)
            self.assertEqual(loaded['settings']['patience'], 4)
            self.assertEqual(loaded['settings']['start_pos'], 0)
            self.assertEqual(loaded['settings']['approach'], 0.0)


class MessageTests(unittest.TestCase):
    def test_bits_are_reported_against_the_reachable_maximum(self):
        result = run(seed=0, episodes=8, batch=4, report_every=8, horizon=4,
                     start_pos=1)
        driver, preparer, encoder, env = result['agents']
        corpus = tasks(3, 1, 'train', 'pair')
        report = messages(driver, preparer, encoder, env, corpus)
        self.assertAlmostEqual(report['max_bits']['zone'], 1.5849625007211559)
        self.assertGreaterEqual(report['distinct_first_messages'], 1)


if __name__ == '__main__':
    unittest.main()
