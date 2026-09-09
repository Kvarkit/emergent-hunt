import unittest

from emergent_hunt.environment import held_out_pairs
from emergent_hunt.trap_prep import (ACTIVATE, ACTIVE, CAUGHT, ESCAPED, HOLD,
                                     MOVE_RIGHT, WAIT, Target, TrapPrepHunt,
                                     TrapTask, blind_reference, blind_search,
                                     drive_action, mechanism_for,
                                     oracle_rollout, prepare_action,
                                     reference_policies, run_episode, tasks)


def script(env, task, moves, driver=None, message=()):
    """Drive the real environment with an explicit preparer action list.

    `driver` is a list of driver actions (default: HOLD). Returns the list of
    per-step info dicts, so tests assert on the simulator's own bookkeeping
    rather than on a reimplementation of it.
    """
    env.reset(task)
    out = []
    for i, action in enumerate(moves):
        env.driver_send(message)
        env.preparer_send(())
        env.driver_act(HOLD if driver is None else driver[i])
        reward, done, info = env.preparer_act(action)
        out.append((reward, done, info))
        if done:
            break
    return out


class SplitTests(unittest.TestCase):
    def test_train_test_partition_and_coverage(self):
        train, test = set(tasks(split='train')), set(tasks(split='test'))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(tasks()))
        for corpus in (train, test):
            for field in ('zone', 'prey_type'):
                self.assertEqual({getattr(t, field) for task in corpus for t in task.targets},
                                 set(range(3)))
            self.assertEqual({t.start_distance for task in corpus for t in task.targets},
                             {1, 2, 3})

    def test_by_pair_withholds_whole_type_zone_pairs(self):
        held = held_out_pairs(3)
        train_pairs = {(t.prey_type, t.zone)
                       for task in tasks(split='train', by='pair') for t in task.targets}
        self.assertFalse(train_pairs & held)
        # Every held-out pair does occur in test, at every distance.
        for pair in held:
            distances = {t.start_distance
                         for task in tasks(split='test', by='pair') for t in task.targets
                         if (t.prey_type, t.zone) == pair}
            self.assertEqual(distances, {1, 2, 3})

    def test_by_triple_is_the_easy_split(self):
        # Under by='triple' every (prey_type, zone) pair still recurs in train,
        # so a table keyed on that pair needs no generalization (cf. #24933).
        train_pairs = {(t.prey_type, t.zone)
                       for task in tasks(split='train', by='triple') for t in task.targets}
        self.assertEqual(len(train_pairs), 9)

    def test_rejects_bad_configurations(self):
        with self.assertRaises(ValueError): tasks(n=1)
        with self.assertRaises(ValueError): tasks(targets=4)
        with self.assertRaises(ValueError): tasks(split='nonsense')
        with self.assertRaises(ValueError): tasks(by='nonsense')
        with self.assertRaises(ValueError): TrapPrepHunt(window=0)
        with self.assertRaises(ValueError): TrapPrepHunt(erasure=2)
        with self.assertRaises(ValueError): TrapPrepHunt(partial=1.5)
        with self.assertRaises(ValueError): TrapPrepHunt(start_pos=3)


class RewardSchemeTests(unittest.TestCase):
    """Spec point 1: correct preparation pays a partial reward, a catch pays the
    rest, and repeating preparation pays nothing extra."""

    def setUp(self):
        self.task = TrapTask((Target(zone=1, prey_type=2, start_distance=1),))
        self.env = TrapPrepHunt(targets=1, split='all', partial=.25)
        self.mechanism = mechanism_for(2, 3)

    def test_full_successful_episode_totals_exactly_one_per_target(self):
        for task in (self.task, TrapTask((Target(0, 0, 3), Target(2, 1, 2)))):
            result = oracle_rollout(task)
            self.assertTrue(result['success'])
            self.assertAlmostEqual(result['total_reward'], float(len(task.targets)))
            self.assertAlmostEqual(result['totals']['prepare'], .25 * len(task.targets))
            self.assertAlmostEqual(result['totals']['catch'], .75 * len(task.targets))

    def test_repeated_preparation_earns_nothing_extra(self):
        steps = script(self.env, self.task,
                       [MOVE_RIGHT, prepare_action(self.mechanism),
                        prepare_action(self.mechanism), prepare_action(self.mechanism)])
        rewards = [r for r, _, _ in steps]
        self.assertAlmostEqual(rewards[1], .25)
        self.assertEqual(rewards[2:], [0.0, 0.0])
        self.assertAlmostEqual(steps[-1][2]['totals']['prepare'], .25)

    def test_wrong_then_corrected_preparation_pays_once(self):
        wrong = (self.mechanism + 1) % 3
        steps = script(self.env, self.task,
                       [MOVE_RIGHT, prepare_action(wrong),
                        prepare_action(self.mechanism), prepare_action(wrong),
                        prepare_action(self.mechanism)])
        self.assertEqual([round(r, 5) for r, _, _ in steps], [0.0, 0.0, .25, 0.0, 0.0])

    def test_preparation_away_from_a_target_pays_nothing(self):
        # Zone 0 holds a trap but no prey: preparing it is not "correct
        # preparation", however well it matches the prey's required mechanism.
        steps = script(self.env, self.task, [prepare_action(self.mechanism)])
        self.assertEqual(steps[0][0], 0.0)
        self.assertEqual(steps[0][2]['totals']['prepare'], 0.0)


class TimingTests(unittest.TestCase):
    """Spec point 3: early activation wastes the charge, late lets it escape."""

    def test_activating_early_wastes_the_charge_permanently(self):
        task = TrapTask((Target(zone=0, prey_type=0, start_distance=2),))
        env = TrapPrepHunt(targets=1, split='all')
        mechanism = mechanism_for(0, 3)
        steps = script(env, task,
                       [ACTIVATE, prepare_action(mechanism), ACTIVATE, ACTIVATE],
                       driver=[HOLD, drive_action(0), drive_action(0), HOLD])
        self.assertIn(('wasted_charge', 0), steps[0][2]['events'])
        # Prey later arrives on a correctly prepared trap, but the charge is gone.
        self.assertIn(('arrived', 0), steps[2][2]['events'])
        self.assertIn(('no_charge', 0), steps[2][2]['events'])
        self.assertEqual(steps[-1][2]['caught'], 0)
        self.assertAlmostEqual(steps[-1][2]['totals']['catch'], 0.0)

    def test_activating_late_lets_the_prey_escape(self):
        task = TrapTask((Target(zone=0, prey_type=0, start_distance=1),))
        env = TrapPrepHunt(targets=1, split='all', window=1)
        mechanism = mechanism_for(0, 3)
        steps = script(env, task, [prepare_action(mechanism), WAIT],
                       driver=[HOLD, drive_action(0)])
        self.assertIn(('arrived', 0), steps[1][2]['events'])
        self.assertIn(('escaped', 0), steps[1][2]['events'])
        self.assertEqual(steps[1][2]['status'], (ESCAPED,))
        self.assertTrue(steps[1][1], 'episode ends once every prey is resolved')

    def test_window_length_controls_how_late_is_too_late(self):
        task = TrapTask((Target(zone=0, prey_type=0, start_distance=1),))
        mechanism = mechanism_for(0, 3)
        for window, expected in ((1, ESCAPED), (2, CAUGHT)):
            env = TrapPrepHunt(targets=1, split='all', window=window)
            steps = script(env, task, [prepare_action(mechanism), WAIT, ACTIVATE],
                           driver=[HOLD, drive_action(0), HOLD])
            self.assertEqual(steps[-1][2]['status'], (expected,), window)

    def test_right_time_wrong_method_fails(self):
        task = TrapTask((Target(zone=0, prey_type=0, start_distance=1),))
        env = TrapPrepHunt(targets=1, split='all')
        wrong = (mechanism_for(0, 3) + 1) % 3
        steps = script(env, task, [prepare_action(wrong), ACTIVATE],
                       driver=[HOLD, drive_action(0)])
        self.assertIn(('wrong_method', 0), steps[1][2]['events'])
        self.assertEqual(steps[1][2]['caught'], 0)


class MultipleTargetTests(unittest.TestCase):
    """Spec point 4: the same fact about a different object needs a different
    decision, and each trap keeps its own charge and its own prepared state."""

    def test_identical_prey_type_at_two_zones_needs_two_different_actions(self):
        task = TrapTask((Target(0, 1, 1), Target(2, 1, 1)))
        result = oracle_rollout(task)
        prepared = [(info['preparer_pos'], info['preparer_action'])
                    for info in result['trace']
                    if info['preparer_action'] >= 4]
        self.assertEqual({zone for zone, _ in prepared}, {0, 2})
        self.assertEqual({action for _, action in prepared},
                         {prepare_action(mechanism_for(1, 3))})
        self.assertTrue(result['success'])

    def test_same_zone_different_prey_needs_a_different_method(self):
        methods = set()
        for prey_type in (0, 1):
            result = oracle_rollout(TrapTask((Target(1, prey_type, 1),)))
            methods |= {info['preparer_action'] - 4 for info in result['trace']
                        if info['preparer_action'] >= 4}
        self.assertEqual(methods, {mechanism_for(0, 3), mechanism_for(1, 3)})

    def test_wasting_one_charge_leaves_the_other_target_playable(self):
        task = TrapTask((Target(0, 0, 2), Target(1, 1, 2)))
        env = TrapPrepHunt(targets=2, split='all')
        steps = script(env, task,
                       [ACTIVATE, MOVE_RIGHT, prepare_action(mechanism_for(1, 3)),
                        ACTIVATE],
                       driver=[HOLD, HOLD, drive_action(1), drive_action(1)])
        self.assertIn(('wasted_charge', 0), steps[0][2]['events'])
        self.assertIn(('caught', 1), steps[-1][2]['events'])
        self.assertEqual(steps[-1][2]['status'], (ACTIVE, CAUGHT))

    def test_oracle_catches_every_prey_on_every_task(self):
        corpus = tasks(split='all')
        self.assertEqual(len(corpus), 486)
        caught = sum(oracle_rollout(task)['caught'] for task in corpus)
        self.assertEqual(caught, sum(len(task.targets) for task in corpus))


class PrivacyTests(unittest.TestCase):
    """Spec point 2: the agent at the trap cannot see the prey. This is what
    makes the message-blind bound in blind_reference exactly computable."""

    def test_preparer_view_never_names_the_prey(self):
        env = TrapPrepHunt(targets=2, split='all')
        _, observation = env.reset(TrapTask((Target(0, 1, 2), Target(2, 0, 3))))
        self.assertEqual(set(observation),
                         {'step', 'steps_left', 'self_pos', 'trap_method', 'trap_charged'})

    def test_driver_view_never_names_the_preparer(self):
        env = TrapPrepHunt(targets=2, split='all')
        observation, _ = env.reset(TrapTask((Target(0, 1, 2), Target(2, 0, 3))))
        self.assertEqual(set(observation),
                         {'step', 'steps_left', 'prey_type', 'zone', 'distance',
                          'window_left', 'status'})

    def test_blind_preparer_stream_is_a_function_of_its_own_actions_only(self):
        # Two tasks differing in prey type, distance and even target zone give a
        # message-blind preparer literally the same observation sequence, so no
        # blind policy can distinguish them. Every zone holds a trap, so the
        # layout leaks nothing either.
        moves = [MOVE_RIGHT, prepare_action(1), ACTIVATE, MOVE_RIGHT, WAIT]
        streams = []
        for task in (TrapTask((Target(0, 0, 1), Target(1, 1, 2))),
                     TrapTask((Target(2, 2, 3), Target(1, 0, 1)))):
            env = TrapPrepHunt(targets=2, split='all')
            env.reset(task)
            stream, done = [], False
            for action in moves:
                if done:
                    break
                stream.append(env.driver_send(())[0])
                env.preparer_send(())
                env.driver_act(drive_action(0))
                _, done, _ = env.preparer_act(action)
            streams.append(stream)
        self.assertEqual(streams[0], streams[1])

    def test_channel_noise_does_not_disturb_task_sampling(self):
        clean, noisy = (TrapPrepHunt(seed=11), TrapPrepHunt(seed=11, erasure=1.0))
        for _ in range(20):
            self.assertEqual(clean.reset()[0], noisy.reset()[0])
            heard_clean = clean.driver_send((1, 2, 0))[1]
            heard_noisy = noisy.driver_send((1, 2, 0))[1]
            self.assertEqual(heard_clean, (1, 2, 0))
            self.assertEqual(heard_noisy, (-1, -1, -1))
            clean.preparer_send(()); noisy.preparer_send(())
            clean.driver_act(HOLD); noisy.driver_act(HOLD)
            clean.preparer_act(WAIT); noisy.preparer_act(WAIT)


class ProtocolValidationTests(unittest.TestCase):
    def test_turn_order_is_enforced(self):
        env = TrapPrepHunt(targets=1, split='all')
        with self.assertRaises(RuntimeError): env.driver_send(())
        env.reset()
        with self.assertRaises(RuntimeError): env.preparer_send(())
        with self.assertRaises(RuntimeError): env.driver_act(HOLD)
        with self.assertRaises(RuntimeError): env.preparer_act(WAIT)
        env.driver_send((0, 0, 0))
        with self.assertRaises(RuntimeError): env.driver_send(())
        env.preparer_send((1,))
        with self.assertRaises(RuntimeError): env.preparer_send(())
        env.driver_act(HOLD)
        with self.assertRaises(RuntimeError): env.driver_act(HOLD)
        env.preparer_act(WAIT)

    def test_message_and_action_validation(self):
        env = TrapPrepHunt(targets=1, split='all', mechanisms=3)
        env.reset()
        with self.assertRaises(ValueError): env.driver_send((0, 0, 0, 0))
        with self.assertRaises(ValueError): env.driver_send((8, 0, 0))
        with self.assertRaises(ValueError): env.driver_send((True, 0, 0))
        env.driver_send(())
        with self.assertRaises(ValueError): env.preparer_send((0, 0))
        env.preparer_send(())
        with self.assertRaises(ValueError): env.driver_act(2)
        env.driver_act(drive_action(0))
        with self.assertRaises(ValueError): env.preparer_act(prepare_action(3))
        with self.assertRaises(ValueError): env.preparer_act(-1)
        env.preparer_act(WAIT)

    def test_reset_rejects_malformed_tasks(self):
        env = TrapPrepHunt(targets=2, split='all')
        with self.assertRaises(ValueError):
            env.reset(TrapTask((Target(0, 0, 1),)))
        with self.assertRaises(ValueError):
            env.reset(TrapTask((Target(0, 0, 1), Target(0, 1, 1))))
        with self.assertRaises(ValueError):
            env.reset(TrapTask((Target(0, 0, 1), Target(9, 1, 1))))

    def test_horizon_terminates_an_idle_episode(self):
        env = TrapPrepHunt(targets=1, split='all', horizon=4)
        steps = script(env, TrapTask((Target(0, 0, 3),)), [WAIT] * 6)
        self.assertEqual(len(steps), 4)
        self.assertTrue(steps[-1][1])
        self.assertEqual(steps[-1][2]['status'], (ACTIVE,))


class CostTests(unittest.TestCase):
    def test_message_and_move_costs_are_charged_and_logged(self):
        env = TrapPrepHunt(targets=1, split='all', message_cost=.1, move_cost=.5)
        steps = script(env, TrapTask((Target(2, 0, 3),)), [MOVE_RIGHT], message=(1, 1, 0))
        reward, _, info = steps[0]
        self.assertAlmostEqual(info['components']['message_cost'], .3)
        self.assertAlmostEqual(info['components']['move_cost'], .5)
        self.assertAlmostEqual(reward, -.8)

    def test_blocked_move_is_not_charged(self):
        env = TrapPrepHunt(targets=1, split='all', move_cost=.5)
        from emergent_hunt.trap_prep import MOVE_LEFT
        steps = script(env, TrapTask((Target(2, 0, 3),)), [MOVE_LEFT])
        self.assertEqual(steps[0][0], 0.0)


class BlindControlTests(unittest.TestCase):
    """The message-blind bound must be exact, not an empirical guess: it is the
    number the communication condition has to beat."""

    def test_reference_values_match_the_closed_form(self):
        self.assertAlmostEqual(blind_reference(split='all')['optimal_blind_success'], 1 / 9)
        self.assertAlmostEqual(blind_reference(split='train', by='pair')['optimal_blind_success'], 1 / 6)
        self.assertAlmostEqual(blind_reference(split='test', by='pair')['optimal_blind_success'], 1 / 3)
        for split in ('all', 'train', 'test'):
            self.assertAlmostEqual(
                blind_reference(split=split, by='triple')['optimal_blind_success'], 1 / 9)

    def test_reference_agrees_with_brute_force_through_the_simulator(self):
        # Small enough to enumerate every fixed preparer sequence and every
        # driver reply: the analytic bound must not exceed or undershoot what
        # the real environment actually permits.
        kwargs = dict(n=2, targets=1, mechanisms=2, horizon=3, by='pair')
        searched = blind_search(env_kwargs=kwargs)
        analytic = blind_reference(n=2, split='all', by='pair', horizon=3, mechanisms=2)
        self.assertAlmostEqual(searched['optimal_blind_catch_rate'],
                               analytic['optimal_blind_success'])

    def test_reference_protocol_beats_the_blind_bound_by_a_wide_margin(self):
        corpus = tasks(targets=1, split='test', by='pair')
        caught = sum(oracle_rollout(task)['caught'] for task in corpus)
        self.assertEqual(caught / len(corpus), 1.0)
        self.assertLess(blind_reference(split='test', by='pair')['optimal_blind_success'], .5)


class MutedProtocolTests(unittest.TestCase):
    """Negative control on the reference protocol itself: with the channel cut,
    the same policies must collapse, otherwise the task would not be about
    communication at all."""

    def test_muting_the_channel_destroys_the_reference_protocol(self):
        env = TrapPrepHunt(targets=2, split='all')
        caught = 0
        for task in tasks(split='all')[:40]:
            driver, preparer = reference_policies()
            result = run_episode(env, driver, preparer, task=task,
                                 perturb=lambda step, message: ())
            caught += result['caught']
        self.assertEqual(caught, 0)


if __name__ == '__main__':
    unittest.main()
