import unittest

from emergent_hunt.line_hunt import LEFT, RIGHT, TRIGGER, LineHunt, oracle_rollout


class LineHuntTests(unittest.TestCase):
    def test_actions_change_next_observation_and_progress(self):
        env = LineHunt(crossed=True, progress_weight=.1, step_cost=.01)
        env.reset(4, 0, start_a=1, start_b=3)
        obs, toward, _, info = env.step(RIGHT, LEFT)
        self.assertEqual((obs['a']['self_pos'], obs['b']['self_pos']), (2,2))
        self.assertEqual(info['distance_progress'], 2)
        _, away, _, info = env.step(LEFT, RIGHT)
        self.assertEqual(info['distance_progress'], -2)
        self.assertGreater(toward, 0)
        self.assertLess(away, 0)
        self.assertAlmostEqual(toward+away, -.02)

    def test_wrong_activation_consumes_recovery_time(self):
        env = LineHunt(crossed=True, trigger_delay=2, step_cost=.01)
        env.reset(4, 0)
        obs, reward, done, info = env.step(2, TRIGGER)
        self.assertEqual(obs['b']['step'], 3)
        self.assertEqual(info['wrong_trigger'], 1)
        self.assertAlmostEqual(reward, -.03)
        self.assertFalse(done)

    def test_terminal_potential_cancels_path_dependent_progress(self):
        totals=[]
        for actions in (((RIGHT,LEFT),(LEFT,RIGHT)), ((2,2),(2,2))):
            env=LineHunt(horizon=2, crossed=True, progress_weight=.1)
            env.reset(4,0)
            total=0
            for a,b in actions:
                _,reward,_,_=env.step(a,b)
                total+=reward
            totals.append(total)
        self.assertAlmostEqual(totals[0],totals[1])

    def test_oracle_reaches_goal_and_trigger(self):
        env, trace = oracle_rollout(goal_pos=3, trap_pos=1)
        self.assertTrue(env.state.success)
        self.assertEqual(trace[-1]["reward"], 1.0)

    def test_private_observations_and_delivery(self):
        env = LineHunt()
        env.reset(goal_pos=1, trap_pos=3)
        self.assertIn("private_goal", env.observe("a"))
        self.assertNotIn("private_trap", env.observe("a"))
        self.assertIn("private_trap", env.observe("b"))
        self.assertEqual(env.deliver(2, 4), (2, 4))

    def test_crossed_mode_swaps_private_facts(self):
        env = LineHunt(crossed=True)
        env.reset(goal_pos=1, trap_pos=3)
        self.assertIn("partner_target", env.observe("a"))
        self.assertNotIn("private_goal", env.observe("a"))
        self.assertIn("partner_target", env.observe("b"))

    def test_transition_order_and_terminal(self):
        env = LineHunt(length=3, horizon=2)
        env.reset(goal_pos=1, trap_pos=2, start_a=0, start_b=1)
        with self.assertRaises(ValueError):
            env.step(99, RIGHT)
        env.reset(goal_pos=1, trap_pos=2, start_a=0, start_b=1)
        _, reward, done, _ = env.step(2, RIGHT)
        self.assertEqual((reward, done), (0.0, False))
        _, reward, done, info = env.step(RIGHT, TRIGGER)
        self.assertEqual((reward, done, info["success"]), (1.0, True, 1))

    def test_no_step_after_terminal(self):
        env = LineHunt(length=3, horizon=1)
        env.reset(goal_pos=0, trap_pos=2)
        env.step(2, TRIGGER)
        with self.assertRaises(RuntimeError):
            env.step(2, TRIGGER)

    def test_default_world_has_no_message_oracle_shortcut(self):
        env = LineHunt(length=5, horizon=8)
        env.reset(goal_pos=1, trap_pos=3)
        for _ in range(8):
            s = env.state
            a = 1 if s.a_pos < s.goal_pos else 0 if s.a_pos > s.goal_pos else 2
            b = 1 if s.b_pos < s.trap_pos else 0 if s.b_pos > s.trap_pos else 3
            _, reward, done, _ = env.step(a, b)
            if done:
                break
        self.assertEqual(reward, 1.0)


if __name__ == "__main__":
    unittest.main()
