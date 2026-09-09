import unittest
from emergent_hunt.coordinated_hunt import Action, CoordinatedHunt


class CoordinatedHuntTests(unittest.TestCase):
    def test_two_targets_require_preparation_and_timed_activation(self):
        e = CoordinatedHunt(horizon=30)
        total = 0
        for target, position in enumerate(e.positions):
            while e.b_pos < position:
                total += e.step(b=Action('right'))[1]
            total += e.step(a=Action('herd', target),
                            b=Action('prepare', target, e.mechanisms[target]))[1]
            while e.time < e.arrival[target]:
                total += e.step()[1]
            _, reward, _, info = e.step(b=Action('activate', target))
            total += reward
        self.assertTrue(info['full_success'])
        self.assertEqual(info['captures'], 2)
        self.assertAlmostEqual(total, 2 - e.time*.01)

    def test_partial_reward_cannot_be_farmed(self):
        e=CoordinatedHunt(positions=(0,), mechanisms=(1,))
        self.assertAlmostEqual(e.step(b=Action('prepare',0,1))[1], .19)
        self.assertAlmostEqual(e.step(b=Action('prepare',0,1))[1], -.01)
        self.assertEqual(e.outcomes, ['pending'])

    def test_early_activation_spends_charge(self):
        e=CoordinatedHunt(positions=(0,), mechanisms=(0,))
        e.step(b=Action('prepare'))
        _,reward,done,info=e.step(b=Action('activate'))
        self.assertTrue(done)
        self.assertFalse(e.charged[0])
        self.assertEqual(info['captures'],0)
        self.assertLess(reward,0)

    def test_late_activation_loses_target(self):
        e=CoordinatedHunt(positions=(0,), mechanisms=(0,), travel_time=1,window=1)
        e.step(a=Action('herd'),b=Action('prepare'))
        e.step()
        self.assertEqual(e.outcomes,['escaped'])

    def test_wrong_mechanism_fails_at_right_time(self):
        e=CoordinatedHunt(positions=(0,), mechanisms=(1,),travel_time=1)
        e.step(a=Action('herd'),b=Action('prepare',0,0))
        self.assertEqual(e.step(b=Action('activate'))[3]['captures'],0)

    def test_target_binding_and_private_delivery(self):
        e=CoordinatedHunt(positions=(0,0), mechanisms=(0,1))
        obs,_,_,_=e.step(b=Action('prepare',1,1),message_a=(7,2),message_b=(4,))
        self.assertEqual(e.prepared,[None,1])
        self.assertNotIn('mechanisms',obs[1])
        self.assertNotIn('positions',obs[1])
        self.assertEqual(obs[1]['incoming'],(7,2))
        self.assertEqual(obs[0]['incoming'],(4,))

    def test_invalid_action_does_not_advance_world(self):
        e=CoordinatedHunt()
        with self.assertRaises(ValueError):
            e.step(a=Action('herd'),b=Action('prepare',99))
        self.assertEqual(e.time,0)
        self.assertEqual(e.arrival,[None,None])
