import unittest
from emergent_hunt.multiround import MultiRoundHunt


class MultiRoundTests(unittest.TestCase):
    def test_private_views_and_bidirectional_rounds(self):
        e = MultiRoundHunt(rounds=2, seed=4)
        a, b = e.reset()
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)
        self.assertNotIn('trap_type', a)
        self.assertNotIn('prey_type', b)
        self.assertEqual(len(e.send(0, 1)), 2)
        self.assertEqual(len(e.send(1, 2)), 2)
        e.act(0, e.state.prey_zone)
        out = e.act(1, e.state.trap_zone)
        self.assertFalse(out[1])
        e.send(0, 3)
        e.send(1, 4)
        e.act(0, e.state.prey_zone)
        out = e.act(1, e.state.trap_zone)
        self.assertTrue(out[1])
        self.assertEqual(out[2]['round'], 2)

    def test_order_and_visibility_invariants(self):
        e = MultiRoundHunt()
        e.reset()
        with self.assertRaises(RuntimeError): e.send(1, 0)
        e.send(0, 0)
        with self.assertRaises(RuntimeError): e.act(0, 0)
        e.send(1, 0)
        with self.assertRaises(ValueError): e.act(0, 99)

    def test_trap_type_can_change_success(self):
        e = MultiRoundHunt(rounds=1, seed=1)
        e.reset()
        e.state = type(e.state)(e.state.prey_type, 1, e.state.prey_type, 2)
        e.send(0, 0); e.send(1, 0); e.act(0, 1)
        reward, done, info = e.act(1, 2)
        self.assertTrue(done)
        self.assertFalse(info['success'])
        self.assertEqual(reward, 0.0)


if __name__ == '__main__':
    unittest.main()
