import unittest
from emergent_hunt.environment import SymbolicHunt, states, held_out_pairs
from emergent_hunt.evaluate import controls, pair_lookup_baseline


class EnvironmentTests(unittest.TestCase):
    def test_split_and_factor_coverage(self):
        train, test = set(states(split='train')), set(states(split='test'))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(states()))
        for field in ('prey', 'direction', 'trap'):
            self.assertEqual({getattr(s, field) for s in train}, set(range(3)))
            self.assertEqual({getattr(s, field) for s in test}, set(range(3)))

    def test_by_pair_holds_out_whole_pairs(self):
        train = set(states(split='train', by='pair'))
        test = set(states(split='test', by='pair'))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(states(by='pair')))
        train_pairs = {(s.prey, s.direction) for s in train}
        test_pairs = {(s.prey, s.direction) for s in test}
        self.assertFalse(train_pairs & test_pairs)
        self.assertEqual(test_pairs, held_out_pairs(3))
        # every held-out pair appears for every trap value: n states each
        for p, d in held_out_pairs(3):
            self.assertEqual({s.trap for s in test if (s.prey, s.direction) == (p, d)},
                              set(range(3)))

    def test_pair_lookup_baseline_shortcut_and_its_closure(self):
        triple = pair_lookup_baseline(by='triple')
        self.assertEqual(triple['train_pair_coverage'], 1.0)
        pair = pair_lookup_baseline(by='pair')
        self.assertEqual(pair['train_pair_coverage'], 0.0)

    def test_reference_protocol(self):
        env = SymbolicHunt(token_cost=.1)
        for _ in range(100):
            observation = env.reset()
            trap, message = env.send(observation)
            reward, done, info = env.step((*message, trap))
            self.assertEqual((reward, done, info['success']), (.8, True, True))

    def test_channel_and_state_rng_independence(self):
        clean, erased = SymbolicHunt(seed=17), SymbolicHunt(seed=17, erasure=1)
        for _ in range(50):
            self.assertEqual(clean.reset(), erased.reset())
            trap, message = clean.send((0, 1))
            trap2, message2 = erased.send((0, 1))
            self.assertEqual(trap, trap2)
            self.assertEqual(message, (0, 1))
            self.assertEqual(message2, (-1, -1))

    def test_protocol_validation(self):
        env = SymbolicHunt()
        with self.assertRaises(RuntimeError): env.send(())
        env.reset()
        with self.assertRaises(ValueError): env.send((True,))
        with self.assertRaises(ValueError): env.send((8,))
        with self.assertRaises(ValueError): env.send((1, 2, 3))
        env.send(())
        with self.assertRaises(RuntimeError): env.send(())
        with self.assertRaises(ValueError): env.step((0, 0))
        env.step((0, 0, 0))
        with self.assertRaises(RuntimeError): env.step((0, 0, 0))

    def test_exact_blind_limits(self):
        self.assertAlmostEqual(controls()['optimal_no_message_success'], 1/9)
        self.assertAlmostEqual(controls(split='train')['optimal_no_message_success'], 1/6)
        self.assertAlmostEqual(controls(split='test')['optimal_no_message_success'], 1/3)


if __name__ == '__main__':
    unittest.main()
