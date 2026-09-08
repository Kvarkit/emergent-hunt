"""One-message cooperative signaling task. No learned policy or hunting physics."""
from dataclasses import dataclass
from itertools import product
import random


@dataclass(frozen=True)
class State:
    prey: int
    direction: int
    trap: int


def states(n=3, split='all'):
    if type(n) is not int or n < 2:
        raise ValueError('n must be an integer >= 2')
    if split not in ('all', 'train', 'test'):
        raise ValueError('unknown split')
    return tuple(State(*x) for x in product(range(n), repeat=3)
                 if split == 'all' or ((sum(x) % n == 0) == (split == 'test')))


class SymbolicHunt:
    """Sender sees (prey, direction); receiver sees trap. One terminal decision.

    reset returns sender observation; send returns receiver observation. Neither
    exposes state, seed, reward or episode index before the terminal decision.
    This Python object is a trusted simulator, not a security sandbox.
    """
    def __init__(self, n=3, split='train', seed=0, vocabulary=8,
                 max_length=2, erasure=0.0, token_cost=0.0):
        self._states = states(n, split)
        if type(vocabulary) is not int or vocabulary < 1:
            raise ValueError('vocabulary must be positive')
        if type(max_length) is not int or max_length < 0:
            raise ValueError('max_length must be nonnegative')
        if not 0 <= erasure <= 1 or not 0 <= token_cost < float('inf'):
            raise ValueError('invalid channel parameters')
        self.n, self.vocabulary, self.max_length = n, vocabulary, max_length
        self.erasure, self.token_cost = erasure, token_cost
        # Channel interventions cannot alter the sampled state sequence.
        self._world_rng = random.Random(seed)
        self._channel_rng = random.Random(seed + 1)
        self._phase = 'done'

    def reset(self):
        self._state = self._world_rng.choice(self._states)
        self._phase = 'send'
        return (self._state.prey, self._state.direction)

    def send(self, message):
        if self._phase != 'send':
            raise RuntimeError('reset before sending; one message per episode')
        message = tuple(message)
        if len(message) > self.max_length or any(
                type(t) is not int or not 0 <= t < self.vocabulary for t in message):
            raise ValueError('invalid message')
        self._length = len(message)
        received = tuple(-1 if self._channel_rng.random() < self.erasure else t
                         for t in message)
        self._phase = 'act'
        return (self._state.trap, received)

    def step(self, action):
        if self._phase != 'act':
            raise RuntimeError('send before acting; one action per episode')
        action = tuple(action)
        if len(action) != 3 or any(type(x) is not int or not 0 <= x < self.n for x in action):
            raise ValueError('action must be three factor indices')
        success = action == (self._state.prey, self._state.direction, self._state.trap)
        self._phase = 'done'
        return float(success) - self.token_cost * self._length, True, {'success': success}
