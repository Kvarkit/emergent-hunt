"""Minimal two-agent, two-round cooperative hunt contract.

This is an environment specification and invariant-tested simulator. It has no
learning code yet. Each agent has a private view; both agents send one discrete
message per round and then choose a joint action.
"""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class HuntState:
    prey_type: int
    prey_zone: int
    trap_type: int
    trap_zone: int


class MultiRoundHunt:
    def __init__(self, prey_types=3, trap_types=3, zones=4, vocabulary=8,
                 rounds=2, seed=0, message_cost=0.0):
        if min(prey_types, trap_types, zones, vocabulary, rounds) < 2:
            raise ValueError('all cardinalities must be >= 2')
        if message_cost < 0:
            raise ValueError('message_cost must be nonnegative')
        self.prey_types, self.trap_types, self.zones = prey_types, trap_types, zones
        self.vocabulary, self.rounds, self.message_cost = vocabulary, rounds, message_cost
        self._rng = random.Random(seed)
        self._phase = 'done'

    def reset(self):
        self.state = HuntState(self._rng.randrange(self.prey_types),
                               self._rng.randrange(self.zones),
                               self._rng.randrange(self.trap_types),
                               self._rng.randrange(self.zones))
        self.round = 0
        self._messages = [None, None]
        self._phase = 'message_a'
        # A sees prey facts, B sees trap facts; zones are deliberately private.
        return ((self.state.prey_type, self.state.prey_zone),
                (self.state.trap_type, self.state.trap_zone))

    def send(self, agent, token):
        if self._phase not in ('message_a', 'message_b'):
            raise RuntimeError('reset or complete the current action first')
        if agent not in (0, 1) or type(token) is not int or not 0 <= token < self.vocabulary:
            raise ValueError('invalid agent or token')
        expected = 0 if self._phase == 'message_a' else 1
        if agent != expected:
            raise RuntimeError('messages must alternate A then B')
        self._messages[agent] = token
        if agent == 0:
            self._phase = 'message_b'
            return (self.state.trap_type, self.state.trap_zone)
        self._phase = 'act_a'
        return (self.state.prey_type, self.state.prey_zone)

    def act(self, agent, action):
        if self._phase not in ('act_a', 'act_b'):
            raise RuntimeError('send both messages before acting')
        expected = 0 if self._phase == 'act_a' else 1
        if agent != expected or type(action) is not int or not 0 <= action < self.zones:
            raise ValueError('invalid acting agent or zone')
        if agent == 0:
            self._action_a = action
            self._phase = 'act_b'
            return None
        self._action_b = action
        success = (self._action_a == self.state.prey_zone and
                   self._action_b == self.state.trap_zone and
                   self.state.prey_type != self.state.trap_type)
        reward = float(success) - self.message_cost * 2 * self.rounds
        self.round += 1
        done = self.round >= self.rounds
        info = {'success': success, 'round': self.round,
                'messages': tuple(self._messages)}
        if done:
            self._phase = 'done'
        else:
            self._messages = [None, None]
            self._phase = 'message_a'
        return reward, done, info
