"""Multi-target physical contract; messages are opaque, observations are private.

A sees target mechanisms and positions and can herd a selected target.
B sees its position, trap preparation and arrival countdowns. It must learn
which target/mechanism to prepare from A's messages. One step delivers the
messages sent on that step, after simultaneous physical actions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    verb: str = 'wait'
    target: int = 0
    mechanism: int = 0


class CoordinatedHunt:
    def __init__(self, positions=(1, 3), mechanisms=(0, 1), length=5,
                 mechanism_count=2, horizon=16, travel_time=3, window=2,
                 step_cost=.01, vocabulary=8, max_message_length=3):
        if (length < 2 or horizon < 1 or mechanism_count < 1 or travel_time < 1
                or window < 1 or not 0 <= step_cost < float('inf')):
            raise ValueError('invalid environment configuration')
        if not positions or len(positions) != len(mechanisms):
            raise ValueError('one mechanism per target required')
        if any(type(x) is not int or not 0 <= x < length for x in positions):
            raise ValueError('invalid position')
        if any(type(x) is not int or not 0 <= x < mechanism_count for x in mechanisms):
            raise ValueError('invalid mechanism')
        self.positions, self.mechanisms = tuple(positions), tuple(mechanisms)
        self.length, self.horizon, self.mechanism_count = length, horizon, mechanism_count
        self.travel_time, self.window, self.step_cost = travel_time, window, step_cost
        if type(vocabulary) is not int or vocabulary < 1 or type(max_message_length) is not int or max_message_length < 1:
            raise ValueError('invalid channel capacity')
        self.vocabulary, self.max_message_length = vocabulary, max_message_length
        self.reset()

    def reset(self):
        self.time = 0
        self.b_pos = 0
        self.prepared = [None] * len(self.positions)
        self.charged = [True] * len(self.positions)
        self.arrival = [None] * len(self.positions)
        self.outcomes = ['pending'] * len(self.positions)
        self.preparation_paid = [False] * len(self.positions)
        self.incoming = (None, None)
        self.done = False
        return self.observe('a'), self.observe('b')

    def observe(self, agent):
        common = dict(time=self.time, outcomes=tuple(self.outcomes))
        if agent == 'a':
            return dict(common, positions=self.positions, mechanisms=self.mechanisms,
                        incoming=self.incoming[0])
        if agent == 'b':
            return dict(common, position=self.b_pos, prepared=tuple(self.prepared),
                        charged=tuple(self.charged),
                        countdowns=tuple(None if t is None else t-self.time for t in self.arrival),
                        incoming=self.incoming[1])
        raise ValueError('unknown agent')

    def _validate(self, action, agent):
        verbs = ('wait', 'herd') if agent == 'a' else ('wait', 'left', 'right', 'prepare', 'activate')
        if not isinstance(action, Action) or action.verb not in verbs:
            raise ValueError('invalid action')
        if type(action.target) is not int or not 0 <= action.target < len(self.positions):
            raise ValueError('invalid target')
        if type(action.mechanism) is not int or not 0 <= action.mechanism < self.mechanism_count:
            raise ValueError('invalid mechanism')

    def step(self, a=Action(), b=Action(), message_a=None, message_b=None):
        if self.done:
            raise RuntimeError('episode ended')
        self._validate(a, 'a'); self._validate(b, 'b')
        # Arbitrary discrete sequences; no token-to-factor mapping is supplied.
        for message in (message_a, message_b):
            if message is not None and (not isinstance(message, tuple) or
                    len(message) > self.max_message_length or
                    any(type(t) is not int or not 0 <= t < self.vocabulary for t in message)):
                raise ValueError('messages must be tuples of nonnegative token ids')
        reward = -self.step_cost
        events = []
        i = a.target
        if a.verb == 'herd' and self.outcomes[i] == 'pending' and self.arrival[i] is None:
            self.arrival[i] = self.time + self.travel_time
        if b.verb == 'left': self.b_pos = max(0, self.b_pos-1)
        if b.verb == 'right': self.b_pos = min(self.length-1, self.b_pos+1)
        j = b.target
        at_target = self.b_pos == self.positions[j]
        available = self.outcomes[j] == 'pending' and self.charged[j]
        if b.verb == 'prepare' and at_target and available:
            self.prepared[j] = b.mechanism
            if b.mechanism == self.mechanisms[j] and not self.preparation_paid[j]:
                self.preparation_paid[j] = True
                reward += .2
                events.append(('prepared', j))
        if b.verb == 'activate' and at_target and available:
            self.charged[j] = False
            arrival = self.arrival[j]
            success = (arrival is not None and arrival <= self.time < arrival+self.window
                       and self.prepared[j] == self.mechanisms[j])
            self.outcomes[j] = 'caught' if success else 'failed'
            reward += .8 if success else -.1
            events.append((self.outcomes[j], j))
        self.time += 1
        for k, arrival in enumerate(self.arrival):
            if (self.outcomes[k] == 'pending' and arrival is not None
                    and self.time >= arrival+self.window):
                self.outcomes[k] = 'escaped'
                events.append(('escaped', k))
        self.done = self.time >= self.horizon or all(x != 'pending' for x in self.outcomes)
        self.incoming = (message_b, message_a)
        info = dict(events=events, captures=self.outcomes.count('caught'),
                    full_success=all(x == 'caught' for x in self.outcomes))
        return (self.observe('a'), self.observe('b')), reward, self.done, info
