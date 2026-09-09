"""Trap preparation and activation: reach -> prepare(method) -> activate(timing).

Task variant proposed by a collaborator on the research board, verbatim spec:

1. *Reach -> prepare -> activate.* Preparation changes the trap's state. Correct
   preparation gives partial reward; a successful catch gives full reward.
   Repeating preparation earns no additional points.
2. *Different preparation methods.* The type of prey determines which mechanism
   is needed; the agent at the trap cannot see the prey. The sender must
   communicate both the location and the required action.
3. *Timing coordination.* One agent drives the prey toward the trap, the other
   prepares/activates it. Activating too early wastes the trap's charge;
   activating too late lets the prey escape. Readiness and situation-change
   messages ("prey approaching", "trap ready", "prey fled") become useful.
4. *Multiple targets.* The message must bind an action to a specific target:
   identical facts attributed to different objects require different decisions.

Design notes (why the contract looks like this):

* Every zone carries a trap with one charge, so the trap *layout* is common
  knowledge and carries no information. What is hidden from the preparer is
  which zones a prey is being driven to, what type each prey is, and where the
  prey currently is. Consequently the preparer's observation stream is a
  deterministic function of its own action sequence alone -- which is what
  makes the message-blind control in `blind_reference` exactly computable
  rather than an empirical guess (see the theorem in its docstring).
* Target binding is *physical*: PREPARE and ACTIVATE apply to the trap at the
  preparer's current position. So a decoded instruction decomposes into
  (target = where it stands, method = which PREPARE, timing = which step it
  ACTIVATEs) -- three separable components, which is what
  `trap_probe.selectivity` perturbs one token at a time to test.
* No token, position or message length is given a meaning by the environment.
  `reference_policies` in this module is a handwritten oracle used as a
  positive control and as ground truth for the probe; it is not emergent
  language and no learned result is claimed from it.

Splits follow environment.py: `by='pair'` withholds whole (prey_type, zone)
combinations -- exactly the two things the sender must transmit -- so a lookup
table keyed on that pair cannot cover the test split, while `by='triple'` is the
easy split in which every such pair still recurs in train.

WARNING about that test split, found by tightening the blind bound: it withholds
the pairs with (prey_type + zone) % n == 0, which leaves exactly ONE prey type
per zone in test. A message-blind preparer that hardcodes the right method per
zone and sweeps the line therefore solves it completely -- if it has time. So a
catch rate on the held-out split is only meaningful under a deadline that makes
sweeping impossible: give the prey a finite `patience` (n=3 wants patience=4)
and always report trap_prep.blind_reference for the exact configuration used.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
import random

from .environment import held_out_pairs

# Preparer actions. PREPARE(m) == PREPARE_BASE + m, so the action space is
# PREPARE_BASE + mechanisms wide.
MOVE_LEFT, MOVE_RIGHT, WAIT, ACTIVATE = range(4)
PREPARE_BASE = 4

# Driver actions. DRIVE(j) == DRIVE_BASE + j advances prey j by one zone.
HOLD = 0
DRIVE_BASE = 1

UNPREPARED = -1

# Per-target status codes reported to the driver (which can see the prey).
ACTIVE, CAUGHT, ESCAPED = 0, 1, 2


def prepare_action(mechanism):
    return PREPARE_BASE + mechanism


def drive_action(target):
    return DRIVE_BASE + target


def mechanism_for(prey_type, mechanisms=3):
    """Which preparation mechanism a prey type requires. Deterministic and
    public; the difficulty is that the preparer cannot see the prey type."""
    return prey_type % mechanisms


@dataclass(frozen=True)
class Target:
    zone: int
    prey_type: int
    start_distance: int


@dataclass(frozen=True)
class TrapTask:
    targets: tuple

    def zones(self):
        return tuple(t.zone for t in self.targets)


def tasks(n=3, targets=2, split='all', by='pair', mechanisms=None):
    """All tasks: `targets` prey on distinct zones, each with a type and a
    starting distance in 1..n.

    by='pair': a task is in test iff ANY of its targets uses a held-out
      (prey_type, zone) pair (environment.held_out_pairs). No held-out pair ever
      appears in train, for any distance or any partner target.
    by='triple': test iff the sum of every field is 0 mod n -- the easy split,
      in which each (prey_type, zone) pair still recurs in train.
    """
    if type(n) is not int or n < 2:
        raise ValueError('n must be an integer >= 2')
    if type(targets) is not int or not 1 <= targets <= n:
        raise ValueError('targets must be an integer in 1..n')
    if split not in ('all', 'train', 'test'):
        raise ValueError('unknown split')
    if by not in ('triple', 'pair'):
        raise ValueError('unknown by')
    mechanisms = n if mechanisms is None else mechanisms
    held = held_out_pairs(n)
    out = []
    for zones in permutations(range(n), targets):
        for types in product(range(n), repeat=targets):
            for distances in product(range(1, n + 1), repeat=targets):
                task = TrapTask(tuple(Target(z, p, d)
                                      for z, p, d in zip(zones, types, distances)))
                if by == 'pair':
                    is_test = any((p, z) in held for z, p in zip(zones, types))
                else:
                    is_test = (sum(zones) + sum(types) + sum(distances)) % n == 0
                if split == 'all' or is_test == (split == 'test'):
                    out.append(task)
    return tuple(out)


class TrapPrepHunt:
    """Two agents, `targets` prey, one trap per zone, one charge each.

    Turn order within a timestep:
      driver_send -> preparer_send -> driver_act -> preparer_act -> transition.
    The preparer therefore hears the driver before acting in the same step, and
    the driver hears the readiness reply before committing its own action, so a
    "ready?" / "now!" handshake is expressible without a one-step lag.

    Views. The driver sees every prey (type, target zone, distance, window,
    status) and nothing about the preparer. The preparer sees its own position,
    the trap under it, and the clock -- never the prey. Neither view exposes the
    other agent's state, the reward, or the split.

    This is a trusted simulator, not a security sandbox.
    """

    def __init__(self, n=3, targets=2, split='train', by='pair', seed=0,
                 mechanisms=None, horizon=16, window=1, vocabulary=8,
                 driver_message_length=3, preparer_message_length=1,
                 erasure=0.0, message_cost=0.0, move_cost=0.0, partial=0.25,
                 approach=0.0, start_pos=0, patience=None,
                 stop_when_resolved=True):
        self._tasks = tasks(n, targets, split, by)
        mechanisms = n if mechanisms is None else mechanisms
        if type(mechanisms) is not int or mechanisms < 1:
            raise ValueError('mechanisms must be a positive integer')
        if type(vocabulary) is not int or vocabulary < 1:
            raise ValueError('vocabulary must be positive')
        for name, value in (('driver_message_length', driver_message_length),
                            ('preparer_message_length', preparer_message_length)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be nonnegative')
        if type(horizon) is not int or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if type(window) is not int or window < 1:
            raise ValueError('window must be a positive integer')
        if not 0 <= erasure <= 1:
            raise ValueError('erasure must be a probability')
        if min(message_cost, move_cost) < 0:
            raise ValueError('costs must be nonnegative')
        if not 0 <= partial <= 1:
            raise ValueError('partial must be in [0, 1]')
        if not 0 <= approach or partial + approach > 1:
            raise ValueError('approach must be nonnegative with partial + approach <= 1')
        if not 0 <= start_pos < n:
            raise ValueError('start_pos outside the line')
        if patience is not None and (type(patience) is not int or patience < 1):
            raise ValueError('patience must be None or a positive integer')
        self.n, self.targets, self.mechanisms = n, targets, mechanisms
        self.horizon, self.window, self.partial = horizon, window, partial
        self.approach = approach
        self.vocabulary = vocabulary
        self.driver_message_length = driver_message_length
        self.preparer_message_length = preparer_message_length
        self.erasure, self.message_cost, self.move_cost = erasure, message_cost, move_cost
        self.start_pos, self.patience = start_pos, patience
        self.stop_when_resolved = stop_when_resolved
        # Channel noise must not perturb the sampled task sequence.
        self._world_rng = random.Random(seed)
        self._channel_rng = random.Random(seed + 1)
        self._phase = 'done'

    # -- observation contract -------------------------------------------------

    @property
    def prepared_methods(self):
        """Diagnostic view of every trap's prepared mechanism (-1 = unprepared).
        Not an observation: neither agent ever sees the whole vector."""
        return tuple(self._prepared)

    @property
    def step_index(self):
        """Number of completed-or-current timesteps; 0 before the first send."""
        return self._step

    def driver_observation(self):
        return {'step': self._step, 'steps_left': self.horizon - self._step,
                'prey_type': tuple(t.prey_type for t in self.task.targets),
                'zone': tuple(t.zone for t in self.task.targets),
                'distance': tuple(self._distance),
                'window_left': tuple(self._window_left),
                'patience_left': tuple(self._patience_left),
                'status': tuple(self._status)}

    def preparer_observation(self):
        return {'step': self._step, 'steps_left': self.horizon - self._step,
                'self_pos': self._pos,
                'trap_method': self._prepared[self._pos],
                'trap_charged': int(self._charged[self._pos])}

    # -- episode protocol -----------------------------------------------------

    def reset(self, task=None):
        if task is None:
            task = self._world_rng.choice(self._tasks)
        if len(task.targets) != self.targets:
            raise ValueError('task has the wrong number of targets')
        for t in task.targets:
            if not (0 <= t.zone < self.n and 0 <= t.prey_type < self.n
                    and 1 <= t.start_distance):
                raise ValueError('target outside the task space')
        if len(set(t.zone for t in task.targets)) != self.targets:
            raise ValueError('targets must occupy distinct zones')
        self.task = task
        self._step = 0
        self._pos = self.start_pos
        self._prepared = [UNPREPARED] * self.n
        self._charged = [True] * self.n
        self._paid = [False] * self.n
        self._distance = [t.start_distance for t in task.targets]
        self._window_left = [0] * self.targets
        # patience=None means "no deadline beyond the horizon", so the counter
        # is initialized to the horizon and never bites before the episode ends.
        deadline = self.horizon if self.patience is None else self.patience
        self._patience_left = [deadline] * self.targets
        self._status = [ACTIVE] * self.targets
        self._approached = [False] * self.targets
        self._totals = {'prepare': 0.0, 'approach': 0.0, 'catch': 0.0,
                        'move_cost': 0.0, 'message_cost': 0.0, 'reward': 0.0}
        self._events = []
        self.trace = []
        self._phase = 'driver_send'
        return self.driver_observation(), self.preparer_observation()

    def _deliver(self, message, length):
        message = tuple(message)
        if len(message) > length or any(
                type(t) is not int or not 0 <= t < self.vocabulary for t in message):
            raise ValueError('invalid message')
        self._step_cost += self.message_cost * len(message)
        return tuple(-1 if self._channel_rng.random() < self.erasure else t
                     for t in message)

    def driver_send(self, message):
        """Driver -> preparer. Returns the preparer's view and what it hears."""
        if self._phase != 'driver_send':
            raise RuntimeError('not the driver\'s turn to send')
        self._step += 1
        self._step_cost = 0.0
        self._step_events = []
        self._driver_sent = tuple(message)
        received = self._deliver(message, self.driver_message_length)
        self._driver_received = received
        self._phase = 'preparer_send'
        return self.preparer_observation(), received

    def preparer_send(self, message):
        """Preparer -> driver (readiness). Returns the driver's view and what it hears."""
        if self._phase != 'preparer_send':
            raise RuntimeError('not the preparer\'s turn to send')
        self._preparer_sent = tuple(message)
        received = self._deliver(message, self.preparer_message_length)
        self._preparer_received = received
        self._phase = 'driver_act'
        return self.driver_observation(), received

    def driver_act(self, action):
        if self._phase != 'driver_act':
            raise RuntimeError('send both messages before the driver acts')
        if type(action) is not int or not 0 <= action < DRIVE_BASE + self.targets:
            raise ValueError('invalid driver action')
        self._driver_action = action
        if action != HOLD:
            j = action - DRIVE_BASE
            if self._status[j] == ACTIVE and self._distance[j] > 0:
                self._distance[j] -= 1
                if self._distance[j] == 0:
                    self._window_left[j] = self.window
                    self._step_events.append(('arrived', j))
            else:
                self._step_events.append(('drive_ignored', j))
        self._phase = 'preparer_act'
        return None

    def _catchable(self, zone):
        for j, t in enumerate(self.task.targets):
            if (t.zone == zone and self._status[j] == ACTIVE and self._distance[j] == 0
                    and self._window_left[j] > 0):
                return j
        return None

    def preparer_act(self, action):
        """Apply the preparer's action, close the timestep, return (reward, done, info)."""
        if self._phase != 'preparer_act':
            raise RuntimeError('the driver must act first')
        if type(action) is not int or not 0 <= action < PREPARE_BASE + self.mechanisms:
            raise ValueError('invalid preparer action')
        reward = -self._step_cost
        components = {'prepare': 0.0, 'approach': 0.0, 'catch': 0.0,
                      'move_cost': 0.0, 'message_cost': self._step_cost}
        # Optional shaping (approach=0 reproduces the original three-part
        # scheme exactly). Paid once per target, BEFORE this step's action is
        # applied, for the state the whole protocol is a plan to reach: the prey
        # standing on a charged trap that already carries the mechanism its type
        # requires. It is the missing rung on the ladder -- reaching it needs the
        # method to have been communicated AND the arrival to have been timed --
        # and without it the driver has no partial credit whatsoever, since
        # DRIVE only ever pays through a catch. Deducted from the catch reward,
        # so a full success still totals exactly 1.0 per target, and the
        # message-blind bound is untouched because that bound counts catches.
        for j, t in enumerate(self.task.targets):
            if (self._status[j] == ACTIVE and not self._approached[j]
                    and self._distance[j] == 0 and self._window_left[j] > 0
                    and self._charged[t.zone]
                    and self._prepared[t.zone] == mechanism_for(
                        t.prey_type, self.mechanisms)):
                self._approached[j] = True
                # The event is logged whether or not it is paid, so
                # `in_position` is available as a diagnostic in the unshaped
                # scheme too.
                self._step_events.append(('in_position', j))
                if self.approach:
                    components['approach'] += self.approach
                    reward += self.approach
        zone = self._pos
        activation = None
        if action in (MOVE_LEFT, MOVE_RIGHT):
            self._pos = (max(0, zone - 1) if action == MOVE_LEFT
                         else min(self.n - 1, zone + 1))
            if self._pos != zone:
                components['move_cost'] = self.move_cost
                reward -= self.move_cost
        elif action >= PREPARE_BASE:
            mechanism = action - PREPARE_BASE
            required = [mechanism_for(t.prey_type, self.mechanisms)
                        for t in self.task.targets if t.zone == zone]
            # Partial reward for CORRECT preparation, and only for the FIRST
            # preparation attempted at this trap: preparation is a one-shot
            # commitment, so the partial reward measures a *guess*.
            #
            # This is not merely "no extra points for repeating". An earlier
            # version paid whenever the current mechanism first matched, which
            # let a message-blind preparer cycle PREPARE(0), PREPARE(1),
            # PREPARE(2) on one trap and collect the partial reward with
            # probability 1, without reading anything. Under that rule the
            # partial reward carried no evidence that the mechanism had been
            # communicated -- and REINFORCE duly converged on a fixed sweep
            # with constant, information-free messages (results/trap-single-
            # target-s*.json). Paying only the first attempt makes the blind
            # expectation exactly partial/mechanisms and gives the channel a
            # dense gradient. The trap's mechanism itself stays mutable, so a
            # later correction can still enable a catch; only the payment is
            # one-shot, and the message-blind catch bound is unaffected because
            # only the mechanism standing at activation time decides a catch.
            first_attempt = not self._paid[zone]
            self._paid[zone] = True
            self._prepared[zone] = mechanism
            if first_attempt and required and mechanism == required[0]:
                components['prepare'] = self.partial
                reward += self.partial
                self._step_events.append(('prepared_correctly', zone))
            else:
                self._step_events.append(('prepared', zone))
        elif action == ACTIVATE:
            if not self._charged[zone]:
                self._step_events.append(('no_charge', zone))
            else:
                self._charged[zone] = False
                j = self._catchable(zone)
                # Ground-truth annotation for diagnostics only; never observed.
                activation = {'zone': zone, 'on_window': j is not None,
                              'method_ok': None if j is None else
                              self._prepared[zone] == mechanism_for(
                                  self.task.targets[j].prey_type, self.mechanisms)}
                if j is not None and self._prepared[zone] == mechanism_for(
                        self.task.targets[j].prey_type, self.mechanisms):
                    self._status[j] = CAUGHT
                    components['catch'] = 1.0 - self.partial - self.approach
                    reward += 1.0 - self.partial - self.approach
                    self._step_events.append(('caught', j))
                elif j is not None:
                    self._step_events.append(('wrong_method', zone))
                else:
                    self._step_events.append(('wasted_charge', zone))
        self._preparer_action = action

        # End of timestep: prey that arrived and were not caught run their window
        # down; every prey also runs down its own patience and flees when it
        # expires. Patience is what stops the driver from stalling indefinitely
        # while a message-blind preparer sweeps and fires every trap in turn --
        # see blind_upper_bound, where it caps the schedule.
        for j in range(self.targets):
            if self._status[j] != ACTIVE:
                continue
            if self._distance[j] == 0:
                self._window_left[j] -= 1
                if self._window_left[j] <= 0:
                    self._status[j] = ESCAPED
                    self._step_events.append(('escaped', j))
                    continue
            self._patience_left[j] -= 1
            if self._patience_left[j] <= 0:
                self._status[j] = ESCAPED
                self._step_events.append(('fled', j))
        for key in ('prepare', 'approach', 'catch'):
            self._totals[key] += components[key]
        for key in ('move_cost', 'message_cost'):
            self._totals[key] += components[key]
        self._totals['reward'] += reward
        self._events.extend(self._step_events)
        done = self._step >= self.horizon or (
            self.stop_when_resolved and all(s != ACTIVE for s in self._status))
        info = {'step': self._step,
                'events': tuple(self._step_events),
                'components': components,
                'totals': dict(self._totals),
                'status': tuple(self._status),
                'caught': sum(s == CAUGHT for s in self._status),
                'success': all(s == CAUGHT for s in self._status),
                'driver_sent': self._driver_sent,
                'driver_received': self._driver_received,
                'preparer_sent': self._preparer_sent,
                'preparer_action': action,
                'preparer_pos': self._pos,
                'activation': activation,
                'driver_action': self._driver_action}
        self.trace.append(info)
        self._phase = 'done' if done else 'driver_send'
        return reward, done, info


# -- handwritten reference protocol -------------------------------------------
# Positive control and probe ground truth. The environment assigns no meaning to
# any token; these two policies agree on one by construction. Slot 0 names the
# target zone, slot 1 the preparation method, slot 2 carries the "now!" cue.
# This is not emergent language and no learned result is claimed from it.

READY, NOT_READY = 1, 0
CUE_NOW, CUE_WAIT = 1, 0


class ReferenceDriver:
    """Announces (zone, method, cue) for the first unresolved target, holds the
    prey one zone out until the preparer reports readiness, then cues and drives."""

    def __init__(self, mechanisms=3):
        self.mechanisms = mechanisms

    def reset(self, observation):
        self._ready = False
        self._cued = False

    def _current(self, observation):
        for j, status in enumerate(observation['status']):
            if status == ACTIVE:
                return j
        return None

    def message(self, observation):
        j = self._current(observation)
        if j is None:
            return (0, 0, CUE_WAIT)
        cue = CUE_NOW if (self._ready and observation['distance'][j] == 1) else CUE_WAIT
        self._cued = cue == CUE_NOW
        return (observation['zone'][j],
                mechanism_for(observation['prey_type'][j], self.mechanisms), cue)

    def act(self, observation, incoming):
        self._ready = bool(incoming) and incoming[0] == READY
        j = self._current(observation)
        if j is None:
            return HOLD
        # Never let a prey arrive before the cue: it would have to escape.
        if observation['distance'][j] > 1 or (self._cued and self._ready):
            return drive_action(j)
        return HOLD


class ReferencePreparer:
    """Reach the announced zone, prepare the announced method, report readiness,
    activate on the cue. Reads slots positionally; the meanings are the
    driver's, not the environment's.

    Readiness is *predictive*: it reports READY as soon as it is standing on the
    announced charged trap, because its own action this step will arm it. A
    retrospective "the trap is armed" reply would cost one extra step per
    target, since the driver's message goes out before it hears the reply. That
    step is not free -- it raises the shortest solvable horizon, and the horizon
    is exactly what decides how many traps a message-blind sweeper can fire (see
    blind_upper_bound), so the lag would inflate the control it is measured
    against.
    """

    def __init__(self, n=3, mechanisms=3):
        self.n, self.mechanisms = n, mechanisms

    def reset(self, observation):
        self._incoming = ()

    def message(self, observation, incoming):
        self._incoming = incoming
        zone, method, _ = self._decode(incoming)
        ready = (zone is not None and observation['self_pos'] == zone
                 and observation['trap_charged'])
        return (READY if ready else NOT_READY,)

    def _decode(self, incoming):
        if len(incoming) < 3 or any(t < 0 for t in incoming):
            return None, None, False
        return (incoming[0] % self.n, incoming[1] % self.mechanisms,
                bool(incoming[2] % 2))

    def act(self, observation):
        zone, method, now = self._decode(self._incoming)
        if zone is None:
            return WAIT
        pos = observation['self_pos']
        if pos != zone:
            return MOVE_RIGHT if pos < zone else MOVE_LEFT
        if observation['trap_method'] != method:
            return prepare_action(method)
        return ACTIVATE if now else WAIT


def reference_policies(n=3, mechanisms=3):
    return ReferenceDriver(mechanisms), ReferencePreparer(n, mechanisms)


def run_episode(env, driver, preparer, task=None, perturb=None):
    """Drive a full episode with two policies. `perturb(step, message)` may
    rewrite the driver's message before it is sent (the token-level
    intervention hook used by trap_probe)."""
    driver_obs, preparer_obs = env.reset(task)
    driver.reset(driver_obs)
    preparer.reset(preparer_obs)
    total, done = 0.0, False
    while not done:
        message = tuple(driver.message(driver_obs))
        if perturb is not None:
            message = tuple(perturb(env.step_index + 1, message))
        preparer_obs, heard = env.driver_send(message)
        reply = tuple(preparer.message(preparer_obs, heard))
        driver_obs, back = env.preparer_send(reply)
        env.driver_act(driver.act(driver_obs, back))
        reward, done, info = env.preparer_act(preparer.act(preparer_obs))
        total += reward
        driver_obs, preparer_obs = env.driver_observation(), env.preparer_observation()
    return {'task': env.task, 'total_reward': total, 'steps': env.step_index,
            'caught': info['caught'], 'success': info['success'],
            'totals': info['totals'], 'trace': tuple(env.trace)}


def oracle_rollout(task, **kwargs):
    """Execute the handwritten protocol on one task; assert every prey is caught."""
    kwargs.setdefault('targets', len(task.targets))
    env = TrapPrepHunt(split='all', **kwargs)
    driver, preparer = reference_policies(env.n, env.mechanisms)
    result = run_episode(env, driver, preparer, task=task)
    if not result['success']:
        raise AssertionError(f'reference protocol failed on {task}: {env.trace[-1]["events"]}')
    return result


# -- exact message-blind control ----------------------------------------------

def blind_reference(n=3, split='all', by='pair', horizon=16, start_pos=0,
                    mechanisms=None, patience=None, window=1):
    """Exact optimal catch rate for a single-target preparer that hears nothing.

    Theorem. Every zone holds a trap and the preparer never observes the prey,
    so its observation stream (self_pos, trap_method, trap_charged, clock) is a
    deterministic function of its own past actions. A message-blind preparer is
    therefore equivalent to a fixed action sequence. Give the *driver* full
    knowledge of the task and of that sequence -- an upper bound, since the
    driver can delay but never hasten an arrival -- and the prey arrives exactly
    when wanted, provided that step is at least its start distance and at most
    its patience.

    CORRECTION (this replaces an earlier, wrong version of this bound). The
    first version assumed the sequence commits to one zone and one method, and
    returned 1/(n*mechanisms). That is false whenever the horizon is long enough
    to walk the line: a blind preparer can prepare and fire SEVERAL traps in
    turn, and the driver -- which knows everything -- simply holds the prey
    until the sweep reaches its zone. At n=3 with horizon >= 8 a blind sweeper
    reaches 1/3 of all tasks and, because the by='pair' test split leaves
    exactly one prey type per zone, *100%* of that split. The bound is therefore
    a function of the horizon and of the prey's patience, and this function now
    maximizes over schedules (see blind_upper_bound); at targets=1 the schedule
    relaxation drops nothing, so the value below is exact.

    Practical consequence: the task is only about communication in the regime
    where a sweep does not fit. Give the prey a finite `patience` (n=3 wants
    patience=4, which the reference protocol meets and a two-trap sweep does
    not) or keep the horizon below the sweep cost, and report this number
    alongside any learned catch rate.
    """
    report = blind_upper_bound(n=n, targets=1, split=split, by=by, horizon=horizon,
                               start_pos=start_pos, mechanisms=mechanisms,
                               patience=patience, window=window)
    return {'split': split, 'by': by, 'tasks': report['tasks'],
            'horizon': horizon, 'patience': patience, 'window': window,
            'optimal_blind_success': report['upper_bound_catches_per_task'],
            'exact': True,
            'argmax_schedule_zone_mechanism_step':
                report['argmax_schedule_zone_mechanism_step']}


def blind_upper_bound(n=3, targets=2, split='all', by='pair', horizon=16,
                      start_pos=0, mechanisms=None, task_list=None,
                      patience=None, window=1):
    """Provable upper bound on the message-blind catch rate for any number of
    targets, by maximizing over *activation schedules* rather than sequences.
    Reproduces `blind_reference` exactly when targets == 1.

    Derivation. By the theorem in `blind_reference`, a blind preparer is one
    fixed action sequence. Traps hold one charge, so all that matters about
    zone z is the step of the FIRST activation there and the method standing in
    that trap at that moment. Any sequence therefore induces a *schedule*: an
    ordered list of distinct zones z_1..z_r with methods m_1..m_r fired at steps
    s_1 < ... < s_r <= horizon. Feasibility is bounded below by counting steps:
    by s_i the body must have walked the path start -> z_1 -> ... -> z_i and
    spent one step preparing and one step firing each of the first i traps, so

        s_i >= travel(start, z_1, ..., z_i) + 2i.

    This is a *lower bound on cost* (a real sequence may also need to walk back
    to pre-prepare a trap), hence an upper bound on what is schedulable.

    A prey at z_i is then caught only if m_i is the method its type requires and
    the driver can make it arrive at s_i, which needs s_i >= its start distance.
    Larger s_i is never worse, so the best timings for a given order are
    s_i = horizon - (r - i). What is dropped, and only ever helps the bound: the
    driver's one-DRIVE-per-step budget across simultaneous prey, and the need to
    re-visit zones whose traps were prepared out of order.

    At targets == 1 only r = 1 schedules can score, and the relaxation drops
    nothing, so the bound is attained -- it equals `blind_reference` and hence
    the closed form. At targets > 1 it is an upper bound only; `blind_search`
    gives the exact but exponential value on small configurations, and the two
    agree on the anchor recorded in tests (n=2, targets=2, horizon=3).
    """
    corpus = tuple(task_list) if task_list is not None else tasks(n, targets, split, by)
    mechanisms = n if mechanisms is None else mechanisms
    prey = sum(len(task.targets) for task in corpus)
    demands = Counter((t.zone, mechanism_for(t.prey_type, mechanisms), t.start_distance)
                      for task in corpus for t in task.targets)
    # A prey must ARRIVE by the end of step `patience`, and once it has arrived
    # it stays catchable for `window` steps. The last step at which any
    # activation can catch anything is therefore patience + window - 1, and that
    # deadline caps every schedule. (Widening the firing window buys the
    # learners timing slack, but it also buys a blind sweeper the same slack --
    # which is exactly why it must enter the bound and not be treated as free.)
    if patience is not None:
        horizon = min(horizon, patience + window - 1)
    best, argmax = 0, None
    for length in range(1, min(n, horizon) + 1):
        for order in permutations(range(n), length):
            travel, position, feasible = 0, start_pos, True
            steps = []
            for i, zone in enumerate(order, start=1):
                travel += abs(position - zone)
                position = zone
                step = horizon - (length - i)
                if step < travel + 2 * i:
                    feasible = False
                    break
                steps.append(step)
            if not feasible:
                continue
            for methods in product(range(mechanisms), repeat=length):
                total = sum(count
                            for (zone, mechanism, distance), count in demands.items()
                            for i, z in enumerate(order)
                            if z == zone and methods[i] == mechanism
                            and distance <= steps[i])
                if total > best:
                    best, argmax = total, tuple(zip(order, methods, steps))
    return {'split': split, 'by': by, 'targets': targets, 'tasks': len(corpus),
            'prey': prey, 'window': window, 'patience': patience,
            'upper_bound_catch_rate': best / prey if prey else None,
            'upper_bound_catches_per_task': best / len(corpus) if corpus else None,
            'exact': targets == 1,
            'argmax_schedule_zone_mechanism_step': argmax}


def blind_preparation_bound(n=3, split='all', by='pair', horizon=16,
                            start_pos=0, mechanisms=None, task_list=None):
    """Exact ceiling on the *partial preparation* reward for a message-blind
    preparer, in prey per task -- the companion of `blind_reference`, which
    bounds catches.

    It exists because the partial reward, not the catch, is what a learner
    actually climbs first, so it is the number a learned `first_guess_rate` has
    to beat before any claim is made that the method was communicated.

    Derivation. By the theorem in `blind_reference` a blind preparer is a fixed
    action sequence, and since the payment is one-shot all that matters about
    zone z is the mechanism of the FIRST preparation attempted there. The
    payment ignores the prey's position and status, so the driver is irrelevant
    and timing drops out entirely: a sequence is characterized by an ordered
    subset of zones z_1..z_r with methods m_1..m_r, feasible iff walking
    start -> z_1 -> ... -> z_r plus one step per preparation fits the horizon,

        travel(start, z_1, ..., z_r) + r <= horizon.

    Given a feasible zone set each zone's method is chosen independently, so the
    optimum is the best feasible set of zones scored by its most common required
    mechanism. Note the horizon enters only through how many zones are
    reachable: at n=3 and horizon >= 5 every zone is coverable and the bound is
    just "the commonest mechanism per zone", which on the by='pair' train split
    is exactly 1/2.
    """
    corpus = tuple(task_list) if task_list is not None else tasks(n, 1, split, by)
    mechanisms = n if mechanisms is None else mechanisms
    prey = sum(len(task.targets) for task in corpus)
    demands = Counter((t.zone, mechanism_for(t.prey_type, mechanisms))
                      for task in corpus for t in task.targets)
    per_zone = {z: max((demands[(z, m)] for m in range(mechanisms)), default=0)
                for z in range(n)}
    best, argmax = 0, ()
    for length in range(0, n + 1):
        for order in permutations(range(n), length):
            travel, position = 0, start_pos
            for zone in order:
                travel += abs(position - zone)
                position = zone
            if travel + length > horizon:
                continue
            total = sum(per_zone[z] for z in order)
            if total > best:
                best, argmax = total, order
    return {'split': split, 'by': by, 'tasks': len(corpus), 'prey': prey,
            'horizon': horizon, 'exact': True,
            'upper_bound_first_guess_rate': best / prey if prey else None,
            'argmax_zone_order': argmax}


def blind_search(env_kwargs=None, task_list=None):
    """Brute-force the same bound through the real simulator: maximize over all
    fixed preparer action sequences the mean, over tasks, of the best driver
    reply. Exponential in the horizon -- for cross-checking small configurations
    against `blind_reference`, not for the default config."""
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.setdefault('split', 'all')
    env = TrapPrepHunt(**env_kwargs)
    corpus = task_list if task_list is not None else env._tasks
    horizon, n_actions = env.horizon, PREPARE_BASE + env.mechanisms
    n_driver = DRIVE_BASE + env.targets
    best, argmax = (-1, -1.0), None
    for prep_seq in product(range(n_actions), repeat=horizon):
        total_caught, total_reward = 0, 0.0
        for task in corpus:
            task_best = (0, 0.0)
            for drive_seq in product(range(n_driver), repeat=horizon):
                env.reset(task)
                reward, done, step = 0.0, False, 0
                while not done:
                    env.driver_send(())
                    env.preparer_send(())
                    env.driver_act(drive_seq[step])
                    r, done, info = env.preparer_act(prep_seq[step])
                    reward += r
                    step += 1
                # Rank by catches first: the partial preparation reward is not
                # comparable with blind_reference's catch rate.
                task_best = max(task_best, (info['caught'], reward))
            total_caught += task_best[0]
            total_reward += task_best[1]
        if (total_caught, total_reward) > best:
            best, argmax = (total_caught, total_reward), prep_seq
    return {'tasks': len(corpus),
            'optimal_blind_catch_rate': best[0] / len(corpus),
            'reward_at_argmax': best[1] / len(corpus),
            'argmax_preparer_sequence': argmax}


if __name__ == '__main__':
    import json
    # The blind bound is a function of the horizon and the prey's patience,
    # because a blind preparer can sweep the line and fire trap after trap.
    print(json.dumps({'single_target_blind_bound_by_deadline': [
        {'patience': p,
         **{s: blind_reference(split=s, by='pair', horizon=8, patience=p)['optimal_blind_success']
            for s in ('all', 'train', 'test')}}
        for p in (4, 6, None)]}, indent=2))
    print(json.dumps({'two_target_upper_bound': [
        {'split': s, **{k: v for k, v in blind_upper_bound(targets=2, split=s, by='pair',
                                                           horizon=12, patience=6).items()
                        if k in ('upper_bound_catch_rate', 'exact')}}
        for s in ('all', 'train', 'test')]}, indent=2))
    corpus = tasks(split='all')
    caught = sum(oracle_rollout(t)['caught'] for t in corpus)
    print(json.dumps({'tasks': len(corpus), 'oracle_prey_caught': caught,
                      'oracle_prey_total': sum(len(t.targets) for t in corpus)}))
