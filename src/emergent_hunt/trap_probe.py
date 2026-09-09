"""Token-perturbation selectivity probe for TrapPrepHunt (protocol EH-TRAP-INT-r0.1).

The existing intervention layer (intervention.py, EH-INT-r0.1) perturbs a *state
factor* and asks which token positions move. This layer runs the other
direction, which is what the trap-preparation spec asks for: perturb one token
position at a time and ask which part of the *decoded action* moves -- target
(which trap was acted on), method (which preparation), timing (when the trap was
activated). A compositional protocol should give a near-diagonal
position-by-component matrix; a holistic code should leak across components.

Nothing here assigns meanings to tokens or positions in advance. Substituted
values are drawn from the values that actually occur at that position in the
corpus, and the position->component association is *read off* the matrix
(`best_component_per_position`) rather than assumed. The two reference policies
in this module and in trap_prep are handwritten controls with known ground
truth; they exist to show that the probe can tell a factorized code from a
holistic one, not as evidence about any learned language.

Limits, stated up front: this measures the *causal localization* of a token's
effect on behaviour. Localization is necessary for a compositional reading but
not sufficient -- a positionally-organized lookup table produces the same
matrix. The held-out-pair split in trap_prep.tasks (by='pair'), not this
matrix, is what tests novel combinations of familiar conditions.
"""
from collections import Counter
import json

from .trap_prep import (ACTIVATE, PREPARE_BASE, ReferenceDriver,
                        ReferencePreparer, TrapPrepHunt, reference_policies,
                        run_episode)

COMPONENTS = ('target', 'method', 'timing')


# -- a second, deliberately non-factorized reference code ---------------------

class HolisticDriver(ReferenceDriver):
    """Same behaviour, entangled code: slots 0 and 1 jointly encode (zone,
    method) through a fixed permutation, so neither slot is about one factor.
    The cue keeps its own slot -- the contrast under test is target vs method."""

    def __init__(self, n=3, mechanisms=3, permutation=None):
        super().__init__(mechanisms)
        self.n = n
        size = n * mechanisms
        self.permutation = tuple(permutation) if permutation is not None else \
            tuple((5 * i + 3) % size for i in range(size))
        if sorted(self.permutation) != list(range(size)):
            raise ValueError('permutation must be a bijection on n*mechanisms')

    def message(self, observation):
        zone, method, cue = super().message(observation)
        code = self.permutation[zone * self.mechanisms + method]
        return (code // self.mechanisms, code % self.mechanisms, cue)


class HolisticPreparer(ReferencePreparer):
    def __init__(self, n=3, mechanisms=3, permutation=None):
        super().__init__(n, mechanisms)
        driver = HolisticDriver(n, mechanisms, permutation)
        self.permutation = driver.permutation
        self.dictionary = {(c // mechanisms, c % mechanisms):
                           divmod(i, mechanisms)
                           for i, c in enumerate(self.permutation)}

    def _decode(self, incoming):
        if len(incoming) < 3 or any(t < 0 for t in incoming):
            return None, None, False
        key = (incoming[0] % self.mechanisms, incoming[1] % self.mechanisms)
        zone, method = self.dictionary[key]
        return zone % self.n, method % self.mechanisms, bool(incoming[2] % 2)


def holistic_policies(n=3, mechanisms=3, permutation=None):
    return (HolisticDriver(n, mechanisms, permutation),
            HolisticPreparer(n, mechanisms, permutation))


POLICIES = {'compositional': reference_policies, 'holistic': holistic_policies}


# -- behavioural decoding -----------------------------------------------------

def decision(info):
    """Decompose one preparer decision into the three parts an instruction can
    bind. `target` is the zone the action lands on (for a move, where it goes;
    for prepare/activate, the trap it touches), `method` which preparation it
    applies, `timing` whether it fires this step.

    These are read off the executed action, not off the policy's internals, so
    the same decomposition applies to a learned preparer.
    """
    action = info['preparer_action']
    return {'target': info['preparer_pos'],
            'method': action - PREPARE_BASE if action >= PREPARE_BASE else None,
            'timing': action == ACTIVATE}


def signature(result):
    """Episode-level summary of the same three parts. Useful for reporting the
    persistent-perturbation controls; NOT used for the selectivity matrix,
    because episode length and ordering are themselves perturbation-dependent
    and would confound the components."""
    target, method, timing, window = [], [], [], []
    for info in result['trace']:
        action = info['preparer_action']
        if action >= PREPARE_BASE:
            target.append(info['preparer_pos'])
            method.append(action - PREPARE_BASE)
        elif action == ACTIVATE:
            target.append(info['preparer_pos'])
            timing.append(info['step'])
            window.append(bool(info['activation'] and info['activation']['on_window']))
    return {'target': tuple(target), 'method': tuple(method),
            'timing': tuple(timing), 'timing_window': tuple(window)}


# -- perturbations ------------------------------------------------------------

def substitute(position, value, at_step=None):
    """Replace one token position with `value`, either at a single step (the
    single-decision counterfactual used by `selectivity`) or at every step."""
    def perturb(step, message):
        if position >= len(message) or (at_step is not None and step != at_step):
            return message
        return message[:position] + (value,) + message[position + 1:]
    return perturb


def mute(step, message):
    return ()


def reverse(step, message):
    return tuple(reversed(message))


def roll(shift):
    def perturb(step, message):
        if not message:
            return message
        k = shift % len(message)
        return message[-k:] + message[:-k] if k else message
    return perturb


# -- the probe ----------------------------------------------------------------

def _episode(policies_factory, env, task, perturb=None):
    driver, preparer = policies_factory(env.n, env.mechanisms)
    return run_episode(env, driver, preparer, task=task, perturb=perturb)


def observed_values(policies_factory, env, corpus):
    """Which token values actually occur at each position, over the corpus."""
    width = env.driver_message_length
    seen = [Counter() for _ in range(width)]
    for task in corpus:
        result = _episode(policies_factory, env, task)
        for info in result['trace']:
            for i, token in enumerate(info['driver_sent'][:width]):
                seen[i][token] += 1
    return [sorted(c) for c in seen]


def selectivity(policy='compositional', corpus=None, n=3, targets=2,
                by='pair', split='all', env_kwargs=None, max_tasks=None,
                min_effect=.25, tolerance=.05):
    """Position-by-component causal matrix plus persistent-perturbation controls.

    Method. For each task, each step `s` of its intact episode, each token
    position `i` and each alternative value `v` seen at that position, replay
    the episode with the token at position `i` set to `v` *at step s only*. The
    two runs share an identical history up to `s`, so the decision at `s` is a
    clean single-token counterfactual, and comparing `decision(...)` says which
    of target/method/timing moved. Perturbing every step instead would let
    episode length and ordering drift and confound the components.

    `changed[c]` is the fraction of counterfactuals in which component `c`
    moved; `exclusive[c]` the fraction in which `c` moved and nothing else. The
    position->component binding is read off `exclusive`, since the environment
    itself couples the components in one direction: a trap can only be prepared
    or fired where the preparer stands, so retargeting can cancel a firing,
    while re-methoding cannot retarget. That asymmetry is a property of the
    task, not of the code under test, and it is why `changed` alone is not a
    fair diagonal test.
    """
    factory = POLICIES[policy] if isinstance(policy, str) else policy
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.update(n=n, targets=targets, by=by, split=split)
    env = TrapPrepHunt(**env_kwargs)
    corpus = list(corpus if corpus is not None else env._tasks)
    if max_tasks is not None:
        corpus = corpus[:max_tasks]
    width = env.driver_message_length
    values = observed_values(factory, env, corpus)
    intact = {task: _episode(factory, env, task) for task in corpus}

    positions = {}
    for position in range(width):
        counts = {c: 0 for c in COMPONENTS}
        exclusive = {c: 0 for c in COMPONENTS}
        runs = unchanged = 0
        for task in corpus:
            base = intact[task]
            for step_info in base['trace']:
                step = step_info['step']
                sent = step_info['driver_sent']
                if position >= len(sent):
                    continue
                for value in values[position]:
                    if value == sent[position]:
                        continue
                    after = _episode(factory, env, task,
                                     perturb=substitute(position, value, at_step=step))
                    before_decision = decision(step_info)
                    after_decision = decision(after['trace'][step - 1])
                    differing = [c for c in COMPONENTS
                                 if before_decision[c] != after_decision[c]]
                    for c in differing:
                        counts[c] += 1
                    if len(differing) == 1:
                        exclusive[differing[0]] += 1
                    unchanged += not differing
                    runs += 1
        positions[str(position)] = {
            'counterfactuals': runs,
            'substituted_values': values[position],
            'changed': {c: counts[c] / runs if runs else 0.0 for c in COMPONENTS},
            'exclusive': {c: exclusive[c] / runs if runs else 0.0 for c in COMPONENTS},
            'no_effect': unchanged / runs if runs else 0.0}

    def persistent(perturb):
        """Whole-episode control: the same message defect at every step."""
        runs = success = 0
        reward = 0.0
        changed = {c: 0 for c in COMPONENTS + ('timing_window',)}
        for task in corpus:
            base = signature(intact[task])
            result = _episode(factory, env, task, perturb=perturb)
            after = signature(result)
            for c in changed:
                changed[c] += base[c] != after[c]
            runs += 1
            success += int(result['success'])
            reward += result['total_reward']
        return {'runs': runs, 'success_rate': success / runs,
                'mean_reward': reward / runs,
                'changed': {c: v / runs for c, v in changed.items()}}

    intact_success = sum(r['success'] for r in intact.values()) / len(corpus)
    intact_reward = sum(r['total_reward'] for r in intact.values()) / len(corpus)
    # Read the binding off the matrix; do not assume slot meanings in advance.
    best = {p: max(COMPONENTS, key=lambda c: positions[p]['exclusive'][c])
            for p in positions}
    # Leakage, not raw diagonal mass, is the discriminating statistic: the
    # task couples target->timing physically, but nothing couples a token to a
    # component it does not encode, so a factorized code should put *zero* mass
    # on off-diagonal exclusive effects while an entangled one cannot.
    leak = max(max(positions[p]['exclusive'][c] for c in COMPONENTS if c != best[p])
               for p in positions)
    localized = (len(set(best.values())) == len(best)
                 # A position whose token never varies in the corpus has not been
                 # tested at all and must not count as evidence of a binding.
                 and all(positions[p]['counterfactuals'] > 0 for p in positions)
                 and all(positions[p]['exclusive'][best[p]] >= min_effect for p in positions)
                 and leak <= tolerance)
    return {'protocol': 'EH-TRAP-INT-r0.1',
            'policy': policy if isinstance(policy, str) else 'custom',
            'n': n, 'targets': targets, 'by': by, 'split': split,
            'tasks': len(corpus), 'width': width,
            'intact': {'success_rate': intact_success, 'mean_reward': intact_reward},
            'positions': positions,
            'controls': {'mute': persistent(mute), 'reverse': persistent(reverse),
                         'roll_1': persistent(roll(1))},
            'best_component_per_position': best,
            'exclusive_leakage': leak,
            'thresholds': {'min_effect': min_effect, 'tolerance': tolerance},
            'candidate_localized_binding': localized,
            'interpretation': 'causal localization only; not evidence of grammar '
                              '-- compare against the by=pair held-out split'}


def leakage(report):
    """Off-diagonal mass: how much a position's perturbation moves components
    other than the one it is best associated with. 0 for a perfectly
    factorized code, large for an entangled one."""
    out = {}
    for position, best in report['best_component_per_position'].items():
        exclusive = report['positions'][position]['exclusive']
        others = [c for c in COMPONENTS if c != best]
        out[position] = {'best': best, 'best_exclusive': exclusive[best],
                         'other_exclusive': {c: exclusive[c] for c in others},
                         'max_other': max(exclusive[c] for c in others),
                         'entangled': report['positions'][position]['changed']}
    return out


if __name__ == '__main__':
    for name in POLICIES:
        report = selectivity(policy=name, max_tasks=40)
        print(json.dumps({'policy': name,
                          'intact': report['intact'],
                          'positions': report['positions'],
                          'controls': report['controls'],
                          'best_component_per_position': report['best_component_per_position'],
                          'exclusive_leakage': report['exclusive_leakage'],
                          'candidate_localized_binding': report['candidate_localized_binding'],
                          'leakage': leakage(report)}, indent=2))
