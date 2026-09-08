"""Intervention-table diagnostic corpus (protocol EH-INT-r0.1, melioralab-agent #25120).

Generates the 162 directed single-factor swaps over the full 27-state corpus
(n=3) and runs three fixed, non-learned control policies over it, exporting
one row per (state, direction, policy) with the schema melioralab specified.
This layer measures observability/localization of an intervention's effect on
sent tokens -- it does not test or claim compositionality (see #25001, #25120).
"""
from dataclasses import asdict
from itertools import product

from .environment import State, states


def held_out_split(n, p, d):
    """train/test label for a (p, d) pair under by='pair' (see environment.held_out_pairs)."""
    return 'test' if (p + d) % n == 0 else 'train'


def counterfactual_pairs(n=3):
    """162 directed (base, counterfactual, factor) triples: for each of the 27
    states and each factor in ('p', 'd', 't'), the two alternate values of
    that factor with everything else held fixed. 27 states * 3 factors * 2
    alternates = 162, 54 per factor.
    """
    out = []
    all_states = [State(*x) for x in product(range(n), repeat=3)]
    assert len(all_states) == n ** 3
    for s in all_states:
        for factor, base_val in (('p', s.prey), ('d', s.direction), ('t', s.trap)):
            for alt in range(n):
                if alt == base_val:
                    continue
                if factor == 'p':
                    cf = State(alt, s.direction, s.trap)
                elif factor == 'd':
                    cf = State(s.prey, alt, s.trap)
                else:
                    cf = State(s.prey, s.direction, alt)
                out.append((s, cf, factor))
    assert len(out) == 162
    return out


# --- three fixed, non-learned control policies -----------------------------
# Each policy is (sender(p, d) -> message, receiver(t, received) -> action).
# All are deterministic and untrained; none is claimed to be an agent.

def handwritten_policy():
    def sender(p, d):
        return (p, d)

    def receiver(t, received):
        return (*received, t)

    return 'handwritten', sender, receiver


def holistic_policy(n=3):
    """#24928's fixed antipode: a permutation-coded 'language' that is not
    factor-aligned, decoded via a dictionary built from every (p, d) pair
    that occurs in the original by='triple' train split -- which, per
    #24928/#24933, is all n*n pairs, so the dictionary is total on this n.
    """
    perm = [8, 3, 6, 1, 7, 0, 5, 2, 4]
    assert len(perm) == n * n

    def code(p, d):
        return divmod(perm[n * p + d], n)

    train_pairs = sorted({(s.prey, s.direction) for s in states(n, 'train', 'triple')})
    assert len(train_pairs) == n * n, 'holistic_policy assumes every pair recurs in triple-train'
    dictionary = {code(p, d): (p, d) for p, d in train_pairs}

    def sender(p, d):
        return code(p, d)

    def receiver(t, received):
        p, d = dictionary[received]
        return (p, d, t)

    return 'holistic', sender, receiver


def no_message_policy():
    def sender(p, d):
        return ()

    def receiver(t, received):
        return (0, 0, t)

    return 'no_message', sender, receiver


# --- corpus export -----------------------------------------------------------

def _run(policy_name, sender, receiver, state, n):
    sent = sender(state.prey, state.direction)
    received = sent  # deterministic layer: erasure=0, no channel noise
    action = receiver(state.trap, received)
    correct = action == (state.prey, state.direction, state.trap)
    reward = float(correct)  # token_cost=0 at this diagnostic layer
    return sent, received, action, correct, reward


def build_rows(n=3, seed=0):
    policies = [handwritten_policy(), holistic_policy(n), no_message_policy()]
    pairs = counterfactual_pairs(n)
    rows = []
    for policy_name, sender, receiver in policies:
        for base, cf, factor in pairs:
            sent_before, received_before, action_before, correct_before, reward_before = \
                _run(policy_name, sender, receiver, base, n)
            sent_after, received_after, action_after, correct_after, reward_after = \
                _run(policy_name, sender, receiver, cf, n)
            changed = [i for i in range(max(len(sent_before), len(sent_after)))
                       if (sent_before[i] if i < len(sent_before) else None) !=
                          (sent_after[i] if i < len(sent_after) else None)]
            rows.append({
                'protocol': 'EH-INT-r0.1',
                'checkpoint': None, 'config': f'n={n},vocabulary=8,max_length=2', 'seed': seed,
                'policy': policy_name,
                'factor': factor,
                'base': asdict(base), 'counterfactual': asdict(cf),
                'base_split': held_out_split(n, base.prey, base.direction),
                'counterfactual_split': held_out_split(n, cf.prey, cf.direction),
                'sender_obs_before': (base.prey, base.direction),
                'sender_obs_after': (cf.prey, cf.direction),
                'sent_before': sent_before, 'sent_after': sent_after,
                'received_before': received_before, 'received_after': received_after,
                'action_before': action_before, 'action_after': action_after,
                'correct_before': correct_before, 'correct_after': correct_after,
                'reward_before': reward_before, 'reward_after': reward_after,
                'changed_token_positions': changed,
            })
    return rows


def control_summary(rows):
    """Per-policy: overall correct/total and per-factor changed-vs-unchanged
    sent-message counts, matching the manual tallies in #25120 (27/27 correct
    for handwritten and holistic, 3/27 for no_message; p 54/54, d 54/54,
    t 0/54 changed for handwritten and holistic, 0/54 all three for
    no_message).
    """
    out = {}
    for row in rows:
        pol = row['policy']
        s = out.setdefault(pol, {'correct_before': 0, 'correct_after': 0, 'n_before': 0,
                                  'n_after': 0, 'changed_by_factor': {'p': 0, 'd': 0, 't': 0},
                                  'total_by_factor': {'p': 0, 'd': 0, 't': 0}})
        s['n_before'] += 1
        s['correct_before'] += int(row['correct_before'])
        s['n_after'] += 1
        s['correct_after'] += int(row['correct_after'])
        s['total_by_factor'][row['factor']] += 1
        if row['sent_before'] != row['sent_after']:
            s['changed_by_factor'][row['factor']] += 1
    return out


if __name__ == '__main__':
    import json
    rows = build_rows()
    print(json.dumps(control_summary(rows), indent=2))
    print(f'{len(rows)} rows ({len(rows) // 3} per policy)')
