"""Where does a learned TrapPrepHunt pair actually fail, and is it talking?

Written because a 60k-episode sweep reported `correct_prep_rate` 0.5 with
`catch_rate` 0.0 and the aggregate numbers could not say which of the four
things that must go right had gone wrong -- nor that the communication arm was
scoring exactly what its own no-message control scored. Both readouts here are
cheap, and either one would have caught that immediately.

Two readouts, both on the greedy (argmax) policy over a whole split:

`stages` walks the causal chain a catch requires, in order, and reports how many
prey reach each rung plus a histogram of the FIRST rung that failed:

    reached      the preparer stood on the prey's zone at some step
    first_guess  the one preparation that could pay used the right mechanism
    in_position  the prey arrived on a charged trap already armed correctly
    fired        the preparer activated the trap at the prey's zone
    caught       it fired while the prey was in the firing window

`messages` asks whether the driver's wire carries anything at all: the empirical
mutual information, in bits, between the driver's first message and each of the
facts only the driver can see (the prey's zone, the mechanism its type requires,
and the pair of the two), plus the number of distinct messages emitted over the
corpus. A pair that has converged on a message-blind policy emits one constant
message and scores 0 bits on every column, which is the signature to look for
before reading anything into a learned score.

Neither readout is evidence of grammar. Mutual information says the channel is
used, not that it is compositional; trap_probe.selectivity is the token-level
test and the by='pair' split is the generalization test.
"""
import argparse
import json
from collections import Counter
from math import log2

import torch

from .trap_prep import (ACTIVATE, PREPARE_BASE, TrapPrepHunt, mechanism_for,
                        tasks)
from .train_trap import (TrapDriver, TrapEncoder, TrapPreparer, evaluate,
                         rollout)

STAGES = ('reached', 'first_guess', 'in_position', 'fired', 'caught')


def load(path, **overrides):
    """Rebuild the environment and both agents from a train_trap checkpoint.

    Checkpoints written before `patience`/`approach`/`start_pos` were recorded
    carry only the older keys, so those fall back to the trainer's defaults and
    can be overridden by the caller.
    """
    blob = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(blob['config'])
    settings = {'n': 3, 'targets': 1, 'by': 'pair', 'horizon': 8, 'window': 1,
                'patience': 4, 'approach': 0.0, 'start_pos': 0, 'seed': 0,
                'mode': 'communication'}
    settings.update({k: v for k, v in config.items() if k in settings})
    settings.update(overrides)
    mode = settings.pop('mode')
    env = TrapPrepHunt(split='train', stop_when_resolved=False, **settings)
    encoder = TrapEncoder(env)
    driver, preparer = TrapDriver(encoder), TrapPreparer(encoder)
    driver.load_state_dict(blob['driver'])
    preparer.load_state_dict(blob['preparer'])
    driver.eval()
    preparer.eval()
    return {'env': env, 'encoder': encoder, 'driver': driver,
            'preparer': preparer, 'mode': mode, 'settings': settings}


def _mutual_information(pairs):
    """I(X;Y) in bits from an iterable of (x, y) observations."""
    joint = Counter(pairs)
    total = sum(joint.values())
    if not total:
        return 0.0
    left = Counter()
    right = Counter()
    for (x, y), count in joint.items():
        left[x] += count
        right[y] += count
    return max(0.0, sum(count / total * log2(count * total / (left[x] * right[y]))
                        for (x, y), count in joint.items()))


@torch.no_grad()
def stages(driver, preparer, encoder, env, corpus, mode='communication'):
    """Per-prey stage counts and the first rung that failed. targets=1 only:
    with several prey the rungs interleave and the histogram is not a partition.
    """
    if env.targets != 1:
        raise ValueError('the stage breakdown assumes a single target')
    reached = Counter()
    first_failure = Counter()
    examples = {}
    for task in corpus:
        target = task.targets[0]
        out = rollout(env, driver, preparer, encoder, task=task, mode=mode,
                      greedy=True)
        events = [event for step in out['trace'] for event in step['events']]
        got = {
            'reached': any(step['preparer_pos'] == target.zone
                           for step in out['trace']),
            'first_guess': out['info']['totals']['prepare'] > 0,
            'in_position': any(name == 'in_position' for name, _ in events),
            'fired': any(step['preparer_action'] == ACTIVATE
                         and step['preparer_pos'] == target.zone
                         for step in out['trace']),
            'caught': out['info']['caught'] > 0}
        stage = next((s for s in STAGES if not got[s]), None)
        first_failure[stage or 'none'] += 1
        for s in STAGES:
            reached[s] += got[s]
        if stage and stage not in examples:
            examples[stage] = {
                'task': {'zone': target.zone, 'prey_type': target.prey_type,
                         'required_mechanism': mechanism_for(target.prey_type,
                                                             env.mechanisms),
                         'start_distance': target.start_distance},
                'driver_messages': [list(step['driver_sent']) for step in out['trace']],
                'readiness_replies': [list(step['preparer_sent']) for step in out['trace']],
                'driver_actions': [step['driver_action'] for step in out['trace']],
                'preparer_actions': [step['preparer_action'] for step in out['trace']],
                'preparer_positions': [step['preparer_pos'] for step in out['trace']],
                'events': [list(event) for event in events]}
    prey = sum(len(task.targets) for task in corpus)
    return {'prey': prey,
            'reached_rate': {s: reached[s] / prey for s in STAGES},
            'first_failed_stage': dict(first_failure),
            'example_failure': examples}


@torch.no_grad()
def messages(driver, preparer, encoder, env, corpus, mode='communication'):
    """Is the driver's wire carrying the two facts only the driver can see?"""
    observed = []
    for task in corpus:
        target = task.targets[0]
        out = rollout(env, driver, preparer, encoder, task=task, mode=mode,
                      greedy=True)
        observed.append((tuple(out['trace'][0]['driver_sent']), target.zone,
                         mechanism_for(target.prey_type, env.mechanisms)))
    return {'distinct_first_messages': len({m for m, _, _ in observed}),
            'first_messages': sorted({m for m, _, _ in observed}),
            'bits': {
                'zone': _mutual_information((m, z) for m, z, _ in observed),
                'mechanism': _mutual_information((m, k) for m, _, k in observed),
                'zone_and_mechanism': _mutual_information(
                    (m, (z, k)) for m, z, k in observed)},
            'max_bits': {
                'zone': _mutual_information((z, z) for _, z, _ in observed),
                'mechanism': _mutual_information((k, k) for _, _, k in observed),
                'zone_and_mechanism': _mutual_information(
                    ((z, k), (z, k)) for _, z, k in observed)}}


def summarize_history(result, split='train', key='catch_rate', bound=None):
    """Final vs best vs time-above-bound for one train_trap result.

    The trainer reports the last evaluation and saves the last weights, which is
    only the right summary if training converges. It does not: on the 200k-
    episode run, seed 0 held a catch rate of 0.278 -- above the 0.167 blind
    bound -- for forty thousand episodes and then fell back to 0.000, so the
    final number alone says the opposite of what happened. `above_bound_evals`
    is the honest middle ground: it says how long the excursion lasted, and it
    is only worth reading next to the same figure for the no_message control,
    since a peak picked out of a hundred evaluations is a selected maximum.
    """
    history = result['history']
    if bound is None:
        bound = (result.get('blind_bound') or {}).get(split) if key == 'catch_rate' \
            else (result.get('blind_preparation_bound') or {}).get(split)
    values = [row[split]['intact'][key] for row in history if key in row[split]['intact']]
    above = [v for v in values if bound is not None and v > bound + 1e-9]
    return {'mode': result['mode'], 'seed': result['seed'], 'split': split,
            'key': key, 'bound': bound, 'evaluations': len(values),
            'final': values[-1] if values else None,
            'best': max(values) if values else None,
            'above_bound_evals': len(above)}


def diagnose(path, splits=('train', 'test'), **overrides):
    loaded = load(path, **overrides)
    env, encoder = loaded['env'], loaded['encoder']
    driver, preparer, mode = loaded['driver'], loaded['preparer'], loaded['mode']
    report = {'checkpoint': str(path), 'mode': mode, 'settings': loaded['settings']}
    for split in splits:
        corpus = tasks(env.n, env.targets, split, loaded['settings']['by'])
        report[split] = {
            'metrics': evaluate(driver, preparer, encoder, env, corpus, mode),
            'stages': stages(driver, preparer, encoder, env, corpus, mode),
            'messages': messages(driver, preparer, encoder, env, corpus, mode)}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',
                        help='a .pt checkpoint, or a train_trap results .json '
                             'with --history')
    parser.add_argument('--history', action='store_true',
                        help='summarize a results .json (final / best / how '
                             'many evaluations cleared the matching bound) '
                             'instead of replaying a checkpoint')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--start-pos', dest='start_pos', type=int)
    parser.add_argument('--approach', type=float)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.history:
        from pathlib import Path
        results = json.loads(Path(args.checkpoint).read_text(encoding='utf-8'))
        report = [summarize_history(result, split, key)
                  for result in results
                  for split in ('train', 'test')
                  for key in ('catch_rate', 'first_guess_rate')]
        print(json.dumps(report, indent=2))
        return
    overrides = {k: v for k, v in (('patience', args.patience),
                                   ('start_pos', args.start_pos),
                                   ('approach', args.approach)) if v is not None}
    report = diagnose(args.checkpoint, **overrides)
    text = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
