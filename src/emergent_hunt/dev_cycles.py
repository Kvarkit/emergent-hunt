"""Resumable development-cycle sweep for emergent communication.

Each cycle trains one configuration, evaluates intact/muted/shifted channels,
and writes the result immediately.  Re-running the command resumes from the
next cycle in the JSON checkpoint, so long 50-cycle audits survive shell time
limits.
"""
import argparse
import json
from pathlib import Path

from .train_multiround import run


def cycle_config(index: int) -> tuple[int, bool, int, str, int]:
    configs = [(r, soft, h, task)
               for r in (2, 3) for soft in (False, True)
               for h in (16, 32) for task in ('one_way', 'symmetric')]
    r, soft, h, task = configs[index % len(configs)]
    return r, soft, h, task, index // len(configs)


def run_cycles(total=50, episodes=100, output='results/dev-cycles.json'):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    start = len(rows)
    for index in range(start, total):
        rounds, soft, hidden, task, seed = cycle_config(index)
        result = run(seed=seed, episodes=episodes, rounds=rounds, vocab=8,
                      zones=4, communication_task=task,
                      soft_curriculum=soft, hidden_dim=hidden)
        intact = result['fixed_grid_eval']['terminal_success']
        muted = result['fixed_grid_nomessage_eval']['terminal_success']
        shifted = result['fixed_grid_shift_eval']['terminal_success']
        rows.append({'cycle': index + 1, 'rounds': rounds, 'soft_curriculum': soft,
                     'hidden_dim': hidden, 'task': task, 'seed': seed,
                     'reward_mean_100': result['history'][-1]['reward_mean_100'],
                     'intact': intact, 'muted': muted, 'shifted': shifted,
                     'causal_gap': intact - muted})
        path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', default='results/dev-cycles.json')
    args = parser.parse_args()
    run_cycles(args.total, args.episodes, args.output)
