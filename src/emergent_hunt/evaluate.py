"""Exact reference controls on a uniform state distribution; no training."""
from collections import Counter
import json
from .environment import states


def controls(n=3, split='all'):
    corpus = states(n, split)
    # Receiver without messages observes only trap. Bayes-optimal action per trap.
    counts = Counter((s.trap, s.prey, s.direction) for s in corpus)
    correct = sum(max(v for (t, _, _), v in counts.items() if t == trap)
                  for trap in range(n))
    # Handwritten two-token sender sends prey,direction; receiver appends trap.
    oracle = sum((s.prey, s.direction, s.trap) ==
                 (*tuple((s.prey, s.direction)), s.trap) for s in corpus)
    return {'split': split, 'states': len(corpus),
            'optimal_no_message_success': correct / len(corpus),
            'handwritten_two_token_success': oracle / len(corpus)}


if __name__ == '__main__':
    print(json.dumps([controls(split=s) for s in ('all', 'train', 'test')], indent=2))
