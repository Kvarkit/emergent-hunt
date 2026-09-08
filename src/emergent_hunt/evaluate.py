"""Exact reference controls on a uniform state distribution; no training."""
from collections import Counter
import json
from .environment import states


def controls(n=3, split='all', by='triple'):
    corpus = states(n, split, by)
    # Receiver without messages observes only trap. Bayes-optimal action per trap.
    counts = Counter((s.trap, s.prey, s.direction) for s in corpus)
    correct = sum(max(v for (t, _, _), v in counts.items() if t == trap)
                  for trap in range(n))
    # Handwritten two-token sender sends prey,direction; receiver appends trap.
    oracle = sum((s.prey, s.direction, s.trap) ==
                 (*tuple((s.prey, s.direction)), s.trap) for s in corpus)
    return {'split': split, 'by': by, 'states': len(corpus),
            'optimal_no_message_success': correct / len(corpus),
            'handwritten_two_token_success': oracle / len(corpus)}


def pair_lookup_baseline(n=3, by='triple'):
    """No-learning shortcut: memorize (prey, direction) -> arbitrary code from
    train, decode back to (prey, direction) at test time, and let the receiver
    copy its directly-observed trap (see environment.py; the receiver already
    sees trap, never has to guess it). This needs no compositional rule, only
    a per-pair lookup table built once from train.

    Per #24928 (melioralab-agent) and #24933 (nadir-codex): under by='triple'
    every (prey, direction) pair recurs in train, so this shortcut reaches
    100% on 'test' with zero generalization. Under by='pair' every held-out
    pair is absent from train, so the table has no entry for it and the
    shortcut necessarily fails there -- the falsifier that closes the gap.
    """
    train_pairs = {(s.prey, s.direction) for s in states(n, 'train', by)}
    test = states(n, 'test', by)
    covered = sum(1 for s in test if (s.prey, s.direction) in train_pairs)
    return {'by': by, 'test_states': len(test),
            'test_pairs_seen_in_train': covered,
            'lookup_would_solve_fraction': covered / len(test) if test else None}


if __name__ == '__main__':
    print(json.dumps([controls(split=s, by=b)
                       for b in ('triple', 'pair') for s in ('all', 'train', 'test')], indent=2))
    print(json.dumps([pair_lookup_baseline(by=b) for b in ('triple', 'pair')], indent=2))
