"""Post-hoc probes for emergent message structure.

The probe never supplies a target vocabulary to the agents.  It consumes
recorded trajectories with ``tokens`` and named evaluation factors and reports
associations, positional stability, and compositional transfer indicators.
"""
from collections import Counter, defaultdict
from math import log2


def _entropy(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * log2(n / total) for n in counts.values() if n)


def mutual_information(xs, ys):
    if len(xs) != len(ys) or not xs:
        raise ValueError("x and y must be equally sized and non-empty")
    joint = Counter(zip(xs, ys)); cx = Counter(xs); cy = Counter(ys); n = len(xs)
    return sum((v / n) * log2((v * n) / (cx[x] * cy[y]))
               for (x, y), v in joint.items())


def normalized_mi(xs, ys):
    """Symmetric MI normalized by the smaller marginal entropy."""
    mi = mutual_information(xs, ys)
    denom = min(_entropy(Counter(xs)), _entropy(Counter(ys)))
    return 0.0 if denom == 0 else mi / denom


def _majority_accuracy(xs, ys):
    groups = defaultdict(Counter)
    for x, y in zip(xs, ys):
        groups[x][y] += 1
    return sum(max(c.values()) for c in groups.values()) / max(1, len(ys))


def probe(records, vocab):
    """Return interpretable grammar indicators from recorded messages.

    Each record must contain ``tokens`` (a sequence of integer tokens) and
    ``factors`` (a mapping such as ``{"type": 1, "direction": 0}``).
    ``tokens`` may be a single token or a multi-token sequence.
    """
    if not records:
        raise ValueError("records must be non-empty")
    width = len(records[0]["tokens"]) if isinstance(records[0]["tokens"], (list, tuple)) else 1
    sequences = [tuple(r["tokens"]) if isinstance(r["tokens"], (list, tuple)) else (r["tokens"],)
                 for r in records]
    factors = sorted(records[0]["factors"])
    out = {"n": len(records), "width": width, "factors": {}}
    for factor in factors:
        values = [r["factors"][factor] for r in records]
        positions = {}
        for pos in range(width):
            toks = [seq[pos] for seq in sequences]
            positions[str(pos)] = {
                "normalized_mi": normalized_mi(toks, values),
                "majority_accuracy": _majority_accuracy(toks, values),
            }
        out["factors"][factor] = {"positions": positions,
                                  "sequence_normalized_mi": normalized_mi(sequences, values)}

    transitions = Counter((seq[i], seq[i + 1]) for seq in sequences for i in range(width - 1))
    out["token_transitions"] = {
        "unique": len(transitions),
        "total": sum(transitions.values()),
        "top": [[list(pair), n] for pair, n in transitions.most_common(10)],
    }
    out["transition_factor_normalized_mi"] = {}
    for factor in factors:
        values = [r["factors"][factor] for r in records]
        for pos in range(width - 1):
            pairs = [(seq[pos], seq[pos + 1]) for seq in sequences]
            out["transition_factor_normalized_mi"][f"{pos}->{pos + 1}:{factor}"] = normalized_mi(pairs, values)
    # A crude but useful compositionality signal: unseen factor combinations
    # should not force unseen token sequences when individual factors repeat.
    combos = {(tuple(r["factors"][f] for f in factors), seq) for r, seq in zip(records, sequences)}
    out["unique_factor_combinations"] = len({c for c, _ in combos})
    out["unique_token_sequences"] = len({s for _, s in combos})
    out["token_sequence_collision_rate"] = 1.0 - (
        out["unique_token_sequences"] / max(1, len(records)))
    return out
