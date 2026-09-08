"""Topographic similarity (kolpaq, board #25662/#25695): the discriminator
between "encodes structure" and "memorized a correct table" that EH-INT-r0.1
does not provide -- do-intervention proves message *sensitivity* to a factor,
not that meaning-distance and message-distance correlate across the corpus
(board #25634).

Two corrections found while running the naive version on-board (#25694,
#25714), both required for the metric to be informative on this environment:

1. Meaning-distance must be Hamming over (prey, direction) only, not the full
   (prey, direction, trap) triple. The receiver observes trap directly
   (environment.SymbolicHunt); no message can encode it, so trap-only state
   pairs always have message-distance 0 regardless of policy. Including trap
   inflates topsim for every policy alike and hides the real separation.

2. Compute over the full corpus, not the held-out test split alone. Under
   by='pair' (environment.held_out_pairs), the held-out set covers only 3
   distinct (prey, direction) pairs -- 9 states are 3 pairs x 3 traps. 3
   meaning-classes give a degenerate, near-uninformative Spearman estimate
   (on-board: learned_greedy 1.00 vs holistic 0.82, no separation) even
   though the same states with the full-corpus + (prey, direction)-only
   correction separate cleanly (0.76 vs 0.18). topsim is a diagnostic on the
   learned representation, not an accuracy metric -- there is no train/test
   leakage in computing it over all 27 states.
"""
from itertools import combinations


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                            dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return dp[m][n]


def _spearman(xs, ys):
    """Spearman rank correlation, average-rank ties, no scipy dependency
    (pyproject.toml declares no runtime deps outside the `train` extra).
    Returns None if either series has zero variance (correlation undefined).
    """
    n = len(xs)

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((r - mx) ** 2 for r in rx)
    vy = sum((r - my) ** 2 for r in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def topographic_similarity(states_and_messages, meaning_factors=(0, 1)):
    """states_and_messages: iterable of (state, message) where state is a
    (prey, direction, trap)-like sequence and message is the emitted symbol
    sequence for that state (e.g. row['sent_before'] from an EH-INT-r0.1
    row, keyed by row['base']).

    meaning_factors: which state indices count toward meaning-distance.
    Default (0, 1) = (prey, direction) only -- see module docstring point 1.
    Pass (0, 1, 2) to reproduce the naive full-triple version for comparison.

    Returns (rho, n_pairs). rho is None if fewer than 2 distinct meaning
    classes are present (degenerate input -- see module docstring point 2).
    """
    items = list(states_and_messages)
    meaning_d, message_d = [], []
    for (s1, m1), (s2, m2) in combinations(items, 2):
        meaning_d.append(sum(1 for i in meaning_factors if s1[i] != s2[i]))
        message_d.append(_edit_distance(m1, m2))
    return _spearman(meaning_d, message_d), len(meaning_d)
