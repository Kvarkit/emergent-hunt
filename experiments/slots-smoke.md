# Compositional architecture positive control

Six CPU runs, seeds 0/1/2, pair split, 2,000 updates x 128 episodes, exact
all-or-nothing reward. Sender has two independent 3->32->8 MLPs; receiver
has two independent 8->32->3 MLPs. Slot 0 sees prey only, slot 1 direction
only. Token meanings are learned by REINFORCE; position-to-factor binding is
hardcoded. This is a positive control, NOT emergent syntax.

| Model, exact reward | Train | Held-out, seeds 0/1/2 |
| --- | --- | --- |
| Previous joint action MLP | 100% | 0/0/0% |
| Previous factorized action MLP | 100% | 0/0/0% |
| Factor-isolated sender and receiver | 100% | 100/100/100% |

No-message controls: 16.7% train, 0% test. Muting communicating policies:
same. Shuffling messages across the full 27-state corpus: 11.1%. Reversing
tokens: full-corpus accuracy 11.1/11.1/33.3%. All seeds end with sampled
reward 1.0 and finite gradients; CPU time 5.3–7.7 seconds per run.

Why this transfers: deterministic outputs have form (f(prey), g(direction)).
Every primitive occurs in training. If every training pair is solved, each
f and g is correct on its entire domain, so every held-out combination is
also solved. Thus 100% transfer conditional on perfect training is structurally
guaranteed here. Different learned token dictionaries across seeds do not alter
this argument. Additional modules also change parameter count/initialization;
this is not a parameter-matched causal isolation study.

The failures in earlier experiments are consistent with a cheap holistic
solution: six training pairs can be memorized within 64 two-token messages,
and no reward tests recombination. A factorized OUTPUT alone did not stop
the shared sender/receiver hidden layers from entangling both factors.
The successful control establishes feasibility, not the unique failure cause.

## Frozen diagnostics

Training now saves final checkpoints per seed/mode. A new command exports
EH-INT-r0.1 rows from a frozen greedy checkpoint, with checkpoint SHA-256.
No training occurs during diagnosis. Seed 0: 162/162 correct before and after,
54/54 changed messages for prey, 54/54 for direction, 0/54 for trap. These are
repeated state comparisons, not independent trials. The current rows use
exact-match diagnostic reward even if a checkpoint was trained with partial
reward; training_reward labels that distinction. Noise is not tested.

```
python -m emergent_hunt.train --by pair --head factorized --architecture slots --steps 2000 --seeds 0 1 2 --output results/slots-smoke.json
python -m emergent_hunt.diagnose_checkpoint results/slots-smoke-0-communication.pt --output results/slots-0-interventions.json
```

Raw training results and seed-0 intervention rows accompany this report.
Checkpoints remain local under results/. Seventeen tests pass, including
factor isolation and the existing exact-gradient and environment checks.

Next experimental question: relax the hard factor-to-slot wiring while
retaining compositional pressure (e.g. controlled learning bottleneck), with
this positive control and the unconstrained MLP as endpoints. Success in a
fixed positional code alone will still not demonstrate role syntax.
