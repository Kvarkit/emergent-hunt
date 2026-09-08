# Pair-held-out smoke experiment

Integrated Zenith's branch through a402d3a, including the intervention summary
regression fix. Added `--by pair` to both training and evaluation; defaults remain
`triple` for backward compatibility. Fifteen tests pass, including the actual
training tensor's sender-pair disjointness.

Command: `python -m emergent_hunt.train --by pair --steps 2000 --seeds 0 1 2 --output results/pair-smoke.json`

Same architecture, optimizer and 256,000 sampled episodes per run as first-smoke.
Six runs took 4.3–5.6 seconds each on CPU. Raw results: `pair-smoke.json`.

| Split | Communicating train | Communicating held-out (seeds 0/1/2) |
| --- | --- | --- |
| Original triple | 100% | 88.9% / 66.7% / 88.9% |
| Whole sender pair | 100% | 0% / 0% / 0% |

Pair-run no-message controls: 16.7% train and 0% test on all seeds. Muting and
shuffling communicating policies reduce train accuracy to 16.7%. Training reward
ends at .992–1.0 and critic MSE at .000019–.00780. Loss/gradients stay finite.

Interpretation: the original successful transfer does not survive withholding
sender observations. Learning coordination on the training support works, but
this architecture does not demonstrate recombination on these three test pairs.

Important confound: the receiver has a JOINT nine-class action head. The three
held-out pair classes are never rewarded during training. This is simultaneously
an unseen sender-input and unseen rewarded-action test. Zero accuracy therefore
does not isolate a failure of language compositionality. Next controlled change:
factorize the action into separate prey/direction heads, retaining identical
reward, split, seeds and budget. That supplies a compositional action prior and
must be reported as such; it does not impose compositional token semantics.

Zenith's 486-row intervention corpus is also integrated. Its handwritten and
holistic controls both solve the task and respond to factor changes: those
diagnostics alone cannot establish compositionality. They have not yet been
applied to trained checkpoints. No trained checkpoint is stored in these smoke
runs; reproduction currently requires rerunning the recorded seeds.
