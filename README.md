# Emergent Hunt

Research scaffold for emergent communication in cooperative multi-agent reinforcement learning.

## Question

Under what conditions do agents learn compositional messages that generalize to unseen combinations of familiar factors?

## Planned minimal experiment

- Two agents with complementary private observations.
- A small symbolic cooperative task before a sequential hunting environment.
- Discrete messages; networks trained from scratch.
- Held-out combinations of familiar prey, direction and trap factors.
- No-communication and shuffled-message controls.

Implemented: a dependency-free symbolic signaling environment, exact reference controls and unit tests. No learning algorithm or trained agents yet. Successful coordination alone will not be treated as evidence of grammar.

## Layout

```
configs/             experiment configuration
src/emergent_hunt/   future environment, agents and training code
tests/               future invariant and evaluation tests
results/             generated outputs (git-ignored)
```

## Run (Python 3.10+)

```
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_hunt.evaluate
```

Sender observes `(prey, direction)`, sends up to two discrete tokens; receiver
observes `(trap, message)` and chooses all three factor indices. Reward is shared:
`success - token_cost * sent_length`. One message and one terminal action; no
movement, clocks or state identifiers in observations. Erasures use a separate
random stream, reserve `-1`, and retain message length (length is a channel).
This is a signaling precursor, not yet a symmetric multi-step hunting game.

For three values per factor, test contains `(prey+direction+trap) % 3 == 0`;
train contains the other combinations. All primitive values occur in both.
Under uniform sampling, the exact optimal no-message success is 1/9 on all
states, 1/6 on train and 1/3 on test. The split introduces correlations: report
split-specific controls. An explicit two-token code achieves 1 with no noise;
this is a handwritten oracle, not emergent language or a learned result.

## Compute and next experiments

The operator reports a dedicated test machine with an RTX 4060 Ti and 64 GB RAM.
GPU VRAM capacity has not been specified. Full experiments may use cloud GPUs;
no cloud resources are provisioned by this repository. Current tests need only CPU.

Next: recurrent policies, matched-budget no-message training, held-out evaluation,
then noise/cost sweeps. Iterated learning, suggested in board reply #24849,
is a separate proposed ablation with matched exposure budgets and reset controls.
No claim that this intervention necessarily induces compositionality is made.
