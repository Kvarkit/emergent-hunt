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

This repository is a scaffold only. No environment, training algorithm or experimental results have been implemented. Successful coordination alone will not be treated as evidence of grammar.

## Layout

```
configs/             experiment configuration
src/emergent_hunt/   future environment, agents and training code
tests/               future invariant and evaluation tests
results/             generated outputs (git-ignored)
```

## Next step

Specify observations, actions, reward and the train/test combination split for a two-agent symbolic task.
