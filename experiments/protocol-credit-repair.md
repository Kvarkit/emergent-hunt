# Symbolic protocol repair

The current multiround trainer is a static symbolic exchange, not a dynamic
hunt. Only final actions affect reward; earlier actions neither move entities
nor change observations. More rounds alone do not create action-conditioned
dialogue. Each agent already sees its own zone, so communication principally
needs to transmit one hidden type. Success here is not evidence of grammar.

Repairs:

- Message REINFORCE uses values computed before this round's message sampling.
  Detaching a post-delivery value did not remove its dependence on the sampled
  token and therefore did not make it a valid message baseline.
- Only final action log probabilities receive terminal credit. Earlier sampled
  actions have no causal path to the outcome in this particular simulator.
- Routing preserves straight-through token vectors and their gradient rather
  than passing floating vectors to integer one-hot encoding.
- Results now declare the task scope and the terminal ceiling. Symmetric
  evaluation includes equal-type states where success is impossible, so with
  three types the ceiling is 2/3, not 1. The one-way ceiling is 1.

Tests cover differentiable routing and training end to end. These repairs do
not establish convergence, compositionality, or dynamic multi-step behavior.
Evaluate sampled REINFORCE and straight-through training separately; they are
different estimators. Token shift and swapping agents' wires measure channel
dependence, not temporal word order.

Validation: full suite 66 tests passed. Sampled REINFORCE, 5000 episodes,
three seeds (0, 1, 2), rounds=2, default symmetric configuration:

| Seed | Intact terminal | Muted terminal | Type accuracy |
| --- | --- | --- | --- |
| 0 | 0.222222 | 0.111111 | 0.500000 |
| 1 | 0.222222 | 0.111111 | 0.500000 |
| 2 | 0.555556 | 0.180556 | 0.763889 |

The fixes do not resolve optimization instability. Do not compare these with
historical scores as a controlled estimate of improvement: the estimator changed.

Next architectural requirement: both location and activation choice must depend
on hidden partner information, and an actual environment transition must change
the next observation. Keep unconstrained token meanings and sequence positions.
Before claiming grammar, test held-out factor combinations and targeted token
substitutions that predictably change one decision component.
