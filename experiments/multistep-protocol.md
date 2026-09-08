# Multi-step communication protocol (proposal)

## Why the current two-round task is insufficient

`MultiRoundHunt` alternates `message -> message -> action -> action`, but the
world does not evolve between actions. A correct zone is therefore a terminal
guess, not an approach to a target. The next benchmark must make actions change
the latent state and make later observations depend on that change.

## Proposed episode structure

Use a fixed horizon of `H` stages. At every stage the agents perform:

```
private observation -> send token(s) -> receive token(s) -> physical action
-> transition -> shaped reward -> next observation
```

The action space should contain at least `approach`, `retreat`, `wait`, and
`capture/trigger` (with invalid terminal actions penalized). A compact state is
two positions on a line or graph: prey position `p`, trap position `t`, and an
agent-controlled safety/energy variable. The prey can move after a noisy action;
the trap becomes armed only after the agents reach compatible positions. This
creates genuine credit assignment: a message can improve a later action rather
than merely select a final label.

The first implementation should keep a deterministic evaluation grid and expose
the full state to an oracle only. Learned agents receive local views, the last
incoming token, remaining horizon, and a recurrent memory state. Compare:

1. no communication;
2. one-way communication;
3. alternating two-way communication;
4. two-way communication with shuffled tokens;
5. oracle communication (upper bound).

All controls must use the same state seeds, horizon, parameter budget and action
costs. Report success, discounted return, progress-at-each-step and episode
length separately; a high shaped return must not be called a capture success.

## Partial reward and preferences

Reward should be decomposed into logged components, not hidden in one scalar:

```
R_t = 0.20 * progress_t
    + 0.15 * safety_t
    + 0.15 * information_gain_t
    + 0.20 * compatibility_t
    + 0.30 * terminal_capture_t
    - movement_cost_t - message_cost_t
```

`progress_t` is decrease in graph distance to a viable prey/trap configuration;
`information_gain_t` is only available when an action/message actually reduces
uncertainty (or is computed as an evaluation diagnostic, never as privileged
training input). Potential-based shaping should be used for distance terms so
the optimal terminal policy is not changed. Always export each component and
the unshaped terminal metric.

To make preferences causally relevant, sample multiple candidate prey/trap
targets. Agent A has a private utility vector over prey types and B has one over
trap types; utilities may be aligned, partially conflicting, or anti-aligned.
The team objective is an explicit weighted sum of both utilities minus energy,
damage and delay costs. A preference that is constant over available actions is
not a preference experiment: it cannot affect communication or behavior.
Evaluate social welfare, each agent's private utility, regret against a central
oracle, and Pareto failures under shared versus conflicting preferences.

## Falsifiable hypotheses

- Longer horizons help only when recurrent memory and two-way communication are
  both available; otherwise the progress curve should match the no-message
  control.
- Shuffling received tokens should reduce coordination if order carries learned
  structure, while preserving performance if communication is genuinely a bag
  of independent symbols.
- Under conflicting preferences, communication should move actions toward the
  declared team welfare, not simply maximize the sender's preferred target.
- Any grammar claim requires held-out combinations of target type, route and
  horizon, plus intervention tests on tokens and order. Reward improvement alone
  is not evidence of compositional syntax.

## Questions for reviewers

1. Should the first transition use a line, a small graph, or a 2-D grid?
2. Is `information_gain` acceptable as an auxiliary diagnostic, or should all
   training rewards remain strictly environment-derived?
3. Which preference regime is the most informative first: aligned, weighted
   compromise, or adversarial/anti-aligned?
4. What minimum held-out route/type split would make a grammar claim credible?
