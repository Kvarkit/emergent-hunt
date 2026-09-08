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

## Transfer test: unseen prey and unseen action grammar

After training, freeze the complete pair (or group) of agents: weights,
vocabulary, recurrent state interface and decoding rule. Evaluate it without
fine-tuning in a new environment family with two independent changes:

1. **Novel prey:** new prey morphology/type IDs and motion dynamics that were
   absent from training (for example, a prey that retreats after `wait`, or a
   prey whose safe approach direction is reversed). The observation encoder may
   expose shared physical features, but must not provide a memorized training
   ID; otherwise this is only an OOD-label test.
2. **Novel action plan:** the shortest successful plan uses a new sequence such
   as `approach → wait → flank → capture`, while training used a different
   order and did not expose `flank` as a successful transition. The action
   vocabulary should therefore be factored into reusable primitives where
   possible, and the transfer environment must document which primitive and
   transition rules are new.

Use four matched evaluations: (A) familiar prey/familiar plan, (B) novel prey
with familiar plan, (C) familiar prey with novel plan, and (D) novel prey plus
novel plan. Include a centrally planned oracle and a re-trained in-domain pair
as upper bounds. Report zero-shot success, discounted return, progress curve,
message length/order, and regret against the oracle. Run at least five fixed
world seeds and retain the exact maps.

The key negative controls are a vocabulary permutation, shuffled received
tokens, and a pair trained from scratch only on the new family. A drop in D is
expected; the claim is specifically that a compositional protocol should retain
more performance than a holistic memorizer, and that token interventions should
preserve factor-level effects across A→D. If the new action primitive has no
shared semantics with training, failure is not evidence against emergent
communication—it is an interface mismatch. The benchmark must therefore
separate reusable physical primitives from genuinely new symbols.

## Questions for reviewers

1. Should the first transition use a line, a small graph, or a 2-D grid?
2. Is `information_gain` acceptable as an auxiliary diagnostic, or should all
   training rewards remain strictly environment-derived?
3. Which preference regime is the most informative first: aligned, weighted
   compromise, or adversarial/anti-aligned?
4. What minimum held-out route/type split would make a grammar claim credible?
