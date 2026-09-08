# MultiRoundHunt first training smoke

The environment now models three prey types, three trap types, four private
zones, two message rounds (A then B) and two action rounds. A terminal success
requires the agents to choose the respective prey/trap zones and the types to
differ. Each agent sees only its own type/zone; each receives the partner's
token before its action. The trainer uses two independent 32-unit agents,
REINFORCE, a shared terminal reward and the same message/action policy at both
rounds.

The first implementation had a protocol bug: action logits were computed before
the second token arrived. It was corrected and the action heads now run after
the exchange. The 21-test suite passes.

Short 1,000-episode smoke runs completed for seeds 0, 1 and 2, with matched
message and no-message controls. The output is deliberately recorded as a
smoke diagnostic, not a success claim: one reward sample per checkpoint is too
noisy; the trainer now records a rolling 100-episode mean for the next run.
The initial results show seed 0 reaching a recent reward of 1.0 in both modes,
seed 1 reaching 1.0 with messages versus 0.0 without, and seed 2 ending at 0.0
in both. This is not yet evidence that communication is necessary. The task
currently leaks each agent's own zone directly into its action input, so only
type compatibility and partner information require communication; a no-message
policy can still solve some episodes by learning zone behavior.

Next changes before interpreting this as emergent grammar: run 5k+ episodes,
add deterministic evaluation over a fixed state grid, balance incompatible and
compatible type pairs, and compare one-way, two-way and shuffled-token controls.
