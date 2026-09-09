# Action-dependent rounds

Use `LineHunt(crossed=True, progress_weight=.1, step_cost=.01,
trigger_delay=2)` for the next dynamic experiment. A knows B's target; B knows
A's target. Physical moves change the next observation. A wrong trigger at
the wrong coordinate consumes up to three remaining time units instead of one.
Time and location are observable physical consequences, hence potential extra
communication channels; channel interventions must keep these rules identical.

Reward = success - elapsed_time_cost + potential(next) - potential(current).
Potential is minus weighted summed distance to the two targets, and is zero
at every terminal state including timeout. On nonterminal steps movement toward
the targets is positive, movement away negative. Potential terms telescope,
so going back and forth cannot farm progress reward. At timeout the terminal
correction can be positive; log task_reward and success separately.

This shaping uses gamma=1. Do not claim policy invariance under trainers using
discount .97. The reward and distance diagnostics are privileged training data;
do not feed them to actors as free target information.

The symbolic multiround trainer remains a separate static benchmark. No training
result for this new reward configuration has been established. Existing LineHunt
defaults preserve previous experiments. Regression tests cover movement, signed
progress, recovery time, and equal total shaping for different failure paths.
