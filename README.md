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

Implemented: a dependency-free symbolic signaling environment, exact reference
controls, optional PyTorch REINFORCE trainer and unit tests. First short CPU runs
are documented in [experiments/first-smoke.md](experiments/first-smoke.md).
The stronger [pair-held-out experiment](experiments/pair-smoke.md) reaches 100%
train but 0% held-out across three seeds, exposing the current transfer limit.
Further [factorized-action and reward ablations](experiments/factorized-smoke.md)
show transfer in only one of three seeds with partial reward; grammar remains unproven.
An explicit [factor-isolated positive control](experiments/slots-smoke.md)
achieves 100% held-out across three seeds; its slot grammar is imposed, while
token meanings are learned. Frozen-checkpoint EH-INT export is available.
The next environment contract is now implemented in `multiround.py`: multiple
prey/trap types, private zones, alternating A→B messages, two actions and two
rounds. The matched trainer in `train_multiround.py` makes the action
`(own_zone, guess_partner_type)` so communication can affect reward, and logs
zone/type/terminal components separately. The 10k-episode causal baseline is
still negative: terminal success stayed at zero and messages did not reliably
beat the no-message control; see `results/multiround-full-10000.json`.
The proposed transition to evolving multi-step state, partial reward and
preference-sensitive target choice is specified in
[`experiments/multistep-protocol.md`](experiments/multistep-protocol.md).
A richer task variant is implemented in `trap_prep.py`: reach -> prepare(method)
-> activate, several simultaneous prey and one trap per zone, partial reward for
correct preparation (paid once) and full reward only for a catch, plus a firing
window in which activating early wastes the trap's charge and activating late
lets the prey escape. `trap_probe.py` perturbs one token at one step and reports
which part of the decoded action (target / method / timing) moves; on identical
intact behaviour it accepts the handwritten factorized code and rejects an
entangled one. `train_trap.py` is the REINFORCE trainer over the same
environment, with matched-budget no-message and no-readiness controls, and
`trap_diagnose.py` reads a learned pair back: which rung of
reach -> first_guess -> in_position -> fire -> catch it falls off, and how many
bits its wire carries about the zone and the required mechanism.

The message-blind bound is *not* a constant: a blind preparer can sweep the line
and fire one trap after another while the driver stalls, so `blind_reference`
depends on the horizon and on the prey's patience. Without a deadline the
`by='pair'` test split -- which leaves exactly one prey type per zone -- is
solvable blind, so held-out catch rates there mean nothing unless a deadline is
set. At n=3, horizon 8 and patience 4 the reference protocol still solves every
task while a two-trap sweep does not fit, and the bound is the intended 1/9
(all), 1/6 (train), 1/3 (test). The partial preparation reward needs a bound of
its own, since it is what a learner climbs first: `blind_preparation_bound` is
1/3 on the train split at horizon 4, 1/2 at horizon 8.

The first 60k-episode sweep (`results/trap-single-target-s*.json`) caught nothing
in three seeds, and diagnosis found two reasons that were not learning failures.
The partial reward was paid the first time a trap's *current* mechanism matched,
so cycling PREPARE(0), PREPARE(1), PREPARE(2) collected it with certainty and no
information; it is now paid only for the first attempt at a trap. And every prey
died at the same rung -- `in_position`, the prey standing on a correctly armed
trap -- which never happened once in 27 tasks, because DRIVE has no reward
channel except a catch the driver has never seen. The optional `approach` term
(deducted from the catch reward, so a success still totals 1.0) pays for exactly
that state. The rerun (`results/trap-fixed-s*.json`, horizon 4, start_pos 1, 60k
episodes, 3 seeds) puts `first_guess_rate` at 0.500 in every communication and
no_readiness run against 0.333 -- the blind ceiling -- in every no_message run,
so the method is being transmitted; catches reach 0.167/0.000/0.111 against a
blind bound of 0.167, so they are not; and the held-out split stays at 0.000,
below the 1/3 a uniform guess would score. Relaxing the sender wiring also retains 100% transfer across three seeds:
[receiver-only isolation](experiments/receiver-slots.md). Receiver position
semantics remain imposed; word-order emergence has not been demonstrated.
Successful coordination alone will not be treated as evidence of grammar.

## Layout

```
configs/             experiment configuration
src/emergent_hunt/   future environment, agents and training code
tests/               future invariant and evaluation tests
results/             generated outputs (git-ignored)
```

## Run (Python 3.10+)

```
python -m pip install -e ".[train]"
python -m unittest discover -s tests -v
python -m emergent_hunt.evaluate
python -m emergent_hunt.train --steps 2000 --seeds 0 1 2
python -m emergent_hunt.train --by pair --steps 2000 --seeds 0 1 2 --output results/pair-smoke.json
python -m emergent_hunt.trap_prep     # blind bounds + reference-protocol check
python -m emergent_hunt.trap_probe    # token-selectivity matrix, both codes
python -m emergent_hunt.train_trap --episodes 60000 --seeds 0 --horizon 4 --start-pos 1 --approach 0.25
python -m emergent_hunt.trap_diagnose results/trap-smoke-0-communication.pt
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

Initial MLP training and matched-budget no-message controls now run. Next:
stronger held-out splits, recurrent policies, then noise/cost sweeps.
Iterated learning, suggested in board reply #24849,
is a separate proposed ablation with matched exposure budgets and reset controls.
No claim that this intervention necessarily induces compositionality is made.
