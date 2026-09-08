# First CPU learning smoke experiment

Six runs: seeds 0, 1, 2, each with communication and a matched no-message
training control. Each run uses 2,000 updates x 128 episodes = 256,000 samples,
drawn with replacement from just 18 training states. CPU time per run: 4.3–5.4 s
(excluding imports). Python 3.12, torch 2.14.0+cpu. No GPU or cloud used.

## Architecture and objective

Sender: 6 one-hot inputs -> 32 tanh -> two independent categorical distributions
over 8 tokens. Receiver: observed trap (3) + token one-hots (16) -> 32 tanh ->
9 actions (prey/direction). Receiver copies its known trap into the final action;
this hardcoded simplification is not learning all three action coordinates.
Central training-only critic: 9 full-state one-hot inputs -> 32 tanh -> scalar.
No recurrence, pretrained weights, differentiable messages or supervised labels.

`A = stop_gradient(reward - V(state))`

`L = -mean(A * (log pi_sender + log pi_receiver))`
`    + 0.5 * mean((V - reward)^2) - 0.02 * mean(entropy)`

Sender log probabilities and entropies sum across two tokens. The no-message
control drops sender loss and feeds zeros into the receiver's message inputs.
Adam learning rate .003, gradient norm clipping 5. Channel is noiseless, fixed
length two, and free in these runs. No length, noise or memory experiments yet.

## Final greedy-policy results

| Seed | Train | Held-out | Train, shuffled messages | Critic MSE, first -> last batch |
| --- | --- | --- | --- | --- |
| 0 | 100% | 88.9% | 11.7% | .1277 -> .00807 |
| 1 | 100% | 66.7% | 13.0% | .1630 -> .0000523 |
| 2 | 100% | 88.9% | 11.7% | .1666 -> .0000381 |

All no-message trained controls: train 16.7%, held-out 0%, all states 11.1%.
Muting trained communicating models: the same three accuracies. Shuffling is
an exact average over all cyclic shifts of the evaluated corpus's messages,
including identity. It breaks correspondence while preserving message marginals.

The 0% held-out blind result is compatible with the 1/3 theoretical test bound:
the latter is an oracle optimized for the test distribution. A blind model
trained on the disjoint train support chooses a pair impossible for that trap
in test. Never label 1/3 the expected accuracy of that train-fitted baseline.

## Does the loss work?

Yes for learning this small coordination task: sampled training reward rises
from roughly .10–.13 to .992–1.0, greedy train accuracy reaches 1 on all seeds,
and channel interventions remove the advantage. Every update checks finite
loss/gradients. Tests compare the REINFORCE gradient against the exact derivative
of expected reward in an enumerated categorical example, verify critic baseline
detachment, and compare vectorized rewards against the environment exhaustively.

Actor loss is not a monotonic error measure: its sign and magnitude depend on
the moving baseline and on-policy distribution. A value near zero can indicate
either successful prediction or uninformative advantages. Critic MSE is a noisy
last-batch diagnostic, not an independent measure of language quality.

Mean held-out accuracy is 81.5%, range 66.7–88.9%, across only three seeds and nine
test states. This is a smoke result, not a statistical claim of robust transfer.
All nine sender observations appear during training. The 64 possible messages
can memorize them holistically; only sender/receiver-context combinations are
held out. No evidence of grammar, token compositionality, symmetric coordination,
or effectiveness of iterated learning has been established.

## Reproduce

Install the optional training dependencies, then run:

```
python -m emergent_hunt.train --steps 2000 --seeds 0 1 2
python -m unittest discover -s tests -v
```

Raw learning curves and codebooks: `first-smoke.json` alongside this report.
Training runs normally write git-ignored `results/smoke.json`. Next useful step:
make sender-level held-out combinations and compare an equal-capacity holistic
baseline before trying to interpret messages as compositional.
