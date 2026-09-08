# Receiver-only factor isolation

Follow-up to slots-smoke: the sender is again the unrestricted 6->32->16 MLP,
seeing both prey and direction at each token output. Only receiver factor
isolation remains: first token decodes prey, second direction. Exact reward,
pair split, 2,000 updates x 128 episodes, seeds 0/1/2, CPU. No tuning between seeds.

All three communicating runs: 100% train, 100% held-out, 100% full-corpus greedy
accuracy. All three separately trained no-message controls: 16.7% train, 0% test.
Shuffling communicating messages: 16.7% train, 33.3% test, 11.1% full corpus.
Muting: train 16.7/0/0%, test 0/33.3/33.3%. Reversing: full-corpus 11.1/11.1/22.2%.
Final sampled reward 1.0 each, finite gradients; 5.4–6.5 seconds per run.

The sender need not be architecturally restricted to one factor per position
to obtain transfer in this tiny task. Unlike the two-sided slot model, perfect
train performance does not logically guarantee transfer: sender outputs may
depend arbitrarily on the previously unseen pair. Empirical transfer here is
therefore an additional observation, still only three seeds and nine test states.

Learned codebook seed 0 (rows prey, columns direction):
```
(4,2) (6,5) (1,3)
(5,2) (5,1) (5,3)
(2,2) (2,5) (2,4)
```
First-position tokens 4/6/1 all decode prey=0. Context-dependent token variants
are allowed; demanding identical tokens whenever a factor is unchanged would
misclassify this successful code. Position-specific interpretation is still
hardcoded in the receiver, so this is not emergence of word order or grammar.
Removing that remaining prior is a separate experimental question.

Frozen seed-0 EH-INT: 162/162 correct before/after; messages change for 54/54
prey interventions and 54/54 direction interventions, 0/54 trap interventions.
The checkpoint hash and all rows are saved in receiver-slots-0-eh-int.json.
No post-diagnostic training. Seventeen tests passed. The previous full-slot
initialization order is preserved; existing experiments remain reproducible.

```
python -m emergent_hunt.train --by pair --head factorized --architecture receiver_slots --steps 2000 --seeds 0 1 2 --output results/receiver-slots.json
python -m emergent_hunt.diagnose_checkpoint results/receiver-slots-0-communication.pt --output results/receiver-slots-0-eh-int.json
```

Raw results are included alongside this report; checkpoint files stay local.
Architecture changes affect parameter counts and initializations; these are
matched-budget exploratory controls, not a parameter-matched causal proof.
