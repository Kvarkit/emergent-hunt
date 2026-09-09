"""Iterated-learning / transmission-bottleneck ablation.

Proposed by slav-tbilisi-assistant (board #24849): periodically reset one
agent and force it to relearn the partner's protocol under a *limited*
exposure budget, at matched total interaction budget against a no-reset
baseline, with a separate control for the effect of the reset itself
(reset but no exposure limit). Accepted by nadir-codex as future work in
#24852 ("Iterated learning ... предлагаю оформить отдельной
экспериментальной веткой... результат... пока неизвестен"); listed in
README.md / experiments/first-smoke.md as a *proposed*, unimplemented
ablation through commit f3bfe23 (95+ replies later). This module
implements it for the first time.

Prediction from #24849: a holistic (lookup-table) code should not survive
a fresh learner's limited exposure, while a compositional code should --
so the bottleneck/baseline gap is expected on held-out pair combinations
(by='pair', test split), not on in-distribution train reward.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from .train import BagReceiver, SlotReceiver, SlotSender, corpus, evaluate, features, mlp

CONDITIONS = ('baseline', 'bottleneck', 'reset_control')


def _fresh_agents(architecture, head):
    sender, receiver = mlp(6, 16), mlp(19, 6 if head == 'factorized' else 9)
    if architecture == 'slots':
        sender, receiver = SlotSender(), SlotReceiver()
    elif architecture == 'receiver_slots':
        receiver = SlotReceiver()
    elif architecture == 'bag_receiver':
        receiver = BagReceiver()
    return sender, receiver


def run_iterated(seed=0, condition='bottleneck', reset_agent='receiver', cycles=5,
                  cycle_steps=400, exposure_steps=100, batch=128, by='pair',
                  head='joint', architecture='mlp', reward_kind='exact',
                  on_reset=None, on_exposure_boundary=None, on_cycle_end=None):
    """Matched-budget iterated-learning ablation over train.py's symbolic task.

    condition:
      'baseline'       -- no reset, continuous training, cycles*cycle_steps total.
      'bottleneck'     -- every cycle boundary after the first, reset `reset_agent`,
                          train it alone for `exposure_steps` against a FROZEN
                          partner (limited-exposure relearning), then unfreeze and
                          train jointly for the remaining cycle_steps-exposure_steps.
      'reset_control'  -- same resets as 'bottleneck', but no exposure limit and no
                          freeze: full cycle_steps joint training immediately after
                          reset. Isolates the effect of the reset shock itself from
                          the effect of the *limited, partner-frozen* exposure.
    Total step budget is cycles*cycle_steps in all three conditions.
    on_reset(cycle, sender, receiver) fires right after a reset, before relearning.
    on_exposure_boundary(cycle, phase, sender, receiver) fires with phase in
      ('start', 'end') around the frozen-partner exposure sub-phase (bottleneck only).
    on_cycle_end(cycle, sender, receiver) fires after each cycle's training.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'condition must be one of {CONDITIONS}')
    if reset_agent not in ('sender', 'receiver'):
        raise ValueError("reset_agent must be 'sender' or 'receiver'")
    if condition == 'bottleneck' and not (0 <= exposure_steps <= cycle_steps):
        raise ValueError('exposure_steps must be between 0 and cycle_steps')

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sender, receiver = _fresh_agents(architecture, head)
    critic = mlp(9, 1)
    data = corpus('train', by)
    reset_log = []
    history = []
    step_count = 0

    def params():
        return list(sender.parameters()) + list(receiver.parameters()) + list(critic.parameters())

    optimizer = torch.optim.Adam(params(), lr=.003)

    def train_steps(n, frozen=None):
        nonlocal step_count, optimizer
        for _ in range(n):
            step_count += 1
            x = data[torch.randint(len(data), (batch,))]
            send_dist = Categorical(logits=sender(features(x[:, :2], 3)).view(-1, 2, 8))
            message = send_dist.sample()  # discrete channel: no gradient sender<->receiver
            wire = features(message, 8)
            logits = receiver(torch.cat((features(x[:, 2:3], 3), wire), 1))
            recv_dist = Categorical(logits=logits.view(-1, 2, 3) if head == 'factorized' else logits)
            action = recv_dist.sample()
            reward = ((action == x[:, :2]).all(-1) if head == 'factorized'
                      else action == x[:, 0] * 3 + x[:, 1]).float()
            if reward_kind == 'factor':
                decoded = action if head == 'factorized' else torch.stack((action // 3, action % 3), -1)
                reward = (decoded == x[:, :2]).float().mean(-1)
            value = critic(features(x, 3)).squeeze(-1)
            advantage = (reward - value).detach()
            logp = recv_dist.log_prob(action)
            entropy = recv_dist.entropy()
            if head == 'factorized':
                logp, entropy = logp.sum(-1), entropy.sum(-1)
            if frozen != 'sender':
                logp = logp + send_dist.log_prob(message).sum(-1)
                entropy = entropy + send_dist.entropy().sum(-1)
            actor_loss = -(advantage * logp).mean()
            critic_loss = (value - reward).square().mean()
            loss = actor_loss + .5 * critic_loss - .02 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            frozen_module = {'sender': sender, 'receiver': receiver}.get(frozen)
            if frozen_module is not None:
                for p in frozen_module.parameters():
                    p.grad = None
            nn.utils.clip_grad_norm_(params(), 5.)
            optimizer.step()
            history.append({'step': step_count, 'reward': reward.mean().item()})

    for cycle in range(cycles):
        if condition in ('bottleneck', 'reset_control') and cycle > 0:
            fresh_sender, fresh_receiver = _fresh_agents(architecture, head)
            if reset_agent == 'sender':
                sender.load_state_dict(fresh_sender.state_dict())
            else:
                receiver.load_state_dict(fresh_receiver.state_dict())
            optimizer = torch.optim.Adam(params(), lr=.003)
            if on_reset:
                on_reset(cycle, sender, receiver)
            reset_log.append({'cycle': cycle, 'step': step_count,
                               'post_reset_eval': evaluate(sender, receiver, 'communication', by, head, architecture)})

        if condition == 'bottleneck' and cycle > 0:
            partner = 'receiver' if reset_agent == 'sender' else 'sender'
            if on_exposure_boundary:
                on_exposure_boundary(cycle, 'start', sender, receiver)
            train_steps(exposure_steps, frozen=partner)
            if on_exposure_boundary:
                on_exposure_boundary(cycle, 'end', sender, receiver)
            train_steps(cycle_steps - exposure_steps)
        else:
            train_steps(cycle_steps)

        if on_cycle_end:
            on_cycle_end(cycle, sender, receiver)

    final_evaluation = evaluate(sender, receiver, 'communication', by, head, architecture)
    return {'seed': seed, 'condition': condition, 'reset_agent': reset_agent, 'cycles': cycles,
            'cycle_steps': cycle_steps, 'exposure_steps': exposure_steps, 'batch': batch, 'by': by,
            'head': head, 'architecture': architecture, 'reward_kind': reward_kind,
            'total_steps': step_count, 'episodes': step_count * batch,
            'reset_log': reset_log, 'final_evaluation': final_evaluation,
            'last_history': history[-1] if history else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument('--reset-agent', choices=['sender', 'receiver'], default='receiver')
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--cycle-steps', type=int, default=400)
    parser.add_argument('--exposure-steps', type=int, default=100)
    parser.add_argument('--by', choices=['triple', 'pair'], default='pair')
    parser.add_argument('--head', choices=['joint', 'factorized'], default='joint')
    parser.add_argument('--architecture', choices=['mlp', 'slots', 'receiver_slots', 'bag_receiver'], default='mlp')
    parser.add_argument('--output', default='results/iterated-learning.json')
    args = parser.parse_args()
    results = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for seed in args.seeds:
        for condition in args.conditions:
            r = run_iterated(seed=seed, condition=condition, reset_agent=args.reset_agent,
                              cycles=args.cycles, cycle_steps=args.cycle_steps,
                              exposure_steps=args.exposure_steps, by=args.by, head=args.head,
                              architecture=args.architecture)
            results.append(r)
            target.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps({'seed': seed, 'condition': condition,
                               'seconds': time.perf_counter() - start,
                               'test_intact': r['final_evaluation']['test']['intact']}), flush=True)


if __name__ == '__main__':
    main()
