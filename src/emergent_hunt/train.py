"""Short CPU REINFORCE smoke experiments, no claim of grammar or optimality."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from .environment import states


def features(x, n):
    return one_hot(x, n).float().flatten(1)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, outputs))


class SlotSender(nn.Module):
    """Explicit factor/position prior; token meanings are learned from reward."""
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([mlp(3, 8), mlp(3, 8)])

    def forward(self, x):
        return torch.cat([net(x[:, i*3:(i+1)*3]) for i, net in enumerate(self.slots)], -1)


class SlotReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([mlp(8, 3), mlp(8, 3)])

    def forward(self, x):
        # Trap is irrelevant to predicting prey/direction; copy occurs outside.
        return torch.cat([net(x[:, 3+i*8:3+(i+1)*8]) for i, net in enumerate(self.slots)], -1)


def corpus(split, by='triple'):
    return torch.tensor([(s.prey, s.direction, s.trap) for s in states(split=split, by=by)])


@torch.no_grad()
def evaluate(sender, receiver, mode, by='triple', head='joint'):
    result = {}
    code = sender(features(torch.cartesian_prod(torch.arange(3), torch.arange(3)), 3)).view(-1, 2, 8).argmax(-1)
    for split in ('train', 'test', 'all'):
        x = corpus(split, by)
        msg = sender(features(x[:, :2], 3)).view(-1, 2, 8).argmax(-1)
        scores = {}
        for intervention in ('intact', 'mute', 'shuffle', 'reverse'):
            # Exact average over all cyclic shifts, avoiding RNG-dependent evaluation.
            shifts = range(len(x)) if intervention == 'shuffle' else (0,)
            accuracies = []
            for shift in shifts:
                altered = msg.roll(shift, 0)
                if intervention == 'reverse':
                    altered = altered.flip(1)
                m = features(altered, 8)
                if mode == 'no_message' or intervention == 'mute':
                    m = torch.zeros_like(m)
                inp = torch.cat((features(x[:, 2:3], 3), m), 1)
                logits = receiver(inp)
                if head == 'factorized':
                    factors = logits.view(-1, 2, 3).argmax(-1)
                    pred = factors[:, 0] * 3 + factors[:, 1]
                else:
                    pred = logits.argmax(-1)
                accuracies.append((pred == x[:, 0] * 3 + x[:, 1]).float().mean().item())
            scores[intervention] = sum(accuracies) / len(accuracies)
        result[split] = scores
    result['codebook_prey_major'] = code.tolist() if mode == 'communication' else None
    return result


def run(seed=0, mode='communication', steps=2000, batch=128, by='triple', head='joint', reward_kind='exact', architecture='mlp', checkpoint=None):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sender, receiver, critic = mlp(6, 16), mlp(19, 6 if head == 'factorized' else 9), mlp(9, 1)
    if architecture in ('slots', 'receiver_slots'):
        if head != 'factorized':
            raise ValueError('slots requires factorized head')
        if architecture == 'slots':
            sender, receiver = SlotSender(), SlotReceiver()
        else:
            receiver = SlotReceiver()
    optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()) + list(critic.parameters()), lr=.003)
    data = corpus('train', by)
    history = []
    start = time.perf_counter()
    initial = evaluate(sender, receiver, mode, by, head)
    for step in range(1, steps + 1):
        x = data[torch.randint(len(data), (batch,))]
        send_dist = Categorical(logits=sender(features(x[:, :2], 3)).view(-1, 2, 8))
        message = send_dist.sample()  # genuinely discrete, no gradient through channel
        wire = features(message, 8)
        if mode == 'no_message':
            wire = torch.zeros_like(wire)
        logits = receiver(torch.cat((features(x[:, 2:3], 3), wire), 1))
        recv_dist = Categorical(logits=logits.view(-1, 2, 3) if head == 'factorized' else logits)
        action = recv_dist.sample()
        # Receiver copies its observed trap; its learned action is prey/direction.
        reward = ((action == x[:, :2]).all(-1) if head == 'factorized'
                  else action == x[:, 0] * 3 + x[:, 1]).float()
        if reward_kind == 'factor':
            decoded = action if head == 'factorized' else torch.stack((action//3, action%3), -1)
            reward = (decoded == x[:, :2]).float().mean(-1)
        value = critic(features(x, 3)).squeeze(-1)
        advantage = (reward - value).detach()
        logp = recv_dist.log_prob(action)
        entropy = recv_dist.entropy()
        if head == 'factorized':
            logp = logp.sum(-1)
            entropy = entropy.sum(-1)
        if mode == 'communication':
            logp = logp + send_dist.log_prob(message).sum(-1)
            entropy = entropy + send_dist.entropy().sum(-1)
        actor_loss = -(advantage * logp).mean()
        critic_loss = (value - reward).square().mean()
        loss = actor_loss + .5 * critic_loss - .02 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        params = list(sender.parameters()) + list(receiver.parameters()) + list(critic.parameters())
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        grad = nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            row = {'step':step, 'sample_reward':reward.mean().item(), 'actor_loss':actor_loss.item(),
                   'critic_mse':critic_loss.item(), 'entropy':entropy.mean().item(), 'grad_norm':grad.item(),
                   'evaluation':evaluate(sender, receiver, mode, by, head)}
            history.append(row)
    if checkpoint:
        torch.save({'sender':sender.state_dict(), 'receiver':receiver.state_dict(),
                    'critic':critic.state_dict(), 'architecture':architecture, 'head':head,
                    'seed':seed, 'mode':mode, 'by':by, 'reward_kind':reward_kind,
                    'steps':steps}, checkpoint)
    return {'seed':seed, 'mode':mode, 'by':by, 'head':head, 'architecture':architecture, 'reward_kind':reward_kind, 'steps':steps, 'batch':batch,
            'episodes':steps*batch, 'seconds':time.perf_counter()-start,
            'torch':torch.__version__, 'initial':initial, 'history':history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output', default='results/smoke.json')
    parser.add_argument('--by', choices=['triple', 'pair'], default='triple')
    parser.add_argument('--head', choices=['joint', 'factorized'], default='joint')
    parser.add_argument('--reward', choices=['exact', 'factor'], default='exact')
    parser.add_argument('--architecture', choices=['mlp', 'slots', 'receiver_slots'], default='mlp')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.architecture in ('slots', 'receiver_slots') and args.head != 'factorized':
        parser.error('slots requires --head factorized')
    results = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for mode in ('communication', 'no_message'):
            checkpoint = target.with_name(f'{target.stem}-{seed}-{mode}.pt')
            r = run(seed, mode, args.steps, by=args.by, head=args.head, reward_kind=args.reward,
                    architecture=args.architecture, checkpoint=checkpoint)
            results.append(r)
            target.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps({'seed':seed,'mode':mode,'seconds':r['seconds'], 'last':r['history'][-1]}), flush=True)


if __name__ == '__main__':
    main()
