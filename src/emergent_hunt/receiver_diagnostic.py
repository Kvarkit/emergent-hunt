"""Train only the receiver/navigation policy with a frozen canonical sender."""
import random
import torch
from torch.distributions import Categorical

from .line_hunt import LineHunt
from .line_policy import GRULineAgent, line_observation


def train_receiver(seed=0, episodes=3000, length=5, horizon=8, hidden_dim=32,
                   lr=.003):
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    receiver = GRULineAgent(vocab=length, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(receiver.parameters(), lr=lr)
    for _ in range(episodes):
        goal, trap = random.sample(range(length), 2)
        env = LineHunt(length, horizon, crossed=True); env.reset(goal, trap)
        h = receiver.initial_state(); incoming = None; logs=[]; rewards=[]
        prev_dist = abs(env.state.b_pos - trap)
        for step in range(horizon):
            obs, _ = line_observation(env, 'b', length, incoming)
            # t=0 is mandatory WAIT while the frozen sender token is delivered.
            _, action_logits, h = receiver(obs, torch.nn.functional.one_hot(
                torch.tensor([incoming if incoming is not None else 0]), length).float(), h)
            dist = Categorical(logits=action_logits)
            action = torch.tensor(2) if step == 0 else dist.sample()
            log = torch.tensor(0.0) if step == 0 else dist.log_prob(action)
            # scripted A receives goal from B at t=0 and moves to it.
            if step == 0:
                action_a = 2
            else:
                action_a = 1 if env.state.a_pos < goal else 0 if env.state.a_pos > goal else 2
            _, terminal, done, _ = env.step(action_a, int(action))
            new_dist = abs(env.state.b_pos - trap)
            rewards.append(float(terminal) + .03 * (prev_dist - new_dist)); prev_dist = new_dist
            logs.append(log)
            incoming = trap  # frozen sender A's canonical token
            if done: break
        returns=[]; running=0.0
        for reward in reversed(rewards): running=reward + .97*running; returns.append(running)
        returns=torch.tensor(list(reversed(returns)))
        loss=sum(-log*ret for log,ret in zip(logs,returns))/max(1,len(logs))
        opt.zero_grad(); loss.backward(); opt.step()
    return receiver


def evaluate_receiver(receiver, length=5, horizon=8):
    good=0
    with torch.no_grad():
        for goal in range(length):
            for trap in range(length):
                if goal == trap: continue
                env=LineHunt(length,horizon,crossed=True); env.reset(goal,trap)
                h=receiver.initial_state(); incoming=None
                for step in range(horizon):
                    obs,_=line_observation(env,'b',length,incoming)
                    _,logits,h=receiver(obs,torch.nn.functional.one_hot(torch.tensor([incoming if incoming is not None else 0]),length).float(),h)
                    action=2 if step==0 else int(logits.argmax())
                    action_a=2 if step==0 else (1 if env.state.a_pos<goal else 0 if env.state.a_pos>goal else 2)
                    _,_,done,_=env.step(action_a,action); incoming=trap
                    if done: break
                good += int(env.state.success)
    return good / (length*(length-1))


def train_sender_with_frozen_receiver(receiver, seed=0, episodes=3000,
                                      length=5, horizon=8, lr=.01):
    """Train only A's message policy against a frozen receiver B."""
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    sender = GRULineAgent(vocab=length, hidden_dim=32)
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.Adam(sender.parameters(), lr=lr)
    for _ in range(episodes):
        goal, trap = random.sample(range(length), 2)
        env = LineHunt(length, horizon, crossed=True); env.reset(goal, trap)
        hs, hr = sender.initial_state(), receiver.initial_state()
        log = None; incoming_receiver = None
        for step in range(horizon):
            # A observes trap and samples the only learned token on step zero.
            oa, _ = line_observation(env, 'a', length, incoming_receiver)
            message_logits, _, hs = sender(oa, torch.zeros(1, length), hs)
            if step == 0:
                dist = Categorical(logits=message_logits)
                token = dist.sample(); log = dist.log_prob(token)
            else:
                token = torch.tensor(trap)
            # B receives A's token one step later and acts from frozen policy.
            ob, _ = line_observation(env, 'b', length, incoming_receiver)
            _, action_logits, hr = receiver(ob, torch.nn.functional.one_hot(
                torch.tensor([incoming_receiver if incoming_receiver is not None else 0]), length).float(), hr)
            action_b = 2 if step == 0 else int(action_logits.argmax())
            action_a = 2 if step == 0 else (1 if env.state.a_pos < goal else 0 if env.state.a_pos > goal else 2)
            _, reward, done, _ = env.step(action_a, action_b)
            incoming_receiver = int(token)
            if done:
                opt.zero_grad(); (-log * reward).backward(); opt.step()
                break
    return sender


def evaluate_staged(sender, receiver, length=5, horizon=8):
    """Evaluate argmax sender + frozen receiver with one-step delivery."""
    good = 0
    with torch.no_grad():
        for goal in range(length):
            for trap in range(length):
                if goal == trap: continue
                env = LineHunt(length, horizon, crossed=True); env.reset(goal, trap)
                hs, hr = sender.initial_state(), receiver.initial_state(); incoming=None
                for step in range(horizon):
                    oa,_=line_observation(env,'a',length,incoming)
                    ms,_,hs=sender(oa,torch.zeros(1,length),hs)
                    token=int(ms.argmax())
                    ob,_=line_observation(env,'b',length,incoming)
                    _,mb,hr=receiver(ob,torch.nn.functional.one_hot(torch.tensor([incoming if incoming is not None else 0]),length).float(),hr)
                    action_b=2 if step==0 else int(mb.argmax())
                    action_a=2 if step==0 else (1 if env.state.a_pos<goal else 0 if env.state.a_pos>goal else 2)
                    _,_,done,_=env.step(action_a,action_b); incoming=token
                    if done: break
                good += int(env.state.success)
    return good / (length*(length-1))
