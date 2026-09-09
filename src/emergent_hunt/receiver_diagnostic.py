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
