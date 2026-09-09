"""Isolate sender learning from navigation and receiver optimisation."""
import torch
from torch import nn
from torch.distributions import Categorical


def exact_sender_objective(logits):
    """Expected coordinate-delivery reward over all trap positions."""
    probs = logits.softmax(-1)
    return probs.diagonal().mean()


def train_sender_exact(seed=0, steps=1000, length=5, lr=.2):
    torch.manual_seed(seed)
    logits = nn.Parameter(torch.randn(length, length) * .1)
    opt = torch.optim.Adam([logits], lr=lr)
    history = []
    for step in range(1, steps + 1):
        loss = -exact_sender_objective(logits)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({'step': step, 'reward': float(-loss.detach())})
    return logits.detach(), history


def train_sender_sampled(seed=0, updates=2000, length=5, lr=.05):
    """REINFORCE sender-only control with fixed one-step delivery."""
    torch.manual_seed(seed)
    logits = nn.Parameter(torch.zeros(length, length))
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(updates):
        trap = torch.randint(length, ()).item()
        dist = Categorical(logits=logits[trap])
        token = dist.sample()
        reward = (token == trap).float()
        loss = -dist.log_prob(token) * reward
        opt.zero_grad(); loss.backward(); opt.step()
    return logits.detach()


def evaluate(logits):
    return float(logits.softmax(-1).argmax(-1).eq(torch.arange(logits.shape[0])).float().mean())
