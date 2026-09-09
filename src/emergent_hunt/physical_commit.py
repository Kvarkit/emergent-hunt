"""One-step physical commit benchmark for the communication pipeline."""
import torch
from torch import nn


def expected_commit_reward(sender_logits, receiver_action_logits):
    """Exact reward for hidden target x, one message, one physical commit."""
    ps = sender_logits.softmax(-1)       # [target, token]
    pa = receiver_action_logits.softmax(-1)  # [token, action/target]
    return sum(.5 * ps[x, m] * pa[m, x]
               for x in range(ps.shape[0]) for m in range(ps.shape[1]))


def train_exact(seed=0, steps=1000, lr=.2):
    torch.manual_seed(seed)
    sender = nn.Parameter(torch.randn(2, 2) * .1)
    receiver = nn.Parameter(torch.randn(2, 2) * .1)
    opt = torch.optim.Adam([sender, receiver], lr=lr)
    history = []
    for step in range(1, steps + 1):
        loss = -expected_commit_reward(sender, receiver)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "reward": float(-loss.detach())})
    return sender.detach(), receiver.detach(), history


def evaluate(sender_logits, receiver_logits, mode="actual"):
    ps = sender_logits.softmax(-1); pa = receiver_logits.softmax(-1)
    score = 0.0
    for x in range(2):
        for m in range(2):
            delivered = m if mode == "actual" else (1 - m if mode == "shuffled" else 0)
            score += .5 * ps[x, m] * pa[delivered, x]
    return float(score)
