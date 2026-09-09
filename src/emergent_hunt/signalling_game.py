"""Tiny exact signalling game used to validate communication gradients."""
import torch
from torch import nn


def expected_reward(sender_logits: torch.Tensor, receiver_logits: torch.Tensor) -> torch.Tensor:
    """Exact E[1[prediction == x]] for binary x, messages and predictions."""
    ps = sender_logits.softmax(-1)       # [x, message]
    pr = receiver_logits.softmax(-1)     # [message, prediction]
    reward = torch.zeros((), dtype=ps.dtype)
    for x in range(ps.shape[0]):
        for message in range(ps.shape[1]):
            reward = reward + .5 * ps[x, message] * pr[message, x]
    return reward


def evaluate(sender_logits, receiver_logits, mode="actual"):
    """Exact accuracy under actual, zero-message, or shuffled-message channels."""
    ps = sender_logits.softmax(-1); pr = receiver_logits.softmax(-1)
    correct = 0.0
    for x in range(2):
        for m in range(2):
            delivered = m if mode == "actual" else 0
            if mode == "shuffled":
                delivered = 1 - m
            correct += .5 * ps[x, m] * pr[delivered, x]
    return float(correct)


def train_exact(seed=0, steps=1000, lr=.2):
    torch.manual_seed(seed)
    sender = nn.Parameter(torch.randn(2, 2) * .1)
    receiver = nn.Parameter(torch.randn(2, 2) * .1)
    opt = torch.optim.Adam([sender, receiver], lr=lr)
    history = []
    for step in range(1, steps + 1):
        loss = -expected_reward(sender, receiver)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "reward": float(-loss.detach())})
    return {"sender": sender.detach(), "receiver": receiver.detach(), "history": history}
