"""Recurrent policies for the dynamic LineHunt environment."""
import torch
from torch import nn


class GRULineAgent(nn.Module):
    def __init__(self, obs_dim: int = 3, vocab: int = 8, hidden_dim: int = 32,
                 action_dim: int = 4):
        super().__init__()
        if min(obs_dim, vocab, hidden_dim, action_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(obs_dim + vocab, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, vocab)
        self.action = nn.Linear(hidden_dim, action_dim)

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, observation: torch.Tensor, incoming: torch.Tensor,
                hidden: torch.Tensor):
        if observation.ndim != 2 or incoming.ndim != 2 or hidden.ndim != 2:
            raise ValueError("observation, incoming and hidden must be rank-2")
        x = torch.tanh(self.encoder(torch.cat((observation, incoming), dim=-1)))
        next_hidden = self.gru(x, hidden)
        return self.message(next_hidden), self.action(next_hidden), next_hidden


def line_observation(env, agent: str, vocab: int, incoming: int | None = None):
    """Encode only the local/private view and the delivered token."""
    raw = env.observe(agent)
    if agent == "a":
        obs = [raw["self_pos"], raw["private_goal"], raw["step"]]
    elif agent == "b":
        obs = [raw["self_pos"], raw["private_trap"], raw["step"]]
    else:
        raise ValueError("agent must be 'a' or 'b'")
    token = torch.zeros(1, vocab)
    if incoming is not None:
        token[0, incoming] = 1.0
    return torch.tensor([obs], dtype=torch.float32), token
