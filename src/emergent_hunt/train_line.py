"""Small policy-gradient trainer for dynamic LineHunt."""
import json
import random
from pathlib import Path

import torch
from torch.distributions import Categorical

from .line_hunt import LineHunt
from .line_policy import GRULineAgent, line_observation
from .line_rollout import rollout_episode


def train(seed=0, episodes=50000, length=5, horizon=8, vocab=8,
          hidden_dim=32, use_messages=True):
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    a, b = GRULineAgent(vocab=vocab, hidden_dim=hidden_dim), GRULineAgent(vocab=vocab, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=3e-3)
    history = []
    for ep in range(1, episodes + 1):
        goal, trap = random.sample(range(length), 2)
        env = LineHunt(length, horizon); env.reset(goal, trap)
        ha, hb = a.initial_state(), b.initial_state()
        incoming_a = incoming_b = None
        logs, rewards = [], []
        prev_da = abs(env.state.a_pos - goal); prev_db = abs(env.state.b_pos - trap)
        for _ in range(horizon):
            oa, ia = line_observation(env, "a", vocab, incoming_a)
            ob, ib = line_observation(env, "b", vocab, incoming_b)
            ma, aa, ha = a(oa, ia, ha); mb, ab, hb = b(ob, ib, hb)
            dm_a, da = Categorical(logits=ma), Categorical(logits=aa)
            dm_b, db = Categorical(logits=mb), Categorical(logits=ab)
            token_a, action_a = dm_a.sample(), da.sample()
            token_b, action_b = dm_b.sample(), db.sample()
            old_a, old_b = env.state.a_pos, env.state.b_pos
            _, terminal_reward, done, _ = env.step(int(action_a), int(action_b))
            progress = ((prev_da - abs(env.state.a_pos - goal)) +
                        (prev_db - abs(env.state.b_pos - trap))) * 0.03
            prev_da, prev_db = abs(env.state.a_pos - goal), abs(env.state.b_pos - trap)
            rewards.append(float(terminal_reward) + float(progress))
            logs.append(dm_a.log_prob(token_a) + da.log_prob(action_a) +
                        dm_b.log_prob(token_b) + db.log_prob(action_b))
            if use_messages:
                incoming_a, incoming_b = int(token_b), int(token_a)
            else:
                incoming_a = incoming_b = None
            if done:
                break
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + 0.97 * running; returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        loss = sum(-log * ret for log, ret in zip(logs, returns)) / max(1, len(logs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(a.parameters()) + list(b.parameters()), 5.0); opt.step()
        if ep == 1 or ep % 1000 == 0 or ep == episodes:
            history.append({"episode": ep, "return": sum(rewards), "loss": float(loss.detach())})
    return {"seed": seed, "episodes": episodes, "length": length,
            "horizon": horizon, "vocab": vocab, "hidden_dim": hidden_dim,
            "use_messages": use_messages, "history": history,
            "agents": (a, b)}


def save_checkpoint(result, path):
    a, b = result.pop("agents")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": result, "a": a.state_dict(), "b": b.state_dict()}, path)
