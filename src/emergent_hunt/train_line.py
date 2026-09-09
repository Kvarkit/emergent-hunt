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
          hidden_dim=32, use_messages=True, crossed=False,
          critic=True, entropy_coef=0.01, canonical_bootstrap_episodes=0,
          sender_aux=0.0, lr=3e-3):
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    a, b = GRULineAgent(vocab=vocab, hidden_dim=hidden_dim), GRULineAgent(vocab=vocab, hidden_dim=hidden_dim)
    value_a = torch.nn.Linear(hidden_dim, 1)
    value_b = torch.nn.Linear(hidden_dim, 1)
    params = list(a.parameters()) + list(b.parameters())
    if critic:
        params += list(value_a.parameters()) + list(value_b.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for ep in range(1, episodes + 1):
        goal, trap = random.sample(range(length), 2)
        env = LineHunt(length, horizon, crossed=crossed); env.reset(goal, trap)
        ha, hb = a.initial_state(), b.initial_state()
        incoming_a = incoming_b = None
        action_logs, message_logs, values, rewards, sender_losses = [], [], [], [], []
        prev_da = abs(env.state.a_pos - goal); prev_db = abs(env.state.b_pos - trap)
        for _ in range(horizon):
            oa, ia = line_observation(env, "a", vocab, incoming_a)
            ob, ib = line_observation(env, "b", vocab, incoming_b)
            ma, aa, ha = a(oa, ia, ha); mb, ab, hb = b(ob, ib, hb)
            dm_a, da = Categorical(logits=ma), Categorical(logits=aa)
            dm_b, db = Categorical(logits=mb), Categorical(logits=ab)
            token_a, action_a = dm_a.sample(), da.sample()
            token_b, action_b = dm_b.sample(), db.sample()
            if crossed and ep <= canonical_bootstrap_episodes:
                token_a, token_b = torch.tensor(trap), torch.tensor(goal)
            old_a, old_b = env.state.a_pos, env.state.b_pos
            _, terminal_reward, done, _ = env.step(int(action_a), int(action_b))
            progress = ((prev_da - abs(env.state.a_pos - goal)) +
                        (prev_db - abs(env.state.b_pos - trap))) * 0.03
            prev_da, prev_db = abs(env.state.a_pos - goal), abs(env.state.b_pos - trap)
            rewards.append(float(terminal_reward) + float(progress))
            action_logs.append(da.log_prob(action_a) + db.log_prob(action_b))
            message_logs.append(dm_a.log_prob(token_a) + dm_b.log_prob(token_b))
            if crossed and sender_aux:
                sender_losses.append(torch.nn.functional.cross_entropy(ma, torch.tensor([trap])) +
                                     torch.nn.functional.cross_entropy(mb, torch.tensor([goal])))
            values.append((value_a(ha).squeeze(-1) + value_b(hb).squeeze(-1)) / 2)
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
        value_tensor = torch.cat(values) if values else torch.zeros(1)
        advantages = returns - value_tensor.detach() if critic else returns
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        policy_loss = sum(-log * adv for log, adv in zip(action_logs, advantages)) / max(1, len(action_logs))
        if use_messages and len(message_logs) > 1:
            # Message at t is causally delivered before action t+1.
            msg_adv = advantages[1:]
            message_loss = sum(-log * adv for log, adv in zip(message_logs[:-1], msg_adv)) / max(1, len(msg_adv))
        else:
            message_loss = torch.tensor(0.0)
        value_loss = ((value_tensor - returns) ** 2).mean() if critic else torch.tensor(0.0)
        decay = max(0.0, 1.0 - ep / episodes)
        sender_loss = (sum(sender_losses) / max(1, len(sender_losses))) if sender_losses else torch.tensor(0.0)
        entropy = torch.stack([da.entropy(), db.entropy()]).mean()
        loss = (policy_loss + message_loss + (value_loss if critic else 0.0) -
                entropy_coef * entropy + sender_aux * decay * sender_loss)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(a.parameters()) + list(b.parameters()), 5.0); opt.step()
        if ep == 1 or ep % 1000 == 0 or ep == episodes:
            history.append({"episode": ep, "return": sum(rewards), "loss": float(loss.detach())})
    return {"seed": seed, "episodes": episodes, "length": length,
            "horizon": horizon, "vocab": vocab, "hidden_dim": hidden_dim,
            "use_messages": use_messages, "history": history,
            "crossed": crossed,
            "critic": critic, "entropy_coef": entropy_coef,
            "canonical_bootstrap_episodes": canonical_bootstrap_episodes,
            "sender_aux": sender_aux,
            "lr": lr,
            "agents": (a, b)}


def save_checkpoint(result, path):
    a, b = result.pop("agents")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": result, "a": a.state_dict(), "b": b.state_dict()}, path)


def train_canonical_control(seed=0, episodes=3000, length=5, horizon=8,
                            vocab=8, hidden_dim=32):
    """Diagnostic: canonical target tokens, supervised physical action only."""
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    a, b = GRULineAgent(vocab=vocab, hidden_dim=hidden_dim), GRULineAgent(vocab=vocab, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=3e-3)
    for _ in range(episodes):
        goal, trap = random.sample(range(length), 2)
        env = LineHunt(length, horizon, crossed=True); env.reset(goal, trap)
        ha, hb = a.initial_state(), b.initial_state()
        for _ in range(horizon):
            # Each receiver is given the canonical token for its own hidden target.
            oa, _ = line_observation(env, "a", vocab, trap)
            ob, _ = line_observation(env, "b", vocab, goal)
            _, aa, ha = a(oa, torch.nn.functional.one_hot(torch.tensor([goal]), vocab).float(), ha)
            _, ab, hb = b(ob, torch.nn.functional.one_hot(torch.tensor([trap]), vocab).float(), hb)
            ta = 1 if env.state.a_pos < goal else 0 if env.state.a_pos > goal else 2
            tb = 1 if env.state.b_pos < trap else 0 if env.state.b_pos > trap else 3
            loss = torch.nn.functional.cross_entropy(aa, torch.tensor([ta])) + torch.nn.functional.cross_entropy(ab, torch.tensor([tb]))
            opt.zero_grad(); loss.backward(); opt.step()
            ha, hb = ha.detach(), hb.detach()
            _, _, done, _ = env.step(ta, tb)
            if done: break
    return a, b
