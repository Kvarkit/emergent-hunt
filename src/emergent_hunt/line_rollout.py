"""Trajectory collection for causal and grammar probes."""
import torch

from .line_hunt import LineHunt
from .line_policy import line_observation


def rollout_episode(agent_a, agent_b, goal_pos, trap_pos, length=5, horizon=8,
                    vocab=8, use_messages=True):
    """Collect a complete deterministic (argmax) episode without state leaks."""
    env = LineHunt(length, horizon)
    env.reset(goal_pos, trap_pos)
    ha, hb = agent_a.initial_state(), agent_b.initial_state()
    incoming_a = incoming_b = None
    trace = []
    for _ in range(horizon):
        oa, ia = line_observation(env, "a", vocab, incoming_a)
        ob, ib = line_observation(env, "b", vocab, incoming_b)
        ma, aa, ha = agent_a(oa, ia, ha)
        mb, ab, hb = agent_b(ob, ib, hb)
        token_a = int(ma.argmax(-1).item())
        token_b = int(mb.argmax(-1).item())
        action_a = int(aa.argmax(-1).item())
        action_b = int(ab.argmax(-1).item())
        trace.append({
            "step": env.state.step,
            "tokens": (token_a, token_b),
            "actions": (action_a, action_b),
            "factors": {"goal_position": goal_pos, "trap_position": trap_pos,
                        "relative_gap": trap_pos - goal_pos},
            "observations": {"a": oa.tolist()[0], "b": ob.tolist()[0]},
        })
        if use_messages:
            incoming_a, incoming_b = token_b, token_a
        else:
            incoming_a = incoming_b = None
        _, reward, done, info = env.step(action_a, action_b)
        trace[-1].update({"reward": reward, "done": done, "info": info})
        if done:
            break
    return trace


def collect_grid(agent_a, agent_b, length=5, horizon=8, vocab=8,
                 use_messages=True):
    """Collect trajectories for all distinct goal/trap positions."""
    records = []
    for goal in range(length):
        for trap in range(length):
            if goal == trap:
                continue
            records.extend(rollout_episode(agent_a, agent_b, goal, trap,
                                            length, horizon, vocab, use_messages))
    return records
