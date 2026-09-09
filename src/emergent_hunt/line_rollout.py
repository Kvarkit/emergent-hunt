"""Trajectory collection for causal and grammar probes."""
import torch

from .line_hunt import LineHunt
from .line_policy import line_observation


def rollout_episode(agent_a, agent_b, goal_pos, trap_pos, length=5, horizon=8,
                    vocab=8, use_messages=True, crossed=False,
                    message_mode="actual"):
    """Collect a complete deterministic (argmax) episode without state leaks."""
    env = LineHunt(length, horizon, crossed=crossed)
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
        if not use_messages or message_mode == "zero":
            incoming_a = incoming_b = None
        elif message_mode == "block_a_to_b":
            incoming_a, incoming_b = token_b, None
        elif message_mode == "block_b_to_a":
            incoming_a, incoming_b = None, token_a
        elif message_mode == "actual":
            incoming_a, incoming_b = token_b, token_a
        else:
            raise ValueError("unknown message_mode")
        _, reward, done, info = env.step(action_a, action_b)
        trace[-1].update({"reward": reward, "done": done, "info": info})
        if done:
            break
    return trace


def collect_grid(agent_a, agent_b, length=5, horizon=8, vocab=8,
                 use_messages=True, crossed=False, message_mode="actual"):
    """Collect trajectories for all distinct goal/trap positions."""
    records = []
    for goal in range(length):
        for trap in range(length):
            if goal == trap:
                continue
            records.extend(rollout_episode(agent_a, agent_b, goal, trap,
                                            length, horizon, vocab, use_messages, crossed,
                                            message_mode))
    return records


def canonical_crossed_rollout(goal_pos, trap_pos, length=5, horizon=8):
    """Upper-bound policy: each agent sends its observed partner target."""
    env = LineHunt(length, horizon, crossed=True)
    env.reset(goal_pos, trap_pos)
    incoming_a = incoming_b = None
    trace = []
    for _ in range(horizon):
        oa, ob = env.observe("a"), env.observe("b")
        token_a, token_b = oa["partner_target"], ob["partner_target"]
        def move(pos, target, trigger=False):
            if pos < target: return 1
            if pos > target: return 0
            return 3 if trigger else 2
        action_a = move(env.state.a_pos, incoming_a, False) if incoming_a is not None else 2
        action_b = move(env.state.b_pos, incoming_b, True) if incoming_b is not None else 2
        trace.append({"step": env.state.step, "tokens": (token_a, token_b),
                      "actions": (action_a, action_b)})
        incoming_a, incoming_b = token_b, token_a
        _, reward, done, info = env.step(action_a, action_b)
        trace[-1].update({"reward": reward, "done": done, "info": info})
        if done: break
    return env, trace
