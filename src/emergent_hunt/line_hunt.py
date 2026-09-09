"""Small deterministic dynamic environment for causal communication tests.

The existing symbolic benchmark is intentionally untouched.  LineHunt gives
two agents a real transition loop: observe -> exchange -> act -> transition.
Agent A privately knows the goal position; agent B privately knows the trap
position.  Success requires A to arrive at the goal while B triggers at the
trap on the same step.
"""
from dataclasses import dataclass
from typing import Dict, Tuple


LEFT, RIGHT, WAIT, TRIGGER = range(4)


@dataclass(frozen=True)
class LineHuntState:
    step: int
    a_pos: int
    b_pos: int
    goal_pos: int
    trap_pos: int
    terminal: bool = False
    success: bool = False


class LineHunt:
    def __init__(self, length: int = 5, horizon: int = 8, crossed: bool = False,
                 progress_weight: float = 0.0, step_cost: float = 0.0,
                 trigger_delay: int = 0):
        if length < 3 or horizon < 1:
            raise ValueError("length must be >=3 and horizon >=1")
        self.length = length
        self.horizon = horizon
        self.crossed = crossed
        if not 0 <= progress_weight < float('inf') or not 0 <= step_cost < float('inf'):
            raise ValueError('reward coefficients must be finite and nonnegative')
        if type(trigger_delay) is not int or trigger_delay < 0:
            raise ValueError('trigger_delay must be a nonnegative integer')
        self.progress_weight, self.step_cost = progress_weight, step_cost
        self.trigger_delay = trigger_delay
        self._state = None

    @property
    def state(self) -> LineHuntState:
        if self._state is None:
            raise RuntimeError("reset() must be called first")
        return self._state

    def reset(self, goal_pos: int, trap_pos: int, start_a: int = 0,
              start_b: int | None = None) -> Dict[str, int]:
        if start_b is None:
            start_b = self.length - 1
        for name, value in (("goal_pos", goal_pos), ("trap_pos", trap_pos),
                            ("start_a", start_a), ("start_b", start_b)):
            if not 0 <= value < self.length:
                raise ValueError(f"{name} outside line")
        if goal_pos == trap_pos:
            raise ValueError("goal and trap must be distinct")
        self._state = LineHuntState(0, start_a, start_b, goal_pos, trap_pos)
        return self.observe("a")

    def observe(self, agent: str) -> Dict[str, int]:
        s = self.state
        if agent == "a":
            key, value = (("partner_target", s.trap_pos) if self.crossed else
                          ("private_goal", s.goal_pos))
            return {"step": s.step, "self_pos": s.a_pos, key: value}
        if agent == "b":
            key, value = (("partner_target", s.goal_pos) if self.crossed else
                          ("private_trap", s.trap_pos))
            return {"step": s.step, "self_pos": s.b_pos, key: value}
        raise ValueError("agent must be 'a' or 'b'")

    def deliver(self, message_to_a: int | None, message_to_b: int | None
                ) -> Tuple[int | None, int | None]:
        """Return exactly what each agent receives; no hidden state is leaked."""
        return message_to_a, message_to_b

    def _move(self, pos: int, action: int) -> int:
        if action == LEFT:
            return max(0, pos - 1)
        if action == RIGHT:
            return min(self.length - 1, pos + 1)
        if action in (WAIT, TRIGGER):
            return pos
        raise ValueError("unknown action")

    def step(self, action_a: int, action_b: int) -> Tuple[Dict[str, int], float, bool, Dict[str, int]]:
        """Apply simultaneous actions and return observations, reward, done, info."""
        s = self.state
        if s.terminal:
            raise RuntimeError("cannot step a terminal episode")
        new_a = self._move(s.a_pos, action_a)
        new_b = self._move(s.b_pos, action_b)
        success = (new_a == s.goal_pos and new_b == s.trap_pos and
                   action_b == TRIGGER)
        wrong_trigger = action_b == TRIGGER and new_b != s.trap_pos
        elapsed = min(1 + self.trigger_delay * int(wrong_trigger),
                      self.horizon - s.step)
        terminal = success or s.step + elapsed >= self.horizon
        self._state = LineHuntState(s.step + elapsed, new_a, new_b, s.goal_pos,
                                    s.trap_pos, terminal, success)
        before_distance = abs(s.a_pos-s.goal_pos) + abs(s.b_pos-s.trap_pos)
        after_distance = abs(new_a-s.goal_pos) + abs(new_b-s.trap_pos)
        # Potential shaping for undiscounted episodic return (gamma=1).
        # Terminal potential is zero even at timeout, preventing agents from
        # accumulating progress reward by oscillating or stopping near a goal.
        potential_before = -self.progress_weight * before_distance
        potential_after = 0.0 if terminal else -self.progress_weight * after_distance
        shaping = potential_after - potential_before
        reward = float(success) - self.step_cost * elapsed + shaping
        info = {"a_pos": new_a, "b_pos": new_b, "success": int(success),
                "distance": after_distance, "distance_progress": before_distance-after_distance,
                "wrong_trigger": int(wrong_trigger), "elapsed": elapsed,
                "task_reward": float(success)-self.step_cost*elapsed,
                "shaping_reward": shaping}
        return {"a": self.observe("a"), "b": self.observe("b")}, reward, terminal, info


def oracle_rollout(goal_pos: int, trap_pos: int, length: int = 5,
                   horizon: int = 8) -> Tuple[LineHunt, list[dict]]:
    """Execute a deterministic valid plan for every solvable line instance."""
    env = LineHunt(length, horizon)
    env.reset(goal_pos, trap_pos)
    trace = []
    for _ in range(horizon):
        s = env.state
        a_action = RIGHT if s.a_pos < goal_pos else LEFT if s.a_pos > goal_pos else WAIT
        if s.b_pos == trap_pos:
            b_action = TRIGGER
        else:
            b_action = LEFT if s.b_pos > trap_pos else RIGHT
        obs, reward, done, info = env.step(a_action, b_action)
        trace.append({"step": s.step, "obs": obs, "actions": (a_action, b_action),
                      "reward": reward, "done": done, "info": info})
        if done:
            break
    if not env.state.success:
        raise AssertionError("oracle failed on a solvable line instance")
    return env, trace
