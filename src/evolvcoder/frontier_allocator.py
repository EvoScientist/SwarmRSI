"""Frontier-pressure discounted bandit over the two research-episode arms.

Each settled episode adds one discounted sample to the arm that initiated it.
The index is an engineering adaptation of Discounted-UCB with cost-normalised
reward and an additive cross-frontier pressure term; it inherits no regret
guarantee and its defaults are unverified starting values, not fitted optima.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from .brs_models import ARMS

NAME = "frontier_pressure_discounted_v1"
_EPS = 1e-9


def _finite(value: Any, name: str, *, minimum: float, maximum: float | None = None, integer: bool = False) -> None:
    ok = type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value >= minimum
    if ok and maximum is not None:
        ok = value <= maximum
    if not ok:
        raise ValueError(f"allocator {name} must be a {'positive integer' if integer else 'finite number'} within bounds")


class FrontierPressureAllocator:
    name = NAME

    def __init__(self, *, name: str = NAME, discount_factor: float = 0.9, exploration_weight: float = 0.5,
                 frontier_pressure_weight: float = 0.25, pressure_saturation: int = 4,
                 max_eligible_wait: int = 8, cost_unit_tokens: int = 1000, reward_scale: float = 1.0,
                 reward_shape: str = "clipped_improvement", reward_scale_mode: str = "fixed",
                 index_rule: str = "discounted", index_seed: int = 0) -> None:
        if name != NAME:
            raise ValueError(f"unsupported allocator {name!r}; only {NAME} is implemented")
        # Reward shape. "clipped_improvement" is the method document's clip((u-u_pre)/scale, 0, 1):
        # every regression scores 0, so at 5-10 evaluations per task 66% of samples were 0 and
        # a -0.69 collapse was indistinguishable from a -0.01 miss. "signed_gap" keeps the sign,
        # tanh((u-u_pre)/scale) in [-1, 1]. Scale mode "reference_gap" sets the scale per task to
        # the distance the search must close, |reference - root|, so a step is rewarded by the
        # fraction of that gap it closes; "fixed" keeps reward_scale.
        # Ablation switch for the index itself. "uniform_random" keeps every
        # deterministic guard below (only_feasible_arm, initialization_sample,
        # starvation_guard) and replaces only the ranked score with a coin flip, so
        # a cell measures the index and nothing else. The flip is derived from the
        # decision counter rather than an RNG object, so it survives checkpointing
        # without extra state and replays bit-identically.
        if index_rule not in {"discounted", "uniform_random"}:
            raise ValueError("index_rule must be discounted or uniform_random")
        if type(index_seed) is not int:
            raise ValueError("index_seed must be an integer")
        if reward_shape not in {"clipped_improvement", "signed_gap"}:
            raise ValueError("reward_shape must be clipped_improvement or signed_gap")
        if reward_scale_mode not in {"fixed", "reference_gap"}:
            raise ValueError("reward_scale_mode must be fixed or reference_gap")
        _finite(discount_factor, "discount_factor", minimum=_EPS, maximum=1.0)
        _finite(exploration_weight, "exploration_weight", minimum=0.0)
        _finite(frontier_pressure_weight, "frontier_pressure_weight", minimum=0.0)
        _finite(pressure_saturation, "pressure_saturation", minimum=1, integer=True)
        _finite(max_eligible_wait, "max_eligible_wait", minimum=1, integer=True)
        _finite(cost_unit_tokens, "cost_unit_tokens", minimum=1, integer=True)
        _finite(reward_scale, "reward_scale", minimum=_EPS)
        self.config = {"name": NAME, "discount_factor": float(discount_factor),
                       "exploration_weight": float(exploration_weight),
                       "frontier_pressure_weight": float(frontier_pressure_weight),
                       "pressure_saturation": int(pressure_saturation), "max_eligible_wait": int(max_eligible_wait),
                       "cost_unit_tokens": int(cost_unit_tokens), "reward_scale": float(reward_scale),
                       "reward_shape": reward_shape, "reward_scale_mode": reward_scale_mode,
                       "index_rule": index_rule, "index_seed": int(index_seed)}
        self.state = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict[str, Any]:
        return {"N": {a: 0.0 for a in ARMS}, "G": {a: 0.0 for a in ARMS}, "C": {a: 0.0 for a in ARMS},
                "samples": {a: 0 for a in ARMS}, "censored": {a: 0 for a in ARMS},
                "wait": {a: 0 for a in ARMS}, "clock": 0, "decisions": 0, "reference_gap": None}

    # --- reward and cost -------------------------------------------------
    def set_reference_gap(self, reference: float | None, root: float | None) -> float | None:
        """Bind the per-task scale: the distance from the root program to the reference.

        Recorded in state so a restart reproduces it. Never smaller than reward_scale,
        so a task whose root already beats the reference keeps a sane scale.
        """
        if self.config["reward_scale_mode"] != "reference_gap":
            return None
        if reference is None or root is None or not (math.isfinite(reference) and math.isfinite(root)):
            return None
        gap = max(abs(float(reference) - float(root)), self.config["reward_scale"])
        self.state["reference_gap"] = gap
        return gap

    def scale(self) -> float:
        gap = self.state.get("reference_gap")
        return float(gap) if (self.config["reward_scale_mode"] == "reference_gap" and gap) else self.config["reward_scale"]

    def reward(self, native_metric: float | None, incumbent_pre: float | None) -> float:
        """Improvement over the pre-episode incumbent, scaled; bootstrap earns nothing.

        clipped_improvement: clip(delta/scale, 0, 1), the method document's definition.
        signed_gap: tanh(delta/scale) in [-1, 1], so a collapse and a near miss differ.
        """
        if native_metric is None or incumbent_pre is None or not math.isfinite(native_metric):
            return 0.0
        delta = (native_metric - incumbent_pre) / self.scale()
        if self.config["reward_shape"] == "signed_gap":
            return math.tanh(delta)
        return min(1.0, max(0.0, delta))

    def cost(self, charged_tokens: int) -> float:
        return max(1.0, float(charged_tokens) / self.config["cost_unit_tokens"])

    def settle(self, arm: str, reward: float, cost: float) -> dict[str, Any]:
        if arm not in ARMS or not (-1.0 <= reward <= 1.0) or cost < 1.0:
            raise ValueError("settle needs a known arm, a reward in [-1, 1] and cost >= 1")
        rho = self.config["discount_factor"]
        for key in ("N", "G", "C"):
            for other in ARMS:
                self.state[key][other] *= rho
        self.state["N"][arm] += 1.0
        self.state["G"][arm] += reward
        self.state["C"][arm] += cost
        self.state["samples"][arm] += 1
        self.state["clock"] += 1
        return {"arm": arm, "reward": reward, "cost": cost, "clock": self.state["clock"]}

    def censor(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.state["censored"][arm] += 1

    # --- selection --------------------------------------------------------
    def rate(self, arm: str) -> float | None:
        n, c = self.state["N"][arm], self.state["C"][arm]
        if n <= _EPS or c <= _EPS:
            return None
        return self.state["G"][arm] / c

    def pressure(self, unmet: dict[str, int]) -> dict[str, float]:
        """Each side's pressure is its own saturated demand, not a share.

        A normalized share lets one weak item on an empty-versus-one split pin
        the other arm's pressure to its maximum, which is what diverted every
        scheduling opportunity away from an open local-refinement lane in the
        first Biology-5 revision. Saturating each side independently keeps a
        single unmet item worth 1/K of the term instead of all of it.
        """
        k = self.config["pressure_saturation"]
        v_h = min(max(0, int(unmet.get("H", 0))), k)
        v_p = min(max(0, int(unmet.get("P", 0))), k)
        return {"H": v_p / k, "P": v_h / k, "v_H": v_h, "v_P": v_p}

    def choose(self, feasible: set[str] | frozenset[str] | tuple[str, ...], unmet: dict[str, int]) -> dict[str, Any]:
        arms = tuple(a for a in ARMS if a in set(feasible))
        psi = self.pressure(unmet)
        decision: dict[str, Any] = {"allocator": NAME, "feasible": list(arms), "pressure": psi,
                                    "unmet": {a: int(unmet.get(a, 0)) for a in ARMS}, "index": {},
                                    "state_before": {k: dict(v) if isinstance(v, dict) else v for k, v in self.state.items()}}
        if not arms:
            decision.update(arm=None, reason="no_feasible_arm")
            return decision
        beta, gamma = self.config["exploration_weight"], self.config["frontier_pressure_weight"]
        total_n = sum(self.state["N"].values())
        for arm in arms:
            rate = self.rate(arm)
            n = self.state["N"][arm]
            explore = beta * math.sqrt(math.log(1.0 + total_n) / max(n, _EPS)) if n > _EPS else None
            decision["index"][arm] = {"rate": rate, "exploration": explore, "pressure": gamma * psi[arm],
                                      "samples": self.state["samples"][arm], "wait": self.state["wait"][arm],
                                      "score": (rate + explore + gamma * psi[arm]) if rate is not None and explore is not None else None}
        if len(arms) == 1:
            decision.update(arm=arms[0], reason="only_feasible_arm")
        else:
            uninitialised = [a for a in arms if self.state["samples"][a] == 0]
            starving = [a for a in arms if self.state["wait"][a] >= self.config["max_eligible_wait"]]
            if uninitialised:
                arm, reason = uninitialised[0], "initialization_sample"
            elif starving:
                arm, reason = starving[0], "starvation_guard"
            elif self.config["index_rule"] == "uniform_random":
                digest = hashlib.sha256(f"{self.config['index_seed']}:{self.state['decisions']}".encode()).digest()
                arm, reason = arms[digest[0] % len(arms)], "uniform_random_index"
            else:
                ranked = sorted(arms, key=lambda a: (decision["index"][a]["score"], -self.state["N"][a], ARMS.index(a) * -1), reverse=True)
                arm, reason = ranked[0], "discounted_index"
            for other in arms:
                self.state["wait"][other] = 0 if other == arm else self.state["wait"][other] + 1
            decision.update(arm=arm, reason=reason)
        self.state["decisions"] += 1
        return decision

    # --- persistence ------------------------------------------------------
    def checkpoint_state(self) -> dict[str, Any]:
        return {"config": dict(self.config), "state": {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.state.items()}}

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("config") != self.config:
            raise ValueError("allocator checkpoint configuration changed")
        state = payload.get("state")
        fresh = self._fresh_state()
        if not isinstance(state, dict) or set(state) != set(fresh):
            raise ValueError("allocator checkpoint state fields invalid")
        for key in ("N", "G", "C", "samples", "censored", "wait"):
            if set(state[key]) != set(ARMS):
                raise ValueError("allocator checkpoint arms invalid")
        self.state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()}
