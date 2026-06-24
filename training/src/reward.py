"""Reward functions for Duck Hunt GRPO training."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .config import RewardConfig
from .utils import Action

logger = logging.getLogger(__name__)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass
class RewardBreakdown:
    """All components of a single reward computation."""
    total: float = 0.0
    base: float = 0.0             # hit/miss/double_kill/shoot_nothing/etc
    horizon_penalty: float = 0.0
    proximity_bonus: float = 0.0
    edge_bonus: float = 0.0       # bonus for hitting ducks away from screen center
    hitbox_center_bonus: float = 0.0  # bonus for hitting close to duck hitbox center
    min_distance: float = -1.0    # -1 = not computed
    outcome: str = ""             # "hit", "miss", "double_kill", "invalid", etc
    duck_a_state: str = ""
    duck_b_state: str = ""


def compute_reward(
    result: dict,
    action: Action | None,
    config: RewardConfig,
) -> float:
    """Compute the scalar reward. Use compute_reward_detailed for breakdown."""
    return compute_reward_detailed(result, action, config).total


def compute_reward_detailed(
    result: dict,
    action: Action | None,
    config: RewardConfig,
) -> RewardBreakdown:
    """Compute reward with full breakdown of all components."""
    bd = RewardBreakdown()

    # ---- unparseable output ----
    if action is None:
        bd.total = config.invalid_action
        bd.base = config.invalid_action
        bd.outcome = "invalid"
        return bd

    hit_a: bool = result.get("hit_a", False)
    hit_b: bool = result.get("hit_b", False)
    had_target: bool = result.get("had_target", False)
    had_target_at_shot: bool = result.get("had_target_at_shot", had_target)
    bd.duck_a_state = result.get("duck_a_state", "unknown")
    bd.duck_b_state = result.get("duck_b_state", "unknown")

    # ---- no ducks were flying at observation time ----
    # Should NOT happen: _collect_batch retries until a flying duck is found.
    # Safety fallback returns neutral reward.
    if not had_target:
        bd.total = 0.0
        bd.base = 0.0
        bd.outcome = "shoot_nothing"
        return bd

    # ---- ducks escaped/fell during latency+horizon (shot too late) ----
    # Should NOT happen: _collect_batch simulates forward and retries if ducks
    # would escape during latency. Safety fallback returns neutral reward.
    if not had_target_at_shot and not (hit_a or hit_b):
        bd.total = 0.0
        bd.base = 0.0
        bd.outcome = "shoot_dead"
        return bd

    # ---- base reward by outcome ----
    if hit_a and hit_b:
        bd.base = config.double_kill
        bd.outcome = "double_kill"
    elif hit_a or hit_b:
        bd.base = config.hit
        bd.outcome = "hit"
    else:
        bd.base = config.miss
        bd.outcome = "miss"

    # ---- horizon penalty ----
    horizon_ratio = action.horizon / max(config.max_horizon, 1)
    if hit_a or hit_b:
        bd.horizon_penalty = config.lambda_horizon * horizon_ratio
    else:
        bd.horizon_penalty = config.lambda_horizon_miss * horizon_ratio

    # ---- proximity bonus (distance to nearest duck hitbox center) ----
    # Works for BOTH hits and misses. Closer to hitbox center = higher bonus.
    # Hitbox center offset in normalized coords (half sprite size / screen size)
    hbox_cx_offset = 0.05  # ~40px / 800px
    hbox_cy_offset = 0.07  # ~35px / 500px

    if config.proximity_bonus > 0:
        shot_pos = result.get("shot_pos")
        duck_a_pos = result.get("duck_a_pos")
        duck_b_pos = result.get("duck_b_pos")
        duck_a_state = result.get("duck_a_state", "flying")
        duck_b_state = result.get("duck_b_state", "flying")

        if shot_pos:
            distances = []
            if duck_a_pos and duck_a_state == "flying":
                center_a = (duck_a_pos[0] + hbox_cx_offset, duck_a_pos[1] + hbox_cy_offset)
                distances.append(_distance(shot_pos, center_a))
            if duck_b_pos and duck_b_state == "flying":
                center_b = (duck_b_pos[0] + hbox_cx_offset, duck_b_pos[1] + hbox_cy_offset)
                distances.append(_distance(shot_pos, center_b))

            if distances:
                bd.min_distance = min(distances)
                bd.proximity_bonus = config.proximity_bonus * math.exp(
                    -config.proximity_decay * bd.min_distance
                )

    bd.total = bd.base - bd.horizon_penalty + bd.proximity_bonus
    return bd
