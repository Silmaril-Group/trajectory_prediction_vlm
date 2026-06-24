"""Custom Scorers for Duck Hunt VLM Evaluation."""

import weave


@weave.op
def accuracy_scorer(output: dict, expected_hit: bool = True) -> dict:
    """
    Score based on whether the shot hit.

    Args:
        output: The prediction output
        expected_hit: Whether we expected a hit (for evaluation scenarios)

    Returns:
        Dict with accuracy score
    """
    result = output.get("result", "miss")
    hit = result in ["hit", "double_kill"]

    if expected_hit:
        return {"accuracy": 1.0 if hit else 0.0}
    else:
        return {"accuracy": 1.0 if not hit else 0.0}


@weave.op
def hit_type_scorer(output: dict) -> dict:
    """
    Score based on hit type.

    Returns:
        Dict with hit_type categorization
    """
    result = output.get("result", "miss")

    scores = {
        "is_hit": 1.0 if result == "hit" else 0.0,
        "is_double_kill": 1.0 if result == "double_kill" else 0.0,
        "is_miss": 1.0 if result == "miss" else 0.0,
        "is_no_target": 1.0 if result == "no_target" else 0.0,
    }

    return scores


@weave.op
def horizon_efficiency_scorer(output: dict) -> dict:
    """
    Score horizon efficiency (lower horizon = better efficiency).

    Horizon penalty exists in the game, so lower horizon is preferred
    when accuracy is maintained.

    Returns:
        Dict with horizon_efficiency score (0-1, higher is better)
    """
    horizon = output.get("horizon", 0)
    max_horizon = 30

    # Efficiency: 0 horizon = 1.0, 30 horizon = 0.0
    efficiency = 1.0 - (horizon / max_horizon)

    return {"horizon_efficiency": efficiency}


@weave.op
def confidence_calibration_scorer(output: dict) -> dict:
    """
    Score how well confidence matches actual result.

    High confidence + hit = good calibration
    High confidence + miss = overconfident
    Low confidence + hit = underconfident

    Returns:
        Dict with calibration metrics
    """
    result = output.get("result", "miss")
    confidence = output.get("confidence", "medium")
    hit = result in ["hit", "double_kill"]

    # Map confidence to numeric
    conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    conf_value = conf_map.get(confidence, 0.5)

    # Calibration error: |confidence - actual_outcome|
    actual = 1.0 if hit else 0.0
    calibration_error = abs(conf_value - actual)

    # Well calibrated = low error
    calibration_score = 1.0 - calibration_error

    return {
        "calibration_score": calibration_score,
        "calibration_error": calibration_error,
        "confidence_level": confidence,
        "was_hit": hit,
    }


@weave.op
def position_validity_scorer(output: dict) -> dict:
    """
    Score whether predicted position is in valid range.

    Returns:
        Dict with position validity scores
    """
    x = output.get("x", 0)
    y = output.get("y", 0)

    # Valid ranges
    x_valid = 0 <= x <= 800
    y_valid = 0 <= y <= 500

    # Ducks fly in upper half, so y should typically be < 250
    y_in_duck_zone = y <= 300  # Some margin

    return {
        "x_valid": 1.0 if x_valid else 0.0,
        "y_valid": 1.0 if y_valid else 0.0,
        "in_duck_zone": 1.0 if y_in_duck_zone else 0.0,
        "position_valid": 1.0 if (x_valid and y_valid) else 0.0,
    }


@weave.op
def reward_scorer(output: dict) -> dict:
    """
    Return the actual game reward as a score.

    Returns:
        Dict with reward
    """
    return {"game_reward": output.get("reward", 0.0)}


@weave.op
def combined_scorer(output: dict) -> dict:
    """
    Combine multiple scoring metrics into one.

    Returns:
        Dict with all combined scores
    """
    # Get individual scores
    hit_scores = hit_type_scorer(output)
    horizon_scores = horizon_efficiency_scorer(output)
    calibration_scores = confidence_calibration_scorer(output)
    position_scores = position_validity_scorer(output)
    reward_scores = reward_scorer(output)

    # Combine all
    combined = {}
    combined.update(hit_scores)
    combined.update(horizon_scores)
    combined.update(calibration_scores)
    combined.update(position_scores)
    combined.update(reward_scores)

    # Overall score: weighted combination
    overall = (
        hit_scores["is_hit"] * 0.4 +
        hit_scores["is_double_kill"] * 0.2 +
        horizon_scores["horizon_efficiency"] * 0.2 +
        calibration_scores["calibration_score"] * 0.1 +
        position_scores["in_duck_zone"] * 0.1
    )
    combined["overall_score"] = overall

    return combined


# List of all scorers for evaluation
ALL_SCORERS = [
    accuracy_scorer,
    hit_type_scorer,
    horizon_efficiency_scorer,
    confidence_calibration_scorer,
    position_validity_scorer,
    reward_scorer,
]
