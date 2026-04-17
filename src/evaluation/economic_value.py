"""Economic value experiment: quantify dollar value of improved probabilistic forecasts."""

import numpy as np


def simulate_reserve_costs(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    reserve_quantile: float = 0.95,
    reserve_cost_per_mwh: float = 50.0,
    penalty_per_mwh: float = 1000.0,
    decision_interval_minutes: int = 5,
    plant_capacity_mw: float = 1000.0,
    dt_seconds: int = 10,
) -> dict[str, float]:
    """Simulate grid operator reserve commitment decisions.

    The operator holds spinning reserve at the (1-α) quantile of the predictive distribution.
    Cost = reserve held × marginal cost + shortfall × penalty.

    Args:
        y_true: Actual GHI, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        reserve_quantile: Quantile for reserve commitment (e.g., 0.95).
        reserve_cost_per_mwh: Marginal reserve cost ($/MWh).
        penalty_per_mwh: Penalty for under-reserve ($/MWh).
        decision_interval_minutes: How often decisions are made.
        plant_capacity_mw: Solar plant capacity for scaling.
        dt_seconds: Data time resolution.

    Returns:
        Dict with 'total_cost', 'reserve_cost', 'penalty_cost', 'annual_cost'.
    """
    # Compute reserve level at each timestep (upper quantile of forecast)
    reserve_level = np.quantile(y_samples, reserve_quantile, axis=1)

    # Subsample to decision interval
    steps_per_decision = (decision_interval_minutes * 60) // dt_seconds
    decision_indices = np.arange(0, len(y_true), steps_per_decision)

    total_reserve_cost = 0.0
    total_penalty_cost = 0.0

    for idx in decision_indices:
        # Convert irradiance (W/m²) to power fraction of capacity
        predicted_max = reserve_level[idx] / 1000.0  # Normalize assuming ~1000 W/m² max
        actual = y_true[idx] / 1000.0

        # Reserve needed = predicted_max × capacity
        reserve_mw = predicted_max * plant_capacity_mw
        actual_mw = actual * plant_capacity_mw

        # Cost of holding reserve
        hours_per_interval = decision_interval_minutes / 60
        reserve_cost = reserve_mw * reserve_cost_per_mwh * hours_per_interval
        total_reserve_cost += reserve_cost

        # Penalty if actual exceeds reserve (shortfall)
        if actual_mw > reserve_mw:
            shortfall_mw = actual_mw - reserve_mw
            penalty = shortfall_mw * penalty_per_mwh * hours_per_interval
            total_penalty_cost += penalty

    total_cost = total_reserve_cost + total_penalty_cost

    # Extrapolate to annual (test period → full year)
    test_hours = len(y_true) * dt_seconds / 3600
    annual_scale = 365.25 * 12 / test_hours  # Assume ~12 daylight hours/day

    return {
        "total_cost": float(total_cost),
        "reserve_cost": float(total_reserve_cost),
        "penalty_cost": float(total_penalty_cost),
        "annual_cost": float(total_cost * annual_scale),
        "annual_cost_per_gw": float(total_cost * annual_scale / (plant_capacity_mw / 1000)),
    }


def compute_savings(
    cost_model: dict[str, float],
    cost_baseline: dict[str, float],
) -> dict[str, float]:
    """Compute cost savings of model vs baseline."""
    return {
        "total_savings": cost_baseline["total_cost"] - cost_model["total_cost"],
        "annual_savings": cost_baseline["annual_cost"] - cost_model["annual_cost"],
        "annual_savings_per_gw": cost_baseline["annual_cost_per_gw"] - cost_model["annual_cost_per_gw"],
        "savings_percent": (
            (cost_baseline["total_cost"] - cost_model["total_cost"])
            / cost_baseline["total_cost"]
            * 100
            if cost_baseline["total_cost"] > 0
            else 0.0
        ),
    }
