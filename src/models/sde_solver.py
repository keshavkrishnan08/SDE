"""Euler-Maruyama SDE solver for generating sample paths from the Latent Neural SDE."""

import torch
from typing import Optional


def euler_maruyama_step(
    drift_fn,
    diffusion_fn,
    z_t: torch.Tensor,
    t: torch.Tensor,
    c_t: torch.Tensor,
    cti_t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Single Euler-Maruyama step.

    z_{t+dt} = z_t + μ(z_t, t, c_t) * dt + σ(z_t, CTI_t) * √dt * ε

    Args:
        drift_fn: Callable(z_t, t, c_t) -> drift vector.
        diffusion_fn: Callable(z_t, cti_t) -> diffusion vector.
        z_t: Current state, shape (B, d_z) or (B, N, d_z) for N sample paths.
        t: Current time, shape (B, 1) or scalar.
        c_t: Covariates, shape (B, d_c).
        cti_t: CTI, shape (B, 1).
        dt: Time step size.

    Returns:
        Next state z_{t+dt}, same shape as z_t.
    """
    mu = drift_fn(z_t, t, c_t)
    sigma = diffusion_fn(z_t, cti_t)
    noise = torch.randn_like(z_t)
    return z_t + mu * dt + sigma * (dt ** 0.5) * noise


def solve_sde(
    sde_model,
    z_0: torch.Tensor,
    t_span: torch.Tensor,
    c_t: torch.Tensor,
    cti_t: torch.Tensor,
    num_samples: int = 100,
    dt: float = 1.0,
    return_paths: bool = False,
) -> dict[str, torch.Tensor]:
    """Solve the Neural SDE forward in time using Euler-Maruyama.

    Args:
        sde_model: LatentNeuralSDE instance.
        z_0: Initial latent state, shape (B, d_z).
        t_span: Time points to solve at, shape (num_steps,).
        c_t: Covariates (assumed constant), shape (B, d_c).
        cti_t: CTI (assumed constant or slowly varying), shape (B, 1).
        num_samples: Number of Monte Carlo sample paths per initial condition.
        dt: Integration step size.
        return_paths: If True, return full trajectories; otherwise just endpoints.

    Returns:
        Dictionary with:
          - 'endpoints': shape (B, N, d_z) — final latent states for each sample path
          - 'paths': shape (B, N, T, d_z) — full trajectories (if return_paths=True)
    """
    B, d_z = z_0.shape
    device = z_0.device
    num_steps = len(t_span)

    # Replicate initial state for N sample paths: (B, d_z) -> (B*N, d_z)
    z = z_0.unsqueeze(1).expand(B, num_samples, d_z).reshape(B * num_samples, d_z)
    c_expanded = c_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
    cti_expanded = cti_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

    if return_paths:
        paths = [z.view(B, num_samples, d_z).clone()]

    for i in range(num_steps - 1):
        t_current = torch.full((B * num_samples, 1), t_span[i].item(), device=device)
        z = euler_maruyama_step(
            sde_model.drift,
            sde_model.diffusion,
            z, t_current, c_expanded, cti_expanded, dt,
        )
        if return_paths:
            paths.append(z.view(B, num_samples, d_z).clone())

    endpoints = z.view(B, num_samples, d_z)
    result = {"endpoints": endpoints}

    if return_paths:
        result["paths"] = torch.stack(paths, dim=2)  # (B, N, T, d_z)

    return result


def solve_sde_multihorizon(
    sde_model,
    z_0: torch.Tensor,
    horizons: list[int],
    c_t: torch.Tensor,
    cti_t: torch.Tensor,
    num_samples: int = 100,
    dt: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Solve SDE and collect endpoints at multiple forecast horizons.

    Args:
        sde_model: LatentNeuralSDE instance.
        z_0: Initial state, shape (B, d_z).
        horizons: List of steps at which to collect predictions.
        c_t: Covariates, shape (B, d_c).
        cti_t: CTI, shape (B, 1).
        num_samples: Number of MC sample paths.
        dt: Step size.

    Returns:
        Dict mapping horizon -> latent endpoints of shape (B, N, d_z).
    """
    B, d_z = z_0.shape
    device = z_0.device
    max_horizon = max(horizons)
    horizon_set = set(horizons)

    z = z_0.unsqueeze(1).expand(B, num_samples, d_z).reshape(B * num_samples, d_z)
    c_expanded = c_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
    cti_expanded = cti_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

    results = {}
    for step in range(max_horizon):
        t_current = torch.full((B * num_samples, 1), float(step), device=device)
        z = euler_maruyama_step(
            sde_model.drift,
            sde_model.diffusion,
            z, t_current, c_expanded, cti_expanded, dt,
        )
        if (step + 1) in horizon_set:
            results[step + 1] = z.view(B, num_samples, d_z).clone()

    return results
