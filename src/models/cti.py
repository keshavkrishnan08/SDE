"""Cloud Turbulence Index (CTI) computation.

CTI is a scalar summary of "how turbulent is the cloud field right now" in latent space.
It is derived (not learned) from the temporal dynamics of CS-VAE latent representations.

CTI ≈ 0: Stable sky (clear or uniformly overcast) → low uncertainty
CTI >> 0: Rapidly changing cloud state → high uncertainty (cloud edges, broken clouds)
"""

import torch
import numpy as np


def compute_cti_single(z_window: torch.Tensor) -> torch.Tensor:
    """Compute CTI for a single window of latent states.

    Args:
        z_window: Latent states of shape (W, d_z) where W is the window size.

    Returns:
        Scalar CTI value.
    """
    # Compute latent velocities (finite differences)
    z_velocities = z_window[1:] - z_window[:-1]  # (W-1, d_z)

    # Variance of velocities across the window, per dimension
    velocity_var = z_velocities.var(dim=0)  # (d_z,)

    # L2 norm of the variance vector → scalar CTI
    cti = torch.norm(velocity_var, p=2)
    return cti


def compute_cti_batch(z_sequence: torch.Tensor, window_size: int = 10) -> torch.Tensor:
    """Compute CTI for each valid timestep in a batch of latent sequences.

    Args:
        z_sequence: Latent states of shape (T, d_z) or (B, T, d_z).
        window_size: Number of frames for the velocity variance window.

    Returns:
        CTI values of shape (T - window_size,) or (B, T - window_size).
    """
    if z_sequence.dim() == 2:
        T, d_z = z_sequence.shape
        cti_values = []
        for t in range(window_size, T):
            window = z_sequence[t - window_size : t]
            cti_values.append(compute_cti_single(window))
        return torch.stack(cti_values)

    elif z_sequence.dim() == 3:
        B, T, d_z = z_sequence.shape
        cti_batch = []
        for b in range(B):
            cti_values = []
            for t in range(window_size, T):
                window = z_sequence[b, t - window_size : t]
                cti_values.append(compute_cti_single(window))
            cti_batch.append(torch.stack(cti_values))
        return torch.stack(cti_batch)

    else:
        raise ValueError(f"Expected 2D or 3D input, got {z_sequence.dim()}D")


def compute_cti_from_numpy(
    latents: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """Compute CTI from a numpy array of latent representations.

    Args:
        latents: Shape (T, d_z) array of latent vectors.
        window_size: Window size for velocity variance computation.

    Returns:
        Shape (T,) array of CTI values. First `window_size` values are set to 0.
    """
    T, d_z = latents.shape
    cti = np.zeros(T, dtype=np.float32)

    for t in range(window_size, T):
        window = latents[t - window_size : t]
        velocities = np.diff(window, axis=0)  # (W-1, d_z)
        velocity_var = np.var(velocities, axis=0)  # (d_z,)
        cti[t] = np.linalg.norm(velocity_var, ord=2)

    return cti
