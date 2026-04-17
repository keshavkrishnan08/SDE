"""Latent Neural SDE with CTI-conditioned diffusion.

dz_t = μ_θ(z_t, t, c_t) dt + σ_θ(z_t, CTI_t) dW_t

The diffusion coefficient is explicitly gated by the Cloud Turbulence Index (CTI),
encoding the physics that uncertainty comes from cloud turbulence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block with SiLU activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DriftNetwork(nn.Module):
    """Drift function μ_θ(z_t, t, c_t) for the Neural SDE.

    Takes the current latent state, physical time, and meteorological covariates.
    Returns the expected rate of change in latent space.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        covariate_dim: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = latent_dim + 1 + covariate_dim  # z_t + t + c_t

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, c_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_t: Latent state, shape (B, d_z).
            t: Scalar time, shape (B, 1).
            c_t: Covariates, shape (B, d_c).

        Returns:
            Drift vector, shape (B, d_z).
        """
        x = torch.cat([z_t, t, c_t], dim=-1)
        return self.net(x)


class CTIDiffusionNetwork(nn.Module):
    """CTI-conditioned diffusion function σ_θ(z_t, CTI_t).

    Core architectural innovation: the diffusion coefficient is gated by CTI.
    When CTI ≈ 0 (stable sky), diffusion → small. When CTI >> 0 (turbulent), diffusion → large.
    All outputs are positive (Softplus).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # CTI gate: maps scalar CTI to a gating vector
        self.cti_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
        )

        # State feature extractor
        self.state_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )

        # Output: element-wise gated features → diagonal diffusion
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # Ensure positive diffusion
        )

    def forward(self, z_t: torch.Tensor, cti_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Latent state, shape (B, d_z).
            cti_t: Cloud Turbulence Index, shape (B, 1).

        Returns:
            Diagonal diffusion coefficient, shape (B, d_z). All positive.
        """
        alpha = self.cti_gate(cti_t)         # (B, hidden_dim)
        h_z = self.state_net(z_t)             # (B, hidden_dim)
        gated = h_z * alpha                   # Element-wise gating
        return self.output(gated)             # (B, d_z), all positive


class LatentNeuralSDE(nn.Module):
    """Full Latent Neural SDE model.

    Combines drift and CTI-conditioned diffusion networks.
    Trained via SDE Matching (simulation-free).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        covariate_dim: int = 5,
        drift_hidden: int = 256,
        diffusion_hidden: int = 64,
        lambda_sigma: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_sigma = lambda_sigma

        self.drift = DriftNetwork(latent_dim, covariate_dim, drift_hidden)
        self.diffusion = CTIDiffusionNetwork(latent_dim, diffusion_hidden)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c_t: torch.Tensor,
        cti_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute drift and diffusion at the current state.

        Returns: (mu, sigma) each of shape (B, d_z).
        """
        mu = self.drift(z_t, t, c_t)
        sigma = self.diffusion(z_t, cti_t)
        return mu, sigma

    def sde_matching_loss(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
        t: torch.Tensor,
        c_t: torch.Tensor,
        cti_t: torch.Tensor,
        dt: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute SDE Matching loss (Bartosh et al., ICLR 2025).

        Simulation-free training:
        - Drift matching: fit μ_θ to observed finite differences
        - Diffusion matching: fit σ_θ² to residual variance

        Args:
            z_t: Current latent state, shape (B, d_z).
            z_next: Next latent state, shape (B, d_z).
            t: Time, shape (B, 1).
            c_t: Covariates, shape (B, d_c).
            cti_t: CTI, shape (B, 1).
            dt: Time step size (normalized).

        Returns:
            Dictionary with loss components.
        """
        mu, sigma = self.forward(z_t, t, c_t, cti_t)

        # Observed finite difference (target for drift)
        dz_observed = (z_next - z_t) / dt

        # Drift matching loss
        drift_loss = F.mse_loss(mu, dz_observed)

        # Residual after drift (should be explained by diffusion)
        residual = z_next - z_t - mu * dt
        residual_sq = residual.pow(2) / dt  # Empirical instantaneous variance

        # Diffusion matching loss
        sigma_sq = sigma.pow(2)
        diffusion_loss = F.mse_loss(sigma_sq, residual_sq)

        total = drift_loss + self.lambda_sigma * diffusion_loss

        return {
            "loss": total,
            "drift_loss": drift_loss,
            "diffusion_loss": diffusion_loss,
        }
