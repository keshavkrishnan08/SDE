"""Conditional Score-Matching Irradiance Decoder (CSMID).

Maps latent SDE trajectories to calibrated irradiance probability distributions
using a conditional denoising diffusion model in 1D (irradiance space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock1D(nn.Module):
    """Residual block for the score network."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ScoreNetwork(nn.Module):
    """Score network s_ω that estimates ∇_{GHI_s} log p_s(GHI_s | z_t, CTI_t, c_t).

    A small MLP (~50K params) since it operates in 1D irradiance space.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        covariate_dim: int = 5,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        # Input: GHI_s (1) + diffusion_time s (1) + z_t (d_z) + CTI (1) + covariates (d_c)
        input_dim = 1 + 1 + latent_dim + 1 + covariate_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(inplace=True)]
        for _ in range(num_res_blocks):
            layers.append(ResBlock1D(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        ghi_s: torch.Tensor,
        s: torch.Tensor,
        z_t: torch.Tensor,
        cti_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ghi_s: Noised irradiance, shape (B, 1).
            s: Diffusion time, shape (B, 1).
            z_t: Latent state, shape (B, d_z).
            cti_t: CTI, shape (B, 1).
            c_t: Covariates, shape (B, d_c).

        Returns:
            Estimated score, shape (B, 1).
        """
        x = torch.cat([ghi_s, s, z_t, cti_t, c_t], dim=-1)
        return self.net(x)


class ConditionalScoreDecoder(nn.Module):
    """Conditional Score-Matching Irradiance Decoder.

    Uses a linear β noise schedule and denoising score matching for training.
    Generates 1D irradiance distributions conditioned on latent state.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        covariate_dim: int = 5,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.score_net = ScoreNetwork(latent_dim, covariate_dim, hidden_dim, num_res_blocks)

        # Precompute noise schedule
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(
        self, ghi_0: torch.Tensor, s_idx: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: add noise at diffusion step s.

        GHI_s = √(ᾱ_s) * GHI_0 + √(1 - ᾱ_s) * ε
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[s_idx].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[s_idx].unsqueeze(-1)
        return sqrt_alpha * ghi_0 + sqrt_one_minus_alpha * noise

    def training_loss(
        self,
        ghi_0: torch.Tensor,
        z_t: torch.Tensor,
        cti_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute denoising score matching loss.

        L = E_{s, ε} [||s_ω(GHI_s, s, z_t, CTI, c_t) + ε / √(1-ᾱ_s)||²]
        """
        B = ghi_0.shape[0]
        device = ghi_0.device

        # Sample random diffusion timesteps
        s_idx = torch.randint(0, self.diffusion_steps, (B,), device=device)
        s_normalized = s_idx.float() / self.diffusion_steps  # Normalize to [0, 1]

        # Sample noise and create noised GHI
        noise = torch.randn_like(ghi_0)
        ghi_s = self.q_sample(ghi_0, s_idx, noise)

        # Predict score
        score_pred = self.score_net(ghi_s, s_normalized.unsqueeze(-1), z_t, cti_t, c_t)

        # Target: -ε / √(1 - ᾱ_s)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[s_idx].unsqueeze(-1)
        target = -noise / sqrt_one_minus_alpha

        loss = F.mse_loss(score_pred, target)
        return {"loss": loss}

    @torch.no_grad()
    def sample(
        self,
        z_t: torch.Tensor,
        cti_t: torch.Tensor,
        c_t: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate irradiance samples via reverse diffusion.

        Args:
            z_t: Latent state, shape (B, d_z).
            cti_t: CTI, shape (B, 1).
            c_t: Covariates, shape (B, d_c).
            num_samples: Number of samples per condition.

        Returns:
            Irradiance samples, shape (B, num_samples).
        """
        B = z_t.shape[0]
        device = z_t.device

        # Expand conditioning for num_samples
        z_exp = z_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
        cti_exp = cti_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
        c_exp = c_t.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

        # Start from noise
        x = torch.randn(B * num_samples, 1, device=device)

        # Reverse diffusion
        for i in reversed(range(self.diffusion_steps)):
            s_normalized = torch.full((B * num_samples, 1), i / self.diffusion_steps, device=device)
            score = self.score_net(x, s_normalized, z_exp, cti_exp, c_exp)

            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]

            # DDPM reverse step
            mean = (1.0 / alpha_t.sqrt()) * (
                x + beta_t * score * (1.0 - alpha_bar_t).sqrt()
            )

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        return x.view(B, num_samples)
