"""Full SolarSDE pipeline combining CS-VAE, Neural SDE, and Score Decoder."""

import torch
import torch.nn as nn
from typing import Optional

from src.models.cs_vae import CloudStateVAE
from src.models.cti import compute_cti_single
from src.models.neural_sde import LatentNeuralSDE
from src.models.score_decoder import ConditionalScoreDecoder
from src.models.sde_solver import solve_sde_multihorizon


class SolarSDE(nn.Module):
    """Full SolarSDE model for probabilistic solar irradiance nowcasting.

    Three-component architecture:
    1. CS-VAE: Encodes sky images → latent cloud state z_t
    2. Latent Neural SDE: Evolves z_t forward with CTI-conditioned diffusion
    3. Score Decoder (CSMID): Maps latent trajectories → irradiance distributions
    """

    def __init__(self, config: dict):
        super().__init__()
        vae_cfg = config["vae"]
        sde_cfg = config["sde"]
        score_cfg = config["score"]

        self.vae = CloudStateVAE(
            latent_dim=vae_cfg["latent_dim"],
            beta=vae_cfg["beta"],
            encoder_channels=vae_cfg.get("encoder_channels"),
        )

        self.sde = LatentNeuralSDE(
            latent_dim=vae_cfg["latent_dim"],
            covariate_dim=sde_cfg["covariate_dim"],
            drift_hidden=sde_cfg["drift_hidden"],
            diffusion_hidden=sde_cfg["diffusion_hidden"],
            lambda_sigma=sde_cfg["lambda_sigma"],
        )

        self.score_decoder = ConditionalScoreDecoder(
            latent_dim=vae_cfg["latent_dim"],
            covariate_dim=sde_cfg["covariate_dim"],
            hidden_dim=score_cfg["hidden_dim"],
            num_res_blocks=score_cfg["num_res_blocks"],
            diffusion_steps=score_cfg["diffusion_steps"],
            beta_start=score_cfg["beta_start"],
            beta_end=score_cfg["beta_end"],
        )

        self.latent_dim = vae_cfg["latent_dim"]
        self.cti_window = config["cti"]["window_size"]

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a sky image to latent mean (deterministic for inference)."""
        return self.vae.encode_to_latent(image)

    def compute_cti(self, z_window: torch.Tensor) -> torch.Tensor:
        """Compute CTI from a window of latent states.

        Args:
            z_window: Shape (B, W, d_z) or (W, d_z).

        Returns:
            CTI values, shape (B, 1) or (1,).
        """
        if z_window.dim() == 2:
            return compute_cti_single(z_window).unsqueeze(0)
        else:
            B = z_window.shape[0]
            ctis = []
            for b in range(B):
                ctis.append(compute_cti_single(z_window[b]))
            return torch.stack(ctis).unsqueeze(-1)

    @torch.no_grad()
    def forecast(
        self,
        images: torch.Tensor,
        covariates: torch.Tensor,
        horizons: list[int],
        num_samples: int = 100,
        dt: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        """Generate probabilistic forecasts at multiple horizons.

        Args:
            images: Past sky images, shape (B, W, 3, H, W) where W >= cti_window.
            covariates: Meteorological covariates, shape (B, d_c).
            horizons: List of forecast horizons (in timesteps).
            num_samples: Number of Monte Carlo sample paths.
            dt: SDE integration step size.

        Returns:
            Dict mapping horizon -> irradiance samples of shape (B, num_samples).
        """
        B = images.shape[0]
        W = images.shape[1]
        device = images.device

        # Encode all images in the window to latent space
        all_images = images.view(B * W, *images.shape[2:])
        all_z = self.vae.encode_to_latent(all_images)  # (B*W, d_z)
        z_window = all_z.view(B, W, self.latent_dim)

        # Current latent state = last in window
        z_0 = z_window[:, -1, :]  # (B, d_z)

        # Compute CTI from the latent window
        cti = self.compute_cti(z_window)  # (B, 1)

        # Solve SDE forward to each horizon
        z_horizons = solve_sde_multihorizon(
            self.sde, z_0, horizons, covariates, cti, num_samples, dt
        )

        # Decode each horizon's latent endpoints to irradiance distributions
        forecasts = {}
        for h, z_endpoints in z_horizons.items():
            # z_endpoints: (B, N, d_z) -> reshape for score decoder
            z_flat = z_endpoints.view(B * num_samples, -1)
            cti_flat = cti.unsqueeze(1).expand(B, num_samples, 1).reshape(B * num_samples, 1)
            c_flat = covariates.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

            # Sample irradiance from score decoder (1 sample per latent endpoint)
            ghi_samples = self.score_decoder.sample(z_flat, cti_flat, c_flat, num_samples=1)
            forecasts[h] = ghi_samples.squeeze(-1).view(B, num_samples)

        return forecasts

    def get_parameter_groups(self) -> list[dict]:
        """Return parameter groups for differential learning rates during fine-tuning."""
        return [
            {"params": self.vae.parameters(), "lr_scale": 0.1},
            {"params": self.sde.parameters(), "lr_scale": 1.0},
            {"params": self.score_decoder.parameters(), "lr_scale": 1.0},
        ]
