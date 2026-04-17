"""Cloud-State Variational Autoencoder (CS-VAE).

Encodes all-sky fisheye images into a low-dimensional latent manifold
capturing cloud morphology, motion, and optical thickness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder with GroupNorm and SiLU activations.

    Architecture:
        256x256x3 -> 128x128x32 -> 64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512
        -> AdaptiveAvgPool -> Flatten -> Linear -> (mu, log_var)
    """

    def __init__(self, latent_dim: int = 64, channels: list[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        layers = []
        in_ch = 3
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, ch), ch),
                nn.SiLU(inplace=True),
            ])
            in_ch = ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed convolutional decoder.

    Architecture:
        z -> Linear -> 512x8x8 -> 256x16x16 -> 128x32x32 -> 64x64x64
        -> 32x128x128 -> 3x256x256
    """

    def __init__(self, latent_dim: int = 64, channels: list[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [512, 256, 128, 64, 32]

        self.fc = nn.Linear(latent_dim, channels[0] * 8 * 8)
        self.init_channels = channels[0]

        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.SiLU(inplace=True),
            ])
        # Final layer: output 3 channels with Sigmoid
        layers.extend([
            nn.ConvTranspose2d(channels[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ])

        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.init_channels, 8, 8)
        return self.deconv(h)


class CloudStateVAE(nn.Module):
    """Cloud-State Variational Autoencoder.

    Learns a continuous low-dimensional representation of cloud morphology
    from fisheye sky images using a β-VAE objective.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        beta: float = 0.1,
        encoder_channels: list[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim, encoder_channels)
        # Decoder channels are reversed encoder channels
        if encoder_channels is not None:
            decoder_channels = list(reversed(encoder_channels))
        else:
            decoder_channels = None
        self.decoder = Decoder(latent_dim, decoder_channels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed image."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode.

        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute β-VAE loss: reconstruction + β * KL divergence."""
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent means (no sampling, for inference)."""
        mu, _ = self.encode(x)
        return mu
