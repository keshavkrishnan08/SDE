"""CSDI baseline: Conditional Score-based Diffusion for time series (Tashiro et al., NeurIPS 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion step embedding."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.unsqueeze(-1) * self.emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CSDITransformerBlock(nn.Module):
    """Single transformer block for CSDI."""

    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CSDIScoreNet(nn.Module):
    """Score network for CSDI."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.diff_emb = DiffusionEmbedding(d_model)
        self.input_proj = nn.Linear(input_dim + 1, d_model)  # +1 for noisy target
        self.diff_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList([
            CSDITransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(
        self,
        x_cond: torch.Tensor,
        y_noisy: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_cond: Conditioning sequence, shape (B, seq_len, input_dim).
            y_noisy: Noised target, shape (B, 1).
            t_idx: Diffusion step index, shape (B,).

        Returns:
            Noise prediction, shape (B, 1).
        """
        B, S, D = x_cond.shape

        # Append noisy target to sequence as an extra timestep
        y_pad = torch.zeros(B, 1, D, device=x_cond.device)
        y_pad[:, 0, 0] = y_noisy.squeeze(-1)
        seq = torch.cat([x_cond, y_pad], dim=1)  # (B, S+1, D)

        # Add noisy target channel
        target_channel = torch.zeros(B, S + 1, 1, device=x_cond.device)
        target_channel[:, -1, 0] = y_noisy.squeeze(-1)
        seq_with_target = torch.cat([seq, target_channel], dim=-1)  # (B, S+1, D+1)

        h = self.input_proj(seq_with_target)

        # Add diffusion time embedding
        t_emb = self.diff_emb(t_idx.float())  # (B, d_model)
        h = h + self.diff_proj(t_emb).unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        # Take the last token's output as noise prediction
        return self.output_proj(h[:, -1, :])


class CSDI(nn.Module):
    """Full CSDI model for probabilistic forecasting."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        self.score_net = CSDIScoreNet(input_dim, d_model, nhead, num_layers, diffusion_steps)
        self.diffusion_steps = diffusion_steps

    def training_loss(self, x_cond: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        device = y.device
        t_idx = torch.randint(0, self.diffusion_steps, (B,), device=device)
        noise = torch.randn_like(y.unsqueeze(-1))

        sqrt_alpha = self.score_net.sqrt_alphas_cumprod[t_idx].unsqueeze(-1)
        sqrt_one_minus = self.score_net.sqrt_one_minus_alphas_cumprod[t_idx].unsqueeze(-1)
        y_noisy = sqrt_alpha * y.unsqueeze(-1) + sqrt_one_minus * noise

        noise_pred = self.score_net(x_cond, y_noisy, t_idx)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_probabilistic(
        self, x_cond: torch.Tensor, num_samples: int = 100
    ) -> torch.Tensor:
        self.eval()
        B = x_cond.shape[0]
        device = x_cond.device

        x_exp = x_cond.unsqueeze(1).expand(B, num_samples, -1, -1).reshape(B * num_samples, *x_cond.shape[1:])
        x_t = torch.randn(B * num_samples, 1, device=device)

        for i in reversed(range(self.diffusion_steps)):
            t_idx = torch.full((B * num_samples,), i, device=device, dtype=torch.long)
            noise_pred = self.score_net(x_exp, x_t, t_idx)

            beta = self.score_net.betas[i]
            alpha = self.score_net.alphas[i]
            alpha_bar = self.score_net.alphas_cumprod[i]

            x_t = (1 / alpha.sqrt()) * (x_t - beta / (1 - alpha_bar).sqrt() * noise_pred)
            if i > 0:
                x_t = x_t + beta.sqrt() * torch.randn_like(x_t)

        return x_t.squeeze(-1).view(B, num_samples)
