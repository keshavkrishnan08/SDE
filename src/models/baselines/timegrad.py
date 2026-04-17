"""TimeGrad baseline: autoregressive diffusion model for time series (Rasul et al., ICML 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeGradRNN(nn.Module):
    """RNN encoder for TimeGrad context."""

    def __init__(self, input_dim: int = 6, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return h_n[-1]  # (B, hidden_size)


class TimeGradDiffusion(nn.Module):
    """Simple DDPM decoder for TimeGrad, predicting one step at a time."""

    def __init__(self, context_dim: int = 64, hidden_dim: int = 128, diffusion_steps: int = 100):
        super().__init__()
        self.diffusion_steps = diffusion_steps

        # Noise prediction network
        self.net = nn.Sequential(
            nn.Linear(1 + 1 + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_noisy, t, context], dim=-1)
        return self.net(inp)

    def training_loss(self, x_0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        t_idx = torch.randint(0, self.diffusion_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].unsqueeze(-1)
        x_noisy = sqrt_alpha * x_0 + sqrt_one_minus * noise

        t_norm = t_idx.float().unsqueeze(-1) / self.diffusion_steps
        noise_pred = self.forward(x_noisy, t_norm, context)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        B = context.shape[0]
        ctx = context.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
        x = torch.randn(B * num_samples, 1, device=context.device)

        for i in reversed(range(self.diffusion_steps)):
            t = torch.full((B * num_samples, 1), i / self.diffusion_steps, device=x.device)
            noise_pred = self.forward(x, t, ctx)

            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]

            x = (1 / alpha.sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * noise_pred)
            if i > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return x.view(B, num_samples)


class TimeGrad(nn.Module):
    """Full TimeGrad model: RNN encoder + DDPM decoder."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        self.encoder = TimeGradRNN(input_dim, hidden_size, num_layers)
        self.decoder = TimeGradDiffusion(hidden_size, hidden_size * 2, diffusion_steps)

    def training_loss(self, x_seq: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        context = self.encoder(x_seq)
        return self.decoder.training_loss(y.unsqueeze(-1), context)

    @torch.no_grad()
    def predict_probabilistic(
        self, x_seq: torch.Tensor, num_samples: int = 100
    ) -> torch.Tensor:
        self.eval()
        context = self.encoder(x_seq)
        return self.decoder.sample(context, num_samples)
