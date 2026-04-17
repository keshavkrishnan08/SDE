"""Tests for Neural SDE components."""

import torch
import pytest

from src.models.neural_sde import DriftNetwork, CTIDiffusionNetwork, LatentNeuralSDE
from src.models.sde_solver import euler_maruyama_step, solve_sde


class TestDriftNetwork:
    def test_output_shape(self):
        drift = DriftNetwork(latent_dim=64, covariate_dim=5)
        z = torch.randn(4, 64)
        t = torch.zeros(4, 1)
        c = torch.randn(4, 5)
        out = drift(z, t, c)
        assert out.shape == (4, 64)


class TestCTIDiffusion:
    def test_output_shape(self):
        diff = CTIDiffusionNetwork(latent_dim=64, hidden_dim=64)
        z = torch.randn(4, 64)
        cti = torch.randn(4, 1).abs()
        out = diff(z, cti)
        assert out.shape == (4, 64)

    def test_output_positive(self):
        """Diffusion coefficients must always be positive (Softplus)."""
        diff = CTIDiffusionNetwork(latent_dim=64)
        z = torch.randn(8, 64)
        cti = torch.randn(8, 1)  # Can be negative input
        out = diff(z, cti)
        assert (out > 0).all(), "Diffusion must be strictly positive"

    def test_cti_sensitivity(self):
        """Higher CTI should generally produce larger diffusion."""
        diff = CTIDiffusionNetwork(latent_dim=32)
        z = torch.randn(1, 32)

        # Low CTI
        sigma_low = diff(z, torch.tensor([[0.01]]))
        # High CTI
        sigma_high = diff(z, torch.tensor([[10.0]]))

        # The mean diffusion should be higher for high CTI
        assert sigma_high.mean() > sigma_low.mean()


class TestLatentNeuralSDE:
    def test_forward(self):
        sde = LatentNeuralSDE(latent_dim=32, covariate_dim=3)
        z = torch.randn(4, 32)
        t = torch.zeros(4, 1)
        c = torch.randn(4, 3)
        cti = torch.rand(4, 1)
        mu, sigma = sde(z, t, c, cti)
        assert mu.shape == (4, 32)
        assert sigma.shape == (4, 32)
        assert (sigma > 0).all()

    def test_sde_matching_loss(self):
        sde = LatentNeuralSDE(latent_dim=16, covariate_dim=2)
        z_t = torch.randn(8, 16)
        z_next = z_t + torch.randn(8, 16) * 0.1
        t = torch.zeros(8, 1)
        c = torch.randn(8, 2)
        cti = torch.rand(8, 1)

        losses = sde.sde_matching_loss(z_t, z_next, t, c, cti)
        assert "loss" in losses
        assert "drift_loss" in losses
        assert "diffusion_loss" in losses
        assert losses["loss"].item() > 0

    def test_backward_pass(self):
        sde = LatentNeuralSDE(latent_dim=16, covariate_dim=2)
        z_t = torch.randn(4, 16)
        z_next = z_t + torch.randn(4, 16) * 0.1
        t = torch.zeros(4, 1)
        c = torch.randn(4, 2)
        cti = torch.rand(4, 1)

        losses = sde.sde_matching_loss(z_t, z_next, t, c, cti)
        losses["loss"].backward()
        for p in sde.parameters():
            assert p.grad is not None


class TestSDESolver:
    def test_euler_maruyama_step(self):
        sde = LatentNeuralSDE(latent_dim=16, covariate_dim=2)
        z = torch.randn(4, 16)
        t = torch.zeros(4, 1)
        c = torch.randn(4, 2)
        cti = torch.rand(4, 1)

        z_next = euler_maruyama_step(sde.drift, sde.diffusion, z, t, c, cti, dt=1.0)
        assert z_next.shape == z.shape

    def test_solve_sde_shapes(self):
        sde = LatentNeuralSDE(latent_dim=16, covariate_dim=2)
        z_0 = torch.randn(2, 16)
        t_span = torch.linspace(0, 1, 11)
        c = torch.randn(2, 2)
        cti = torch.rand(2, 1)

        result = solve_sde(sde, z_0, t_span, c, cti, num_samples=10)
        assert result["endpoints"].shape == (2, 10, 16)
