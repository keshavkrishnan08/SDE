"""Tests for Conditional Score-Matching Irradiance Decoder."""

import torch
import pytest

from src.models.score_decoder import ScoreNetwork, ConditionalScoreDecoder


class TestScoreNetwork:
    def test_output_shape(self):
        net = ScoreNetwork(latent_dim=32, covariate_dim=3)
        ghi_s = torch.randn(4, 1)
        s = torch.rand(4, 1)
        z = torch.randn(4, 32)
        cti = torch.rand(4, 1)
        c = torch.randn(4, 3)

        out = net(ghi_s, s, z, cti, c)
        assert out.shape == (4, 1)


class TestConditionalScoreDecoder:
    def test_training_loss(self):
        decoder = ConditionalScoreDecoder(latent_dim=32, covariate_dim=3, diffusion_steps=50)
        ghi = torch.randn(8, 1)
        z = torch.randn(8, 32)
        cti = torch.rand(8, 1)
        c = torch.randn(8, 3)

        losses = decoder.training_loss(ghi, z, cti, c)
        assert "loss" in losses
        assert losses["loss"].item() > 0

    def test_backward_pass(self):
        decoder = ConditionalScoreDecoder(latent_dim=16, covariate_dim=2, diffusion_steps=20)
        ghi = torch.randn(4, 1)
        z = torch.randn(4, 16)
        cti = torch.rand(4, 1)
        c = torch.randn(4, 2)

        losses = decoder.training_loss(ghi, z, cti, c)
        losses["loss"].backward()
        for p in decoder.parameters():
            assert p.grad is not None

    def test_sample_shape(self):
        decoder = ConditionalScoreDecoder(latent_dim=16, covariate_dim=2, diffusion_steps=10)
        z = torch.randn(3, 16)
        cti = torch.rand(3, 1)
        c = torch.randn(3, 2)

        samples = decoder.sample(z, cti, c, num_samples=20)
        assert samples.shape == (3, 20)

    def test_noise_schedule_monotone(self):
        decoder = ConditionalScoreDecoder(diffusion_steps=100)
        # alphas_cumprod should be monotonically decreasing
        ac = decoder.alphas_cumprod.numpy()
        assert all(ac[i] >= ac[i+1] for i in range(len(ac) - 1))
