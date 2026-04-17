"""Tests for Cloud-State VAE."""

import torch
import pytest

from src.models.cs_vae import CloudStateVAE, Encoder, Decoder


class TestEncoder:
    def test_output_shape(self):
        encoder = Encoder(latent_dim=64)
        x = torch.randn(2, 3, 256, 256)
        mu, logvar = encoder(x)
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_different_latent_dims(self):
        for d in [16, 32, 128]:
            encoder = Encoder(latent_dim=d)
            x = torch.randn(1, 3, 256, 256)
            mu, logvar = encoder(x)
            assert mu.shape == (1, d)


class TestDecoder:
    def test_output_shape(self):
        decoder = Decoder(latent_dim=64)
        z = torch.randn(2, 64)
        x_recon = decoder(z)
        assert x_recon.shape == (2, 3, 256, 256)

    def test_output_range(self):
        decoder = Decoder(latent_dim=64)
        z = torch.randn(2, 64)
        x_recon = decoder(z)
        assert x_recon.min() >= 0
        assert x_recon.max() <= 1


class TestCloudStateVAE:
    def test_forward_shapes(self):
        vae = CloudStateVAE(latent_dim=64)
        x = torch.randn(2, 3, 256, 256)
        recon, mu, logvar = vae(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_loss_computation(self):
        vae = CloudStateVAE(latent_dim=64, beta=0.1)
        x = torch.randn(2, 3, 256, 256)
        recon, mu, logvar = vae(x)
        losses = vae.loss(x, recon, mu, logvar)
        assert "loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        assert losses["loss"].item() > 0

    def test_encode_to_latent(self):
        vae = CloudStateVAE(latent_dim=32)
        x = torch.randn(4, 3, 256, 256)
        z = vae.encode_to_latent(x)
        assert z.shape == (4, 32)
