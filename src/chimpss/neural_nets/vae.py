"""
Variational AutoEncoder models in JAX/Flax.

Provides:
  BatchNorm_VAE  — dense (fully-connected) VAE, ported from Deep-MMS.
                   Symmetric encoder/decoder with BatchNorm + Dropout.

  WaveTransformVAE — 3D convolutional VAE for wave-transformed molecular
                     grids (Kuzminykh et al. 2018). Uses Xception-style
                     depthwise-separable Conv3D blocks. Default architecture
                     matches Table 1 of the paper (384-dim latent).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def reparameterize(
    z_rng: jax.Array,
    z_mean: jax.Array,
    z_logvar: jax.Array,
) -> jax.Array:
    """Sample z ~ N(z_mean, exp(z_logvar)) via the reparameterisation trick."""
    z_std = jnp.exp(0.5 * z_logvar)
    eps = jax.random.normal(z_rng, z_logvar.shape)
    return z_mean + eps * z_std


# ---------------------------------------------------------------------------
# Dense (fully-connected) VAE  — ported from Deep-MMS/pyscripts/NN_models.py
# ---------------------------------------------------------------------------

class BVEncoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list

    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(len(self.d_hidden)):
            x = nn.Dense(self.d_hidden[i])(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=not train)
        mean_x = nn.Dense(self.latents, name='fc_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc_logvar')(x)
        return mean_x, logvar_x


class BVDecoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list

    @nn.compact
    def __call__(self, z, train: bool):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.Dense(self.d_hidden[i])(z)
            z = nn.leaky_relu(z, negative_slope=0.2)
            z = nn.BatchNorm(use_running_average=not train)(z)
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=not train)
        z = nn.Dense(self.out_dim, name='fc_out')(z)
        return z


class BatchNorm_VAE(nn.Module):
    """
    Symmetric dense VAE with BatchNorm and Dropout.

    Hidden layers are typically square (input_size × input_size Dense blocks)
    so every layer except the latent projection is a square matrix transform.
    Architecture is configured via JSON-style parameters (see Deep-MMS usage).

    Args:
        input_size: dimensionality of the input/output vector.
        hidden_layers: tuple of ints, hidden layer widths (symmetric encoder=decoder).
        latents: latent space dimension.
        dropout_rates: dropout rate per hidden layer.
    """
    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list

    def setup(self):
        self.encoder = BVEncoder(
            list(self.hidden_layers), self.latents, self.dropout_rates
        )
        self.decoder = BVDecoder(
            list(self.hidden_layers), self.input_size, self.dropout_rates
        )

    def __call__(self, x, z_rng, train: bool):
        z_mean, z_logvar = self.encoder(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train), z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train)

    def encode(self, x, z_rng=None, train=False):
        return self.encoder(x, train=train)

    def decode(self, z, z_rng=None, train=False):
        return self.decoder(z, train=train)


# ---------------------------------------------------------------------------
# 3D Convolutional VAE — Kuzminykh et al. 2018
# ---------------------------------------------------------------------------

class _DWSConv3DBlock(nn.Module):
    """
    Depthwise-separable 3D convolution block (Xception-style).

    Depthwise: spatial filtering per channel (kernel_size³ params per channel).
    Pointwise: 1×1×1 cross-channel mixing → target features.
    Followed by ReLU + BatchNorm.
    """
    features: int
    kernel_size: int = 7
    strides: Tuple[int, int, int] = (1, 1, 1)

    @nn.compact
    def __call__(self, x, train: bool):
        in_features = x.shape[-1]
        # Depthwise
        x = nn.Conv(
            features=in_features,
            kernel_size=(self.kernel_size,) * 3,
            strides=self.strides,
            feature_group_count=in_features,
            padding='SAME',
            use_bias=False,
        )(x)
        # Pointwise
        x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1, 1),
            padding='SAME',
        )(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class _DWSConvTranspose3DBlock(nn.Module):
    """
    Depthwise-separable 3D transposed convolution block for the decoder.

    Pointwise: 1×1×1 cross-channel projection first.
    Depthwise-transposed: spatial upsampling per channel.
    Followed by ReLU + BatchNorm (skipped for the final output layer).
    """
    features: int
    kernel_size: int = 7
    strides: Tuple[int, int, int] = (1, 1, 1)
    activate: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        # Pointwise projection
        x = nn.Conv(features=self.features, kernel_size=(1, 1, 1), padding='SAME')(x)
        # Transposed depthwise upsampling
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(self.kernel_size,) * 3,
            strides=self.strides,
            padding='SAME',
        )(x)
        if self.activate:
            x = nn.relu(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class Conv3DEncoder(nn.Module):
    """
    3D convolutional encoder for wave-transformed molecular grids.

    Architecture (Table 1, Kuzminykh et al. 2018):
      Stage 1 : 1  DWS block,  64 ch,  kernel 7, stride 2
      Stage 2 : 1  DWS block, 128 ch,  kernel 7, stride 2
      Stage 3 : 8  DWS blocks,256 ch,  kernel 7, stride 2 (stride on last block)
      Stage 4 : 8  DWS blocks,512 ch,  kernel 7, stride 2 (stride on last block)
      Stage 5 : 1  DWS block, 128 ch,  kernel 1, stride 1
      Flatten → Dense(dense_hidden, relu) → Dense(latents) × 2 (mean, logvar)

    Args:
        latents: latent dimension (paper: 384).
        channel_schedule: output channels per stage.
        n_blocks: number of DWS blocks per stage.
        kernel_schedule: kernel size per stage.
        dense_hidden: width of the pre-latent dense layer.
    """
    latents: int = 384
    channel_schedule: Sequence[int] = (64, 128, 256, 512, 128)
    n_blocks: Sequence[int] = (1, 1, 8, 8, 1)
    kernel_schedule: Sequence[int] = (7, 7, 7, 7, 1)
    dense_hidden: int = 1024

    @nn.compact
    def __call__(self, x, train: bool):
        # x: (batch, X, Y, Z, C)
        strides_schedule = [(2,2,2), (2,2,2), (2,2,2), (2,2,2), (1,1,1)]

        for ch, n, ks, strides in zip(
            self.channel_schedule,
            self.n_blocks,
            self.kernel_schedule,
            strides_schedule,
        ):
            for block_i in range(n):
                # Apply stride only on the last block of each stage
                s = strides if block_i == n - 1 else (1, 1, 1)
                x = _DWSConv3DBlock(features=ch, kernel_size=ks, strides=s)(x, train)

        # Flatten spatial dims
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.dense_hidden)(x)
        x = nn.relu(x)
        z_mean = nn.Dense(self.latents, name='fc_mean')(x)
        z_logvar = nn.Dense(self.latents, name='fc_logvar')(x)
        return z_mean, z_logvar


class Conv3DDecoder(nn.Module):
    """
    3D convolutional decoder (mirror of Conv3DEncoder).

    Args:
        out_channels: number of atom-type output channels.
        bottleneck_shape: (X, Y, Z) spatial dims at encoder bottleneck.
                          Default (4, 4, 2) matches a (64, 64, 32) input grid.
        bottleneck_channels: channels at the bottleneck (encoder stage 5 output).
        channel_schedule: output channels per decoder stage (reverse of encoder).
        n_blocks: DWS blocks per stage.
        kernel_schedule: kernel sizes per stage.
        dense_hidden: dense layer width.
    """
    out_channels: int = 9
    bottleneck_shape: Tuple[int, int, int] = (4, 4, 2)
    bottleneck_channels: int = 128
    channel_schedule: Sequence[int] = (512, 256, 128, 64, 9)
    n_blocks: Sequence[int] = (1, 8, 8, 1, 1)
    kernel_schedule: Sequence[int] = (1, 7, 7, 7, 7)
    dense_hidden: int = 1024

    @nn.compact
    def __call__(self, z, train: bool):
        # z: (batch, latents)
        bx, by, bz = self.bottleneck_shape
        x = nn.Dense(self.dense_hidden)(z)
        x = nn.relu(x)
        x = nn.Dense(bx * by * bz * self.bottleneck_channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], bx, by, bz, self.bottleneck_channels))

        strides_schedule = [(1,1,1), (2,2,2), (2,2,2), (2,2,2), (2,2,2)]

        for i, (ch, n, ks, strides) in enumerate(zip(
            self.channel_schedule,
            self.n_blocks,
            self.kernel_schedule,
            strides_schedule,
        )):
            is_output_stage = (i == len(self.channel_schedule) - 1)
            for block_i in range(n):
                # Apply upsampling stride on the first block of each stage
                s = strides if block_i == 0 else (1, 1, 1)
                # Final stage: no activation (raw output)
                activate = not (is_output_stage and block_i == n - 1)
                x = _DWSConvTranspose3DBlock(
                    features=ch, kernel_size=ks, strides=s, activate=activate
                )(x, train)
        return x


class WaveTransformVAE(nn.Module):
    """
    3D convolutional Variational AutoEncoder for wave-transformed molecular grids.

    Follows Kuzminykh et al. (2018) with Xception-style depthwise-separable
    Conv3D blocks. Intended to encode molecular 3D structure into a continuous
    latent space for generative molecular design.

    Input should be a wave-transformed voxel grid (from wave_transform.apply_wave_transform).
    The decoder outputs in wave space; apply wiener_deconvolve to recover atom positions.

    Default grid_shape=(64,64,32) gives a 32Å × 32Å × 16Å box at 0.5 Å/voxel,
    sufficient for most drug-like molecules, and keeps bottleneck dimensions
    exact powers of 2 (4,4,2) for clean stride-2 reconstruction.

    Args:
        in_channels: number of atom-type channels (default 9: H,C,N,O,F,S,Cl,Br,I).
        latents: latent space dimension (paper default 384).
        grid_shape: (X, Y, Z) of the input grid (must each be divisible by 16).
        dense_hidden: width of the dense bottleneck layers.

    Usage::

        model = WaveTransformVAE(in_channels=9, latents=384)
        params = model.init(rng, x, z_rng, train=False)
        recon, z_mean, z_logvar = model.apply(params, x, z_rng, train=True)
    """
    in_channels: int = 9
    latents: int = 384
    grid_shape: Tuple[int, int, int] = (64, 64, 32)
    dense_hidden: int = 1024

    def setup(self):
        bx = self.grid_shape[0] // 16
        by = self.grid_shape[1] // 16
        bz = self.grid_shape[2] // 16

        self.encoder = Conv3DEncoder(
            latents=self.latents,
            dense_hidden=self.dense_hidden,
        )
        self.decoder = Conv3DDecoder(
            out_channels=self.in_channels,
            bottleneck_shape=(bx, by, bz),
            bottleneck_channels=128,
            dense_hidden=self.dense_hidden,
        )

    def __call__(self, x, z_rng, train: bool):
        """
        Forward pass.

        Args:
            x: (batch, X, Y, Z, C) wave-transformed input grid.
            z_rng: JAX PRNG key for reparameterisation.
            train: enables BatchNorm/Dropout training mode.

        Returns:
            (reconstruction, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encoder(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        recon = self.decoder(z, train=train)
        return recon, z_mean, z_logvar

    def encode(self, x, train=False):
        """Encode x → (z_mean, z_logvar)."""
        return self.encoder(x, train=train)

    def decode(self, z, train=False):
        """Decode latent z → reconstructed grid."""
        return self.decoder(z, train=train)

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        """Sample z from (z_mean, z_logvar) and decode."""
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train)

    def sample(self, z_rng, n_samples: int = 1, train=False):
        """Sample n_samples molecules from the prior N(0,I)."""
        z = jax.random.normal(z_rng, (n_samples, self.latents))
        return self.decoder(z, train=train)
