"""
chimpss.neural_nets — JAX/Flax neural network models.

Modules
-------
wave_transform
    3D wave transform preprocessing for molecular voxel grids
    (Kuzminykh et al. 2018). Converts sparse atom grids to dense
    wave-interference representations suitable for 3D CNNs.

vae
    Variational AutoEncoder models:
      - BatchNorm_VAE : dense VAE (ported from Deep-MMS)
      - WaveTransformVAE : 3D convolutional VAE for wave-transformed grids
"""

from chimpss.neural_nets.wave_transform import (
    wave_kernel_3d,
    gaussian_kernel_3d,
    apply_wave_transform,
    apply_gaussian_transform,
    wiener_deconvolve,
    compute_channel_weights,
    smiles_to_voxel_grid,
    voxelize_mol,
    ATOM_TYPES_DEFAULT,
)
from chimpss.neural_nets.vae import (
    reparameterize,
    BatchNorm_VAE,
    BVEncoder,
    BVDecoder,
    Conv3DEncoder,
    Conv3DDecoder,
    WaveTransformVAE,
)

__all__ = [
    # wave_transform
    'wave_kernel_3d',
    'gaussian_kernel_3d',
    'apply_wave_transform',
    'apply_gaussian_transform',
    'wiener_deconvolve',
    'compute_channel_weights',
    'smiles_to_voxel_grid',
    'voxelize_mol',
    'ATOM_TYPES_DEFAULT',
    # vae
    'reparameterize',
    'BatchNorm_VAE',
    'BVEncoder',
    'BVDecoder',
    'Conv3DEncoder',
    'Conv3DDecoder',
    'WaveTransformVAE',
]
