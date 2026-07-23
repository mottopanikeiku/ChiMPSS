"""
3D wave transform for molecular voxel grids.

Implements Kuzminykh et al. (2018) "3D Molecular Representations Based on
the Wave Transform for Convolutional Neural Networks", Mol. Pharmaceutics.

The wave kernel replaces each atom with concentric decaying waves:
    k(x,y,z) = exp(-r² / 2σ²) · cos(2πω·r),  r = sqrt(x²+y²+z²)

with ω = 1/σ and clipping beyond 4σ. This is denser and more informative
than Gaussian smoothing, and is invertible via Wiener deconvolution.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def wave_kernel_3d(sigma: float) -> jnp.ndarray:
    """
    Build the 3D wave transform kernel.

    Args:
        sigma: spatial spread parameter in voxel units. Paper uses sigma=4
               (= 2 Å at 0.5 Å/voxel). Controls wave decay rate; omega=1/sigma.

    Returns:
        kernel: (K, K, K) float32 array, K = 2*ceil(4*sigma)+1.
    """
    h = int(np.ceil(4 * sigma))
    idx = jnp.arange(-h, h + 1, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(idx, idx, idx, indexing='ij')
    r2 = gx**2 + gy**2 + gz**2
    r = jnp.sqrt(r2)
    omega = 1.0 / sigma
    kernel = jnp.exp(-r2 / (2.0 * sigma**2)) * jnp.cos(2.0 * jnp.pi * omega * r)
    return jnp.where(r <= 4.0 * sigma, kernel, 0.0)


def gaussian_kernel_3d(sigma: float) -> jnp.ndarray:
    """Gaussian smoothing kernel for comparison (eq. 1 from paper)."""
    h = int(np.ceil(4 * sigma))
    idx = jnp.arange(-h, h + 1, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(idx, idx, idx, indexing='ij')
    r2 = gx**2 + gy**2 + gz**2
    return jnp.exp(-r2 / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def apply_wave_transform(voxel_grid: jnp.ndarray, sigma: float = 4.0) -> jnp.ndarray:
    """
    Apply the 3D wave transform to a sparse molecular voxel grid.

    Convolves each atom-type channel independently with the wave kernel.
    The interference pattern between atoms in different channels is captured
    by the CNN's first layer (which mixes channels).

    Args:
        voxel_grid: (X, Y, Z, C) one-hot sparse atom grid. Each channel c
                    is 1.0 where atom type c is present, 0 elsewhere.
        sigma: wave kernel spread in voxels (paper default: 4, = 2 Å at 0.5 Å).

    Returns:
        transformed: (X, Y, Z, C) dense wave-smoothed grid.
    """
    kernel = wave_kernel_3d(sigma)   # (K, K, K)
    w = kernel[..., None, None]      # (K, K, K, 1, 1) — shared across channels

    def _conv_channel(channel):
        # channel: (X, Y, Z) — single atom-type plane
        x = channel[None, ..., None]   # (1, X, Y, Z, 1)
        out = jax.lax.conv_general_dilated(
            x, w,
            window_strides=(1, 1, 1),
            padding='SAME',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        )
        return out[0, ..., 0]          # (X, Y, Z)

    # vmap over C channels: (C, X, Y, Z) → (C, X, Y, Z)
    channels_first = jnp.moveaxis(voxel_grid, -1, 0)
    result = jax.vmap(_conv_channel)(channels_first)
    return jnp.moveaxis(result, 0, -1)  # (X, Y, Z, C)


def apply_gaussian_transform(voxel_grid: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Gaussian smoothing transform for comparison."""
    kernel = gaussian_kernel_3d(sigma)
    w = kernel[..., None, None]

    def _conv_channel(channel):
        x = channel[None, ..., None]
        out = jax.lax.conv_general_dilated(
            x, w,
            window_strides=(1, 1, 1),
            padding='SAME',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        )
        return out[0, ..., 0]

    channels_first = jnp.moveaxis(voxel_grid, -1, 0)
    result = jax.vmap(_conv_channel)(channels_first)
    return jnp.moveaxis(result, 0, -1)


# ---------------------------------------------------------------------------
# Wiener deconvolution (eq. 4-5 from paper)
# ---------------------------------------------------------------------------

def wiener_deconvolve(
    transformed: jnp.ndarray,
    sigma: float = 4.0,
    snr: float = 1000.0,
) -> jnp.ndarray:
    """
    Invert the wave transform via Wiener deconvolution in the Fourier domain.

    Solves:  g* = argmin_g E||X̂*g - X||²
    Solution: G = K* / (|K|² + 1/SNR)  in Fourier space.

    High SNR (e.g. 1000) approaches the exact inverse; lower values add
    noise regularisation.  Tune SNR on a validation set (paper eq. 5).

    Args:
        transformed: (X, Y, Z, C) wave-transformed grid from autoencoder output.
        sigma: same sigma used during forward transform.
        snr: signal-to-noise ratio (constant approximation across frequencies).

    Returns:
        recovered: (X, Y, Z, C) approximately recovered sparse atom grid.
    """
    X, Y, Z, C = transformed.shape
    kernel = wave_kernel_3d(sigma)
    K = kernel.shape[0]
    h = K // 2

    # Embed kernel in zero-padded volume, centred at origin.
    # Clip to grid bounds when the kernel extends beyond the grid.
    k_padded = jnp.zeros((X, Y, Z))
    cx, cy, cz = X // 2, Y // 2, Z // 2

    # Source/destination slice limits
    sx0, sx1 = max(0, cx - h), min(X, cx + h + 1)
    sy0, sy1 = max(0, cy - h), min(Y, cy + h + 1)
    sz0, sz1 = max(0, cz - h), min(Z, cz + h + 1)
    kx0 = sx0 - (cx - h)
    ky0 = sy0 - (cy - h)
    kz0 = sz0 - (cz - h)
    k_padded = k_padded.at[sx0:sx1, sy0:sy1, sz0:sz1].set(
        kernel[kx0:kx0 + (sx1 - sx0), ky0:ky0 + (sy1 - sy0), kz0:kz0 + (sz1 - sz0)]
    )

    K_fft = jnp.fft.fftn(jnp.fft.ifftshift(k_padded))
    K_conj = jnp.conj(K_fft)
    K_abs2 = jnp.abs(K_fft) ** 2
    G = K_conj / (K_abs2 + 1.0 / snr)   # Wiener filter

    def _deconv_channel(ch):
        return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(ch) * G))

    channels_first = jnp.moveaxis(transformed, -1, 0)
    result = jax.vmap(_deconv_channel)(channels_first)
    return jnp.moveaxis(result, 0, -1)


# ---------------------------------------------------------------------------
# Channel reweighting (eq. 6 from paper)
# ---------------------------------------------------------------------------

def compute_channel_weights(
    dataset_grids: jnp.ndarray,
    epsilon: float = 0.05,
) -> jnp.ndarray:
    """
    Compute per-channel loss weights that upweight rare atom types.

    w[c] = (ε + ||X_c||² / max_c'||X_c'||²)^{-1}

    Args:
        dataset_grids: (N, X, Y, Z, C) collection of voxel grids.
        epsilon: floor weight to prevent division by zero (paper default 0.05).

    Returns:
        weights: (C,) float32 array, normalised so max weight = 1.
    """
    # Sum of squared values per channel across the entire dataset
    channel_norms = jnp.sum(dataset_grids ** 2, axis=(0, 1, 2, 3))   # (C,)
    normed = channel_norms / (jnp.max(channel_norms) + 1e-12)
    weights = 1.0 / (epsilon + normed)
    return weights


# ---------------------------------------------------------------------------
# Voxelisation (SMILES / RDKit mol → sparse grid)  [requires rdkit + sklearn]
# ---------------------------------------------------------------------------

ATOM_TYPES_DEFAULT = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']


def voxelize_mol(
    mol,
    grid_shape: tuple,
    grid_spacing: float = 0.5,
    atom_types: list = None,
) -> np.ndarray:
    """
    Convert an RDKit molecule (with 3D coordinates) to a sparse voxel grid.

    Applies PCA alignment (no dimensionality reduction) to orient the molecule
    consistently, then discretises atom positions onto a regular grid.

    Args:
        mol: RDKit Mol object with a 3D conformer.
        grid_shape: (X, Y, Z) target grid dimensions in voxels. Molecule is
                    centred within this box; atoms outside are clipped.
        grid_spacing: Å per voxel (paper uses 0.5 Å).
        atom_types: ordered list of atom symbols to include. Atoms not in this
                    list are ignored.

    Returns:
        grid: (X, Y, Z, C) float32 numpy array where C = len(atom_types).
    """
    try:
        from sklearn.decomposition import PCA
        from rdkit.Chem import rdMolTransforms
    except ImportError as e:
        raise ImportError("voxelize_mol requires rdkit and scikit-learn.") from e

    if atom_types is None:
        atom_types = ATOM_TYPES_DEFAULT

    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=np.float32)  # (N_atoms, 3)

    # PCA alignment — reorient along principal axes (paper section 3.1)
    pca = PCA(n_components=3)
    coords = pca.fit_transform(coords)
    coords -= coords.mean(axis=0)

    # Discretise to grid indices
    grid_coords = np.round(coords / grid_spacing).astype(int)
    gx, gy, gz = grid_shape
    offset = np.array([gx // 2, gy // 2, gz // 2])
    grid_coords += offset

    atom_idx = {sym: i for i, sym in enumerate(atom_types)}
    C = len(atom_types)
    grid = np.zeros((*grid_shape, C), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        if sym not in atom_idx:
            continue
        ix, iy, iz = grid_coords[i]
        if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
            grid[ix, iy, iz, atom_idx[sym]] = 1.0

    return grid


def smiles_to_voxel_grid(
    smiles: str,
    grid_shape: tuple = (64, 64, 32),
    grid_spacing: float = 0.5,
    atom_types: list = None,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Convenience wrapper: SMILES string → 3D voxel grid.

    Generates a 3D conformer using MMFF94, then calls voxelize_mol.

    Args:
        smiles: SMILES string.
        grid_shape: (X, Y, Z) grid dimensions in voxels.
        grid_spacing: Å per voxel.
        atom_types: atom symbols to include (default: H,C,N,O,F,S,Cl,Br,I).
        random_seed: RDKit embedding seed.

    Returns:
        grid: (X, Y, Z, C) float32 numpy array.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise ImportError("smiles_to_voxel_grid requires rdkit.") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return voxelize_mol(mol, grid_shape=grid_shape, grid_spacing=grid_spacing, atom_types=atom_types)
