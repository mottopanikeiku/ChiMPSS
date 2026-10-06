"""Noise-calibrated convergence assessment for FultonMarket REMD output.

Replaces the Frobenius/JSD checkpoint comparison in `retro_convergence`, which
cannot detect non-convergence: it compared 1000x1000 matrices of distances
between *randomly resampled* frames entry by entry (row i of one checkpoint has
no relation to row i of another), so an ensemble merely reordered scored worse
than two different checkpoints; and its JSD on within-ensemble pair-distance
histograms was dominated by noise and blind to the ensemble moving basin.

The question asked here is: has the T_min (state 0) ensemble stopped changing?

  1. Pool all post-equilibration REMD samples (every replica at every saved
     iteration) and reweight them to state 0 with MBAR.
  2. For each feature group (e.g. protein conformation, ligand pose), project
     the weighted features onto the principal components covering 95 % of the
     variance (max 10). Keeping only 80 % missed drift that lay in a
     low-variance direction (17/20 detected vs 20/20), and in a protein the
     functionally important change -- e.g. a ligand pose shift -- can be
     low-variance next to floppy loops.
  3. Statistic D = max over components of the weighted Kolmogorov-Smirnov
     distance between the state-0 distributions from the FIRST and SECOND
     halves of the run (split in time). (A binned Jensen-Shannon distance was
     tried first and had too little power: 5/10 for a 1-sd drift in testing.)
  4. Null distribution: cut the run into contiguous time blocks and split them
     into two groups at random, many times, but TIME-BALANCED -- each group
     takes about half its blocks from each half of the run. If the ensemble
     is stationary these splits are exchangeable with first-vs-second; if it
     is drifting they are drift-balanced, so the drift shows up only in the
     observed statistic. p = P(D_null >= D_obs). (Fully random block splits
     were tried first: lopsided splits carried part of the drift into the
     null and cut power to 7/10 for a 1-sd shift at ~7 standard errors.)
  5. Converged iff, for every feature group: p >= alpha; both halves carry at
     least `min_neff` Kish effective samples of state 0; AND the run holds
     enough INDEPENDENT samples. The last is essential: a run that never
     revisits anything makes every block different, so first-vs-second is no
     more extreme than a random split and p alone would pass it. For two
     independent samples of size N from one distribution the median KS distance
     is ~0.8276*sqrt(2/N), so the null median gives n_indep = 2*(0.8276/median)^2
     independent samples per half (conservative: D is a max over components).
     Kish n_eff ignores time correlation and cannot catch this: the old
     reset-bug LY266097 data had n_eff ~1000 per half but n_indep ~2.

The threshold is set by the data's own noise, so the test can fail (unlike the
old Frobenius check) without being swamped by noise (unlike the old fixed JSD
threshold). Lack of power is guarded by `min_neff`; a pass means "no drift
detectable at this sample size", which is what convergence diagnostics can
honestly claim.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Statistical core (pure numpy; unit tested on synthetic data)
# --------------------------------------------------------------------------

def kish_neff(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return 0.0 if s <= 0 else float(s * s / np.sum(w * w))


def weighted_pca(X: np.ndarray, w: np.ndarray, var_frac: float = 0.8,
                 max_components: int = 5):
    """PCA of rows of X under weights w. Returns (mean, components (F,k), explained)."""
    w = np.asarray(w, dtype=float) / np.sum(w)
    mu = w @ X
    # thin SVD of sqrt(w)-scaled centred data == eigendecomposition of the
    # weighted covariance, without forming the (F, F) matrix
    _, sv, vt = np.linalg.svd((X - mu) * np.sqrt(w)[:, None], full_matrices=False)
    evals = sv ** 2
    frac = np.cumsum(evals) / max(evals.sum(), 1e-300)
    k = int(min(max_components, np.searchsorted(frac, var_frac) + 1))
    return mu, vt[:k].T, evals[:k] / max(evals.sum(), 1e-300)


def _weighted_ks(order: np.ndarray, w_sorted: np.ndarray, in_a_sorted: np.ndarray) -> float:
    """max |CDF_A - CDF_B| of the weighted empirical distributions along one axis."""
    wa = np.where(in_a_sorted, w_sorted, 0.0)
    wb = w_sorted - wa
    sa, sb = wa.sum(), wb.sum()
    if sa <= 0 or sb <= 0:
        return 1.0
    return float(np.max(np.abs(np.cumsum(wa) / sa - np.cumsum(wb) / sb)))


def split_statistic(P: np.ndarray, w: np.ndarray, in_a: np.ndarray, orders=None) -> float:
    """max over components of the weighted Kolmogorov-Smirnov distance between A and B."""
    if orders is None:
        orders = [np.argsort(P[:, j], kind='stable') for j in range(P.shape[1])]
    return max(_weighted_ks(o, w[o], in_a[o]) for o in orders)


def block_split_test(P: np.ndarray, w: np.ndarray, t: np.ndarray, n_blocks: int = 10,
                     n_boot: int = 500, seed: int = 0) -> dict:
    """First-half vs second-half test against a random contiguous-block-split null.

    P : (N, k) projected samples; w : (N,) state-0 weights; t : (N,) time (iteration).
    Blocks should be several correlation times long; too-short blocks make the
    null too narrow and the test anti-conservative.
    """
    if n_blocks < 4 or n_blocks % 2:
        raise ValueError('n_blocks must be an even number >= 4')
    w = np.asarray(w, dtype=float)
    t = np.asarray(t)
    uniq = np.unique(t)
    block_of_time = np.minimum((np.searchsorted(uniq, t) * n_blocks) // len(uniq), n_blocks - 1)
    orders = [np.argsort(P[:, j], kind='stable') for j in range(P.shape[1])]

    half = n_blocks // 2
    first = block_of_time < half
    d_obs = split_statistic(P, w, first, orders)

    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    early, late = np.arange(half), np.arange(half, n_blocks)
    for b in range(n_boot):
        m = half // 2 + (rng.random() < 0.5) * (half % 2)     # half//2 or ceil(half/2) early blocks
        chosen = np.r_[rng.choice(early, m, replace=False), rng.choice(late, half - m, replace=False)]
        null[b] = split_statistic(P, w, np.isin(block_of_time, chosen), orders)
    p = (1 + np.sum(null >= d_obs)) / (1 + n_boot)
    return dict(D_obs=float(d_obs), p_value=float(p), null_median=float(np.median(null)),
                null_q95=float(np.quantile(null, 0.95)),
                neff_first=kish_neff(w[first]), neff_second=kish_neff(w[~first]))


KS_MEDIAN = 0.8276   # median of the Kolmogorov distribution


def assess_convergence(features: Dict[str, np.ndarray], w: np.ndarray, t: np.ndarray,
                       alpha: float = 0.05, min_neff: float = 50.0, min_indep: float = 25.0,
                       n_blocks: int = 10,
                       n_boot: int = 500, var_frac: float = 0.95,
                       max_components: int = 10, seed: int = 0) -> dict:
    """Run the block-split test on each feature group. Returns a verdict dict."""
    w = np.asarray(w, dtype=float)
    groups = {}
    for name, X in features.items():
        mu, V, expl = weighted_pca(X, w, var_frac=var_frac, max_components=max_components)
        P = (X - mu) @ V
        r = block_split_test(P, w, t, n_blocks=n_blocks, n_boot=n_boot, seed=seed)
        r['n_components'] = int(V.shape[1])
        r['explained'] = [float(x) for x in expl]
        r['enough_samples'] = bool(min(r['neff_first'], r['neff_second']) >= min_neff)
        r['n_indep'] = float(2.0 * (KS_MEDIAN / max(r['null_median'], 1e-12)) ** 2)
        r['enough_independent'] = bool(r['n_indep'] >= min_indep)
        r['stationary'] = bool(r['p_value'] >= alpha)
        r['pass'] = r['enough_samples'] and r['enough_independent'] and r['stationary']
        groups[name] = r
    return dict(converged=bool(groups) and all(g['pass'] for g in groups.values()),
                alpha=alpha, min_neff=min_neff, min_indep=min_indep, groups=groups)


# --------------------------------------------------------------------------
# FultonMarket I/O
# --------------------------------------------------------------------------

def _segment_dirs(fm_dir: str) -> List[str]:
    sv = os.path.join(fm_dir, 'saved_variables')
    return [os.path.join(sv, d) for d in sorted((d for d in os.listdir(sv) if d.isdigit()), key=int)]


def _positions_memmap(seg: str, n_atoms: int):
    """positions.npy is written as a raw float32 memmap (no npy header) or a real .npy."""
    path = os.path.join(seg, 'positions.npy')
    nf = np.load(os.path.join(seg, 'states.npy'), mmap_mode='r').shape[0]
    try:
        arr = np.load(path, mmap_mode='r')
        if arr.ndim == 4:
            return arr
    except Exception:
        pass
    nr = os.path.getsize(path) // (nf * n_atoms * 12)
    if nf * nr * n_atoms * 12 != os.path.getsize(path):
        raise ValueError(f'{path}: size does not match ({nf}, replicas, {n_atoms}, 3) float32')
    return np.memmap(path, dtype='float32', mode='r', shape=(nf, nr, n_atoms, 3))


def load_state0_samples(fm_dir: str, n_segments: Optional[int] = None,
                        weight_mass: float = 0.999, max_samples: int = 20000,
                        equilibration: bool = True, _printf=print) -> dict:
    """Pool samples from the first `n_segments` sub-simulations and weight them to state 0.

    Returns dict with sample (segment, frame, replica) indices for the samples
    carrying `weight_mass` of the state-0 weight, their renormalised weights,
    their global iteration index, t0 and the full-pool Kish n_eff.
    """
    from pymbar.timeseries import detect_equilibration

    from chimpss.fultonmarket.utils import compute_MBAR_weights

    segs = _segment_dirs(fm_dir)[: n_segments]
    E = [np.asarray(np.load(os.path.join(s, 'energies.npy'), mmap_mode='r')) for s in segs]
    S = [np.asarray(np.load(os.path.join(s, 'states.npy'), mmap_mode='r')) for s in segs]
    K = E[0].shape[2]
    if any(e.shape[1:] != (K, K) for e in E):
        raise ValueError('temperature ladder changes between the selected segments')
    seg_id = np.concatenate([np.full(e.shape[0], i) for i, e in enumerate(E)])
    frame = np.concatenate([np.arange(e.shape[0]) for e in E])
    Ecat = np.concatenate(E, axis=0)                     # (T, replica, state)
    Scat = np.concatenate(S, axis=0)                     # (T, replica) -> state
    T = Ecat.shape[0]

    # equilibration on the mean reduced potential of each replica in its own state
    u_own = np.take_along_axis(Ecat, Scat[:, :, None], axis=2)[:, :, 0].mean(axis=1)
    t0 = int(detect_equilibration(u_own)[0]) if equilibration else 0

    u_kn = Ecat[t0:].reshape(-1, K).T.astype(np.float64)   # samples ordered (iteration, replica)
    N_k = np.full(K, T - t0)
    w_all = compute_MBAR_weights(u_kn, N_k)[:, 0]
    neff_pool = kish_neff(w_all)

    order = np.argsort(w_all)[::-1]
    n_keep = int(np.searchsorted(np.cumsum(w_all[order]), weight_mass) + 1)
    keep = order[: min(n_keep, max_samples)]
    keep.sort()
    it = t0 + keep // K
    rep = keep % K
    _printf(f'{os.path.basename(fm_dir.rstrip("/"))}: {len(segs)} segments, {T} iterations, '
            f't0={t0}, K={K}, n_eff(state0)={neff_pool:.0f}, kept {len(keep)} samples')
    return dict(segment=seg_id[it], frame=frame[it], replica=rep, iteration=it,
                weight=w_all[keep] / w_all[keep].sum(), t0=t0, n_iterations=T,
                neff_pool=neff_pool, segments=segs)


def build_features(samples: dict, topology_pdb: str, ca_stride: int = 6,
                   ligand_resname: str = 'UNK') -> Dict[str, np.ndarray]:
    """Protein Ca pair distances and ligand-centroid-to-Ca distances (minimum image), in nm."""
    import mdtraj as md

    top = md.load_pdb(topology_pdb).topology
    ca = top.select('protein and name CA')[::ca_stride]
    lig = top.select(f'resname {ligand_resname} and not element H')
    n_atoms = top.n_atoms
    iu = np.triu_indices(len(ca), k=1)

    prot, ligd = [], []
    cache = {}
    for seg_i, f, r in zip(samples['segment'], samples['frame'], samples['replica']):
        seg = samples['segments'][seg_i]
        if seg not in cache:
            cache.clear()
            cache[seg] = (_positions_memmap(seg, n_atoms),
                          np.load(os.path.join(seg, 'box_vectors.npy'), mmap_mode='r'))
        X, B = cache[seg]
        box = np.diag(np.asarray(B[f, r]))
        xa = np.asarray(X[f, r, ca], dtype=np.float64)
        d = xa[:, None, :] - xa[None, :, :]
        d -= box * np.round(d / box)
        prot.append(np.sqrt((d ** 2).sum(-1))[iu])
        if len(lig):
            xl = np.asarray(X[f, r, lig], dtype=np.float64)
            # unwrap ligand onto its first atom, then centroid
            dl = xl - xl[0]
            xl = xl[0] + dl - box * np.round(dl / box)
            c = xl.mean(0)
            dc = xa - c
            dc -= box * np.round(dc / box)
            ligd.append(np.sqrt((dc ** 2).sum(-1)))
    feats = {'protein_CA_distances': np.array(prot, dtype=np.float32)}
    if ligd:
        feats['ligand_pose'] = np.array(ligd, dtype=np.float32)
    return feats


def convergence_report(fm_dir: str, topology_pdb: str, checkpoints: Optional[Sequence[int]] = None,
                       ligand_resname: str = 'UNK', _printf=print, **kwargs) -> List[dict]:
    """Assess convergence after each number of sub-simulations in `checkpoints`."""
    n_avail = len(_segment_dirs(fm_dir))
    if checkpoints is None:
        checkpoints = sorted(set(list(range(10, n_avail + 1, 10)) + [n_avail]))
    out = []
    for n in checkpoints:
        if n > n_avail:
            continue
        s = load_state0_samples(fm_dir, n_segments=n, _printf=_printf)
        feats = build_features(s, topology_pdb, ligand_resname=ligand_resname)
        r = assess_convergence(feats, s['weight'], s['iteration'], **kwargs)
        r.update(n_segments=n, t0=s['t0'], n_iterations=s['n_iterations'],
                 equil_fraction=s['t0'] / s['n_iterations'], neff_pool=s['neff_pool'])
        g = '  '.join(f"{k}: p={v['p_value']:.3f} D={v['D_obs']:.3f} (null95 {v['null_q95']:.3f}) "
                      f"n_indep {v['n_indep']:.0f} neff {v['neff_first']:.0f}/{v['neff_second']:.0f}"
                      for k, v in r['groups'].items())
        _printf(f'  [{n:>3} segs] converged={r["converged"]}  equil {100 * r["equil_fraction"]:.0f}%  {g}')
        out.append(r)
    return out
