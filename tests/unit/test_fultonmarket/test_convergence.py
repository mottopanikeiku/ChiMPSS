"""chimpss.fultonmarket.convergence: calibration, power and sample bookkeeping."""

import os

import numpy as np
import pytest

from chimpss.fultonmarket.convergence import (
    _positions_memmap,
    assess_convergence,
    kish_neff,
    weighted_pca,
)


def _ar1(n, rho, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return x


def _trajectory(rng, n_iter=2000, per_iter=8, n_feat=12, drift=0.0, rho=0.8, rho_mode=0.9):
    """Correlated, bimodal features; `drift` shifts the second half (in sd units)."""
    t = np.repeat(np.arange(n_iter), per_iter)
    latent = np.stack([_ar1(n_iter, rho, rng) for _ in range(3)], 1)
    mode = (_ar1(n_iter, rho_mode, rng) > 0).astype(float)      # two-state switching
    base = np.concatenate([latent, 2.0 * mode[:, None]], 1)[t]
    base = base + 0.3 * rng.standard_normal(base.shape)
    base[t >= n_iter // 2, 0] += drift
    mix = rng.standard_normal((base.shape[1], n_feat))
    return base @ mix, t


def test_weighted_pca_recovers_dominant_axis():
    rng = np.random.default_rng(1)
    X = np.c_[5 * rng.standard_normal(4000), 0.5 * rng.standard_normal(4000)]
    _, V, expl = weighted_pca(X, np.ones(4000), var_frac=0.9)
    assert abs(abs(V[0, 0]) - 1) < 1e-2 and expl[0] > 0.95


def test_stationary_runs_pass_at_the_nominal_rate():
    passes = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        X, t = _trajectory(rng)
        r = assess_convergence({'f': X}, np.ones(len(t)), t, n_boot=200, min_neff=10, seed=seed)
        passes += r['converged']
    assert passes >= 15                     # alpha = 0.05 -> expect ~19/20


def _detection_rate(drift, n=10):
    fails = 0
    for seed in range(n):
        rng = np.random.default_rng(100 + seed)
        X, t = _trajectory(rng, drift=drift)
        r = assess_convergence({'f': X}, np.ones(len(t)), t, n_boot=200, min_neff=10, seed=seed)
        fails += not r['converged']
    return fails / n


def test_drift_between_halves_is_detected():
    # The drifted coordinate is AR(1) with rho=0.8 (tau_int ~ 9 iterations), so a
    # 1000-iteration half holds ~110 independent samples: a 1-sd shift is ~7 SE.
    assert _detection_rate(1.0) >= 0.9


def test_power_grows_with_drift():
    assert _detection_rate(0.0) < _detection_rate(0.3) <= _detection_rate(1.0)


def test_run_that_never_revisits_is_not_converged():
    """A slow random walk makes every block different, so the half split is no more
    extreme than a random split and p alone would pass it. The independent-sample
    requirement must catch it."""
    rng = np.random.default_rng(11)
    n_iter, per_iter = 2000, 8
    t = np.repeat(np.arange(n_iter), per_iter)
    walk = np.cumsum(rng.standard_normal((n_iter, 3)), 0)[t]
    X = walk @ rng.standard_normal((3, 12)) + 0.3 * rng.standard_normal((len(t), 12))
    g = assess_convergence({'f': X}, np.ones(len(t)), t, n_boot=200, min_neff=10)['groups']['f']
    assert not g['enough_independent'] and not g['pass']
    assert g['neff_first'] > 1000            # Kish n_eff is blind to it


def test_reweighting_is_respected():
    """Drift confined to samples with zero state-0 weight must not count."""
    rng = np.random.default_rng(7)
    X, t = _trajectory(rng)
    hot = np.zeros(len(t), bool)
    hot[1::2] = True                        # half the samples are 'other temperatures'
    X = X.copy()
    X[hot & (t >= t.max() // 2)] += 25.0    # ...and only they drift
    w = np.where(hot, 0.0, 1.0)
    assert assess_convergence({'f': X}, w, t, n_boot=200, min_neff=10)['converged']
    assert not assess_convergence({'f': X}, np.ones(len(t)), t, n_boot=200, min_neff=10)['converged']


def test_too_few_effective_samples_gives_no_verdict():
    rng = np.random.default_rng(3)
    X, t = _trajectory(rng)
    w = np.zeros(len(t))
    w[rng.choice(len(t), 12, replace=False)] = 1.0
    r = assess_convergence({'f': X}, w, t, n_boot=100, min_neff=50)
    assert not r['converged'] and not r['groups']['f']['enough_samples']
    assert kish_neff(w) == pytest.approx(12)


def test_loader_maps_samples_to_segment_frame_replica(tmp_path):
    """Pool two fake FultonMarket segments; every kept sample must point at its own coordinates."""
    pytest.importorskip("pymbar")
    from chimpss.fultonmarket.convergence import load_state0_samples

    rng = np.random.default_rng(0)
    K, n_atoms = 3, 4
    betas = np.array([1.0, 0.97, 0.94])
    for seg, nf in ((0, 6), (1, 5)):
        d = tmp_path / 'saved_variables' / str(seg)
        d.mkdir(parents=True)
        U = -1000 + 5 * rng.standard_normal((nf, K))                  # (frame, replica)
        np.save(d / 'energies.npy', (U[:, :, None] * betas[None, None, :]).astype(np.float32))
        np.save(d / 'states.npy', np.array([rng.permutation(K) for _ in range(nf)], np.int32))
        pos = np.zeros((nf, K, n_atoms, 3), np.float32)
        for f in range(nf):
            for r in range(K):
                pos[f, r, 0] = (seg, f, r)                             # encode identity
        pos.tofile(d / 'positions.npy')                                 # raw memmap, as Randolph writes
        np.save(d / 'box_vectors.npy', np.tile(np.eye(3, dtype=np.float32) * 5, (nf, K, 1, 1)))

    s = load_state0_samples(str(tmp_path), equilibration=False, weight_mass=1.0, _printf=lambda *_: None)
    assert s['n_iterations'] == 11 and s['t0'] == 0
    assert np.isclose(s['weight'].sum(), 1.0)
    for seg_i, f, r, it in zip(s['segment'], s['frame'], s['replica'], s['iteration']):
        X = _positions_memmap(s['segments'][seg_i], n_atoms)
        assert tuple(X[f, r, 0]) == (seg_i, f, r)
        assert it == (f if seg_i == 0 else 6 + f)
    assert os.path.basename(s['segments'][1]) == '1'
