"""compute_MBAR_weights must converge on REMD-scale energies and never return
an unconverged solution.

Reduced potentials in the 5-HT2B runs are ~5e5 kT, so the f_k span ~1e5 kT
across the ladder. pymbar's default zeros initialisation could not reach that
(10,000 iterations, "No solution found", weights not even normalised) and the
old code resampled with whatever came back.
"""

import numpy as np
import pytest


pytest.importorskip("pymbar", reason="pymbar not installed")

from chimpss.fultonmarket import utils


def _harmonic_ladder(K=20, n_per_state=200, d=2000, U0=-6.0e5, seed=0):
    """d-dim harmonic oscillator U = U0 + |x|^2 / 2 sampled at K inverse temps.

    |x|^2 at inverse temperature b is chi2(d) / b, so x itself is never needed.
    Exact: <U>_b = U0 + d / (2 b).
    """
    rng = np.random.default_rng(seed)
    betas = np.linspace(1.0, 300.0 / 367.0, K)
    U = np.concatenate([U0 + 0.5 * rng.chisquare(d, n_per_state) / b for b in betas])
    u_kn = betas[:, None] * U[None, :]
    N_k = np.full(K, n_per_state)
    return u_kn, N_k, U, betas, d, U0


def test_converges_and_reweights_correctly_at_remd_scale():
    u_kn, N_k, U, betas, d, U0 = _harmonic_ladder()
    assert np.ptp(u_kn.mean(axis=1)) > 1e5          # same regime as the real data

    w = utils.compute_MBAR_weights(u_kn, N_k)

    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-8)
    exact = U0 + d / (2.0 * betas[0])
    estimate = float(np.sum(w[:, 0] * U))
    assert abs(estimate - exact) < 0.02 * (d / (2.0 * betas[0]))


def test_unconverged_solution_is_rejected(monkeypatch):
    u_kn, N_k, *_ = _harmonic_ladder(K=5, n_per_state=50)

    class _WrongMBAR:
        def __init__(self, u, n, **kwargs):
            self.f_k = np.zeros(u.shape[0])          # wrong by ~1e5 kT

        def weights(self):
            raise AssertionError("weights must not be read from an unconverged solve")

    monkeypatch.setattr(utils, "MBAR", _WrongMBAR)

    with pytest.raises(RuntimeError, match="did not converge"):
        utils.compute_MBAR_weights(u_kn, N_k)
