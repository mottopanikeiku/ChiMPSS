"""Regression test: every replica must start a sub-simulation from its OWN state.

Randolph._build_simulation once passed `self.sampler_states[0]` to
ParallelTemperingSampler.create. openmmtools copies a single SamplerState to
every replica, so each sub-simulation restarted all replicas from the T_min
configuration. That silently reset the whole 5-HT2B REMD campaign every
sub-simulation (all replicas byte-identical at frame 0 of every segment).
"""

import types

import pytest


pytest.importorskip("openmm", reason="openmm not installed")
pytest.importorskip("openmmtools", reason="openmmtools not installed")

from chimpss.fultonmarket import randolph as randolph_mod
from chimpss.fultonmarket import Randolph


class _RecordingSampler:
    last_create_kwargs = None

    def __init__(self, *args, **kwargs):
        pass

    def create(self, **kwargs):
        _RecordingSampler.last_create_kwargs = kwargs


def _randolph(tmp_path, monkeypatch, n_replicates, sampler_states):
    monkeypatch.setattr(randolph_mod, "ParallelTemperingSampler", _RecordingSampler)
    monkeypatch.setattr(randolph_mod, "MultiStateReporter", lambda *a, **k: object())
    monkeypatch.setattr(randolph_mod.mcmc, "LangevinDynamicsMove", lambda **k: object())

    r = Randolph.__new__(Randolph)
    r.dt = 3.5
    r.n_steps_per_iter = 5000
    r.n_iters = 8
    r.checkpoint_interval = 1
    r.output_ncdf = str(tmp_path / "output.ncdf")
    r.n_replicates = n_replicates
    r.temperatures = [300.0 + i for i in range(n_replicates)]
    r.sampler_states = sampler_states
    system = types.SimpleNamespace(getNumParticles=lambda: 10)
    r.reference_state = types.SimpleNamespace(system=system)
    return r


def test_create_receives_one_sampler_state_per_replica(tmp_path, monkeypatch):
    states = [object() for _ in range(4)]
    r = _randolph(tmp_path, monkeypatch, n_replicates=4, sampler_states=states)

    r._build_simulation()

    passed = _RecordingSampler.last_create_kwargs["sampler_states"]
    assert isinstance(passed, list), "a single SamplerState is broadcast to every replica"
    assert passed == states


def test_mismatched_sampler_state_count_is_rejected(tmp_path, monkeypatch):
    r = _randolph(tmp_path, monkeypatch, n_replicates=4,
                  sampler_states=[object() for _ in range(3)])

    with pytest.raises(ValueError, match="one per replica"):
        r._build_simulation()
