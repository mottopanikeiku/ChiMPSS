"""An interrupted sub-simulation 0 must restart with the ladder it had reached.

Previously a run killed inside sub-sim 0 left output.ncdf behind with no saved
sub-simulations, and every later run printed 'Trajectory already exists' and
exited 0 having done nothing (5-HT2B methylergonovine_T140A_mutseq stalled this
way 40 min short of finishing a 180-state ladder).
"""

import types

import pytest


pytest.importorskip("openmm", reason="openmm not installed")
multistate = pytest.importorskip("openmmtools.multistate", reason="openmmtools not installed")

from openmm import unit

from chimpss.fultonmarket import FultonMarket

GROWN = [300.0, 300.4, 301.1, 333.3, 367.0]


class _Reporter:
    def __init__(self, path, open_mode='r'):
        self.path = path

    def read_thermodynamic_states(self):
        return [types.SimpleNamespace(temperature=t * unit.kelvin) for t in GROWN], []

    def close(self):
        pass


def _market(tmp_path):
    m = FultonMarket.__new__(FultonMarket)
    m.output_ncdf = str(tmp_path / 'output.ncdf')
    m.checkpoint_ncdf = str(tmp_path / 'output_checkpoint.ncdf')
    for p in (m.output_ncdf, m.checkpoint_ncdf):
        open(p, 'w').write('partial')
    m.temperatures = [300.0 * unit.kelvin, 367.0 * unit.kelvin]   # the initial ladder
    m.n_replicates = 2
    return m


def test_restarts_subsim0_with_the_ladder_it_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(multistate, 'MultiStateReporter', _Reporter)
    m = _market(tmp_path)
    m._resume_interrupted_first_subsim()
    assert [t.value_in_unit(unit.kelvin) for t in m.temperatures] == GROWN
    assert m.n_replicates == len(GROWN)
    # partial files moved aside (kept), so a fresh sub-sim 0 can be created
    assert not (tmp_path / 'output.ncdf').exists() and not (tmp_path / 'output_checkpoint.ncdf').exists()
    assert (tmp_path / 'output.ncdf.interrupted_subsim0').exists()
    assert (tmp_path / 'output_checkpoint.ncdf.interrupted_subsim0').exists()


def test_unreadable_file_falls_back_to_initial_ladder_but_still_runs(tmp_path, monkeypatch):
    class _Broken(_Reporter):
        def read_thermodynamic_states(self):
            raise OSError('truncated netCDF header')

    monkeypatch.setattr(multistate, 'MultiStateReporter', _Broken)
    m = _market(tmp_path)
    m._resume_interrupted_first_subsim()
    assert m.n_replicates == 2
    assert not (tmp_path / 'output.ncdf').exists()          # never left blocking the run


def test_run_no_longer_bails_out_on_an_existing_trajectory():
    import inspect
    src = inspect.getsource(FultonMarket.run)
    assert 'Trajectory already exists' not in src
    assert '_resume_interrupted_first_subsim' in src
