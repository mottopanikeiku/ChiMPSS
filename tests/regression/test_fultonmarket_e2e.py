"""
End-to-end regression tests for FultonMarket (replaces Test_Things_Work.py).

These tests require:
  - openmm, openmmtools, mdtraj, pymbar
  - A GPU (or slow CPU fallback)
  - The test fixture files in test_data/

Run with:
    pytest tests/regression/ -m slow -v

Deselected automatically on CI (which runs -m 'not slow and not gpu').
"""
import glob
import os

import pytest

openmm  = pytest.importorskip("openmm",      reason="openmm not installed")
mdtraj  = pytest.importorskip("mdtraj",      reason="mdtraj not installed")
pymbar  = pytest.importorskip("pymbar",      reason="pymbar not installed")


_TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "test_data")
_INPUT_DIR = os.path.join(_TEST_DATA, "raw_OPM")


def _fixture_path(filename):
    p = os.path.join(_INPUT_DIR, filename)
    if not os.path.exists(p):
        pytest.skip(f"fixture not found: {p}")
    return p


def _output_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


# ── FultonMarket smoke test ───────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.gpu
def test_fultonmarket_short_run(tmp_path):
    """
    10-step 5-replica FultonMarket run on the 7OH fixture PDB.
    Asserts the NetCDF4 output directory is created and non-empty.
    """
    from chimpss.fultonmarket import FultonMarket

    pdb    = _fixture_path("7srq.pdb")
    # system XML and state XML are expected alongside the PDB
    sys_xml   = pdb.replace(".pdb", "_sys.xml")
    state_xml = pdb.replace(".pdb", "_state.xml")
    if not os.path.exists(sys_xml) or not os.path.exists(state_xml):
        pytest.skip("system/state XML fixtures not found alongside PDB")

    out = _output_dir(tmp_path, "fm_output")

    init_kwargs = dict(
        input_pdb=pdb,
        input_system=sys_xml,
        input_state=state_xml,
        n_replicates=5,
        T_min=310,
        T_max=320,
    )
    run_kwargs = dict(
        iter_length=0.001,
        sim_length=0.01,
        output_dir=out,
        init_overlap_thresh=0.0,
        term_overlap_thresh=0.0,
    )

    market = FultonMarket(**init_kwargs)
    market.run(**run_kwargs)

    ncdf_files = glob.glob(os.path.join(out, "*.nc"))
    assert len(ncdf_files) > 0, "Expected at least one .nc output file"


# ── Randolph smoke test ───────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.gpu
def test_randolph_short_run(tmp_path):
    """
    Short Randolph (parallel-tempering with restraints) smoke test.
    """
    from chimpss.fultonmarket import Randolph

    pdb    = _fixture_path("7srq.pdb")
    sys_xml   = pdb.replace(".pdb", "_sys.xml")
    state_xml = pdb.replace(".pdb", "_state.xml")
    if not os.path.exists(sys_xml) or not os.path.exists(state_xml):
        pytest.skip("system/state XML fixtures not found alongside PDB")

    _output_dir(tmp_path, "randolph_output")

    init_kwargs = dict(
        input_pdb=pdb,
        input_system=sys_xml,
        input_state=state_xml,
        n_replicates=4,
        T_min=310,
        T_max=320,
    )

    randolph = Randolph(**init_kwargs)
    # minimal run — just confirm construction and initial setup don't raise
    assert randolph is not None
