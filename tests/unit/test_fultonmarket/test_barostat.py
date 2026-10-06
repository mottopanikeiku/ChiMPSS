"""FultonMarket pressure coupling: membrane barostat option for bilayer systems.

Without it openmmtools adds an isotropic MonteCarloBarostat, which fixes the
box aspect ratio, so a bilayer cannot relax its area and thickness separately.
"""

import pytest


pytest.importorskip("openmm", reason="openmm not installed")
testsystems = pytest.importorskip("openmmtools.testsystems", reason="openmmtools not installed")

from openmm import MonteCarloBarostat, MonteCarloMembraneBarostat, unit

from chimpss.fultonmarket import FultonMarket


def _market(membrane, tension=0.0):
    m = FultonMarket.__new__(FultonMarket)
    m.system = testsystems.WaterBox(box_edge=2.0 * unit.nanometer).system
    m.temperatures = [300.0 * unit.kelvin, 310.0 * unit.kelvin]
    m.membrane_barostat = membrane
    m.surface_tension = tension
    m._build_thermodynamic_states()
    return m


def _barostats(system):
    return [f for f in system.getForces() if 'Barostat' in type(f).__name__]


def test_membrane_barostat_is_used_when_requested():
    m = _market(membrane=True)
    state = m.thermodynamic_states[0]
    (baro,) = _barostats(state.system)
    assert isinstance(baro, MonteCarloMembraneBarostat)
    assert baro.getXYMode() == MonteCarloMembraneBarostat.XYIsotropic
    assert baro.getZMode() == MonteCarloMembraneBarostat.ZFree
    assert state.pressure.value_in_unit(unit.bar) == pytest.approx(1.0)
    assert state.surface_tension.value_in_unit(unit.bar * unit.nanometer) == pytest.approx(0.0)
    assert state.temperature.value_in_unit(unit.kelvin) == pytest.approx(300.0)
    # the input system itself is left untouched
    assert _barostats(m.system) == []


def test_isotropic_default_is_unchanged():
    m = _market(membrane=False)
    (baro,) = _barostats(m.thermodynamic_states[0].system)
    assert isinstance(baro, MonteCarloBarostat)
