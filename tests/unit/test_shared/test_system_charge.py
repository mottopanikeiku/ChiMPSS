"""chimpss.shared.system_charge: exact net charge and counterion neutralization."""

import numpy as np
import pytest


pytest.importorskip("openmm", reason="openmm not installed")

from openmm import NonbondedForce, System, Vec3, unit
from openmm.app import Topology, element

from chimpss.shared.system_charge import (
    neutralize_with_counterions,
    system_net_charge,
)

BOX = 4.0  # nm


def _solvated_solute():
    """A compact 'LIG' solute at the box centre plus waters on a 0.31 nm grid."""
    top = Topology()
    top.setPeriodicBoxVectors([Vec3(BOX, 0, 0), Vec3(0, BOX, 0), Vec3(0, 0, BOX)] * unit.nanometer)
    chain = top.addChain()
    pos = []
    lig = top.addResidue('LIG', chain)
    for k in range(10):
        top.addAtom(f'C{k}', element.carbon, lig)
        pos.append(np.array([BOX / 2 + 0.1 * (k % 3), BOX / 2 + 0.1 * (k // 3 % 3), BOX / 2]))
    wchain = top.addChain()
    g = np.arange(0.15, BOX, 0.31)
    for x in g:
        for y in g:
            for z in g:
                o = np.array([x, y, z])
                if np.linalg.norm(o - BOX / 2) < 0.5:
                    continue
                r = top.addResidue('HOH', wchain)
                top.addAtom('O', element.oxygen, r)
                top.addAtom('H1', element.hydrogen, r)
                top.addAtom('H2', element.hydrogen, r)
                pos += [o, o + [0.0957, 0, 0], o + [-0.024, 0.0927, 0]]
    return top, unit.Quantity([Vec3(*p) for p in pos], unit.nanometer)


def _xyz(positions):
    return np.array(positions.value_in_unit(unit.nanometer))


def _min_image(a, b):
    d = a - b
    d -= BOX * np.round(d / BOX)
    return np.linalg.norm(d)


def test_system_net_charge_sums_nonbonded_charges():
    s = System()
    nb = NonbondedForce()
    for q in (0.4, -0.1, 0.7):
        s.addParticle(12.0)
        nb.addParticle(q, 0.3, 0.5)
    s.addForce(nb)
    assert system_net_charge(s) == pytest.approx(1.0)


@pytest.mark.parametrize("charge, ion", [(+3, 'CL'), (-2, 'NA')])
def test_replaces_bulk_waters_with_counterions(charge, ion):
    top, pos = _solvated_solute()
    n_atoms, n_wat = top.getNumAtoms(), sum(r.name == 'HOH' for r in top.residues())

    new_top, new_pos, replaced = neutralize_with_counterions(top, pos, charge + 0.003)

    n = abs(charge)
    assert len(replaced) == n
    assert sum(r.name == 'HOH' for r in new_top.residues()) == n_wat - n
    assert new_top.getNumAtoms() == n_atoms - 3 * n + n
    ions = [a.index for a in new_top.atoms() if a.residue.name == ion]
    assert len(ions) == n
    xyz = _xyz(new_pos)
    lig = [a.index for a in new_top.atoms() if a.residue.name == 'LIG']
    for i in ions:
        assert min(_min_image(xyz[i], xyz[j]) for j in lig) >= 0.6
        for j in ions:
            if j != i:
                assert _min_image(xyz[i], xyz[j]) >= 0.5
    assert np.allclose(new_top.getPeriodicBoxVectors().value_in_unit(unit.nanometer),
                       top.getPeriodicBoxVectors().value_in_unit(unit.nanometer))


def test_neutral_system_is_unchanged():
    top, pos = _solvated_solute()
    new_top, new_pos, replaced = neutralize_with_counterions(top, pos, 0.004)
    assert replaced == [] and new_top is top and new_pos is pos


def test_non_integer_charge_is_refused():
    top, pos = _solvated_solute()
    with pytest.raises(ValueError, match="not near an integer"):
        neutralize_with_counterions(top, pos, 2.4)


def test_impossible_spacing_raises_instead_of_crowding():
    top, pos = _solvated_solute()
    with pytest.raises(RuntimeError, match="bulk waters satisfy"):
        neutralize_with_counterions(top, pos, 40, min_ion_dist=2.0)
