"""
Verifies that LJr and electrostatic scaling factors computed from the ligand
prmtop (OpenMM path) match the validated MMTK-derived benchmark values.

Skips automatically if:
  - MMTK is not installed (benchmark generation requires it), OR
  - test input files are not present on the filesystem.
"""

import os
import pytest
import numpy as np

ALGDOCK_TEST_INPUT = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', '..',
    'AlGDock', 'tests', 'input',
)
ALGDOCK_TEST_INPUT = os.path.abspath(ALGDOCK_TEST_INPUT)

LIGAND_PRMTOP = os.path.join(ALGDOCK_TEST_INPUT, 'ligand.prmtop')
LIGAND_INPCRD = os.path.join(ALGDOCK_TEST_INPUT, 'ligand.trans.inpcrd')
LIGAND_DB = os.path.join(ALGDOCK_TEST_INPUT, 'ligand.db')
GAFF_DAT = os.path.join(ALGDOCK_TEST_INPUT, 'gaff2.dat')
LIGAND_FRCMOD = os.path.join(ALGDOCK_TEST_INPUT, 'ligand.frcmod')

inputs_present = pytest.mark.skipif(
    not os.path.isfile(LIGAND_PRMTOP),
    reason=f'AlGDock test inputs not found at {ALGDOCK_TEST_INPUT}',
)

try:
    import MMTK
    mmtk_available = True
except ImportError:
    mmtk_available = False

requires_mmtk = pytest.mark.skipif(
    not mmtk_available,
    reason='MMTK not installed — benchmark generation unavailable',
)


@inputs_present
@requires_mmtk
def test_ljr_and_ele_scaling_factors_match_mmtk():
    """LJr and ELE scaling factors from OpenMM prmtop must match MMTK values."""
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
    from openmm.unit import (
        elementary_charge, nanometer, kilojoule_per_mole,
    )
    from MMTK.ForceFields import Amber12SBForceField

    MMTK.Database.molecule_types.directory = ALGDOCK_TEST_INPUT
    molecule = MMTK.Molecule('ligand.db')
    universe = MMTK.Universe.InfiniteUniverse()
    universe.addObject(molecule)
    universe.setForceField(Amber12SBForceField(
        parameter_file=GAFF_DAT, mod_files=[LIGAND_FRCMOD],
    ))

    # MMTK reference values
    mmtk_ele = {}
    mmtk_ljr = {}
    for a in molecule.atomList():
        name = molecule.getAtomProperty(a, 'name').split('i')[0]
        mmtk_ele[name] = float(molecule.getAtomProperty(a, 'scaling_factor_electrostatic'))
        mmtk_ljr[name] = float(molecule.getAtomProperty(a, 'scaling_factor_LJr'))

    # OpenMM / vendored IO path
    import chimpss.algdock.IO
    prmtop_io = chimpss.algdock.IO.prmtop()
    varnames = [
        'POINTERS', 'ATOM_NAME', 'AMBER_ATOM_TYPE', 'CHARGE', 'MASS',
        'NONBONDED_PARM_INDEX', 'LENNARD_JONES_ACOEF', 'LENNARD_JONES_BCOEF',
        'ATOM_TYPE_INDEX', 'BONDS_INC_HYDROGEN', 'BONDS_WITHOUT_HYDROGEN',
        'RADII', 'SCREEN',
    ]
    prmtop_data = prmtop_io.read(LIGAND_PRMTOP, varnames)
    NATOM = prmtop_data['POINTERS'][0]
    NTYPES = prmtop_data['POINTERS'][1]

    LJ_radius = np.zeros(NTYPES)
    LJ_depth = np.zeros(NTYPES)
    for i in range(NTYPES):
        idx = prmtop_data['NONBONDED_PARM_INDEX'][NTYPES * i + i] - 1
        A = prmtop_data['LENNARD_JONES_ACOEF'][idx]
        B = prmtop_data['LENNARD_JONES_BCOEF'][idx]
        if A > 1e-6:
            factor = 2 * A / B
            LJ_radius[i] = pow(factor, 1.0 / 6.0) * 0.5
            LJ_depth[i] = B / 2 / factor

    root_LJ_depth = np.sqrt(LJ_depth)
    LJ_diameter = LJ_radius * 2
    type_indices = [prmtop_data['ATOM_TYPE_INDEX'][i] - 1 for i in range(NATOM)]
    scaling_ljr = {
        name.strip(): round(4.184 * root_LJ_depth[ti] * (LJ_diameter[ti] ** 6), 6)
        for name, ti in zip(prmtop_data['ATOM_NAME'], type_indices)
    }

    prmtop = AmberPrmtopFile(LIGAND_PRMTOP)
    inpcrd = AmberInpcrdFile(LIGAND_INPCRD)
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometer, constraints=None,
    )
    nb_force = system.getForce(3)
    assert nb_force.getName() == 'NonbondedForce'

    atoms = list(prmtop.topology.atoms())
    omm_ele = []
    omm_ljr = []
    ref_ele = []
    ref_ljr = []
    for i in range(system.getNumParticles()):
        name = atoms[i].name.strip()
        charge, _sigma, epsilon = nb_force.getParticleParameters(i)
        eps_val = epsilon.value_in_unit(kilojoule_per_mole)
        if eps_val == 0:
            continue
        charge_val = charge.value_in_unit(elementary_charge)
        omm_ele.append(round(charge_val * 4.184, 6))
        omm_ljr.append(scaling_ljr[name])
        ref_ele.append(mmtk_ele[name])
        ref_ljr.append(mmtk_ljr[name])

    assert omm_ele == ref_ele, 'ELE scaling factors differ from MMTK benchmark'
    assert omm_ljr == ref_ljr, 'LJr scaling factors differ from MMTK benchmark'
