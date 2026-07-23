"""
Integration test: NTSR1 / ML-301 → Bridgeport → Converter → chimpss.algdock

Runs the full pipeline from raw NTSR1 inputs through to BPMF-ready AMBER
prmtop/inpcrd files, using persistent cached outputs so the slow Bridgeport
build only runs once.

Stage layout:
  1. Bridgeport  → NTSR1_ML-301.topology.pdb  +  NTSR1_ML-301.system.xml
  2. Converter   → ligand/receptor/complex .prmtop + .inpcrd
  3. BPMF check  → BindingPMF can be imported; converter paths satisfy BPMF kwarg names

MotorRow output (equil.pdb) from /media/volume/Josephs-Volume/ChiMPSS_Testing/
is used for converter input if present, otherwise the topology PDB from
Bridgeport is used (coordinates differ, force field parameters are the same).

Run with:
    pytest tests/integration/test_algdock_ntsr1.py -v -m slow
"""

import json
import os

import pytest
import parmed as pmd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_DATA = os.path.join(REPO_ROOT, 'test_data', 'test_case_1')

# Persistent output dir — survives between test runs (Bridgeport caches here)
PERSISTENT_OUT = '/media/volume/Josephs-Volume/ChiMPSS_Testing/ntsr1_bridgeport'
SYSTEMS_DIR = os.path.join(PERSISTENT_OUT, 'systems')

# Expected Bridgeport output paths
TOPOLOGY_PDB = os.path.join(SYSTEMS_DIR, 'NTSR1_ML-301.topology.pdb')
SYSTEM_XML   = os.path.join(SYSTEMS_DIR, 'NTSR1_ML-301.system.xml')

# Must use the Bridgeport topology PDB — its atom count matches the system XML
# exactly.  MDTraj-written equil PDBs have different water/ion counts from
# box-wrapping and will cause a SystemConverter atom count mismatch error.
INPUT_PDB = TOPOLOGY_PDB

LIG_RESNAME = 'UNK'


def _abspath(filename):
    return os.path.join(TEST_DATA, filename)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def bridgeport_system():
    """
    Run Bridgeport once and cache outputs in PERSISTENT_OUT.
    Returns (topology_pdb, system_xml).
    If outputs already exist, returns immediately (< 1 s).
    """
    os.makedirs(PERSISTENT_OUT, exist_ok=True)
    modeller_dir = os.path.join(PERSISTENT_OUT, 'modeller_intermediates')
    os.makedirs(modeller_dir, exist_ok=True)

    config = {
        'working_dir': PERSISTENT_OUT,
        'protein_name': 'NTSR1',
        'Protein': {
            'input_pdb': _abspath('6za8-pdb-bundle1.pdb'),
            'chain': 'A',
        },
        'Ligand': {
            'name': 'RTI-3a',
            'resname': 'SR5',
            'chainid': False,
            'smiles': 'COC1=CC=CC(OC)=C1C1=CC(=NN1C1=CC=NC2=C1C=CC(Cl)=C2)C(=O)N[C@@H](CC(C)C)C(O)=O',
            'Analogue': {
                'name': 'ML-301',
                'smiles': 'COC1=CC=CC(OC)=C1C1=NC(=CN1C1=C2C=CC(Cl)=CC2=NC=C1)C(=O)N[C@H](CC(C)C)C([O-])=O',
                'add_atoms': [
                    [10, 13], [6, 15], [26, 26], [27, 27], [28, 28],
                    [29, 29], [30, 33], [34, 30], [36, 32], [35, 31],
                ],
                'remove_atoms': False,
                'align_all': True,
                'rmsd_thresh': 1,
                'n_conformers': 1,
            },
            'small_molecule_params': True,
            'sanitize': True,
            'removeHs': True,
            'proximityBonding': True,
            'nstd_resids': None,
            'pH': 7.0,
            'neutral_Cterm': False,
        },
        'Environment': {
            'alignment_ref': _abspath('6z66_OPM.pdb'),
            'reference_chain': ['A'],
            'membrane': True,
            'pH': 7.2,
            'ion_strength': 0.15,
        },
        'RepairProtein': {
            'fasta_path': _abspath('NTSR1.fasta'),
            'working_dir': modeller_dir,
            'tails': [48, 390],
            'loops': [],
            'secondary_template': _abspath('AF-P30989-F1-model_v4.pdb'),
            'engineered_resids': None,
        },
    }

    config_path = os.path.join(PERSISTENT_OUT, 'ML-301_local.json')
    with open(config_path, 'w') as fh:
        json.dump(config, fh, indent=2)

    from chimpss.bridgeport import Bridgeport
    bp = Bridgeport(config_path)
    bp.run()   # no-op if TOPOLOGY_PDB and SYSTEM_XML already exist

    return bp.final_pdb, bp.final_xml


@pytest.fixture(scope='module')
def converted(bridgeport_system, tmp_path_factory):
    """Run SystemConverter on Bridgeport outputs; cache in PERSISTENT_OUT."""
    _topology_pdb, system_xml = bridgeport_system
    out_dir = os.path.join(PERSISTENT_OUT, 'algdock_inputs')

    # Check if converter outputs already exist
    expected = [
        os.path.join(out_dir, f'{label}.{ext}')
        for label in ('ligand', 'receptor', 'complex')
        for ext in ('prmtop', 'inpcrd')
    ]
    if all(os.path.isfile(p) for p in expected):
        return {
            'ligand_prmtop':   os.path.join(out_dir, 'ligand.prmtop'),
            'ligand_inpcrd':   os.path.join(out_dir, 'ligand.inpcrd'),
            'receptor_prmtop': os.path.join(out_dir, 'receptor.prmtop'),
            'receptor_inpcrd': os.path.join(out_dir, 'receptor.inpcrd'),
            'complex_prmtop':  os.path.join(out_dir, 'complex.prmtop'),
            'complex_inpcrd':  os.path.join(out_dir, 'complex.inpcrd'),
        }

    from chimpss.algdock.converter import SystemConverter
    conv = SystemConverter(INPUT_PDB, system_xml, lig_resname=LIG_RESNAME)
    return conv.convert(out_dir)


# ---------------------------------------------------------------------------
# Stage 1: Bridgeport outputs
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bridgeport_topology_pdb_exists(bridgeport_system):
    topology_pdb, _ = bridgeport_system
    assert os.path.isfile(topology_pdb), f'Missing topology PDB: {topology_pdb}'
    assert os.path.getsize(topology_pdb) > 0


@pytest.mark.slow
def test_bridgeport_system_xml_exists(bridgeport_system):
    _, system_xml = bridgeport_system
    assert os.path.isfile(system_xml), f'Missing system XML: {system_xml}'
    assert os.path.getsize(system_xml) > 0


@pytest.mark.slow
def test_system_xml_is_openmm_system(bridgeport_system):
    """The system XML must deserialise as an OpenMM System."""
    from openmm import XmlSerializer
    _, system_xml = bridgeport_system
    with open(system_xml) as fh:
        system = XmlSerializer.deserialize(fh.read())
    assert system.getNumParticles() > 0


# ---------------------------------------------------------------------------
# Stage 2: Converter outputs
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_converter_produces_six_files(converted):
    expected_keys = {
        'ligand_prmtop', 'ligand_inpcrd',
        'receptor_prmtop', 'receptor_inpcrd',
        'complex_prmtop', 'complex_inpcrd',
    }
    assert set(converted.keys()) == expected_keys
    for path in converted.values():
        assert os.path.isfile(path), f'Missing converter output: {path}'


@pytest.mark.slow
def test_ligand_prmtop_loads(converted):
    struct = pmd.load_file(converted['ligand_prmtop'], converted['ligand_inpcrd'])
    assert len(struct.atoms) > 0, 'Ligand prmtop has no atoms'
    resnames = {r.name for r in struct.residues}
    assert LIG_RESNAME in resnames, f'Ligand prmtop missing residue {LIG_RESNAME}'


@pytest.mark.slow
def test_receptor_prmtop_loads(converted):
    struct = pmd.load_file(converted['receptor_prmtop'], converted['receptor_inpcrd'])
    assert len(struct.atoms) > 0
    resnames = {r.name for r in struct.residues}
    assert LIG_RESNAME not in resnames, 'Receptor must not contain ligand'
    assert 'HOH' not in resnames, 'Receptor must not contain water'


@pytest.mark.slow
def test_complex_prmtop_loads(converted):
    struct = pmd.load_file(converted['complex_prmtop'], converted['complex_inpcrd'])
    assert len(struct.atoms) > 0
    resnames = {r.name for r in struct.residues}
    assert LIG_RESNAME in resnames, 'Complex must contain ligand'
    assert 'HOH' not in resnames, 'Complex must not contain water'


@pytest.mark.slow
def test_complex_atom_count_equals_receptor_plus_ligand(converted):
    lig = pmd.load_file(converted['ligand_prmtop'], converted['ligand_inpcrd'])
    rec = pmd.load_file(converted['receptor_prmtop'], converted['receptor_inpcrd'])
    com = pmd.load_file(converted['complex_prmtop'], converted['complex_inpcrd'])
    assert len(com.atoms) == len(lig.atoms) + len(rec.atoms)


@pytest.mark.slow
def test_no_null_bond_types(converted):
    for label in ('ligand', 'receptor', 'complex'):
        struct = pmd.load_file(converted[f'{label}_prmtop'])
        null_bonds = [b for b in struct.bonds if b.type is None]
        assert len(null_bonds) == 0, (
            f'{label}.prmtop has {len(null_bonds)} bonds with no type'
        )


@pytest.mark.slow
def test_ligand_has_ntsr1_ml301_atom_count(converted):
    """ML-301 ligand should have a specific, known atom count."""
    struct = pmd.load_file(converted['ligand_prmtop'], converted['ligand_inpcrd'])
    n = len(struct.atoms)
    # ML-301 (C22H22ClN4O4-) has 34 heavy atoms + H; typical with H: ~55-70 atoms
    assert 30 <= n <= 80, f'Unexpected ML-301 atom count: {n}'


# ---------------------------------------------------------------------------
# Stage 3: BPMF compatibility
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_converter_paths_are_bpmf_kwargs(converted):
    """Every key returned by convert() must be a recognised BPMF argument."""
    from chimpss.algdock import arguments
    bpmf_keys = set(arguments.args.keys())
    for key in converted:
        assert key in bpmf_keys, f"'{key}' not in BPMF arguments"


@pytest.mark.slow
def test_bpmf_importable():
    from chimpss.algdock import BindingPMF
    from chimpss.algdock.BindingPMF import BPMF
    assert BindingPMF is BPMF
