"""
Tests for chimpss.algdock.converter.SystemConverter.

Uses the RS1126 test system from ChiMPSS test_data (46 907 atoms, membrane
protein with POPC bilayer, 1 UNK ligand).
"""

import os
import pytest
import parmed as pmd

# Locate the test data relative to this file
_HERE = os.path.dirname(__file__)
_CHIMPSS_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_TEST_PDB = os.path.join(_CHIMPSS_ROOT, 'test_data', 'RS_sets', 'RS1126.pdb')
_TEST_XML = os.path.join(_CHIMPSS_ROOT, 'test_data', 'RS_sets', 'RS1126_sys.xml')
_LIG_RESNAME = 'UNK'

requires_test_data = pytest.mark.skipif(
    not os.path.isfile(_TEST_PDB),
    reason='RS1126 test data not found — skipping converter tests',
)


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    """Run the converter once; reuse across all tests in this module."""
    from chimpss.algdock.converter import SystemConverter

    tmp = tmp_path_factory.mktemp('algdock_inputs')
    conv = SystemConverter(_TEST_PDB, _TEST_XML, lig_resname=_LIG_RESNAME)
    return conv.convert(str(tmp))


@requires_test_data
def test_converter_import():
    from chimpss.algdock.converter import SystemConverter  # noqa: F401
    from chimpss.algdock import SystemConverter as SC2  # noqa: F401
    assert SystemConverter is SC2


@requires_test_data
def test_convert_returns_six_paths(converted):
    expected_keys = {
        'ligand_prmtop', 'ligand_inpcrd',
        'receptor_prmtop', 'receptor_inpcrd',
        'complex_prmtop', 'complex_inpcrd',
    }
    assert set(converted.keys()) == expected_keys


@requires_test_data
def test_all_files_exist(converted):
    for path in converted.values():
        assert os.path.isfile(path), f'Missing: {path}'


@requires_test_data
def test_ligand_atom_count(converted):
    struct = pmd.load_file(converted['ligand_prmtop'], converted['ligand_inpcrd'])
    # RS1126 UNK ligand has 68 atoms
    assert len(struct.atoms) == 68


@requires_test_data
def test_receptor_excludes_ligand_and_solvent(converted):
    struct = pmd.load_file(converted['receptor_prmtop'], converted['receptor_inpcrd'])
    resnames = {r.name for r in struct.residues}
    assert 'UNK' not in resnames, 'Receptor should not contain ligand'
    assert 'HOH' not in resnames, 'Receptor should not contain water'
    assert 'NA' not in resnames, 'Receptor should not contain ions'
    # Protein residues present
    assert 'ALA' in resnames or 'GLY' in resnames or 'LEU' in resnames


@requires_test_data
def test_complex_excludes_solvent(converted):
    struct = pmd.load_file(converted['complex_prmtop'], converted['complex_inpcrd'])
    resnames = {r.name for r in struct.residues}
    assert 'HOH' not in resnames, 'Complex should not contain water'
    assert 'NA' not in resnames, 'Complex should not contain sodium ions'
    assert 'UNK' in resnames, 'Complex must contain the ligand'


@requires_test_data
def test_complex_atom_count_equals_receptor_plus_ligand(converted):
    lig = pmd.load_file(converted['ligand_prmtop'], converted['ligand_inpcrd'])
    rec = pmd.load_file(converted['receptor_prmtop'], converted['receptor_inpcrd'])
    com = pmd.load_file(converted['complex_prmtop'], converted['complex_inpcrd'])
    assert len(com.atoms) == len(lig.atoms) + len(rec.atoms)


@requires_test_data
def test_no_null_bond_types(converted):
    """All bonds in every output file must have explicit types."""
    for label in ('ligand', 'receptor', 'complex'):
        struct = pmd.load_file(converted[f'{label}_prmtop'])
        null_bonds = [b for b in struct.bonds if b.type is None]
        assert len(null_bonds) == 0, (
            f'{label}.prmtop has {len(null_bonds)} bonds without type'
        )


@requires_test_data
def test_paths_map_to_bpmf_kwargs(converted):
    """The returned dict should match BPMF keyword argument names."""
    from chimpss.algdock import arguments
    bpmf_keys = set(arguments.args.keys())
    for key in converted:
        assert key in bpmf_keys, f"'{key}' is not a recognised BPMF argument"
