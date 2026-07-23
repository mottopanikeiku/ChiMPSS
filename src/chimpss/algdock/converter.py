"""
chimpss.algdock.converter
=========================
Convert ChiMPSS MotorRow outputs (solvated PDB + OpenMM system XML) to the
AMBER prmtop/inpcrd format consumed natively by the vendored BPMF engine.

Produces three file pairs in the output directory:

  ligand.prmtop / ligand.inpcrd
      Ligand alone (residue name == lig_resname).

  receptor.prmtop / receptor.inpcrd
      Everything that is *not* the ligand and *not* solvent/ions —
      i.e. protein chains, lipid bilayer, any cofactors.

  complex.prmtop / complex.inpcrd
      Receptor + ligand together (no solvent, no ions).

Usage::

    from chimpss.algdock.converter import SystemConverter

    conv = SystemConverter(
        pdb_file='equil/step5_final.pdb',
        system_xml='systems/system.xml',
        lig_resname='UNK',
    )
    paths = conv.convert(output_dir='algdock_inputs/')
    # paths keys: ligand_prmtop, ligand_inpcrd,
    #             receptor_prmtop, receptor_inpcrd,
    #             complex_prmtop, complex_inpcrd

The returned dict maps directly to BPMF keyword arguments::

    from chimpss.algdock import BindingPMF
    BindingPMF(**paths, dir_CD='results/CD', dir_BC='results/BC', run_type='timed')

Background
----------
OpenMM system XMLs do not store force constants for constrained bonds (H–X bonds
are represented only as distance constraints, not HarmonicBondForce entries).
AMBER prmtop requires explicit bond parameters for every bond.  The converter
fills in canonical AMBER/GAFF force constants for these constrained bonds using
the constraint distance and element pair as the lookup key.  Force constants for
constrained bonds do not affect MD dynamics (the integrator enforces the fixed
length regardless); they only affect energy minimisation.
"""

import os
from typing import Optional

import parmed as pmd
import openmm.unit as unit
from openmm.app import PDBFile
from openmm import XmlSerializer

from chimpss.shared.logging import printf

# ---------------------------------------------------------------------------
# Solvent/ion residue names excluded from receptor and complex files
# ---------------------------------------------------------------------------
_SOLVENT_RESNAMES: frozenset = frozenset({
    'HOH', 'WAT', 'SOL', 'TIP3', 'TIP4', 'TIP5', 'TP3', 'TP4', 'TP5',
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN',
    'Na+', 'Cl-', 'K+',
})

# ---------------------------------------------------------------------------
# Canonical AMBER force constants for H–X constrained bonds
# key: (heavy_atom_Z, round(constraint_dist_Å, 2))
# value: (k in kcal/mol/Å², req in Å)
# ---------------------------------------------------------------------------
_HX_BOND_PARAMS: dict = {
    (8, 0.96): (553.0, 0.9572),   # TIP3P O–H
    (8, 0.97): (553.0, 0.9572),   # TIP3P O–H (slightly elongated)
    (8, 0.98): (553.0, 0.9790),   # OPC3 / SPC/E O–H
    (7, 1.01): (434.0, 1.0100),   # N–H (protein backbone / GAFF n-hn)
    (7, 1.02): (434.0, 1.0230),   # N–H (GAFF ligand amide, e.g. ML-301 at 1.0226 Å)
    (6, 1.08): (367.0, 1.0820),   # sp² / aromatic C–H (GAFF ca-ha)
    (6, 1.09): (340.0, 1.0900),   # sp³ C–H (GAFF c3-hc)
    (6, 1.10): (340.0, 1.0900),   # sp³ C–H alt
    (6, 1.11): (340.0, 1.1000),   # terminal methyl C–H
    (16, 1.34): (274.0, 1.3360),  # S–H (CYS thiol)
}

# Fallback parameters by heavy-atom element when exact distance not in table.
# Force constants for constrained bonds do not affect dynamics — only minimisation.
_HX_ELEMENT_FALLBACK: dict = {
    6:  (340.0, 1.0900),   # any C–H
    7:  (434.0, 1.0100),   # any N–H
    8:  (553.0, 0.9572),   # any O–H
    16: (274.0, 1.3360),   # any S–H
    15: (320.0, 1.4000),   # any P–H
}


class SystemConverter:
    """
    Convert a ChiMPSS solvated system (OpenMM XML + PDB) to AMBER files for the BPMF engine.

    Parameters
    ----------
    pdb_file : str
        Path to the equilibrated complex PDB (protein + ligand + membrane + solvent).
        Typically ``equil/step5_final.pdb`` from MotorRow.
    system_xml : str
        Path to the serialised OpenMM System XML.
        Typically ``systems/system.xml`` from Bridgeport.
    lig_resname : str
        Three-character residue name of the ligand (e.g. ``'UNK'``, ``'SR5'``).
    extra_solvent_resnames : list of str, optional
        Additional residue names to treat as solvent/ions (excluded from
        receptor and complex files).  The built-in list covers common water
        models and monovalent ions.
    """

    def __init__(
        self,
        pdb_file: str,
        system_xml: str,
        lig_resname: str,
        extra_solvent_resnames: Optional[list] = None,
    ):
        self.pdb_file = pdb_file
        self.system_xml = system_xml
        self.lig_resname = lig_resname.upper()

        self._solvent = set(_SOLVENT_RESNAMES)
        if extra_solvent_resnames:
            self._solvent.update(r.upper() for r in extra_solvent_resnames)

        printf(f'SystemConverter: loading {pdb_file}')
        self._pdb = PDBFile(pdb_file)
        with open(system_xml) as fh:
            self._system = XmlSerializer.deserialize(fh.read())

        n_pdb = self._pdb.topology.getNumAtoms()
        n_sys = self._system.getNumParticles()
        if n_pdb != n_sys:
            raise ValueError(
                f'Topology and System have different numbers of atoms '
                f'({n_pdb} vs. {n_sys}).\n'
                f'pdb_file must match the system XML exactly (use the Bridgeport '
                f'topology PDB, not an MDTraj-written equilibration PDB — MDTraj '
                f'box-wrapping can add/remove solvent atoms relative to the '
                f'original topology).'
            )

        printf('SystemConverter: building ParmEd structure from OpenMM topology…')
        self._struct = pmd.openmm.load_topology(
            self._pdb.topology, self._system, self._pdb.positions,
        )

        n_null = sum(1 for b in self._struct.bonds if b.type is None)
        printf(f'SystemConverter: {len(self._struct.atoms)} atoms, '
               f'{len(self._struct.bonds)} bonds '
               f'({n_null} constrained bonds need type assignment)')

        if n_null > 0:
            self._fill_constrained_bonds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, output_dir: str) -> dict:
        """
        Write ligand, receptor, and complex prmtop/inpcrd files.

        Parameters
        ----------
        output_dir : str
            Directory where the six output files are written.  Created if absent.

        Returns
        -------
        paths : dict
            Keys: ``ligand_prmtop``, ``ligand_inpcrd``,
                  ``receptor_prmtop``, ``receptor_inpcrd``,
                  ``complex_prmtop``, ``complex_inpcrd``.
            Maps directly to ``BPMF`` keyword arguments.
        """
        os.makedirs(output_dir, exist_ok=True)

        masks = self._build_masks()
        paths = {}

        for label, mask in (
            ('ligand',   masks['ligand']),
            ('receptor', masks['receptor']),
            ('complex',  masks['complex']),
        ):
            sub = self._struct[mask]
            prmtop_path = os.path.join(output_dir, f'{label}.prmtop')
            inpcrd_path = os.path.join(output_dir, f'{label}.inpcrd')

            sub.save(prmtop_path, overwrite=True)
            sub.save(inpcrd_path, overwrite=True)

            printf(f'  {label}: {len(sub.atoms)} atoms → {prmtop_path}')

            paths[f'{label}_prmtop'] = os.path.abspath(prmtop_path)
            paths[f'{label}_inpcrd'] = os.path.abspath(inpcrd_path)

        printf('SystemConverter: done.')
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_constrained_bonds(self) -> None:
        """Assign AMBER force constants to constrained bonds with no type."""
        constraint_map: dict = {}
        for i in range(self._system.getNumConstraints()):
            p1, p2, dist = self._system.getConstraintParameters(i)
            constraint_map[tuple(sorted([p1, p2]))] = dist.value_in_unit(unit.angstrom)

        filled = 0
        unfilled_samples: list = []

        for bond in self._struct.bonds:
            if bond.type is not None:
                continue

            pair_key = tuple(sorted([bond.atom1.idx, bond.atom2.idx]))
            dist = constraint_map.get(pair_key)
            if dist is None:
                unfilled_samples.append(bond)
                continue

            if bond.atom1.element == 1:
                heavy_z = bond.atom2.element
            else:
                heavy_z = bond.atom1.element

            params = _HX_BOND_PARAMS.get((heavy_z, round(dist, 2)))
            if params is None:
                params = _HX_ELEMENT_FALLBACK.get(heavy_z)
            if params is None:
                unfilled_samples.append(bond)
                continue

            k, req = params
            bond.type = pmd.BondType(k=k, req=req)
            filled += 1

        printf(f'  filled {filled} constrained bond types')
        if unfilled_samples:
            printf(
                f'  WARNING: {len(unfilled_samples)} bonds remain without type '
                f'(first 3: '
                + ', '.join(
                    f'{b.atom1.element}-{b.atom2.element}@{b.atom1.residue.name}'
                    for b in unfilled_samples[:3]
                )
                + '). prmtop write may fail — add entries to _HX_BOND_PARAMS.'
            )

    def _build_masks(self) -> dict:
        """Return ParmEd AMBER-style selection masks for the three subsets."""
        lig = self.lig_resname
        sol = ','.join(sorted(self._solvent))

        return {
            'ligand':   f':{lig}',
            'receptor': f'!:{lig},{sol}',
            'complex':  f'!:{sol}',
        }
