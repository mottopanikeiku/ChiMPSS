"""Rebuild one 5-HT2B system for the v2 campaign (run in the prep1 env).

Identical to the original Bridgeport build except for two deliberate changes:
  1. the ligand SMILES is protonated on its basic amine (formal charge +1);
  2. the final assembled system is neutralized with counterions.

The repaired, solvated protein environment from the original build
(work_dir/proteins/<name>_env.pdb) is reused verbatim, so the protein model,
membrane and water packing are unchanged. Alignment and ligand extraction are
re-run; both are deterministic and are checked against the original build.

Then writes <name>_FG_HMR.xml exactly as /home/fcetin/MotorRow/generate_hmr_xmls.py
did (chimpss.bridgeport._utils.apply_force_groups + apply_hmr).

Usage: python build_v2.py <name>
"""
import json
import os
import shutil
import sys

import numpy as np

PROJ = '/expanse/lustre/projects/iit127/fcetin/5ht2b'
V2 = f'{PROJ}/v2'
sys.path.insert(0, '/home/fcetin/Bridgeport/Bridgeport')
sys.path.insert(0, '/home/fcetin/ChiMPSS/src')

from Bridgeport import Bridgeport  # noqa: E402  (old standalone repo, as originally built)
from openmm import XmlSerializer  # noqa: E402
from openmm.app import PDBFile  # noqa: E402

from chimpss.shared.system_charge import neutralize_with_counterions, system_net_charge  # noqa: E402

# The only SMILES edit per ligand: protonate the basic amine.
#   ergolines: N6 tertiary amine (ring D, bears the N-methyl)
#   LY266097:  tetrahydro-beta-carboline secondary amine
PROTONATE = {
    'ergoline': ('CN([C@@H]2', 'C[NH+]([C@@H]2'),
    'LY266097': ('[C@@H](NCC3)', '[C@@H]([NH2+]CC3)'),
}


def log(msg):
    print(f'[build_v2] {msg}', flush=True)


def heavy_xyz(pdb):
    out = []
    for line in open(pdb):
        if line.startswith(('ATOM', 'HETATM')) and line[76:78].strip() != 'H':
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(out)


def main(name):
    old_json = f'{PROJ}/prep/json/{name}.json'
    params = json.load(open(old_json))
    smi = params['Ligand']['smiles']
    key = 'LY266097' if name == 'LY266097' else 'ergoline'
    a, b = PROTONATE[key]
    assert smi.count(a) == 1, f'{name}: protonation site pattern {a!r} not unique in {smi}'
    params['Ligand']['smiles'] = smi.replace(a, b)
    params['working_dir'] = f'{V2}/work_dir'
    new_json = f'{V2}/json/{name}.json'
    json.dump(params, open(new_json, 'w'), indent=6)
    log(f'{name}: SMILES {smi}  ->  {params["Ligand"]["smiles"]}')

    bp = Bridgeport(input_json=new_json)
    bp.align_to_reference()
    bp.separate_lig_prot()

    # Reuse the original repaired + solvated environment verbatim.
    os.makedirs(bp.prot_only_dir, exist_ok=True)
    old_env = f'{PROJ}/work_dir/proteins/{name}_env.pdb'
    bp.env_pdb = os.path.join(bp.prot_only_dir, f'{name}_env.pdb')
    shutil.copy(old_env, bp.env_pdb)
    log(f'reusing environment {old_env}')

    bp.ligand_prep()

    # Check: ligand heavy atoms identical to the original build's (same frame).
    old_lig = heavy_xyz(f'{PROJ}/work_dir/ligands/{name}.pdb')
    new_lig = heavy_xyz(bp.lig_pdb)
    assert old_lig.shape == new_lig.shape, (old_lig.shape, new_lig.shape)
    dmax = float(np.abs(old_lig - new_lig).max())
    assert dmax < 1e-3, f'ligand heavy atoms moved by {dmax} A vs original build'
    log(f'ligand heavy atoms match original build (max |d| {dmax:.1e} A)')

    # Build, then neutralize the FINAL system and rebuild.
    bp.generate_systems()
    q0 = system_net_charge(bp.sys)
    log(f'net charge before neutralization: {q0:+.4f} e')
    if round(q0) != 0:
        pdb = PDBFile(bp.env_pdb)
        top, pos, replaced = neutralize_with_counterions(pdb.topology, pdb.positions, q0)
        with open(bp.env_pdb, 'w') as f:
            PDBFile.writeFile(top, pos, f, keepIds=True)
        log(f'replaced {len(replaced)} bulk waters with {"Cl-" if q0 > 0 else "Na+"}')
        bp.generate_systems()
    q1 = system_net_charge(bp.sys)
    assert abs(q1) < 1e-3, f'system still charged after neutralization: {q1:+.6f}'
    log(f'net charge after neutralization: {q1:+.6f} e')

    bp.reposition_at_origin()

    # HMR + force groups, identical to generate_hmr_xmls.py
    from chimpss.bridgeport._utils import apply_force_groups, apply_hmr
    topology = PDBFile(bp.final_pdb).topology
    system = XmlSerializer.deserialize(open(bp.final_xml).read())
    m0 = sum(system.getParticleMass(i)._value for i in range(system.getNumParticles()))
    apply_force_groups(system)
    apply_hmr(system, topology)
    m1 = sum(system.getParticleMass(i)._value for i in range(system.getNumParticles()))
    assert abs(m1 - m0) < 1e-3, (m0, m1)
    out = f'{V2}/work_dir/systems_hmr/{name}_FG_HMR.xml'
    with open(out, 'w') as f:
        f.write(XmlSerializer.serialize(system))
    log(f'HMR system written: {out} (mass {m0:.3f} -> {m1:.3f})')
    log('DONE')


if __name__ == '__main__':
    main(sys.argv[1])
