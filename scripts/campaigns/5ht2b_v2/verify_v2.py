"""Verify a v2 5-HT2B system against its original build (chimpss env).

The v2 build may differ from the original in exactly three ways:
  * the ligand gains one H (protonated amine) and carries +1 e;
  * n bulk waters are replaced by n counterions, making the system neutral;
  * nothing else -- protein, lipids, remaining water and ions are identical.
Also checks the HMR variant. Exit code 1 on any failure.

Usage: python verify_v2.py <name> [<name> ...]
"""
import sys

import numpy as np
import openmm as mm
from openmm import unit

P = '/expanse/lustre/projects/iit127/fcetin/5ht2b'
NEUTRAL_H = {'LSD': 25, 'LSD_L362F_mutseq': 25, 'lisuride': 26, 'lisuride_L362F_mutseq': 26,
             'methylergonovine': 25, 'methylergonovine_T140A_mutseq': 25,
             'methysergide': 27, 'methysergide_A225G_mutseq': 27, 'LY266097': 23}
WATER = {'HOH', 'WAT'}
IONS = {'NA', 'CL'}


def read_pdb(path):
    res, name, el, xyz, rid = [], [], [], [], []
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')):
            res.append(line[17:20].strip()); name.append(line[12:16].strip())
            el.append(line[76:78].strip()); rid.append(line[22:26].strip())
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(res), np.array(name), np.array(el), np.array(xyz), np.array(rid)


def load(path):
    return mm.XmlSerializer.deserialize(open(path).read())


def charges(system):
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    return np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                     for i in range(system.getNumParticles())])


def masses(system):
    return np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                     for i in range(system.getNumParticles())])


def check(name):
    fails = []

    def need(ok, msg):
        print(f'   [{"ok" if ok else "FAIL"}] {msg}')
        if not ok:
            fails.append(msg)

    print(f'=== {name} ===')
    o_res, o_nm, o_el, o_xyz, _ = read_pdb(f'{P}/work_dir/systems/{name}.pdb')
    n_res, n_nm, n_el, n_xyz, _ = read_pdb(f'{P}/v2/work_dir/systems/{name}.pdb')
    plain = load(f'{P}/v2/work_dir/systems/{name}.xml')
    hmr = load(f'{P}/v2/work_dir/systems_hmr/{name}_FG_HMR.xml')
    q = charges(plain)

    n_cl_o, n_na_o = (o_res == 'CL').sum(), (o_res == 'NA').sum()
    n_cl, n_na = (n_res == 'CL').sum(), (n_res == 'NA').sum()
    added = (n_cl - n_cl_o) + (n_na - n_na_o)
    need(plain.getNumParticles() == len(n_res), f'system/PDB atom counts match ({len(n_res)})')
    need(len(n_res) == len(o_res) - 2 * added + 1,
         f'atoms {len(o_res)} -> {len(n_res)} = old - 2x{added} ions + 1 ligand H')
    need(abs(q.sum()) < 1e-3, f'net charge {q.sum():+.5f} e')
    lig = n_res == 'UNK'
    need(abs(q[lig].sum() - 1) < 1e-3, f'ligand charge {q[lig].sum():+.5f} e')
    nH = int((n_el[lig] == 'H').sum())
    need(nH == NEUTRAL_H[name] + 1, f'ligand H {nH} = neutral {NEUTRAL_H[name]} + 1')
    w_o, w_n = (o_res == 'HOH').sum() // 3, (n_res == 'HOH').sum() // 3   # 3-site water, per-atom arrays
    need(w_n == w_o - added,
         f'waters {w_o} -> {w_n} (-{added}); ions Na {n_na_o}->{n_na}, Cl {n_cl_o}->{n_cl}')

    # Everything except ligand H, ions and water must be identical, atom for atom, in order.
    keep_o = ~np.isin(o_res, list(WATER | IONS)) & ~((o_res == 'UNK') & (o_el == 'H'))
    keep_n = ~np.isin(n_res, list(WATER | IONS)) & ~((n_res == 'UNK') & (n_el == 'H'))
    same_ids = (np.array_equal(o_res[keep_o], n_res[keep_n]) and
                np.array_equal(o_nm[keep_o], n_nm[keep_n]))
    need(same_ids, f'protein/lipid/ligand-heavy atoms identical in identity and order ({keep_n.sum()})')
    if same_ids:
        d = np.abs(o_xyz[keep_o] - n_xyz[keep_n]).max()
        need(d < 2e-3, f'...and in position (max |d| {d:.1e} A)')

    # New ions sit in bulk solvent.
    solute = ~np.isin(n_res, list(WATER | IONS))
    new_ion_idx = np.where(np.isin(n_res, list(IONS)))[0][-added:] if added else []
    if added:
        box = np.array([v.value_in_unit(unit.angstrom) for v in
                        [plain.getDefaultPeriodicBoxVectors()[k][k] for k in range(3)]])
        dd = n_xyz[new_ion_idx][:, None, :] - n_xyz[solute][None, :, :]
        dd -= box * np.round(dd / box)
        dmin = np.sqrt((dd ** 2).sum(-1)).min()
        need(dmin >= 5.9, f'new ions >= 6 A from protein/lipid/ligand (closest {dmin:.1f} A)')

    # HMR variant: same as original recipe (validated bit-for-bit on the old LSD build).
    mp, mh = masses(plain), masses(hmr)
    need(abs(mh.sum() - mp.sum()) < 1e-3, f'HMR mass conserved ({mp.sum():.2f} Da)')
    raised = mh - mp > 1e-6
    water = n_res == 'HOH'
    need(not (raised & water).any(), 'HMR leaves water untouched')
    ligH = lig & (n_el == 'H')
    need((raised & ligH).sum() == ligH.sum(), f'HMR raises all {ligH.sum()} ligand H')
    need(sorted(set(np.round(mh[raised], 2))) == [2.02, 2.52], 'HMR H masses are {2.02, 2.52} Da')
    need(hmr.getNumConstraints() == plain.getNumConstraints() > 0,
         f'constraints {hmr.getNumConstraints()}')
    need(not any('Barostat' in type(f).__name__ or type(f).__name__ == 'CustomExternalForce'
                 for f in hmr.getForces()), 'no barostat or restraint forces in the system')

    # Salt bridge geometry at the start: protonated N-H -> D135 carboxylate.
    i_np = [i for i in np.where(lig)[0] if n_el[i] == 'N']
    od = np.where((n_res == 'ASP') & np.isin(n_nm, ['OD1', 'OD2']) &
                  (np.array([r for r in read_pdb(f'{P}/v2/work_dir/systems/{name}.pdb')[4]]) == '135'))[0]
    dN = min(np.linalg.norm(n_xyz[i] - n_xyz[j]) for i in i_np for j in od)
    print(f'   [info] closest ligand N to D135 carboxylate O at build: {dN:.2f} A')
    return fails


if __name__ == '__main__':
    bad = {n: f for n in sys.argv[1:] if (f := check(n))}
    print('\nALL PASS' if not bad else f'\nFAILURES: {bad}')
    sys.exit(1 if bad else 0)
