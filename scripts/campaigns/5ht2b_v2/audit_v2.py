"""Science sanity audit of v2 production output (chimpss env; numpy only).

Per system, over every saved frame of the 300 K state (state 0):
  * salt bridge: ligand protonated N (4 covalent neighbours) to D135 OD1/OD2
  * pose: ligand heavy-atom centroid to D135 CA, vs the equilibrated start
  * membrane: area per lipid (box Lx*Ly / lipids per leaflet)
and, over all replicas, temperature-space mixing: walkers are followed across
sub-simulation boundaries (new replica i continues the walker that ended in
state i; inserted states start new walkers), counting full
bottom -> top -> bottom round trips.

Usage: python audit_v2.py [system ...]
"""
import os
import sys

import numpy as np

V2 = '/expanse/lustre/projects/iit127/fcetin/5ht2b/v2'
SYSTEMS = ['LSD', 'lisuride', 'methylergonovine', 'methysergide', 'LY266097',
           'lisuride_L362F_mutseq', 'methylergonovine_T140A_mutseq',
           'methysergide_A225G_mutseq', 'LSD_L362F_mutseq']


def read_pdb(path):
    rows = [l for l in open(path) if l.startswith(('ATOM', 'HETATM'))]
    res = np.array([l[17:20].strip() for l in rows])
    name = np.array([l[12:16].strip() for l in rows])
    rid = np.array([l[22:26].strip() for l in rows])
    el = np.array([l[76:78].strip() for l in rows])
    xyz = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in rows]) / 10.0  # nm
    return res, name, rid, el, xyz


def positions(seg, n_atoms):
    nf = np.load(os.path.join(seg, 'states.npy'), mmap_mode='r').shape[0]
    path = os.path.join(seg, 'positions.npy')
    nr = os.path.getsize(path) // (nf * n_atoms * 12)
    return np.memmap(path, dtype='float32', mode='r', shape=(nf, nr, n_atoms, 3))


def mic(d, box):
    return d - box * np.round(d / box)


def audit(system):
    res, name, rid, el, xyz0 = read_pdb(f'{V2}/equil_hmr/{system}/step_5.pdb')
    n_atoms = len(res)
    lig = np.where(res == 'UNK')[0]
    lig_heavy = lig[el[lig] != 'H']
    # protonated amine: the ligand N with 4 covalent neighbours (incl. its new H)
    nN = [i for i in lig if el[i] == 'N']
    nbrs = {i: int((np.linalg.norm(xyz0[lig] - xyz0[i], axis=1) < 0.165).sum() - 1) for i in nN}
    nplus = [i for i, k in nbrs.items() if k == 4]
    od = np.where((res == 'ASP') & (rid == '135') & np.isin(name, ['OD1', 'OD2']))[0]
    ca135 = np.where((res == 'ASP') & (rid == '135') & (name == 'CA'))[0]
    n_lipid = len(set(zip(res[res == 'POP'], rid[res == 'POP'])))
    assert len(nplus) == 1 and len(od) == 2 and len(ca135) == 1, (system, nbrs, len(od))
    nplus = nplus[0]

    sv = f'{V2}/fultonmarket/{system}/saved_variables'
    segs = [os.path.join(sv, d) for d in sorted(os.listdir(sv), key=int)]
    salt, pose, apl = [], [], []
    walkers_state, walker_of = [], None
    trips, phase, next_id = {}, {}, 0
    for k, seg in enumerate(segs):
        S = np.load(os.path.join(seg, 'states.npy'))
        T = np.load(os.path.join(seg, 'temperatures.npy'))
        B = np.load(os.path.join(seg, 'box_vectors.npy'))
        X = positions(seg, n_atoms)
        K = len(T)
        # --- 300 K observables
        for f in range(S.shape[0]):
            r = int(np.where(S[f] == 0)[0][0])
            box = np.diag(B[f, r]).astype(float)
            p = np.asarray(X[f, r], dtype=float) if False else None
            pl = np.asarray(X[f, r, lig_heavy], dtype=float)
            pl = pl[0] + mic(pl - pl[0], box)
            pn = np.asarray(X[f, r, nplus], dtype=float)
            po = np.asarray(X[f, r, od], dtype=float)
            pca = np.asarray(X[f, r, ca135[0]], dtype=float)
            salt.append(np.min(np.linalg.norm(mic(po - pn, box), axis=1)))
            pose.append(np.linalg.norm(mic(pl.mean(0) - pca, box)))
            apl.append(box[0] * box[1] / (n_lipid / 2))
        # --- walkers: replica i of this segment continues the walker that ended in state i
        if walker_of is None:
            walker_of = list(range(K)); next_id = K
        else:
            prevT = prev_T
            new = []
            for i in range(K):
                j = np.where(np.abs(prevT - T[i]) < 1e-6)[0]
                if len(j):
                    r_old = int(np.where(prev_S_last == j[0])[0][0])
                    new.append(walker_of[r_old])
                else:
                    new.append(next_id); next_id += 1
            walker_of = new
        for f in range(S.shape[0]):
            for r in range(K):
                w, st = walker_of[r], int(S[f, r])
                ph = phase.get(w, None)
                if st == 0:
                    if ph == 'top':
                        trips[w] = trips.get(w, 0) + 1
                    phase[w] = 'bottom'
                elif st == K - 1 and ph == 'bottom':
                    phase[w] = 'top'
        prev_T, prev_S_last = T, S[-1]
    salt, pose, apl = map(np.array, (salt, pose, apl))
    start_pose = np.linalg.norm(xyz0[lig_heavy].mean(0) - xyz0[ca135[0]])
    start_salt = np.min(np.linalg.norm(xyz0[od] - xyz0[nplus], axis=1))
    n_walk = next_id
    h = len(apl) // 2
    print(f'{system:30s} segs {len(segs):2d} | salt bridge N+..OD: start {start_salt*10:.2f} A, '
          f'300K median {np.median(salt)*10:.2f} A, <4 A {100*np.mean(salt < 0.4):.0f}% | '
          f'pose (lig-D135 CA) start {start_pose*10:.1f} A, 300K median {np.median(pose)*10:.1f} A, max {pose.max()*10:.1f} A | '
          f'APL {apl[:h].mean():.3f} -> {apl[h:].mean():.3f} nm^2 | '
          f'round trips {sum(trips.values())} over {n_walk} walkers ({sum(trips.values())/n_walk:.2f}/walker)',
          flush=True)


if __name__ == '__main__':
    for s in (sys.argv[1:] or SYSTEMS):
        audit(s)
