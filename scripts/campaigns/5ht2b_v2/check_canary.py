"""Check a FultonMarket v2 canary run on real output. Exit 1 on any failure."""
import os
import sys

import numpy as np

out, pdb = sys.argv[1], sys.argv[2]
n_atoms = sum(1 for l in open(pdb) if l.startswith(('ATOM', 'HETATM')))
sv = os.path.join(out, 'saved_variables')
segs = sorted((int(d) for d in os.listdir(sv)), key=int)
fails = []


def need(ok, msg):
    print(f'[{"ok" if ok else "FAIL"}] {msg}', flush=True)
    if not ok:
        fails.append(msg)


def pos(k):
    d = os.path.join(sv, str(k))
    nf = np.load(os.path.join(d, 'states.npy')).shape[0]
    nr = os.path.getsize(os.path.join(d, 'positions.npy')) // (nf * n_atoms * 12)
    return np.memmap(os.path.join(d, 'positions.npy'), dtype='float32', mode='r', shape=(nf, nr, n_atoms, 3))


need(len(segs) >= 3, f'{len(segs)} sub-simulations saved (need >= 3)')
for k in segs[1:]:
    X, Xp = pos(k), pos(k - 1)
    nr = X.shape[1]
    last_states = np.load(os.path.join(sv, str(k - 1), 'states.npy'))[-1]
    # replica i of the new sub-sim starts in state i, from whichever replica held state i
    worst = 0.0
    for i in range(nr):
        r = int(np.where(last_states == i)[0][0])
        worst = max(worst, float(np.abs(np.asarray(X[0, i]) - np.asarray(Xp[-1, r])).max()))
    need(worst < 1e-3, f'sub-sim {k}: every replica starts where its state ended in sub-sim {k-1} (max |d| {worst:.1e} nm)')
    spread = max(float(np.sqrt(((np.asarray(X[0, 0]) - np.asarray(X[0, i])) ** 2).sum(-1).mean())) for i in range(1, nr))
    need(spread > 0.02, f'sub-sim {k}: replicas start from DIFFERENT structures (max RMS vs replica 0 {spread:.3f} nm)')
B = np.concatenate([np.load(os.path.join(sv, str(k), 'box_vectors.npy')) for k in segs])
Lx, Ly, Lz = B[..., 0, 0], B[..., 1, 1], B[..., 2, 2]
need(np.allclose(Lx, Ly, rtol=1e-6), 'membrane barostat: box x == y (XY isotropic)')
need(np.std(Lx / Lz) > 1e-5, f'membrane barostat: x/z ratio varies (std {np.std(Lx / Lz):.2e}; isotropic would be ~1e-7)')
print('CANARY PASS' if not fails else f'CANARY FAIL: {fails}')
sys.exit(1 if fails else 0)
