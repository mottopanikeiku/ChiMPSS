"""Net-charge accounting and counterion neutralization for built systems.

Bridgeport builds the solvent/membrane environment around the protein alone
(PDBFixer -> Modeller.addMembrane, which neutralizes), then trims the box and
inserts the ligand. The trim removes ions with everything else outside the
cutoff, and the ligand's charge is never counted, so the final system can carry
a net charge. The 5-HT2B systems ended up at +7 to +9 e with exactly equal
Na+/Cl- counts. Under PME a net charge is silently cancelled by a uniform
background, which is a known artefact source in membrane systems.

The robust fix is to neutralize the FINAL assembled system: read its exact
charge from the parameterized System, then swap that many bulk waters for
counterions and re-parameterize.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from openmm import NonbondedForce, unit
from openmm.app import Element, Modeller, Topology

WATER_RESNAMES = frozenset({'HOH', 'WAT', 'TIP3', 'TIP4', 'SOL', 'OPC', 'OPC3', 'SPC'})
ION_RESNAMES = frozenset({'NA', 'CL', 'K', 'Na+', 'Cl-', 'K+', 'SOD', 'CLA', 'POT'})


def system_net_charge(system) -> float:
    """Sum of NonbondedForce particle charges, in elementary charges."""
    nbs = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(nbs) != 1:
        raise ValueError(f'expected exactly one NonbondedForce, found {len(nbs)}')
    nb = nbs[0]
    return float(sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                     for i in range(nb.getNumParticles())))


def _orthorhombic_lengths(topology: Topology) -> np.ndarray:
    vecs = topology.getPeriodicBoxVectors()
    if vecs is None:
        raise ValueError('topology has no periodic box vectors')
    m = np.array([[v[0], v[1], v[2]] for v in vecs.value_in_unit(unit.nanometer)], dtype=float)
    if not np.allclose(m - np.diag(np.diag(m)), 0.0, atol=1e-6):
        raise ValueError('only orthorhombic boxes are supported')
    return np.diag(m).copy()


def _min_image_dist(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image distances from each row of a (n,3) to each row of b (m,3)."""
    d = a[:, None, :] - b[None, :, :]
    d -= box * np.round(d / box)
    return np.sqrt((d * d).sum(-1))


def neutralize_with_counterions(
    topology: Topology,
    positions,
    net_charge: float,
    positive_ion: Tuple[str, str, str] = ('NA', 'Na', 'Na'),
    negative_ion: Tuple[str, str, str] = ('CL', 'Cl', 'Cl'),
    min_solute_dist: float = 0.6,
    min_ion_dist: float = 0.5,
    charge_tol: float = 0.01,
) -> Tuple[Topology, list, List[int]]:
    """Replace bulk waters with counterions so the system becomes neutral.

    Parameters
    ----------
    net_charge : float
        Net charge of the parameterized system (e.g. from `system_net_charge`).
        Must be within `charge_tol` of an integer.
    positive_ion, negative_ion : (resname, atom name, element symbol)
        Must match the ion templates of the force field. Defaults are Amber/OPC3.
    min_solute_dist, min_ion_dist : float, nm
        Water oxygens closer than `min_solute_dist` to any non-water, non-ion atom
        (protein, lipid, ligand) are never replaced, and new ions are kept at
        least `min_ion_dist` from every ion. Waters are taken farthest-first
        from the solute, i.e. from bulk solvent.

    Returns
    -------
    topology, positions, replaced_water_residue_indices
    """
    q = int(round(net_charge))
    if abs(net_charge - q) > charge_tol:
        raise ValueError(f'net charge {net_charge:.4f} is not near an integer; '
                         f'refusing to guess how many counterions to add')
    if q == 0:
        return topology, positions, []

    resname, atom_name, symbol = negative_ion if q > 0 else positive_ion
    n_ions = abs(q)

    box = _orthorhombic_lengths(topology)
    xyz = np.array(positions.value_in_unit(unit.nanometer)
                   if hasattr(positions, 'value_in_unit') else
                   [p.value_in_unit(unit.nanometer) for p in positions], dtype=float)

    waters, water_O, solute_idx, ion_idx = [], [], [], []
    for res in topology.residues():
        if res.name in WATER_RESNAMES:
            o = [a.index for a in res.atoms() if a.element is not None and a.element.symbol == 'O']
            if len(o) == 1:
                waters.append(res)
                water_O.append(o[0])
        elif res.name in ION_RESNAMES:
            ion_idx.extend(a.index for a in res.atoms())
        else:
            solute_idx.extend(a.index for a in res.atoms()
                              if a.element is None or a.element.symbol != 'H')
    if not waters:
        raise ValueError('no water residues found to replace')

    O = xyz[np.array(water_O)]
    S = xyz[np.array(solute_idx)]
    d_solute = np.empty(len(O))
    for start in range(0, len(O), 2048):
        d_solute[start:start + 2048] = _min_image_dist(O[start:start + 2048], S, box).min(1)

    order = np.argsort(-d_solute)
    taken_xyz = [xyz[i] for i in ion_idx]
    chosen: List[int] = []
    for w in order:
        if d_solute[w] < min_solute_dist:
            break
        if taken_xyz and _min_image_dist(O[w:w + 1], np.array(taken_xyz), box).min() < min_ion_dist:
            continue
        chosen.append(int(w))
        taken_xyz.append(O[w])
        if len(chosen) == n_ions:
            break
    if len(chosen) < n_ions:
        raise RuntimeError(f'only {len(chosen)}/{n_ions} bulk waters satisfy '
                           f'min_solute_dist={min_solute_dist} nm and '
                           f'min_ion_dist={min_ion_dist} nm')

    modeller = Modeller(topology, positions)
    modeller.delete([waters[w] for w in chosen])

    ion_top = Topology()
    ion_top.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    chain = ion_top.addChain()
    element = Element.getBySymbol(symbol)
    for _ in chosen:
        r = ion_top.addResidue(resname, chain)
        ion_top.addAtom(atom_name, element, r)
    modeller.add(ion_top, [unit.Quantity(O[w], unit.nanometer) for w in chosen])
    renumber_solvent_residues(modeller.topology)

    return modeller.topology, modeller.positions, [waters[w].index for w in chosen]


def renumber_solvent_residues(topology: Topology) -> None:
    """Give residues in solvent-only chains sequential ids (1..9999, wrapping).

    Water chains hold far more than 9999 residues, so printed PDB residue ids
    repeat. That is harmless until two residues with the same printed id end up
    ADJACENT -- e.g. after deleting the water between them -- at which point
    OpenMM's PDB reader merges them and silently drops the second one's atoms
    (seen on 5-HT2B methysergide_A225G_mutseq: 3 atoms lost). Sequential ids
    guarantee neighbours always differ. Chains containing any non-solvent
    residue (protein, ligand, lipid) are left untouched.
    """
    solvent = WATER_RESNAMES | ION_RESNAMES
    for chain in topology.chains():
        residues = list(chain.residues())
        if residues and all(r.name in solvent for r in residues):
            for k, res in enumerate(residues):
                res.id = str(k % 9999 + 1)


__all__: Sequence[str] = ('system_net_charge', 'neutralize_with_counterions', 'renumber_solvent_residues',
                          'WATER_RESNAMES', 'ION_RESNAMES')
