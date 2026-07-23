#!/usr/bin/env python

"""
OpenMM version of rigid body analysis for pose-specific BPMF

Identifies rings, rigid bodies, and soft torsions from OpenMM topology.
Replicates functionality of rigid_bodies.py but works with OpenMM instead of MMTK.
"""

import numpy as np
import openmm
from openmm import app

def dihedral(p1, p2, p3, p4):
    """
    Calculate dihedral angle between four points

    Based on algorithm from chimpss.algdock.BAT.dihedral

    Parameters
    ----------
    p1, p2, p3, p4 : array-like
        Four 3D positions

    Returns
    -------
    float
        Dihedral angle in radians [-pi, pi]
    """
    from numpy.linalg import norm
    v1 = p2 - p1
    v2 = p2 - p3
    v3 = p3 - p4
    a = np.cross(v1, v2)
    a_norm = norm(a)
    if a_norm < 1e-10:
        return 0.0
    a = a / a_norm
    b = np.cross(v3, v2)
    b_norm = norm(b)
    if b_norm < 1e-10:
        return 0.0
    b = b / b_norm
    c = np.sum(a * b)
    v2_norm = norm(v2)
    if v2_norm < 1e-10:
        return 0.0
    s = np.sum(np.cross(b, a) * (v2 / v2_norm))
    return np.arctan2(s, c)


def _join_sets(sets):
    """
    Join together sets that have intersecting elements

    Parameters
    ----------
    sets : list of sets
        Sets to join if they intersect

    Returns
    -------
    list of sets
        Merged sets with no intersections
    """
    while not _sets_are_unique(sets):
        output_sets = []
        for new_set in sets:
            joined = False
            for old_set in output_sets:
                if len(new_set.intersection(old_set)) > 0:
                    output_sets.remove(old_set)
                    joined_set = old_set.union(new_set)
                    output_sets.append(joined_set)
                    joined = True
            if not joined:
                output_sets.append(new_set)
        sets = output_sets
    return sets


def _sets_are_unique(sets):
    """
    Returns True if each set has no intersecting elements with every other set

    Parameters
    ----------
    sets : list of sets
        Sets to check

    Returns
    -------
    bool
        True if all sets are disjoint
    """
    nsets = len(sets)
    for a in range(nsets):
        for b in range(a + 1, nsets):
            if len(sets[a].intersection(sets[b])) > 0:
                return False
    return True


class TopologyGraph:
    """
    Graph representation of OpenMM topology for ring detection

    Stores atoms, bonds, and connectivity information
    """

    def __init__(self, topology):
        """
        Build graph from OpenMM topology

        Parameters
        ----------
        topology : openmm.app.Topology
            OpenMM topology object
        """
        self.topology = topology
        self.atoms = list(topology.atoms())
        self.bonds = list(topology.bonds())

        # Build adjacency list: atom_index -> [bonded_atom_indices]
        self.adjacency = {i: [] for i in range(len(self.atoms))}
        for bond in self.bonds:
            i = bond.atom1.index
            j = bond.atom2.index
            self.adjacency[i].append(j)
            self.adjacency[j].append(i)

    def get_bonded_atoms(self, atom_idx):
        """
        Get indices of atoms bonded to this atom

        Parameters
        ----------
        atom_idx : int
            Index of atom

        Returns
        -------
        list of int
            Indices of bonded atoms
        """
        return self.adjacency[atom_idx]

    def get_atom_mass(self, atom_idx):
        """
        Get mass of atom

        Parameters
        ----------
        atom_idx : int
            Index of atom

        Returns
        -------
        float
            Atomic mass in amu
        """
        atom = self.atoms[atom_idx]
        if atom.element is not None:
            return atom.element.mass.value_in_unit(openmm.unit.amu)
        return 0.0


def find_rings(graph):
    """
    Find all rings in the topology using depth-first search

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation of topology

    Returns
    -------
    list of sets
        Each set contains atom indices in a ring
    """
    unique_rings = []

    # Try starting from each atom
    for start_atom_idx in range(len(graph.atoms)):
        ring = _find_ring_from_atom(graph, start_atom_idx)
        if ring and ring not in unique_rings:
            unique_rings.append(ring)

    # Join fused rings (e.g., naphthalene)
    return _join_sets(unique_rings)


def _find_ring_from_atom(graph, start_atom_idx):
    """
    Find a ring starting from a specific atom using DFS

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    start_atom_idx : int
        Starting atom index

    Returns
    -------
    set or None
        Set of atom indices in ring, or None if no ring found
    """
    ancestors = [start_atom_idx]
    ring = _in_ring(graph, ancestors)
    return set(ring) if ring else None


def _in_ring(graph, ancestors):
    """
    Recursive search for circular path

    Replicates the MMTK _in_ring function but works with OpenMM topology

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    ancestors : list of int
        Path of atom indices explored so far

    Returns
    -------
    list of int
        Atom indices forming a ring, or empty list if no ring
    """
    # Get atoms bonded to last atom in path, excluding recent ancestors
    current_atom = ancestors[-1]
    bonded_atoms = graph.get_bonded_atoms(current_atom)

    # Don't go back to immediate ancestors (avoid oscillation)
    exclude_atoms = ancestors[1:] if len(ancestors) > 1 else []
    children = [a for a in bonded_atoms if a not in exclude_atoms]

    for child in children:
        # Found a ring!
        if child == ancestors[0]:
            if len(ancestors) > 2:
                return ancestors
            else:
                # Ring of size 2 (bond) - not valid
                return []

        # Continue search
        in_ring = _in_ring(graph, ancestors + [child])
        if len(in_ring) > 0:
            return in_ring

    return []


def identify_rigid_bodies(graph, rings):
    """
    Identify rigid portions of molecule

    Rigid body = ring atoms + terminal atoms bonded to ring
    Terminal atom = atom with only 1 bond

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    rings : list of sets
        Ring atom index sets

    Returns
    -------
    list of sets
        Each set contains atom indices in a rigid body
    """
    rigid_bodies = []

    for ring in rings:
        # Find terminal atoms attached to this ring
        terminal_atoms = set()
        for ring_atom_idx in ring:
            for neighbor_idx in graph.get_bonded_atoms(ring_atom_idx):
                # Terminal = only 1 bond
                if len(graph.get_bonded_atoms(neighbor_idx)) == 1:
                    terminal_atoms.add(neighbor_idx)

        # Rigid body = ring + terminal atoms
        rigid_bodies.append(ring.union(terminal_atoms))

    # Join intersecting rigid bodies
    return _join_sets(rigid_bodies)


def identify_soft_torsions(graph, rigid_bodies, torsion_list):
    """
    Find rotatable bonds that connect rigid bodies

    Soft torsion = dihedral angle where central two atoms are in different rigid bodies

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    rigid_bodies : list of sets
        Rigid body atom index sets
    torsion_list : list of tuples
        All torsions as (i, j, k, l) atom index tuples

    Returns
    -------
    list of tuples
        Soft torsions as (i, j, k, l) atom index tuples
    """
    soft_torsions = []

    def find_rigid_body(atom_idx):
        """Find which rigid body contains this atom, or -1 if none"""
        for body_id, body in enumerate(rigid_bodies):
            if atom_idx in body:
                return body_id
        return -1

    for (i, j, k, l) in torsion_list:
        # Check if central atoms j and k are in different rigid bodies
        j_body = find_rigid_body(j)
        k_body = find_rigid_body(k)

        # Soft torsion if in different bodies (or not in any rigid body)
        if j_body != k_body:
            soft_torsions.append((i, j, k, l))

    return soft_torsions


def select_root_atoms(graph, rings, positions):
    """
    Choose 3 atoms to define reference frame for external restraints

    Preference:
    1. Heaviest terminal atom attached to largest ring
    2. Two atoms in that ring bonded to atom 1
    3. Third atom defining the plane

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    rings : list of sets
        Ring atom index sets
    positions : np.ndarray
        Atomic positions (natoms, 3)

    Returns
    -------
    tuple of int
        (root1, root2, root3) atom indices
    """
    if len(rings) == 0:
        # No rings - choose heaviest atoms
        return _select_root_atoms_acyclic(graph, positions)

    # Sort rings by size (largest first)
    ordered_rings = sorted(rings, key=lambda r: len(r), reverse=True)

    # Find heaviest terminal atom attached to largest ring
    for ring in ordered_rings:
        terminal_atoms = []
        for ring_atom_idx in ring:
            for neighbor_idx in graph.get_bonded_atoms(ring_atom_idx):
                if len(graph.get_bonded_atoms(neighbor_idx)) == 1:
                    mass = graph.get_atom_mass(neighbor_idx)
                    terminal_atoms.append((mass, neighbor_idx, ring_atom_idx))

        if len(terminal_atoms) > 0:
            # Sort by mass (heaviest last)
            terminal_atoms.sort()
            mass, root1, ring_atom = terminal_atoms[-1]

            # root2 is the ring atom bonded to root1
            root2 = ring_atom

            # root3 is another atom bonded to root2 (preferably in ring)
            root3_candidates = [idx for idx in graph.get_bonded_atoms(root2) if idx != root1]
            if len(root3_candidates) > 0:
                # Prefer atoms in the ring
                ring_candidates = [idx for idx in root3_candidates if idx in ring]
                root3 = ring_candidates[0] if ring_candidates else root3_candidates[0]
            else:
                # Fallback: use another ring atom
                ring_list = sorted(list(ring))
                root3 = ring_list[0] if ring_list[0] != root2 else ring_list[1]

            return (root1, root2, root3)

    # No terminal atoms found - use ring atoms
    largest_ring = ordered_rings[0]
    ring_list = sorted(list(largest_ring))
    return (ring_list[0], ring_list[1], ring_list[2])


def _select_root_atoms_acyclic(graph, positions):
    """
    Select root atoms for molecule without rings

    Choose 3 heaviest atoms

    Parameters
    ----------
    graph : TopologyGraph
        Graph representation
    positions : np.ndarray
        Atomic positions

    Returns
    -------
    tuple of int
        (root1, root2, root3) atom indices
    """
    # Get all atoms with masses
    atoms_with_mass = [(graph.get_atom_mass(i), i) for i in range(len(graph.atoms))]
    atoms_with_mass.sort(reverse=True)

    # Return 3 heaviest
    if len(atoms_with_mass) >= 3:
        return (atoms_with_mass[0][1], atoms_with_mass[1][1], atoms_with_mass[2][1])
    elif len(atoms_with_mass) == 2:
        return (atoms_with_mass[0][1], atoms_with_mass[1][1], atoms_with_mass[1][1])
    elif len(atoms_with_mass) == 1:
        return (atoms_with_mass[0][1], atoms_with_mass[0][1], atoms_with_mass[0][1])
    else:
        return (0, 0, 0)


class RigidBodyIdentifier:
    """
    OpenMM equivalent of MMTK rigid_bodies.identifier

    Identifies rings, rigid bodies, soft torsions, and generates
    restraint specifications for pose-specific BPMF
    """

    def __init__(self, topology, positions):
        """
        Analyze topology and identify structural features

        Parameters
        ----------
        topology : openmm.app.Topology
            OpenMM topology object
        positions : np.ndarray or openmm.unit.Quantity
            Atomic positions, shape (natoms, 3)
            If Quantity, will be converted to nm
        """
        self.topology = topology

        # Convert positions to numpy array in nm
        if hasattr(positions, 'value_in_unit'):
            self.positions = positions.value_in_unit(openmm.unit.nanometer)
        else:
            self.positions = np.array(positions)

        self.natoms = topology.getNumAtoms()

        # Build graph representation
        self.graph = TopologyGraph(topology)

        # Identify rings
        self.rings = find_rings(self.graph)

        # Identify rigid bodies
        self.rigid_bodies = identify_rigid_bodies(self.graph, self.rings)

        # Select root atoms for external restraints
        self.root_atoms = select_root_atoms(self.graph, self.rings, self.positions)

        # Generate all torsions in molecule
        self._all_torsions = self._generate_all_torsions()

        # Identify soft torsions
        self.soft_torsions = identify_soft_torsions(
            self.graph, self.rigid_bodies, self._all_torsions)

    def _generate_all_torsions(self):
        """
        Generate all proper torsions in the molecule

        A torsion i-j-k-l exists if:
        - j and k are bonded
        - i is bonded to j (i != k)
        - l is bonded to k (l != j)

        Returns
        -------
        list of tuples
            Torsions as (i, j, k, l) atom index tuples
        """
        torsions = []

        for bond in self.graph.bonds:
            j = bond.atom1.index
            k = bond.atom2.index

            # Find outer atoms
            i_candidates = [idx for idx in self.graph.get_bonded_atoms(j) if idx != k]
            l_candidates = [idx for idx in self.graph.get_bonded_atoms(k) if idx != j]

            # Create torsions for all combinations
            for i in i_candidates:
                for l in l_candidates:
                    torsions.append((i, j, k, l))

        return torsions

    def poseInp(self):
        """
        Generate restraint specifications for pose force field

        Returns
        -------
        tuple
            (TorsionRestraintSpecs, ExternalRestraintSpecs)

            TorsionRestraintSpecs : list of lists
                [[i, j, k, l, torsion_value, half_width], ...]

            ExternalRestraintSpecs : list
                [root1, root2, root3, X1, Y1, Z1, phi, theta, psi]
                For OrientationRestraintForce, only root atoms and position needed
        """
        # Internal torsion restraints
        TorsionRestraintSpecs = []
        for (i, j, k, l) in self.soft_torsions:
            # Calculate current dihedral value
            torsion_value = dihedral(
                self.positions[i], self.positions[j],
                self.positions[k], self.positions[l])

            # Calculate half-width based on number of bonds
            # More bonds = narrower restraint
            max_nbonded = max(
                len(self.graph.get_bonded_atoms(j)),
                len(self.graph.get_bonded_atoms(k)))
            hwidth = 2 * np.pi / (max_nbonded - 1) / 2.0

            TorsionRestraintSpecs.append([i, j, k, l, torsion_value, hwidth])

        # External restraints
        # For now, return simplified specs (root atoms + position)
        # Will be expanded when implementing OrientationRestraintForce
        root1, root2, root3 = self.root_atoms

        # Use position of root1 as reference center
        X1, Y1, Z1 = self.positions[root1]

        # Placeholder for orientation angles (will compute properly later)
        phi = 0.0
        theta = 0.0
        psi = 0.0

        ExternalRestraintSpecs = [root1, root2, root3, X1, Y1, Z1, phi, theta, psi]

        return [TorsionRestraintSpecs, ExternalRestraintSpecs]
