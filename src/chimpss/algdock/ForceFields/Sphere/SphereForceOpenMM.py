"""
OpenMM implementation of a flat-bottom harmonic spherical restraint.

This restraint keeps the center of mass of a molecule within a sphere.
If the COM is outside the sphere, a harmonic restoring force is applied.
"""

import openmm
import openmm.unit as unit
import numpy as np


class SphereForceOpenMM:
    """
    Flat-bottom harmonic potential for a sphere

    Applies a restraint force when the center of mass of the molecule
    moves outside a spherical region.

    Energy: E = 0.5 * k * (r - max_R)^2 if r > max_R, else 0
    where r is the distance from COM to center, k = 10000 kJ/mol/nm^2
    """

    def __init__(self, center, max_R, name='Sphere'):
        """
        Initialize the spherical restraint.

        Parameters
        ----------
        center : numpy.array or list
            The center of the sphere (x, y, z) in nanometers
        max_R : float
            The maximum radius in nanometers
        name : str
            Name for this force field term
        """
        self.center = np.array(center)
        self.max_R = max_R
        self.name = name
        self.k = 10000.0  # kJ/mol/nm^2, matching MMTK implementation
        # Calculate the sphere volume
        self.volume = 4.0/3.0 * np.pi * (self.max_R ** 3)

    def add_to_system(self, system, topology):
        """
        Add the spherical restraint force to an OpenMM system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system to add the force to
        topology : openmm.app.Topology
            The topology containing the atoms

        Returns
        -------
        openmm.Force
            The created force object
        """
        # Create a CustomCentroidBondForce that restrains the COM to a sphere
        # We use 2 groups: group 1 is the COM, group 2 is the fixed center point
        # The energy is: 0.5 * k * max(0, r - max_R)^2
        # where r is the distance between the two groups

        force = openmm.CustomCentroidBondForce(
            2,  # number of groups (COM and center)
            "0.5 * k * max(0, distance(g1, g2) - max_R)^2"
        )

        # Define per-bond parameters
        force.addPerBondParameter("k")
        force.addPerBondParameter("max_R")

        # First, add a dummy particle to the system to represent the sphere center
        # We need to do this BEFORE adding groups so we can exclude it from group 1
        num_real_atoms = system.getNumParticles()
        system.addParticle(0.0 * unit.dalton)  # massless particle
        dummy_idx = system.getNumParticles() - 1

        # Add group 1: all REAL atoms (weighted by mass for COM calculation)
        # IMPORTANT: Don't include the dummy particle in the COM calculation!
        group_atoms = []
        group_weights = []
        for atom_idx in range(num_real_atoms):  # Only iterate over real atoms
            group_atoms.append(atom_idx)
            mass = system.getParticleMass(atom_idx).value_in_unit(unit.dalton)
            group_weights.append(mass)

        group1_index = force.addGroup(group_atoms, group_weights)

        # Add the dummy particle to all existing forces that track particles
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            if hasattr(f, 'addParticle'):
                # Add dummy particle with zero parameters to existing forces
                if f.__class__.__name__ == 'NonbondedForce':
                    f.addParticle(0.0, 1.0, 0.0)  # zero charge, unit sigma, zero epsilon
                elif f.__class__.__name__ == 'GBSAOBCForce':
                    f.addParticle(0.0, 0.15, 1.0)  # zero charge, default radius, default scale

        # Add group 2 with just the dummy particle
        group2_index = force.addGroup([dummy_idx], [1.0])

        # Add the bond (restraint) between the two groups
        force.addBond(
            [group1_index, group2_index],  # groups involved
            [self.k, self.max_R]  # k and max_R parameters
        )

        # Set the force name
        force.setName(self.name)

        # Add force to system
        system.addForce(force)

        # Store the dummy particle index so we can set its position later
        self.dummy_particle_index = dummy_idx

        return force

    def randomPoint(self):
        """
        Returns a random point within the sphere.

        Returns
        -------
        tuple
            (x, y, z) coordinates in nanometers
        """
        x, y, z = self._randomPointInSphere()
        return (x * self.max_R + self.center[0],
                y * self.max_R + self.center[1],
                z * self.max_R + self.center[2])

    def _randomPointInSphere(self):
        """
        Returns a random point within a unit sphere.

        Returns
        -------
        tuple
            (x, y, z) coordinates in the unit sphere
        """
        r2 = 2
        while r2 > 1:
            x, y, z = np.random.uniform(-1, 1, size=3)
            r2 = x*x + y*y + z*z
        return (x, y, z)
