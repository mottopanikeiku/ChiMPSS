#!/usr/bin/env python

"""
OpenMM implementation of pose-specific restraint force fields

Provides flat-bottom harmonic restraints on internal torsions and external
position/orientation for pose-specific BPMF calculations.

This module replicates the functionality of PoseFF.py (MMTK version) using
OpenMM Custom Forces and the new OrientationRestraintForce (OpenMM 2025).
"""

import numpy as np
import openmm
from openmm import unit


class InternalRestraintForceOpenMM:
    """
    Flat-bottom harmonic restraints on internal torsions

    Uses OpenMM CustomTorsionForce to implement:
        E = k * max(0, (theta - theta0)^2 - b^2)

    where:
        theta = current dihedral angle
        theta0 = reference dihedral angle
        b = half-width of flat bottom region
        k = force constant (scaled by alpha_r during CD)
    """

    def __init__(self, torsion_specs, k=200.0):
        """
        Initialize internal torsion restraints

        Parameters
        ----------
        torsion_specs : list of lists
            Each spec is [i, j, k, l, theta0, hwidth]
            where i,j,k,l are atom indices, theta0 is reference angle (radians),
            and hwidth is half-width of flat region (radians)
        k : float, optional
            Initial force constant in kJ/mol (default: 200.0)
            Will be scaled by alpha_r = tanh(16*alpha^2) during CD
        """
        self.torsion_specs = torsion_specs
        self.k = k
        self.force = None
        self.force_index = None

    def _create_force(self):
        """
        Create CustomTorsionForce with flat-bottom harmonic potential

        Energy expression uses max() function directly.
        OpenMM automatically keeps theta in [-pi, pi] range.
        """
        # Simple flat-bottom harmonic potential
        # theta is automatic variable in CustomTorsionForce
        energy_expr = "k_internal * max(0, (theta - theta0)^2 - b^2)"

        force = openmm.CustomTorsionForce(energy_expr)
        force.addPerTorsionParameter('theta0')  # Reference angle (radians)
        force.addPerTorsionParameter('b')       # Half-width (radians)
        force.addGlobalParameter('k_internal', self.k)   # Force constant (kJ/mol)
        force.setName("PoseInternalRestraint")

        # Add all torsion restraints
        for spec in self.torsion_specs:
            i, j, k_idx, l, theta0, hwidth = spec
            force.addTorsion(i, j, k_idx, l, [theta0, hwidth])

        return force

    def add_to_system(self, system, topology):
        """
        Add internal restraint force to OpenMM system

        Parameters
        ----------
        system : openmm.System
            OpenMM system to add force to
        topology : openmm.app.Topology
            OpenMM topology (not used, for interface compatibility)

        Returns
        -------
        int
            Force index in system
        """
        self.force = self._create_force()
        self.force_index = system.addForce(self.force)
        return self.force_index

    def setParams(self, k):
        """
        Update force constant

        Parameters
        ----------
        k : float
            New force constant in kJ/mol
            During CD: k = k_angular_int * alpha_r
            where alpha_r = tanh(16 * alpha^2)
        """
        self.k = k
        # Update global parameter in context
        # This will be called via context.setParameter() when context exists

    def set_k(self, k):
        """
        Compatibility method for system.py interface

        Parameters
        ----------
        k : float
            New force constant in kJ/mol
        """
        self.setParams(k)

    def updateParametersInContext(self, context):
        """
        Update parameters in an existing context

        Parameters
        ----------
        context : openmm.Context
            OpenMM context to update
        """
        context.setParameter('k_internal', self.k)

    def getEnergy(self, context):
        """
        Get energy of this force in current context

        Parameters
        ----------
        context : openmm.Context
            OpenMM context

        Returns
        -------
        float
            Potential energy in kJ/mol
        """
        # Get state with only this force group
        force_group = self.force.getForceGroup()
        state = context.getState(getEnergy=True, groups={force_group})
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


class ExternalRestraintForceOpenMM_PathA:
    """
    External restraints using OrientationRestraintForce (Path A - recommended)

    Combines:
    1. Spatial restraint: Flat-bottom distance from reference position
    2. Orientation restraint: Using built-in OrientationRestraintForce

    This is simpler and more robust than manual angle/dihedral restraints.
    """

    def __init__(self, topology, external_specs,
                 k_spatial=200.0, k_angular=200.0,
                 hwidth_spatial=0.1):
        """
        Initialize external restraints

        Parameters
        ----------
        topology : openmm.app.Topology
            OpenMM topology
        external_specs : list
            [root1, root2, root3, X1, Y1, Z1, phi, theta, psi]
            root1,2,3: root atom indices
            X1,Y1,Z1: reference position (nm)
            phi,theta,psi: orientation angles (radians) - used to compute reference positions
        k_spatial : float, optional
            Spatial restraint force constant in kJ/mol (default: 200.0)
        k_angular : float, optional
            Angular restraint force constant in kJ/mol (default: 200.0)
        hwidth_spatial : float, optional
            Spatial restraint half-width in nm (default: 0.1)
        """
        self.topology = topology
        self.root_atoms = external_specs[:3]
        self.ref_position = np.array(external_specs[3:6])
        self.ref_angles = external_specs[6:9]  # phi, theta, psi
        self.k_spatial = k_spatial
        self.k_angular = k_angular
        self.hwidth_spatial = hwidth_spatial
        self.spatial_force = None
        self.orientation_force = None

    def _create_spatial_restraint(self):
        """
        Create flat-bottom spatial restraint

        Restrains root atom (root1) to reference position with flat-bottom potential:
            E = k * max(0, (r - r0)^2 - b^2)

        where r is distance from reference position
        """
        # Use CustomExternalForce (acts on single particle)
        energy_expr = """
            k_spatial_ext * max(0, r^2 - b^2);
            r = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)
        """

        self.spatial_force = openmm.CustomExternalForce(energy_expr)
        self.spatial_force.addGlobalParameter('k_spatial_ext', self.k_spatial)
        self.spatial_force.addPerParticleParameter('x0')
        self.spatial_force.addPerParticleParameter('y0')
        self.spatial_force.addPerParticleParameter('z0')
        self.spatial_force.addPerParticleParameter('b')  # Half-width

        # Apply to root atom (restrains to reference position)
        self.spatial_force.addParticle(
            self.root_atoms[0],
            [self.ref_position[0], self.ref_position[1], self.ref_position[2],
             self.hwidth_spatial]
        )
        self.spatial_force.setName("PoseSpatialRestraint")

    def _create_orientation_restraint(self):
        """
        Create orientation restraint using OrientationRestraintForce

        This is a built-in OpenMM force (added in 2025) that restrains orientation:
            E = 2*k*sin^2(theta/2)

        where theta is rotation angle from reference orientation.
        Approximately harmonic for small angles: E ~ (k/2)*theta^2
        """
        # Get current positions of root atoms to use as reference
        # In practice, these would be transformed by phi/theta/psi angles
        # For now, use simple approach: root atom positions define reference

        # Create list of positions for all atoms in topology
        natoms = self.topology.getNumAtoms()
        ref_positions = [openmm.Vec3(0, 0, 0)] * natoms

        # Set reference positions for root atoms
        # TODO: Properly transform these using phi/theta/psi angles
        # For now, just use reference position as center
        for i, root_idx in enumerate(self.root_atoms):
            offset = openmm.Vec3(i * 0.1, 0, 0)  # Simple offset for now
            ref_positions[root_idx] = openmm.Vec3(
                self.ref_position[0] + offset[0],
                self.ref_position[1] + offset[1],
                self.ref_position[2] + offset[2]
            )

        # Create OrientationRestraintForce
        # API: OrientationRestraintForce(k, referencePositions, particles)
        self.orientation_force = openmm.OrientationRestraintForce(
            self.k_angular,
            ref_positions,
            self.root_atoms  # Only restrain root atoms
        )
        self.orientation_force.setName("PoseOrientationRestraint")

    def add_to_system(self, system, topology):
        """
        Add external restraint forces to OpenMM system

        Parameters
        ----------
        system : openmm.System
            OpenMM system to add forces to
        topology : openmm.app.Topology
            OpenMM topology (not used, for interface compatibility)

        Returns
        -------
        int
            Index of the spatial force (for interface compatibility)
        """
        # Always create fresh forces for each system (can't reuse Force objects)
        self._create_spatial_restraint()
        self._create_orientation_restraint()
        spatial_idx = system.addForce(self.spatial_force)
        orientation_idx = system.addForce(self.orientation_force)
        # Store both indices but return spatial for interface compatibility
        self.spatial_force_index = spatial_idx
        self.orientation_force_index = orientation_idx
        return spatial_idx

    def setParams(self, k_spatial, k_angular):
        """
        Update force constants

        Parameters
        ----------
        k_spatial : float
            New spatial force constant in kJ/mol
        k_angular : float
            New angular force constant in kJ/mol

        During CD: k = k_ext * alpha_r where alpha_r = tanh(16 * alpha^2)
        """
        self.k_spatial = k_spatial
        self.k_angular = k_angular

    def set_k_spatial(self, k_spatial):
        """
        Compatibility method for system.py interface

        Parameters
        ----------
        k_spatial : float
            New spatial force constant in kJ/mol
        """
        self.k_spatial = k_spatial

    def set_k_angular(self, k_angular):
        """
        Compatibility method for system.py interface

        Parameters
        ----------
        k_angular : float
            New angular force constant in kJ/mol
        """
        self.k_angular = k_angular

    def updateParametersInContext(self, context):
        """
        Update parameters in existing context

        Parameters
        ----------
        context : openmm.Context
            OpenMM context to update
        """
        context.setParameter('k_spatial_ext', self.k_spatial)
        self.orientation_force.setK(self.k_angular)
        self.orientation_force.updateParametersInContext(context)

    def getEnergy(self, context):
        """
        Get total external restraint energy

        Parameters
        ----------
        context : openmm.Context
            OpenMM context

        Returns
        -------
        dict
            {'spatial': float, 'orientation': float, 'total': float}
            Energies in kJ/mol
        """
        # Get spatial energy
        spatial_group = self.spatial_force.getForceGroup()
        spatial_state = context.getState(getEnergy=True, groups={spatial_group})
        spatial_energy = spatial_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

        # Get orientation energy
        orientation_group = self.orientation_force.getForceGroup()
        orientation_state = context.getState(getEnergy=True, groups={orientation_group})
        orientation_energy = orientation_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

        return {
            'spatial': spatial_energy,
            'orientation': orientation_energy,
            'total': spatial_energy + orientation_energy
        }


class ExternalRestraintForceOpenMM_PathB:
    """
    External restraints using CustomCompoundBondForce (Path B - fallback)

    Traditional approach with explicit angle/dihedral restraints.
    Use this if OrientationRestraintForce validation fails.

    Uses LocalCoordinatesSite for virtual sites instead of manual dummy atoms.
    """

    def __init__(self, system, topology, external_specs,
                 k_spatial=200.0, k_angular=200.0,
                 hwidth_spatial=0.1, hwidth_angular=np.pi/4):
        """
        Initialize external restraints with manual angle/dihedral implementation

        Parameters
        ----------
        system : openmm.System
            OpenMM system
        topology : openmm.app.Topology
            OpenMM topology
        external_specs : list
            [root1, root2, root3, X1, Y1, Z1, phi, theta, omega]
        k_spatial : float
            Spatial force constant (kJ/mol)
        k_angular : float
            Angular force constant (kJ/mol)
        hwidth_spatial : float
            Spatial half-width (nm)
        hwidth_angular : float
            Angular half-width (radians)
        """
        raise NotImplementedError(
            "Path B (CustomCompoundBondForce) not yet implemented. "
            "Use ExternalRestraintForceOpenMM_PathA (OrientationRestraintForce) instead. "
            "Path B will only be implemented if Path A validation fails."
        )


# For backward compatibility and ease of use
InternalRestraintForce = InternalRestraintForceOpenMM
ExternalRestraintForce = ExternalRestraintForceOpenMM_PathA
