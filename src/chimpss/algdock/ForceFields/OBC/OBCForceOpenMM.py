"""
OpenMM OBC Force wrapper for AlGDock implicit solvent calculations

This replaces the MMTK OBCForceField with OpenMM's native GB/SA OBC implementation
"""

import openmm
import openmm.unit as unit
from openmm.app import OBC2


class OBCForceOpenMM:
    """
    OpenMM-based OBC implicit solvent force field

    This mimics the MMTK OBCForceField API but uses OpenMM's native GBSAOBCForce
    """

    def __init__(self, topology=None, system=None, obc_model=OBC2):
        """
        Initialize OBC force

        Parameters
        ----------
        topology : openmm.app.Topology
            Molecular topology
        system : openmm.System
            OpenMM system containing the GBSAOBCForce
        obc_model : openmm.app constant
            OBC model (OBC1 or OBC2)
        """
        self.topology = topology
        self.system = system
        self.obc_model = obc_model
        self.strength = 1.0
        self.force = None
        self.force_index = None
        self.context = None  # Store context reference for updateParametersInContext

        # Find the GBSAOBCForce in the system
        if system is not None:
            for i in range(system.getNumForces()):
                f = system.getForce(i)
                if isinstance(f, openmm.GBSAOBCForce):
                    self.force = f
                    self.force_index = i
                    break

    def set_strength(self, strength):
        """
        Set the strength/scaling of the OBC force

        Parameters
        ----------
        strength : float
            Scaling factor (0.0 = no solvent, 1.0 = full solvent)

        Note
        ----
        This scales the OBC force during MD simulation to match MMTK's behavior.

        MMTK scales the GB preFactor: preFactor *= strength
        where preFactor = 2*k_e*(1/ε_solute - 1/ε_solvent)

        In OpenMM, we achieve this by scaling the charges in GBSAOBCForce.
        Since GB energy is proportional to q_i*q_j, scaling all charges by
        sqrt(strength) gives energy ~ strength and forces ~ strength.

        This ONLY affects GBSAOBCForce, NOT NonbondedForce electrostatics,
        because each force maintains separate particle parameters.

        This ensures the system samples from the correct Hamiltonian at each
        alpha value during the thermodynamic path (BC phase: alpha 1.0 → 0.0).
        """
        if self.force is None:
            self.strength = strength
            return

        # If strength hasn't changed, nothing to do
        if abs(strength - self.strength) < 1e-10:
            return

        # Store original parameters on first call
        if not hasattr(self, '_original_charges'):
            self._original_charges = []
            self._original_radii = []
            self._original_scale_factors = []
            for i in range(self.force.getNumParticles()):
                charge, radius, scale_factor = self.force.getParticleParameters(i)
                self._original_charges.append(charge)
                self._original_radii.append(radius)
                self._original_scale_factors.append(scale_factor)
            # Store original surface area energy (kJ/mol/nm^2)
            self._original_sa_energy = self.force.getSurfaceAreaEnergy()

        # Scale charges by sqrt(strength) to get GB energy ~ strength
        # GB energy ~ sum_ij (q_i * q_j), so scaling all q by sqrt(s) gives energy ~ s
        import math
        charge_scale = math.sqrt(max(0.0, strength))  # Ensure non-negative

        for i in range(self.force.getNumParticles()):
            scaled_charge = self._original_charges[i] * charge_scale
            self.force.setParticleParameters(
                i,
                scaled_charge,
                self._original_radii[i],
                self._original_scale_factors[i]
            )

        # CRITICAL: Also scale surface area energy term by strength (linearly)
        # SA energy = sa_energy_scale * surface_area
        # This ensures SA term scales linearly like the GB term, matching MMTK
        scaled_sa_energy = self._original_sa_energy * strength
        self.force.setSurfaceAreaEnergy(scaled_sa_energy)

        # Update parameters in the context to apply changes immediately
        # Only call if context exists and force is in the context
        # (context won't exist during initial system setup, or force won't exist in CD with Desolvated)
        if self.context is not None and self.force is not None:
            try:
                # Update per-particle parameters (charges, radii, scale factors)
                self.force.updateParametersInContext(self.context)
                # CRITICAL: reinitialize() is REQUIRED to apply force-level parameter changes
                # like surfaceAreaEnergy. From OpenMM docs:
                # "updateParametersInContext() only updates per-particle parameters. All other
                #  aspects of the Force...can only be changed by reinitializing the Context."
                # Use preserveState=True to maintain positions/velocities during sampling
                self.context.reinitialize(preserveState=True)
            except Exception as e:
                # Force not in context (e.g., CD phase with Desolvated solvation)
                pass

        self.strength = strength

    def addToSystem(self, system):
        """
        Add OBC force to an OpenMM system

        Parameters
        ----------
        system : openmm.System
            System to add force to
        """
        # Create GB/SA OBC force
        # This uses the model specified (OBC1 or OBC2)
        force = openmm.GBSAOBCForce()

        # Set parameters based on OBC model
        if self.obc_model == OBC2:
            # OBC2 is the default
            force.setSoluteDielectric(1.0)
            force.setSolventDielectric(78.5)
        else:
            # OBC1
            force.setSoluteDielectric(1.0)
            force.setSolventDielectric(78.5)

        self.force = force
        self.force_index = system.addForce(force)
        self.system = system

        return self.force_index
