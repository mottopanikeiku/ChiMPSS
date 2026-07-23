"""
OpenMM GridForce wrapper for AlGDock grid-based energy calculations

This replaces the MMTK InterpolationForceField with OpenMM's GridForce plugin
"""

import ctypes
import numpy as np
import openmm
import openmm.unit as unit
import gridforceplugin as gfp

# C API wrappers — bypass SWIG cross-module type mismatch.
# gridforceplugin.GridForce inherits from OpenMM::Force in C++ but SWIG
# registers separate type tables per .so, so openmm Force methods fail when
# called on a GridForce object.  The C API uses void* and works correctly.
_libomm = ctypes.CDLL('libOpenMM.so', mode=ctypes.RTLD_GLOBAL)
_libgf  = ctypes.CDLL('libOpenMMGridForce.so', mode=ctypes.RTLD_GLOBAL)

_libomm.OpenMM_System_addForce.restype  = ctypes.c_int
_libomm.OpenMM_System_addForce.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_libomm.OpenMM_Force_setName.restype  = None
_libomm.OpenMM_Force_setName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_libomm.OpenMM_Force_setForceGroup.restype  = None
_libomm.OpenMM_Force_setForceGroup.argtypes = [ctypes.c_void_p, ctypes.c_int]

_libomm.OpenMM_Force_getForceGroup.restype  = ctypes.c_int
_libomm.OpenMM_Force_getForceGroup.argtypes = [ctypes.c_void_p]

def _force_ptr(force):
    return ctypes.c_void_p(int(force.this))

def _system_add_force(system, force):
    return _libomm.OpenMM_System_addForce(
        ctypes.c_void_p(int(system.this)), _force_ptr(force))

def _force_set_name(force, name):
    _libomm.OpenMM_Force_setName(_force_ptr(force), name.encode())

def _force_set_group(force, group):
    _libomm.OpenMM_Force_setForceGroup(_force_ptr(force), int(group))

def _force_get_group(force):
    return _libomm.OpenMM_Force_getForceGroup(_force_ptr(force))

# Bypass SWIG type mismatch for GridForce.updateParametersInContext(context):
# _gridforceplugin.so's SWIG registry doesn't recognize openmm.Context objects.
# Call the C++ mangled symbol directly with void* pointers instead.
_gridforce_update_params_in_context = getattr(
    _libgf,
    '_ZN15GridForcePlugin9GridForce25updateParametersInContextERN6OpenMM7ContextE'
)
_gridforce_update_params_in_context.restype  = None
_gridforce_update_params_in_context.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

def _gridforce_updateParametersInContext(gridforce, context):
    _gridforce_update_params_in_context(
        ctypes.c_void_p(int(gridforce.this)),
        ctypes.c_void_p(int(context.this))
    )


class GridForceOpenMM:
    """
    OpenMM-based grid force field using the GridForce plugin

    This mimics the MMTK InterpolationForceField API but uses OpenMM's GridForce
    """

    def __init__(self, FN,
                 name='GridForce',
                 strength=1.0,
                 scaling_property='amber_charge',
                 scaling_prefactor=None,
                 inv_power=None,
                 grid_thresh=-1.0,
                 energy_thresh=-1.0):
        """
        Initialize grid force from file

        Parameters
        ----------
        FN : str
            Grid file name (.dx or .nc format)
        name : str
            Name for the grid force
        strength : float
            Scaling factor for energies/gradients
        scaling_property : str
            Atomic property for per-atom scaling (e.g., 'amber_charge')
        scaling_prefactor : float
            Additional scaling factor
        inv_power : float
            Transform grid values by grid^(1/inv_power)
        grid_thresh : float
            Cap grid values using tanh
        energy_thresh : float
            Maximum allowed energy
        """
        self.name = name
        self.strength = strength
        self.scaling_property = scaling_property
        self.scaling_prefactor = scaling_prefactor
        self.inv_power = inv_power
        self.grid_thresh = grid_thresh
        self.energy_thresh = energy_thresh

        # Load grid data
        import chimpss.algdock.IO
        IO_Grid = chimpss.algdock.IO.Grid()
        self.grid_data = IO_Grid.read(FN, multiplier=0.1)  # Convert Angstrom to nm

        # Convert grid values from kcal/mol to kJ/mol for OpenMM
        # Grid files are created with MMTK which uses kcal/mol, but OpenMM uses kJ/mol
        self.grid_data['vals'] = self.grid_data['vals'] * 4.184

        # DEBUG: Print grid statistics BEFORE transformation
        import numpy as np
        # print(f"DEBUG GridForceOpenMM: Loading {name} grid from {FN}")
        # print(f"  Grid vals BEFORE transform - shape: {self.grid_data['vals'].shape}")
        # print(f"  Grid vals BEFORE transform - min/max/mean: {self.grid_data['vals'].min():.3f}/{self.grid_data['vals'].max():.3f}/{self.grid_data['vals'].mean():.3f}")
        # print(f"  inv_power={self.inv_power}, grid_thresh={self.grid_thresh}")
        # print(f"  Grid spacing (nm): {self.grid_data['spacing']}")
        # print(f"  Grid counts: {self.grid_data['counts']}")

        if not (self.grid_data['origin'] == 0.0).all():
            raise Exception(f'Grid origin in {FN} not at (0, 0, 0)!')

        # Transform grid values
        self._transform_grid_values()

        # Create OpenMM GridForce
        self.gridforce = gfp.GridForce()
        self._setup_gridforce()

    def set_strength(self, strength):
        """
        Update the strength/scaling factor for this grid force.

        Note: For OpenMM, this requires rebuilding the system since forces
        are immutable once added. The system.py setParams method handles this.

        Parameters
        ----------
        strength : float
            New scaling factor
        """
        self.strength = strength
        # Recalculate final scaling with new strength using stored _neg_vals flag
        if self.scaling_prefactor is not None:
            self._final_scaling = self.scaling_prefactor * self.strength
        else:
            # Use the flag set during grid transformation (not the transformed vals!)
            self._final_scaling = (-1.0 if self._neg_vals else 1.0) * self.strength

    def update_strength_in_context(self, gridforce, context, new_strength, base_scaling_factors):
        """
        Update strength in an existing context without rebuilding system.

        Parameters
        ----------
        gridforce : openmm.GridForce
            The GridForce object in the system
        context : openmm.Context
            The context to update
        new_strength : float
            New strength value
        base_scaling_factors : np.array
            BASE per-atom scaling factors (NOT pre-scaled by strength)
        """
        # Update strength and recalculate final scaling
        old_strength = self.strength
        old_final_scaling = self._final_scaling

        self.strength = new_strength
        if self.scaling_prefactor is not None:
            self._final_scaling = self.scaling_prefactor * self.strength
        else:
            self._final_scaling = (-1.0 if self._neg_vals else 1.0) * self.strength

        # Calculate new scaling factors from BASE factors
        # This ensures we don't compound scaling across multiple updates
        new_scaling_factors = np.array(base_scaling_factors) * self._final_scaling

        for i in range(len(new_scaling_factors)):
            gridforce.setScalingFactor(i, float(new_scaling_factors[i]))

        # Apply changes to context (C++ bypass: avoids SWIG cross-module type mismatch)
        _gridforce_updateParametersInContext(gridforce, context)

    def _transform_grid_values(self):
        """Apply transformations to grid values (inv_power, grid_thresh)"""
        vals = self.grid_data['vals'].copy()

        # Check if ALL grid values are negative BEFORE any transformation
        # This applies to ALL grids, not just those with inv_power
        # Logic mirrors MMTK: if no positive values exist, then all are negative
        self._neg_vals = False
        if not (vals > 0).any():  # No positive values
            if (vals < 0).any():  # Has some negative values = all negative
                self._neg_vals = True
                vals = -vals  # Flip sign to make positive

        # Apply inv_power transformation to smooth the grid
        # Grid files contain UNTRANSFORMED values G
        # We transform to G^(1/inv_power) for smoother interpolation
        # Then C++ plugin reverses this: (G^(1/n))^n = G
        if self.inv_power is not None:
            nonzero = vals != 0
            vals[nonzero] = vals[nonzero] ** (1.0 / self.inv_power)

        # Cap grid values
        if self.grid_thresh > 0.0:
            vals = self.grid_thresh * np.tanh(vals / self.grid_thresh)

        # Store transformed values
        self.grid_data['vals'] = vals

        # DEBUG: Print grid statistics AFTER transformation
        # print(f"DEBUG GridForceOpenMM: {self.name} grid AFTER transform:")
        # print(f"  Grid vals AFTER transform - min/max/mean: {vals.min():.3f}/{vals.max():.3f}/{vals.mean():.3f}")
        # print(f"  _neg_vals={self._neg_vals}")

        # Set scaling prefactor
        if self.scaling_prefactor is not None:
            self._final_scaling = self.scaling_prefactor * self.strength
        else:
            self._final_scaling = (-1.0 if self._neg_vals else 1.0) * self.strength

        # print(f"  _final_scaling={self._final_scaling}")

    def _setup_gridforce(self):
        """Configure the OpenMM GridForce with grid data (for initial creation)"""
        self._setup_gridforce_instance(self.gridforce)

    def _setup_gridforce_instance(self, gridforce):
        """Configure a GridForce instance with grid data"""
        # Set grid dimensions
        counts = self.grid_data['counts']
        gridforce.addGridCounts(int(counts[0]), int(counts[1]), int(counts[2]))

        # Set grid spacing (in nm)
        spacing = self.grid_data['spacing']
        gridforce.addGridSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))

        # Add grid values (flattened)
        for val in self.grid_data['vals'].flat:
            gridforce.addGridValue(float(val))

    def add_to_system(self, system, topology, scaling_factors=None):
        """
        Add this grid force to an OpenMM System

        Parameters
        ----------
        system : openmm.System
            The OpenMM system to add the force to
        topology : openmm.app.Topology
            Topology to get per-atom scaling factors
        scaling_factors : list or np.array, optional
            Per-atom scaling factors. If None, uses charges from topology

        Returns
        -------
        int
            Index of the force in the system
        """
        # Get per-atom scaling factors
        if scaling_factors is None:
            # TODO: Extract from topology based on scaling_property
            # For now, assume all atoms have scaling factor of 1.0
            n_atoms = sum(1 for _ in topology.atoms())
            scaling_factors = np.ones(n_atoms)

        # Store BASE scaling factors (before applying final scaling) for updates
        base_scaling_factors = np.array(scaling_factors)

        # Apply final scaling for initial force creation
        scaled_factors = base_scaling_factors * self._final_scaling

        # Create a fresh GridForce (can't reuse Force objects across systems)
        gridforce = gfp.GridForce()
        self._setup_gridforce_instance(gridforce)

        # Set inverse power if specified
        # This tells the plugin to apply interpolated^inv_power during energy calculation
        # to reverse the grid transformation: (G^(1/n))^n = G
        if self.inv_power is not None:
            gridforce.setInvPowerMode(gfp.InvPowerMode_STORED, float(self.inv_power))
        else:
            gridforce.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        # Add scaling factors to force
        for sf in scaled_factors:
            gridforce.addScalingFactor(float(sf))

        # Set a custom name so we can identify this force later (C API)
        _force_set_name(gridforce, self.name)

        # Add force to system via C API (bypasses SWIG cross-module type mismatch)
        force_index = _system_add_force(system, gridforce)

        # Store reference to GridForce and BASE scaling factors for later updates
        self.gridforce_ref = gridforce
        self.base_scaling_factors = base_scaling_factors  # Store UNSCALED base factors

        # gridforce.setForceGroup(force_index)  # Disabled: handled by system.py after all forces added

        return force_index

    def calculate_energy(self, context, scaling_factors=None):
        """
        Calculate grid energy for current configuration

        Parameters
        ----------
        context : openmm.Context
            OpenMM context with positions set
        scaling_factors : array-like, optional
            Per-atom scaling factors

        Returns
        -------
        float
            Grid energy in kJ/mol
        """
        # Create temporary system and context if needed
        # This is for compatibility with MMTK-style usage
        # In practice, the force should be added to the main system
        state = context.getState(getEnergy=True, groups={_force_get_group(self.gridforce)})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return energy
