# This module implements am External Monte Carlo move "integrator"

try:
  from MMTK import Configuration, Dynamics, Environment, Features, Trajectory, Units
  import MMTK_dynamics
  MMTK = True
except ImportError:
  MMTK = None
  Configuration = None
  Dynamics = None
  # Define fallback Units for unit conversions
  class _Units:
    Ang = 1e-10  # Angstrom to meters
    kcal = 4184.0  # kcal to joules
    kJ = 1000.0  # kJ to joules
    mol = 1.0  # mole
    J = 1.0  # joule
    K = 1.0  # kelvin
  Units = _Units()

try:
  import openmm
  import openmm.unit as unit
except ImportError:
  openmm = None

import numpy as np
import random

R = 8.3144621*Units.J/Units.mol/Units.K

def random_rotate():
  """
  Return a random rotation matrix
  """
  u = np.random.uniform(size=3)

  # Random quaternion
  q = np.array([np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),
               np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
               np.sqrt(u[0])*np.sin(2*np.pi*u[2]),
               np.sqrt(u[0])*np.cos(2*np.pi*u[2])])
  
  # Convert the quaternion into a rotation matrix 
  rotMat = np.array([[q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3],
                     2*q[1]*q[2] - 2*q[0]*q[3],
                     2*q[1]*q[3] + 2*q[0]*q[2]],
                    [2*q[1]*q[2] + 2*q[0]*q[3],
                     q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3],
                     2*q[2]*q[3] - 2*q[0]*q[1]],
                    [2*q[1]*q[3] - 2*q[0]*q[2],
                     2*q[2]*q[3] + 2*q[0]*q[1],
                     q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]])
  return rotMat

#
# External Monte Carlo move integrator (MMTK version)
#
if MMTK:
  class ExternalMCIntegrator(Dynamics.Integrator):
    def __init__(self, universe, molecule, step_size, sampling_universe=None, \
        **options):
      """
      confs - configurations to dart to
      extended - whether or not to use external coordinates
      """
      Dynamics.Integrator.__init__(self, universe, options)
      # Supported features: none for the moment, to keep it simple
      self.features = []

      self.molecule = molecule
      self.step_size = step_size
      self.sampling_universe = sampling_universe

    def __call__(self, **options):
      # Process the keyword arguments
      self.setCallOptions(options)
      # Check if the universe has features not supported by the integrator
      Features.checkFeatures(self, self.universe)

      RT = R*self.getOption('T')
      ntrials = self.getOption('ntrials')
      natoms = self.universe.numberOfAtoms()

      acc = 0
      xo = np.copy(self.universe.configuration().array)
      com = self.universe.centerOfMass().array
      if self.sampling_universe is None:
        eo = self.universe.energy()
      else:
        self.sampling_universe.configuration().array[-natoms:,:] = xo
        eo = self.sampling_universe.energy() # <- Using sampling Hamiltonian

      for c in range(ntrials):
        step = np.random.randn(3)*self.step_size
        if c%2==0:
          # Random translation and full rotation
          xn = np.dot((xo - com), random_rotate()) + com + step
        else:
          # Random translation
          xn = xo + step
        if self.sampling_universe is None:
          self.universe.setConfiguration(Configuration(self.universe,xn))
          en = self.universe.energy()
        else:
          self.sampling_universe.configuration().array[-natoms:,:] = xn
          en = self.sampling_universe.energy() # <- Using sampling Hamiltonian
        if ((en<eo) or (np.random.random()<np.exp(-(en-eo)/RT))):
          acc += 1
          xo = xn
          eo = en
          com += step
        else:
          self.universe.setConfiguration(Configuration(self.universe,xo))

      return ([np.copy(xo)], [en], acc, ntrials, 0.0)


#
# OpenMM version of External Monte Carlo move integrator
#
class ExternalMCIntegratorOpenMM:
  """
  OpenMM-based External Monte Carlo integrator

  Performs random translations and rotations of the ligand molecule
  with Metropolis acceptance criterion.
  """
  def __init__(self, simulation, molecule, step_size=0.25, topology=None):
    """
    Parameters
    ----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object (NOTE: not stored, fetched from topology)
    molecule : topology molecule object
        The ligand molecule (OpenMM topology)
    step_size : float
        Step size for random translations in Angstroms (default: 0.25)
    topology : TopologyOpenMM object, optional
        The AlGDock topology object containing L_first_atom info
    """
    # Fetch simulation from topology each time to get the current system with grid forces
    # (simulation gets recreated in setParams() so storing a reference becomes stale)
    self.molecule = molecule
    self.step_size = step_size * 0.1  # Convert Angstrom to nm (OpenMM units)
    self.topology = topology

    # Get ligand atom indices
    # For OpenMM topology, atoms() is a method that returns iterator
    n_atoms = sum(1 for _ in molecule.atoms())

    # If we have topology info, use L_first_atom to get ligand indices
    if topology is not None and hasattr(topology, 'L_first_atom'):
      # Ligand atoms start at L_first_atom
      self.ligand_indices = list(range(topology.L_first_atom, n_atoms))
    else:
      # Assume all atoms are ligand (ligand-only system)
      self.ligand_indices = list(range(n_atoms))

  def __call__(self, ntrials=5, T=300.0, **options):
    """
    Perform External MC moves

    Parameters
    ----------
    ntrials : int
        Number of MC trial moves
    T : float
        Temperature in Kelvin

    Returns
    -------
    tuple
        (positions_list, energies_list, accepted, attempted, delta_t)
    """
    kB = 0.008314462  # Boltzmann constant in kJ/mol/K
    RT = kB * T

    # Fetch current simulation from topology (gets recreated in setParams)
    # This ensures we use the system with current grid forces at correct strength
    context = self.topology.OMM_simulation.context

    # Get current positions
    state = context.getState(getPositions=True, getEnergy=True)
    positions_with_units = state.getPositions(asNumpy=True)  # Quantity with units
    eo = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # Strip units to get numpy array in nm
    positions = positions_with_units.value_in_unit(unit.nanometer)

    # Extract ligand positions
    xo = positions[self.ligand_indices, :].copy()  # Shape: (natoms, 3)

    # Calculate center of mass
    com = np.mean(xo, axis=0)

    acc = 0

    for c in range(ntrials):
      # Random translation step
      step = np.random.randn(3) * self.step_size

      if c % 2 == 0:
        # Random translation and full rotation
        xn = np.dot((xo - com), random_rotate()) + com + step
      else:
        # Random translation only
        xn = xo + step

      # Update positions in full system
      positions_new = positions.copy()
      positions_new[self.ligand_indices, :] = xn

      # Set new positions and get energy (need to add units back)
      context.setPositions(positions_new * unit.nanometer)
      state_new = context.getState(getEnergy=True)
      en = state_new.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

      # Metropolis acceptance
      if (en < eo) or (np.random.random() < np.exp(-(en - eo) / RT)):
        acc += 1
        xo = xn
        eo = en
        positions = positions_new
        com += step
      else:
        # Reject: restore old positions
        context.setPositions(positions * unit.nanometer)

    # Return in same format as MMTK version
    return ([xo.copy()], [eo], acc, ntrials, 0.0)
