import os, sys
import numpy as np

try:
  import MMTK
  from MMTK.ParticleProperties import Configuration
  from MMTK.ForceFields import ForceField
  from MMTK.ForceFields import Amber12SBForceField
except ImportError:
  MMTK = None

try:
  import openmm
  from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff
  import openmm.unit as unit

except ImportError:
  openmm = None

from chimpss.algdock.logger import NullDevice

class TopologyMMTK:
  """Describes the system to simulate

  ...

  Attributes
  ----------
  molecule : MMTK.Molecule
    Ligand molecule, like an OpenMM Chain
  molecule_R : MMTK.Molecule
    If includeReceptor, Receptor molecule, like an OpenMM Chain
  universe : MMTK.InfiniteUniverse
    A universe containing all the molecules

  L_first_atom : int
    Index of the first ligand atom in the AMBER prmtop file
  inv_prmtop_atom_order_L : np.array
    Indices to convert from prmtop to MMTK ordering
  prmtop_atom_order_L : np.array
    Indices to convert from prmtop to MMTK ordering

  """
  def __init__(self, args, includeReceptor=False):
    """Initializes the class

    Parameters
    ----------
    args : simulation_arguments.SimulationArguments
      Simulation arguments
    includeReceptor : bool
      Includes the receptor in the topology
    """

    # Set up the system
    original_stderr = sys.stderr
    sys.stderr = NullDevice()
    MMTK.Database.molecule_types.directory = \
      os.path.dirname(args.FNs['ligand_database'])
    self.molecule = MMTK.Molecule(\
      os.path.basename(args.FNs['ligand_database']))
    if includeReceptor and \
        (args.FNs['receptor_database'] is not None) and \
        os.path.isfile(args.FNs['receptor_database']):
      self.molecule_R = MMTK.Molecule(\
        os.path.basename(args.FNs['receptor_database']))
    else:
      self.molecule_R = None
    sys.stderr = original_stderr

    # Hydrogen Mass Repartitioning
    # (sets hydrogen mass to H_mass and scales other masses down)
    if args.params['BC']['H_mass'] > 0.:
      from chimpss.algdock.HMR import hydrogen_mass_repartitioning
      self.molecule = hydrogen_mass_repartitioning(self.molecule, \
        args.params['BC']['H_mass'])

    # # Helpful variables for referencing and indexing atoms in the molecule
    # self.molecule.heavy_atoms = [ind for (atm,ind) in \
    #   zip(self.molecule.atoms,range(self.molecule.numberOfAtoms())) \
    #   if atm.type.name!='hydrogen']
    # self.molecule.nhatoms = len(self.molecule.heavy_atoms)

    self.prmtop_atom_order_L = np.array([atom.number \
      for atom in self.molecule.prmtop_order], dtype=int)
    self.inv_prmtop_atom_order_L = \
      np.zeros(shape=len(self.prmtop_atom_order_L), dtype=int)
    for i in range(len(self.prmtop_atom_order_L)):
      self.inv_prmtop_atom_order_L[self.prmtop_atom_order_L[i]] = i

    # Create universe and add molecule to universe
    self.universe = MMTK.Universe.InfiniteUniverse()
    self.universe.addObject(self.molecule)
    if includeReceptor:
      if self.molecule_R is not None:
        self.universe.addObject(self.molecule_R)
      else:
        self.universe = None

    # Define L_first_atom
    if includeReceptor:
      if (args.FNs['prmtop']['R'] is not None) and \
         (args.FNs['prmtop']['RL'] is not None):
        import chimpss.algdock.IO
        IO_prmtop = chimpss.algdock.IO.prmtop()
        prmtop_R = IO_prmtop.read(args.FNs['prmtop']['R'])
        prmtop_RL = IO_prmtop.read(args.FNs['prmtop']['RL'])
        ligand_ind = [
          ind for ind in range(len(prmtop_RL['RESIDUE_LABEL']))
          if prmtop_RL['RESIDUE_LABEL'][ind] not in prmtop_R['RESIDUE_LABEL']
        ]
        if len(ligand_ind) == 0:
          raise Exception('Ligand not found in complex prmtop')
        elif len(ligand_ind) > 1:
          print('  possible ligand residue labels: '+\
            ', '.join([prmtop_RL['RESIDUE_LABEL'][ind] for ind in ligand_ind]))
        print('ligand residue name: ' + \
          prmtop_RL['RESIDUE_LABEL'][ligand_ind[0]].strip())
        self.L_first_atom = prmtop_RL['RESIDUE_POINTER'][ligand_ind[0]] - 1
      else:
        self.L_first_atom = 0
    else:
      self.L_first_atom = 0

class TopologyUsingOpenMM:
  """Describes the system to simulate using OpenMM
  ...
  Attributes
  ----------
  molecule : openmm.app.Topology
    Ligand topology
  OMM_system : openmm.System
    OpenMM system object
  OMM_simulation : openmm.app.Simulation
    OpenMM simulation object
  context : openmm.Context
    OpenMM context for energy/force calculations

  L_first_atom : int
    Index of the first ligand atom in the AMBER prmtop file
  inv_prmtop_atom_order_L : np.array
    Indices to convert from prmtop to OpenMM ordering
  prmtop_atom_order_L : np.array
    Indices to convert from prmtop to OpenMM ordering
  """
  def __init__(self, args, includeReceptor=False):
    self.args = args  # Store args reference
    original_stderr = sys.stderr
    sys.stderr = NullDevice()
    prmtopL = AmberPrmtopFile(args.FNs['prmtop']["L"])
    inpcrdL = AmberInpcrdFile(args.FNs['inpcrd']["L"])
    # Use OBC2 implicit solvent model with NoCutoff for all forces
    # NoCutoff is the most accurate approach and avoids cutoff artifacts
    # Note: MMTK uses Wolf correction with 1.5nm cutoff which approximates NoCutoff,
    # so using NoCutoff directly in OpenMM is actually more accurate
    from openmm.app import OBC2, NoCutoff
    self.OMM_system = prmtopL.createSystem(nonbondedMethod=NoCutoff,
                                           constraints=None,
                                           implicitSolvent=OBC2)

    self.molecule = prmtopL.topology  # Full simulation topology
    self.ligand_molecule = prmtopL.topology  # Keep reference to ligand topology for ExternalMC

    if includeReceptor and args.FNs['prmtop']["RL"] is not None:
      prmtopRL = AmberPrmtopFile(args.FNs['prmtop']["RL"])
      inpcrdRL = AmberInpcrdFile(args.FNs['inpcrd']["RL"])

      # Use NoCutoff for receptor-ligand complex as well
      self.OMM_system = prmtopRL.createSystem(nonbondedMethod=NoCutoff,
                                              constraints=None,
                                              implicitSolvent=OBC2)
      self.molecule = prmtopRL.topology  # Complex topology for simulation
      # ligand_molecule still points to ligand-only topology from above

    sys.stderr = original_stderr
    dummy_integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)

    # Hydrogen Mass Repartitioning
    # (sets hydrogen mass to H_mass and scales other masses down)
    if args.params['BC']['H_mass'] > 0.:
      from chimpss.algdock.HMR import hydrogen_mass_repartitioning_openmm
      self.OMM_system = hydrogen_mass_repartitioning_openmm(self.molecule,  self.OMM_system,\
        args.params['BC']['H_mass'])

    # Try to get fastest available platform — CUDA/OpenCL first for GPU acceleration
    platform = None
    for platform_name in ['CUDA', 'OpenCL', 'CPU']:
      try:
        test_platform = openmm.Platform.getPlatformByName(platform_name)
        # Try to create a test context to verify platform works
        test_integrator = openmm.VerletIntegrator(0.001)
        test_system = openmm.System()
        test_system.addParticle(1.0)
        test_context = openmm.Context(test_system, test_integrator, test_platform)
        del test_context, test_integrator, test_system
        platform = test_platform
        print(f"Using OpenMM {platform_name} platform")
        break
      except Exception as e:
        continue

    if platform is None:
      # Fall back to Reference platform as last resort
      platform = openmm.Platform.getPlatformByName('Reference')
      print(f"Using OpenMM Reference platform (slowest)")

    # Create simulation with appropriate topology
    if includeReceptor and args.FNs['prmtop']["RL"] is not None: # this is complex
      self.OMM_simulation = openmm.app.Simulation(prmtopRL.topology, self.OMM_system, dummy_integrator, platform)
      self.OMM_simulation.context.setPositions(inpcrdRL.positions)
      self.context = self.OMM_simulation.context
    else:
      self.OMM_simulation = openmm.app.Simulation(prmtopL.topology, self.OMM_system, dummy_integrator, platform)
      self.OMM_simulation.context.setPositions(inpcrdL.positions)
      self.context = self.OMM_simulation.context  # Convenient reference

    self.inv_prmtop_atom_order_L = self.prmtop_atom_order_L = np.array([atom.index \
      for atom in self.molecule.atoms()], dtype=int)

    if (args.FNs['prmtop']['R'] is not None) and \
       (args.FNs['prmtop']['RL'] is not None):
      prmtopR = AmberPrmtopFile(args.FNs['prmtop']["R"])
      # prmtopRL already loaded above if includeReceptor=True
      if not includeReceptor:
        prmtopRL = AmberPrmtopFile(args.FNs['prmtop']["RL"])
      receptor_atoms = set(atom.index for atom in prmtopR.topology.atoms())
      complex_atoms = set(atom.index for atom in prmtopRL.topology.atoms())
      ligand_atoms = complex_atoms - receptor_atoms
      if ligand_atoms:
        # L_first_atom is only meaningful when includeReceptor=True (complex system)
        # When includeReceptor=False (ligand-only), ligand starts at index 0
        self.L_first_atom = min(ligand_atoms) if includeReceptor else 0
      else:
        print("No ligand atoms found in complex prmtop file.")
        self.L_first_atom = 0
    else:
      self.L_first_atom = 0

  # Helper methods for MMTK-style API (incremental refactoring)

  def setConfiguration(self, conf):
    """Set positions in OpenMM context

    Parameters
    ----------
    conf : np.array
      Configuration array in nanometers
    """
    # Check if the system has more particles than the conf array
    # This happens when force fields add virtual particles (e.g., sphere site dummy particle)
    num_particles = self.OMM_system.getNumParticles()
    num_conf_atoms = conf.shape[0]

    if num_particles > num_conf_atoms:
      # Get positions for virtual/dummy particles from the current context state
      # This preserves positions set for virtual particles (like sphere center dummy particle)
      import numpy as np
      state = self.context.getState(getPositions=True)
      current_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

      # Create new position array: real atom positions from conf + virtual particle positions from current state
      padded_conf = np.vstack([conf, current_positions[num_conf_atoms:]])
      self.context.setPositions(padded_conf * unit.nanometer)
    else:
      self.context.setPositions(conf * unit.nanometer)

  def configuration(self):
    """Get current configuration from OpenMM context

    Returns
    -------
    object with .array attribute containing positions in nm
    """
    class ConfigWrapper:
      def __init__(self, positions):
        self.array = positions

    state = self.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Only return positions for real atoms (from Topology), excluding virtual particles
    # In OpenMM, topology atoms are mapped to particle indices 0..N-1 in order.
    # Virtual particles added by force fields (e.g., sphere site dummy) get indices >= N.
    # So we can safely slice to numberOfAtoms() to get only real atom positions.
    num_real_atoms = self.numberOfAtoms()
    if positions.shape[0] > num_real_atoms:
      positions = positions[:num_real_atoms]

    return ConfigWrapper(positions)

  def energy(self):
    """Get potential energy from OpenMM context

    Returns
    -------
    float
      Potential energy in kJ/mol
    """
    state = self.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

  def numberOfAtoms(self):
    """Get number of atoms

    Returns
    -------
    int
      Number of atoms in the topology
    """
    return sum(1 for _ in self.molecule.atoms())

  def centerOfMass(self):
    """Calculate center of mass

    Returns
    -------
    np.array
      Center of mass coordinates in nm
    """
    state = self.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Get masses directly from the OpenMM system
    n_particles = self.OMM_system.getNumParticles()
    masses = np.array([self.OMM_system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(n_particles)])

    # Calculate COM: sum(mass_i * pos_i) / sum(mass_i)
    total_mass = np.sum(masses)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

    return com

  def setForceField(self, forcefield):
    """Placeholder for setting force field (grid forces)

    For now, this is a no-op. Grid forces will be handled separately.
    """
    # TODO: Implement when grid energy calculator is ready
    pass

