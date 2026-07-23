# This module implements a Hamiltonian Monte Carlo "integrator"
# It is a stripped-down version of the velocity verlet integrator
# with a Metropolis acceptance criterion.
# It requires the option 'T' in addition to velocity verlet options.

import numpy as np

try:
  from MMTK import Dynamics, Environment, Features, Units
  import MMTK_dynamics
  from MMTK.ParticleProperties import Configuration
  from Scientific import N
  MMTK = True
except ImportError:
  MMTK = None
  Dynamics = None
  Configuration = None
  # Define fallback Units for unit conversions
  class _Units:
    Ang = 1e-10  # Angstrom to meters
    kcal = 4184.0  # kcal to joules
    kJ = 1000.0  # kJ to joules
    mol = 1.0  # mole
    J = 1.0  # joule
    K = 1.0  # kelvin
    fs = 1e-15  # femtosecond
  Units = _Units()

try:
  import openmm
  import openmm.unit as unit
  from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff
  from openmm import *
  from openmmtools.integrators import VelocityVerletIntegrator
except ImportError:
  OpenMM = None

R = 8.3144621*Units.J/Units.mol/Units.K

#
# Hamiltonian Monte Carlo integrator (MMTK version)
#
if MMTK:
    class HamiltonianMonteCarloIntegrator(Dynamics.Integrator):
  
      def __init__(self, universe, **options):
          Dynamics.Integrator.__init__(self, universe, options)
          # Supported features: none for the moment, to keep it simple
          self.features = []
  
      def __call__(self, **options):
          # Process the keyword arguments
          self.setCallOptions(options)
          # Check if the universe has features not supported by the integrator
          Features.checkFeatures(self, self.universe)
          print('options:', options)
  
          RT = R*self.getOption('T')
          delta_t = self.getOption('delta_t')
          
          if 'steps_per_trial' in self.call_options.keys():
            steps_per_trial = self.getOption('steps_per_trial')
            ntrials = self.getOption('steps')/steps_per_trial
          else:
            steps_per_trial = self.getOption('steps')
            ntrials = 1
    
          if 'normalize' in self.call_options.keys():
            normalize = self.getOption('normalize')
          else:
            normalize = False          
  
          # Seed the random number generator
          if 'random_seed' in self.call_options.keys():
            np.random.seed(self.getOption('random_seed'))
  
          self.universe.initializeVelocitiesToTemperature(self.getOption('T'))
          
          # Get the universe variables needed by the integrator
          masses = self.universe.masses()
          fixed = self.universe.getAtomBooleanArray('fixed')
          nt = self.getOption('threads')
          comm = self.getOption('mpi_communicator')
          evaluator = self.universe.energyEvaluator(threads=nt,
                                                    mpi_communicator=comm)
          evaluator = evaluator.CEvaluator()
  
          late_args = (
                  masses.array, fixed.array, evaluator,
                  N.zeros((0, 2), N.Int), N.zeros((0, ), N.Float),
                  N.zeros((1,), N.Int),
                  N.zeros((0,), N.Float), N.zeros((2,), N.Float),
                  N.zeros((0,), N.Float), N.zeros((1,), N.Float),
                  delta_t, self.getOption('first_step'),
                  steps_per_trial, self.getActions(),
                  'Hamiltonian Monte Carlo step')
  
          # Variables for velocity assignment
          m3 = np.repeat(np.expand_dims(masses.array,1),3,axis=1)
          sigma_MB = np.sqrt((self.getOption('T')*Units.k_B)/m3)
          natoms = self.universe.numberOfAtoms()
  
          xs = []
          energies = []
  
          # Store initial configuration and potential energy
          xo = np.copy(self.universe.configuration().array)
          pe_o = self.universe.energy()
  
          acc = 0
          for t in range(ntrials):
            # Initialize the velocity
            v = self.universe.velocities()
            v.array = np.multiply(sigma_MB,np.random.randn(natoms,3))
      
            # Store total energy
            eo = pe_o + 0.5*np.sum(np.multiply(m3,np.square(v.array)))
  
            # Run the velocity verlet integrator
            self.run(MMTK_dynamics.integrateVV,
              (self.universe,
               self.universe.configuration().array,
               self.universe.velocities().array) + late_args)
  
            # Decide whether to accept the move
            pe_n = self.universe.energy()
            en = pe_n + 0.5*np.sum(np.multiply(m3,np.square(v.array)))
            
            if ((en<eo) or (np.random.random()<np.exp(-(en-eo)/RT))) and \
               ((abs(pe_o-pe_n)/RT<250.) or (abs(eo-en)/RT<250.)):
              xo = np.copy(self.universe.configuration().array)
              pe_o = pe_n
              acc += 1
              if normalize:
                self.universe.normalizePosition()
            else:
              self.universe.setConfiguration(Configuration(self.universe,xo))
            
            xs.append(np.copy(self.universe.configuration().array))
            energies.append(pe_o)
    
          return (xs, energies, acc, ntrials, delta_t)


class HamiltonianMonteCarloIntegratorUsingOpenMM:
    def __init__(self, molecule, top, OMM_system):
        self.options = {'first_step': 0, 'steps': 100, 'delta_t': 1. * Units.fs,
                           'background': False, 'threads': None,
                           'mpi_communicator': None, 'actions': []}
        self.top = top
        self.molecule = molecule
        # Note: We use self.top.OMM_system (current system) at runtime, not the
        # OMM_system passed to constructor (which may become stale after setParams)

    def __call__(self, **options):
        # Update options with any passed parameters
        self.options.update(options)
        # R is in J/(mol*K), but OpenMM energies are in kJ/mol
        # So RT must be in kJ/mol to match energy units
        RT = (R / 1000.0) * self.options['T']  # Convert J/mol to kJ/mol
        delta_t = self.options['delta_t']

        # Get seed_index and state_id for trajectory saving (default to 0 if not provided)
        seed_index = self.options.get('seed_index', 0)
        state_id = self.options.get('state_id', 0)

        if 'steps_per_trial' in self.options.keys():
            steps_per_trial = self.options['steps_per_trial']
            ntrials = self.options['steps'] / steps_per_trial
        else:
            steps_per_trial = self.options['steps']
            ntrials = 1

        if 'normalize' in self.options.keys():
            normalize = self.options['normalize']
        else:
            normalize = False

            # Seed the random number generator
        if 'random_seed' in self.options.keys():
            np.random.seed(self.options['random_seed'])

        context = self.top.OMM_simulation.context
        context.setVelocitiesToTemperature(self.options['T'] * unit.kelvin)
        #self.universe.initializeVelocitiesToTemperature(self.options('T'))

        # Get the universe variables needed by the integrator
        # Use self.top.OMM_system (current system) with up-to-date particle masses
        current_system = self.top.OMM_system
        masses = []
        for atom in self.molecule.atoms():
            mass = current_system.getParticleMass(atom.index)  # Get mass from CURRENT system
            masses.append(mass.value_in_unit(unit.dalton))  # Convert to Daltons
        masses = np.array(masses)

        # fixed = self.universe.getAtomBooleanArray('fixed')
        # What is fixed? It's [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        nt = self.options['threads']
        comm = self.options['mpi_communicator']
        # evaluator = self.universe.energyEvaluator(threads=nt,
        #                                           mpi_communicator=comm)
        # evaluator = evaluator.CEvaluator()

        # late_args = (
        #     masses.array, #fixed.array, evaluator,
        #     N.zeros((0, 2), N.Int), N.zeros((0,), N.Float),
        #     N.zeros((1,), N.Int),
        #     N.zeros((0,), N.Float), N.zeros((2,), N.Float),
        #     N.zeros((0,), N.Float), N.zeros((1,), N.Float),
        #     delta_t, options('first_step'),
        #     steps_per_trial, getActions(),
        #     'Hamiltonian Monte Carlo step')

        # Variables for velocity assignment
        # Convert masses to OpenMM units for proper calculation
        m3_with_units = np.repeat(np.expand_dims(masses, 1), 3, axis=1) * unit.dalton
        k_B = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        # sigma = sqrt(kT/m) for Maxwell-Boltzmann velocity distribution
        # Result will be in correct velocity units (nm/ps)
        # Need to use numpy sqrt for arrays, then attach units
        kT = (k_B * self.options['T'] * unit.kelvin)
        kT_over_m = kT / m3_with_units
        # Extract value, take sqrt, then reattach units
        sigma_MB_squared_value = kT_over_m.value_in_unit((unit.nanometer / unit.picosecond)**2)
        sigma_MB = np.sqrt(sigma_MB_squared_value)  # Now in nm/ps

        # Keep m3 as unitless array (Daltons) for KE calculation
        m3 = np.repeat(np.expand_dims(masses, 1), 3, axis=1)

        # In OpenMM's unit system: 1 Dalton * (nm/ps)^2 = 1 kJ/mol
        # So the conversion factor is simply 1.0
        ke_conversion = 1.0

        # Use number of real atoms from topology, not particles from system
        # (System may have virtual particles added by force fields)
        natoms = len(masses)

        xs = []
        energies = []

        # Trajectory saving: Save trajectories for specified seeds if enabled
        # Create a time-series trajectory that appends across all states
        import os
        save_traj_env = os.environ.get('ALGDOCK_SAVE_TRAJECTORIES', 'false').lower() == 'true'
        max_seeds_to_save = int(os.environ.get('ALGDOCK_MAX_SEEDS_TO_SAVE', '3'))
        save_trajectory = save_traj_env and (seed_index < max_seeds_to_save)

        if save_trajectory:
            self._trajectory_positions = []
            self._trajectory_energies = []
            self._trajectory_state_ids = []  # Track which state each frame belongs to
            import sys
            # Energy range will be printed after collection
            self._traj_seed_index = seed_index
            self._traj_state_id = state_id

        # Get initial configuration and potential energy from main simulation context
        # (iteration() sets this before calling HMC)
        state_main = self.top.OMM_simulation.context.getState(getPositions=True, getEnergy=True)
        all_positions = state_main.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        xo = all_positions[:natoms]  # Only real atoms for trajectory tracking
        # Save virtual particle positions for padding HMC simulation setPositions calls
        virtual_positions = all_positions[natoms:] if all_positions.shape[0] > natoms else None
        pe_o = state_main.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)


        if save_trajectory:
            self._trajectory_positions.append(np.copy(xo))
            self._trajectory_energies.append(pe_o)
            self._trajectory_state_ids.append(state_id)

        # Create HMC simulation using current System from main simulation
        # Unlike MMTK which can specify integrator per-call, OpenMM integrators are bound to Context
        # So we must create a separate simulation with VelocityVerlet integrator
        # using self.top.OMM_system (current System) which has up-to-date forces
        integrator = VelocityVerletIntegrator(delta_t)

        # Use Reference platform (CPU platform has known multiprocessing issues)
        # Reference is slower but more stable for parallel/multiprocessing scenarios
        platform = openmm.Platform.getPlatformByName('Reference')

        # Create HMC simulation with CURRENT System
        # Note: positions will be set in the trial loop (line ~307)
        hmc_simulation = Simulation(self.molecule, self.top.OMM_system, integrator, platform)

        simulation = hmc_simulation

        acc = 0

        for t in range(int(ntrials)):
            # Set initial positions for this trial
            # Pad with virtual particle positions if needed (similar to velocity padding below)
            if virtual_positions is not None and len(virtual_positions) > 0:
                positions_to_set = np.vstack([xo, virtual_positions])
                simulation.context.setPositions(positions_to_set * unit.nanometer)
            else:
                simulation.context.setPositions(xo * unit.nanometer)

            # Initialize velocities from Maxwell-Boltzmann distribution
            v_initial = np.multiply(sigma_MB, np.random.randn(natoms, 3))

            # Pad velocities with zeros for virtual particles if needed
            # Check the actual system in the simulation context, not cached systems
            num_particles = simulation.context.getSystem().getNumParticles()
            if num_particles > natoms:
                # Virtual particles should have zero velocity
                v_padded = np.vstack([v_initial, np.zeros((num_particles - natoms, 3))])
                simulation.context.setVelocities(v_padded * unit.nanometer / unit.picosecond)
            else:
                simulation.context.setVelocities(v_initial * unit.nanometer / unit.picosecond)

            # Calculate initial kinetic energy (in kJ/mol)
            ke_o = 0.5 * np.sum(np.multiply(m3, np.square(v_initial))) * ke_conversion
            eo = pe_o + ke_o

            # Run integration steps
            simulation.step(int(steps_per_trial))

            # OPTIMIZATION: Only get energy and velocities initially (not positions)
            # Positions are only needed if we accept the move
            state = simulation.context.getState(getEnergy=True, getVelocities=True)
            pe_n = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

            # Get final velocities and calculate final kinetic energy (in kJ/mol)
            v_final = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
            # Strip virtual particle velocities if present (only use real atom velocities)
            if v_final.shape[0] > natoms:
                v_final = v_final[:natoms]
            ke_n = 0.5 * np.sum(np.multiply(m3, np.square(v_final))) * ke_conversion
            en = pe_n + ke_n

            # Metropolis acceptance criterion
            accept_energy = (en < eo) or (np.random.random() < np.exp(-(en - eo) / RT))
            accept_stability = (abs(pe_o - pe_n) / RT < 250.) or (abs(eo - en) / RT < 250.)

            if accept_energy and accept_stability:
                # Accept the move - NOW get positions
                positions_state = simulation.context.getState(getPositions=True)
                positions = positions_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                xo = np.copy(positions[:natoms])  # Only store real atoms, not virtual particles
                pe_o = pe_n
                acc += 1

                if save_trajectory:
                    self._trajectory_positions.append(np.copy(xo))
                    self._trajectory_energies.append(pe_o)
                    self._trajectory_state_ids.append(state_id)
            else:
                # Reject the move - reset to previous accepted position
                if virtual_positions is not None and len(virtual_positions) > 0:
                    positions_to_set = np.vstack([xo, virtual_positions])
                    simulation.context.setPositions(positions_to_set * unit.nanometer)
                else:
                    simulation.context.setPositions(xo * unit.nanometer)

            # Normalize positions if requested (applies to both accepted and rejected moves)
            if normalize:
                # Normalize using LOCAL HMC simulation context (like MMTK's universe.normalizePosition())
                # enforcePeriodicBox wraps positions back into the periodic box
                if virtual_positions is not None and len(virtual_positions) > 0:
                    positions_to_set = np.vstack([xo, virtual_positions])
                    simulation.context.setPositions(positions_to_set * unit.nanometer)
                else:
                    simulation.context.setPositions(xo * unit.nanometer)
                state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                xo = np.copy(positions[:natoms])  # Only store real atoms, not virtual particles
                # Update virtual positions after normalization
                virtual_positions = positions[natoms:] if positions.shape[0] > natoms else None
                # HMC simulation context now has normalized position for next trial

            # Always append the current accepted position (which is xo)
            xs.append(np.copy(xo))
            energies.append(pe_o)

        # NOTE: Do NOT delete simulation/integrator - they are cached for reuse

        # Update main simulation context with final accepted position
        # This ensures the next HMC call starts from the final accepted position
        if virtual_positions is not None and len(virtual_positions) > 0:
            positions_to_set = np.vstack([xo, virtual_positions])
            self.top.OMM_simulation.context.setPositions(positions_to_set * unit.nanometer)
        else:
            self.top.OMM_simulation.context.setPositions(xo * unit.nanometer)

        # Save trajectory if collected - APPEND to time-series file for this seed
        if save_trajectory:
            import sys
            # One trajectory file per seed, appended across all states
            traj_file = f'hmc_trajectory_seed{seed_index}.pdb'

            # Check if file exists to determine starting frame number
            import os as os_module
            if os_module.path.exists(traj_file):
                # Count existing frames to continue numbering
                with open(traj_file, 'r') as f:
                    frame_offset = sum(1 for line in f if line.startswith('MODEL'))
                file_mode = 'a'  # Append mode
            else:
                frame_offset = 0
                file_mode = 'w'  # Write mode

            # Append trajectory frames to file
            try:
                from openmm.app import PDBFile
                # Get topology from the molecule
                topology = self.top.molecule
                n_atoms_topology = topology.getNumAtoms()

                # Append/write multi-frame PDB
                with open(traj_file, file_mode) as f:
                    for frame_idx, (pos, energy, state) in enumerate(zip(self._trajectory_positions,
                                                                          self._trajectory_energies,
                                                                          self._trajectory_state_ids)):
                        n_atoms_pos = len(pos)

                        # Handle virtual/dummy particles (e.g., sphere center for site restraint)
                        # Position array may have extra virtual particles at the end
                        if n_atoms_pos > n_atoms_topology:
                            # Only save positions for real atoms (exclude virtual particles)
                            pos = pos[:n_atoms_topology]

                        # Frame number continues from existing frames
                        global_frame_num = frame_offset + frame_idx + 1
                        f.write(f"MODEL     {global_frame_num:4d}\n")
                        f.write(f"REMARK State_ID {state:.5f} Energy {energy:.2f} kJ/mol\n")
                        # Convert positions to Quantity with units
                        pos_with_units = pos * unit.nanometer
                        PDBFile.writeModel(topology, pos_with_units, f)
                        f.write(f"ENDMDL\n")

                # Single-line summary
                # sys.stderr.write(f"Seed {self._traj_seed_index}, state {self._traj_state_id:.5f}, energy range: {min(self._trajectory_energies):.2f} to {max(self._trajectory_energies):.2f} kJ/mol\n")
                # sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"Error saving trajectory: {e}\n")
                import traceback
                traceback.print_exc(file=sys.stderr)

        # Print profiling results
        # if profile_this_call:
        #     # Write to dedicated profiling file
        #     with open('hmc_profiling.txt', 'w') as f:
        #         f.write(f"\n{'='*70}\n")
        #         f.write(f"HMC PROFILING RESULTS (first call, {int(ntrials)} trials)\n")
        #         f.write(f"{'='*70}\n")
        #         total_time = 0
        #         for key in ['setPositions', 'setVelocities', 'step', 'getState_energy_vel', 'acceptance', 'getState_positions']:
        #             times = timings[key]
        #             if len(times) > 0:
        #                 avg = np.mean(times) * 1000  # Convert to ms
        #                 total = np.sum(times) * 1000
        #                 total_time += np.sum(times)
        #                 f.write(f"  {key:25s}: avg={avg:7.3f} ms/trial, total={total:8.1f} ms\n")
        #         f.write(f"  {'TOTAL':25s}:                       total={total_time*1000:8.1f} ms\n")
        #         f.write(f"  Per trial: {total_time*1000/int(ntrials):.3f} ms\n")
        #         f.write(f"{'='*70}\n\n")
        #     print(f"\n*** HMC profiling results written to hmc_profiling.txt ***\n")

        return (xs, energies, acc, ntrials, delta_t)


class NUTSIntegratorUsingOpenMM:
    """
    No-U-Turn Sampler (NUTS) implementation using OpenMM

    Port of David Minh's NUTS.pyx to work with OpenMM instead of MMTK.
    NUTS is an extension of HMC that adaptively determines trajectory length
    using a recursive tree-building algorithm.

    Reference: Hoffman & Gelman (2011) "The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo"
    """

    def __init__(self, molecule, top, OMM_system):
        self.options = {
            'first_step': 0,
            'steps': 100,
            'delta_t': 1. * Units.fs,
            'background': False,
            'threads': None,
            'mpi_communicator': None,
            'actions': [],
            'delta': 0.6,  # Target acceptance rate for dual averaging
            'adapt': False  # Whether to adapt step size
        }
        self.top = top
        self.molecule = molecule

    def __call__(self, **options):
        """
        Run NUTS sampling

        Returns
        -------
        xs : list of ndarray
            Sampled configurations
        energies : list of float
            Potential energies
        acc_rate : float
            Average acceptance probability (from dual averaging)
        nsteps_total : int
            Total number of leapfrog steps taken
        delta_t_final : float
            Final adapted step size (or initial if adapt=False)
        """
        # Update options
        self.options.update(options)

        # Extract parameters
        T = self.options['T']
        delta_t = self.options['delta_t']
        nsteps = self.options['steps']
        normalize = self.options.get('normalize', False)
        delta = self.options.get('delta', 0.6)
        adapt = self.options.get('adapt', False)

        # Convert delta_t to unitless float if it has MMTK or OpenMM units
        # (needed for np.log operations in dual averaging)
        if hasattr(delta_t, 'value_in_unit'):
            # OpenMM Quantity - extract in femtoseconds (matches initialization units)
            delta_t = delta_t.value_in_unit(unit.femtosecond)
        elif MMTK and hasattr(delta_t, 'value'):
            # MMTK Quantity - extract numeric value
            delta_t = delta_t.value

        # Seed RNG
        if 'random_seed' in self.options:
            np.random.seed(self.options['random_seed'])

        # RT in kJ/mol (OpenMM energy units)
        RT = (R / 1000.0) * T

        # Get masses
        current_system = self.top.OMM_system
        masses = []
        for atom in self.molecule.atoms():
            mass = current_system.getParticleMass(atom.index)
            masses.append(mass.value_in_unit(unit.dalton))
        masses = np.array(masses)
        m3 = np.repeat(np.expand_dims(masses, 1), 3, axis=1)
        natoms = len(masses)

        # Maxwell-Boltzmann sigma for velocity sampling
        m3_with_units = m3 * unit.dalton
        k_B = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = k_B * T * unit.kelvin
        kT_over_m = kT / m3_with_units
        sigma_MB_squared_value = kT_over_m.value_in_unit((unit.nanometer / unit.picosecond)**2)
        sigma_MB = np.sqrt(sigma_MB_squared_value)

        # KE conversion factor (1 Da * (nm/ps)^2 = 1 kJ/mol in OpenMM)
        ke_conversion = 1.0

        # Dual averaging parameters
        if adapt:
            gamma = 0.05
            t0 = 10
            kappa = 0.75
            mu = np.log(10 * delta_t)
            delta_t_bar = 1.0
            Hbar = 0.0
        else:
            delta_t_bar = delta_t
            Hbar = 0.0

        # Get main simulation context
        context = self.top.OMM_simulation.context

        # Initialize velocities
        context.setVelocitiesToTemperature(T * unit.kelvin)

        # Get initial state
        state = context.getState(getPositions=True, getEnergy=True)
        x_all = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Only sample over real atoms, not virtual particles
        # Virtual particles (if any) are at indices >= natoms
        x_m = x_all[:natoms]
        e_m = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Storage for samples
        # Store full configurations (real atoms + virtual particles) for compatibility with energyTerms
        xs = [np.copy(x_all)]
        energies = [e_m]

        # Main sampling loop
        elapsed_steps = 0
        m = 1

        while elapsed_steps < nsteps:
            # Resample velocities from Maxwell-Boltzmann
            v = np.multiply(sigma_MB, np.random.randn(natoms, 3))
            ke = 0.5 * np.sum(np.multiply(m3, np.square(v))) * ke_conversion

            # Joint log-probability
            joint = -(e_m + ke) / RT

            # Sample slice variable u ~ Uniform[0, exp(joint)]
            logu = joint - np.random.exponential(1)

            # Initialize tree
            xminus = np.copy(x_m)
            xplus = np.copy(x_m)
            vminus = np.copy(v)
            vplus = np.copy(v)

            # Initial tree depth and valid point count
            j = 0
            n = 1
            steps_m = 0

            # Build tree until U-turn or max depth
            s = True
            while s:
                # Choose direction: backward (-1) or forward (+1)
                direction = -1 if np.random.rand() < 0.5 else 1

                if direction == -1:
                    # Build tree backward
                    (xminus, vminus, _, _, xprime, eprime, nprime, sprime,
                     steps_m, alpha, nalpha) = self._build_tree(
                        xminus, vminus, logu, direction, j, delta_t,
                        context, m3, ke_conversion, RT, joint, steps_m
                    )
                else:
                    # Build tree forward
                    (_, _, xplus, vplus, xprime, eprime, nprime, sprime,
                     steps_m, alpha, nalpha) = self._build_tree(
                        xplus, vplus, logu, direction, j, delta_t,
                        context, m3, ke_conversion, RT, joint, steps_m
                    )

                # Metropolis-Hastings: accept sample from new subtree?
                if sprime and (np.random.rand() < float(nprime) / n):
                    x_m = xprime
                    e_m = eprime

                # Update valid point count
                n += nprime

                # Increment tree depth
                j += 1

                # Check stopping criteria
                s = (sprime and
                     self._stop_criterion(xminus, xplus, vminus, vplus) and
                     (elapsed_steps + steps_m * 2 < nsteps))

            # Update dual averaging statistics and adapt step size
            if adapt:
                eta = 1.0 / (m + t0)
                Hbar = (1 - eta) * Hbar + eta * (delta - alpha / nalpha)
                delta_t = np.exp(mu - np.sqrt(m) / gamma * Hbar)
                eta = m ** (-kappa)
                delta_t_bar = np.exp((1 - eta) * np.log(delta_t_bar) + eta * np.log(delta_t))

            # Store sample (only real atoms, exclude virtual particles)
            state_sample = context.getState(getPositions=True)
            x_sample_full = state_sample.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            xs.append(np.copy(x_sample_full[:natoms]))  # Only store real atoms, not virtual particles
            energies.append(e_m)

            # Update counters
            m += 1
            elapsed_steps += steps_m

            # Normalize if requested
            if normalize:
                # Get full state and update real atom positions
                state_full = context.getState(getPositions=True)
                x_all = state_full.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                x_all[:natoms] = x_m
                context.setPositions(x_all * unit.nanometer)

                state = context.getState(getPositions=True, enforcePeriodicBox=True)
                x_all_normalized = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                x_m = x_all_normalized[:natoms]

        # Update main context with final position (real atoms + virtual particles)
        state_final = context.getState(getPositions=True)
        x_all_final = state_final.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        x_all_final[:natoms] = x_m
        context.setPositions(x_all_final * unit.nanometer)

        # Return: acceptance rate from dual averaging (Hbar*nsteps gives total acc)
        # Convert delta_t_bar back to Quantity with femtosecond units for consistency
        delta_t_bar_with_units = delta_t_bar * unit.femtosecond
        return (xs, energies, Hbar * nsteps, nsteps, delta_t_bar_with_units)

    def _leapfrog_step(self, x, v, delta_t, context, m3, ke_conversion):
        """
        Single leapfrog integration step

        Parameters
        ----------
        x : ndarray (natoms, 3)
            Positions of real atoms only (no virtual particles)
        v : ndarray (natoms, 3)
            Velocities of real atoms only

        Returns
        -------
        x : ndarray
            New positions (real atoms only)
        v : ndarray
            New velocities (real atoms only)
        pe : float
            Potential energy at new position
        """
        natoms = len(x)

        # Get current full state including dummy particles (e.g., site center)
        # Dummy particles are massless fixed points and should not be moved
        state_full = context.getState(getPositions=True)
        x_all = state_full.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Update only real atom positions, preserve dummy particle positions
        x_all[:natoms] = x
        context.setPositions(x_all * unit.nanometer)

        # Half step in velocity using forces at current position
        state = context.getState(getForces=True, getEnergy=True)
        forces_all = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        # Extract forces for real atoms only (dummy particles have zero mass/force)
        forces = forces_all[:natoms]

        # OpenMM forces are F = -grad(U), so gradients g = -F
        # Note: forces are in kJ/(mol*nm), need to convert properly
        # In leapfrog: v += -0.5*dt*g/m where g is energy gradient
        # Since F = -g, we have: v += 0.5*dt*F/m
        v = v + 0.5 * delta_t * forces / m3

        # Full step in position (real atoms only)
        x = x + delta_t * v

        # Update real atom positions again, keep dummy particles fixed
        x_all[:natoms] = x
        context.setPositions(x_all * unit.nanometer)

        # Half step in velocity at new position
        state = context.getState(getForces=True, getEnergy=True)
        forces_all = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        forces = forces_all[:natoms]
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        v = v + 0.5 * delta_t * forces / m3

        return x, v, pe

    def _build_tree(self, x, v, logu, direction, depth, delta_t,
                    context, m3, ke_conversion, RT, joint_o, steps):
        """
        Recursively build tree for NUTS

        Returns
        -------
        xminus, vminus : ndarray
            Leftmost (backward) position and velocity
        xplus, vplus : ndarray
            Rightmost (forward) position and velocity
        xprime : ndarray
            Proposed sample from this subtree
        eprime : float
            Energy of proposed sample
        nprime : int
            Number of valid points in subtree
        sprime : bool
            Whether subtree is valid (no U-turn, energies reasonable)
        steps : int
            Updated step counter
        alpha : float
            Sum of acceptance probabilities
        nalpha : int
            Number of acceptance probability terms
        """
        if depth == 0:
            # Base case: single leapfrog step
            x_new, v_new, pe_new = self._leapfrog_step(
                x, v, direction * delta_t, context, m3, ke_conversion
            )
            steps += 1

            # Calculate kinetic energy
            ke_new = 0.5 * np.sum(np.multiply(m3, np.square(v_new))) * ke_conversion
            joint_new = -(pe_new + ke_new) / RT

            # Is new point in slice?
            nprime = int(logu < joint_new)

            # Is simulation stable?
            sprime = (np.abs(joint_o - joint_new) < 200.0 and
                     np.abs((pe_new - (joint_o * (-RT) - ke_new)) / RT) < 200.0)

            # Acceptance probability (clamped at 1)
            alpha = min(1.0, np.exp(joint_new - joint_o)) if sprime else 0.0

            # Return: minus=plus for depth 0
            return (np.copy(x_new), np.copy(v_new),
                   np.copy(x_new), np.copy(v_new),
                   np.copy(x_new), pe_new, nprime, sprime, steps, alpha, 1)

        else:
            # Recursion: build left and right subtrees
            # First subtree
            (xminus, vminus, xplus, vplus, xprime, eprime, nprime, sprime,
             steps, alpha, nalpha) = self._build_tree(
                x, v, logu, direction, depth - 1, delta_t,
                context, m3, ke_conversion, RT, joint_o, steps
            )

            # Second subtree (only if first didn't fail)
            if sprime:
                if direction == -1:
                    # Build second subtree on left
                    (xminus, vminus, _, _, xprime2, eprime2, nprime2, sprime2,
                     steps, alpha2, nalpha2) = self._build_tree(
                        xminus, vminus, logu, direction, depth - 1, delta_t,
                        context, m3, ke_conversion, RT, joint_o, steps
                    )
                else:
                    # Build second subtree on right
                    (_, _, xplus, vplus, xprime2, eprime2, nprime2, sprime2,
                     steps, alpha2, nalpha2) = self._build_tree(
                        xplus, vplus, logu, direction, depth - 1, delta_t,
                        context, m3, ke_conversion, RT, joint_o, steps
                    )

                # Choose sample from subtrees based on valid point counts
                if (nprime + nprime2) > 0 and np.random.rand() < float(nprime2) / (nprime + nprime2):
                    xprime = xprime2
                    eprime = eprime2

                # Update statistics
                nprime = nprime + nprime2
                sprime = sprime and sprime2 and self._stop_criterion(xminus, xplus, vminus, vplus)
                alpha = alpha + alpha2
                nalpha = nalpha + nalpha2

            return (xminus, vminus, xplus, vplus, xprime, eprime,
                   nprime, sprime, steps, alpha, nalpha)

    def _stop_criterion(self, xminus, xplus, vminus, vplus):
        """
        Check if trajectory has made a U-turn

        A U-turn is detected when the dot product of the trajectory vector
        with either endpoint velocity becomes negative.
        """
        thetavec = np.ravel(xplus - xminus)
        return (np.dot(thetavec, np.ravel(vminus)) > 0 and
                np.dot(thetavec, np.ravel(vplus)) > 0)