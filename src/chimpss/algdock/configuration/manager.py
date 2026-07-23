"""Configuration management for loading, minimizing, and scoring configurations

Handles all aspects of configuration management including:
- Loading configurations from various file formats
- Energy minimization with crash checking
- Configuration scoring and energy calculation
- Debug trajectory I/O
"""

import sys
import os
import time
import gzip
import pickle
import numpy as np

try:
    import MMTK
    from MMTK.ParticleProperties import Configuration
except ImportError:
    MMTK = None

try:
    import openmm
    import openmm.unit as unit
except ImportError:
    pass

from chimpss.algdock.logger import NullDevice
from chimpss.algdock.IO import load_pkl_gz, write_pkl_gz, HMStime
DEBUG = False


class ConfigurationManager:
    """Manages configuration loading, minimization, and scoring
    
    Instance methods that require access to simulation context and state.
    """
    
    def __init__(self, context):
        """Initialize with BPMF context
        
        Parameters
        ----------
        context : BPMF instance
            The BindingPMF instance with access to all simulation state
        """
        self.ctx = context
    
    def save_debug_trajectory(self, confs, filename, process='CD'):
        """Save conformations as multi-model PDB for visualization and debugging

        Parameters
        ----------
        confs : list of np.array
            List of configurations (each is Nx3 array in nanometers)
        filename : str
            Output PDB filename (relative to current directory)
        process : str
            'BC' or 'CD' to determine which topology to use
        """
        try:
          from openmm.app import PDBFile, AmberPrmtopFile
          import openmm.unit as unit

          # Load ligand-only topology (always use ligand topology for debug output)
          prmtop = AmberPrmtopFile(self.ctx.args.FNs['prmtop']['L'])
          topology = prmtop.topology

          with open(filename, 'w') as f:
            for idx, conf in enumerate(confs):
              f.write(f"MODEL     {idx+1:4d}\n")
              pos_with_units = conf * unit.nanometer
              PDBFile.writeModel(topology, pos_with_units, f)
              f.write(f"ENDMDL\n")

          self.ctx.log.tee(f"  Saved {len(confs)} structures to {filename}")
        except Exception as e:
          self.ctx.log.tee(f"  Warning: Could not save trajectory to {filename}: {e}")
    
    def checkedMinimizer(self, confs):
        """Minimizes configurations while checking for crashes and overflows

        Parameters
        ----------
        confs : list of np.array
          Configurations to minimize

        Returns
        -------
        confs : list of np.array
          Minimized configurations
        energies : list of float
          Energies of the minimized configurations
        """
        # Check for detailed minimization debug mode
        DEBUG_MINIMIZATION = os.environ.get('ALGDOCK_DEBUG_MINIMIZATION', '0') == '1'

        original_stderr = sys.stderr
        sys.stderr = NullDevice()  # Suppresses warnings for minimization

        minimized_confs = []
        minimized_energies = []
        self.ctx.log.recordStart('minimization')

        # DEBUG: Print what forces are active during minimization
        print("\n" + "="*80)
        print("DEBUG: _checkedMinimizer - Force setup at start of minimization")
        print("="*80)
        if hasattr(self.ctx.top, 'context'):
            # OpenMM
            system = self.ctx.top.context.getSystem()
            print(f"Number of forces: {system.getNumForces()}")
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force_name = force.__class__.__name__
                print(f"  Force {i}: {force_name}")
        else:
            # MMTK
            print(f"MMTK universe energy evaluator: {type(self.ctx.top.universe._energy_evaluator)}")
        print("="*80 + "\n")

        initial_energies_debug = []

        if MMTK:
          from MMTK.Minimization import SteepestDescentMinimizer  # @UnresolvedImport
          minimizer = SteepestDescentMinimizer(self.ctx.top.universe)

          for conf_idx, conf in enumerate(confs):
            self.ctx.top.universe.setConfiguration(
              Configuration(self.ctx.top.universe, conf))
            x_o = np.copy(self.ctx.top.universe.configuration().array)
            e_o = self.ctx.top.universe.energy()

            # DEBUG: Store initial energy
            if conf_idx < 10:
                initial_energies_debug.append(e_o)

            for rep in range(50):
              minimizer(steps=25)
              x_n = np.copy(self.ctx.top.universe.configuration().array)
              e_n = self.ctx.top.universe.energy()
              diff = abs(e_o - e_n)
              # Only revert if energy INCREASED significantly (not just changed)
              # This allows minimization of high-energy structures (e.g., Vina poses with grid clashes)
              # that may improve by >1000 kJ/mol in a single step
              if np.isnan(e_n) or diff < 0.05 or e_n > e_o + 1000.:
                self.ctx.top.universe.setConfiguration(
                  Configuration(self.ctx.top.universe, x_o))
                break
              else:
                x_o = x_n
                e_o = e_n
            if not np.isnan(e_o):
              minimized_confs.append(x_o)
              minimized_energies.append(e_o)

        else:
          # OpenMM minimization using LocalEnergyMinimizer
          import openmm
          import openmm.unit as unit

          # Always log detailed minimization for first 3 poses
          sys.stderr = original_stderr
          self.ctx.log.tee("\n" + "="*80)
          self.ctx.log.tee("Detailed minimization logging (first 3 poses)")
          self.ctx.log.tee("="*80)
          sys.stderr = NullDevice()

          for conf_idx, conf in enumerate(confs):
            self.ctx.top.setConfiguration(conf)
            x_o = np.copy(self.ctx.top.configuration().array)
            e_o = self.ctx.top.energy()

            # DEBUG: Store initial energy
            if conf_idx < 10:
                initial_energies_debug.append(e_o)

            # Always log for first 3 poses
            log_this_pose = (conf_idx < 3)

            if log_this_pose:
                sys.stderr = original_stderr
                self.ctx.log.tee(f"\n--- Pose {conf_idx+1} ---")
                self.ctx.log.tee(f"  Initial energy: {e_o:.2f} kJ/mol")
                self.ctx.log.tee(f"  Initial COM: [{np.mean(conf[:,0]):.3f}, {np.mean(conf[:,1]):.3f}, {np.mean(conf[:,2]):.3f}]")

                # Print per-force energies
                system = self.ctx.top.context.getSystem()
                self.ctx.log.tee(f"  Initial per-force energies:")
                for i in range(system.getNumForces()):
                    force = system.getForce(i)
                    force_class = force.__class__.__name__
                    # Try to get the force name (for grid forces)
                    try:
                        force_label = force.getName()
                        if force_label:
                            force_name = f"{force_class} ({force_label})"
                        else:
                            force_name = force_class
                    except:
                        force_name = force_class
                    state = self.ctx.top.context.getState(getEnergy=True, groups={i})
                    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    self.ctx.log.tee(f"    {force_name:40s}: {energy:10.2f} kJ/mol")

                self.ctx.log.tee(f"  Tolerance: 1.0 kJ/(mol*nm), MaxIterations: 25")
                sys.stderr = NullDevice()

            # Debug logging for first pose if requested
            if DEBUG_MINIMIZATION and conf_idx == 0:
                sys.stderr = original_stderr  # Temporarily restore stderr for debug output
                print(f"\n{'='*80}")
                print(f"OpenMM Minimization Debug (Pose 1, tolerance=1.0 kJ/(mol*nm))")
                print(f"{'='*80}")
                print(f"Initial energy: {e_o:.2f} kJ/mol")
                print(f"Initial COM: {np.mean(conf, axis=0)}")
                print(f"\n{'Iter':<6} {'Steps':<10} {'Energy (kJ/mol)':<20} {'Change':<15} {'Status'}")
                print("-" * 80)
                sys.stderr = NullDevice()

            for rep in range(50):
              # Minimize using LocalEnergyMinimizer
              openmm.LocalEnergyMinimizer.minimize(
                self.ctx.top.context,
                tolerance=1.0,  # kJ/mol/nm (tightened from 10.0 for better convergence)
                maxIterations=25
              )

              # Get new configuration and energy
              x_n = np.copy(self.ctx.top.configuration().array)
              e_n = self.ctx.top.energy()
              diff = abs(e_o - e_n)

              # Always log for first 3 poses
              if log_this_pose:
                  sys.stderr = original_stderr
                  status = ""
                  if np.isnan(e_n):
                      status = " (NaN!)"
                  elif diff < 0.05:
                      status = " (converged)"
                  elif e_n > e_o + 1000.:
                      status = " (DIVERGING!)"
                  self.ctx.log.tee(f"    Iter {rep+1:2d}: {e_n:10.2f} kJ/mol (change: {e_n-e_o:+8.2f}){status}")
                  sys.stderr = NullDevice()

              # Debug logging
              if DEBUG_MINIMIZATION and conf_idx == 0:
                  sys.stderr = original_stderr
                  status = ""
                  if np.isnan(e_n):
                      status = "NaN!"
                  elif diff < 0.05:
                      status = "Converged"
                  elif e_n > e_o + 1000.:
                      status = "Diverging!"
                  print(f"{rep+1:<6} {rep*25:3d}-{(rep+1)*25:<3d}  {e_n:<20.2f} {e_n-e_o:+14.2f}  {status}")
                  sys.stderr = NullDevice()

              # Only revert if energy INCREASED significantly (not just changed)
              # This allows minimization of high-energy structures (e.g., Vina poses with grid clashes)
              # that may improve by >1000 kJ/mol in a single step
              if np.isnan(e_n) or diff < 0.05 or e_n > e_o + 1000.:
                # Revert to previous configuration
                self.ctx.top.setConfiguration(x_o)
                break
              else:
                x_o = x_n
                e_o = e_n

            if not np.isnan(e_o):
              minimized_confs.append(x_o)
              minimized_energies.append(e_o)

              # Always log summary for first 3 poses
              if log_this_pose:
                  sys.stderr = original_stderr
                  self.ctx.log.tee(f"  Final energy: {e_o:.2f} kJ/mol (total change: {e_o - initial_energies_debug[conf_idx]:+.2f})")
                  self.ctx.log.tee(f"  Completed {rep+1} iterations")

                  # Print final per-force energies
                  system = self.ctx.top.context.getSystem()
                  self.ctx.log.tee(f"  Final per-force energies:")
                  for i in range(system.getNumForces()):
                      force = system.getForce(i)
                      force_class = force.__class__.__name__
                      # Try to get the force name (for grid forces)
                      try:
                          force_label = force.getName()
                          if force_label:
                              force_name = f"{force_class} ({force_label})"
                          else:
                              force_name = force_class
                      except:
                          force_name = force_class
                      state = self.ctx.top.context.getState(getEnergy=True, groups={i})
                      energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                      self.ctx.log.tee(f"    {force_name:40s}: {energy:10.2f} kJ/mol")

                  sys.stderr = NullDevice()

              # Debug final summary
              if DEBUG_MINIMIZATION and conf_idx == 0:
                  sys.stderr = original_stderr
                  print("-" * 80)
                  print(f"Final energy: {e_o:.2f} kJ/mol")
                  print(f"Total change: {e_o - initial_energies_debug[0]:+.2f} kJ/mol")
                  print(f"Final COM: {np.mean(x_o, axis=0)}")
                  print("="*80 + "\n")
                  sys.stderr = NullDevice()

        sys.stderr = original_stderr  # Restores error reporting

        confs = minimized_confs
        energies = minimized_energies

        # DEBUG: Print before and after energies
        print("\n" + "="*80)
        print("DEBUG: Energy comparison (first 10 configurations)")
        print("="*80)
        print(f"{'Pose':<6} {'Before (kJ/mol)':<20} {'After (kJ/mol)':<20} {'Change':<15}")
        print("-" * 80)
        for i in range(min(len(initial_energies_debug), len(energies))):
            before = initial_energies_debug[i]
            after = energies[i]
            change = after - before
            print(f"{i+1:<6} {before:<20.2f} {after:<20.2f} {change:<15.2f}")
        print("="*80 + "\n")

        self.ctx.log.tee("  minimized %d configurations in "%len(confs) + \
          HMStime(self.ctx.log.timeSince('minimization')) + \
          "\n  the first %d energies are:\n  "%min(len(confs),10) + \
          ', '.join(['%.2f'%e for e in energies[:10]]))
        return confs, energies

    def get_scaling_factors(self, scaling_property):
        """
        Get per-atom scaling factors for grid forces (OpenMM only)

        Parameters
        ----------
        scaling_property : str
            Property name from ligand database (e.g., 'scaling_factor_electrostatic')

        Returns
        -------
        np.array
            Array of scaling factors in prmtop order
        """
        import numpy as np

        # Get scaling factors from molecule database
        # For OpenMM, atoms are in prmtop order
        scaling_factors = []
        for atom in self.ctx.top.molecule.atoms():
          # Try to get property from atom object
          # This requires the property to be set during topology initialization
          # For now, return ones - this needs to be connected to ligand database
          scaling_factors.append(1.0)

        return np.array(scaling_factors)

