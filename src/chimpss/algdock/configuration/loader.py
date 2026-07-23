"""Configuration loading and filtering

Handles loading configurations from various file formats and filtering them
based on site constraints.
"""

import os
import pickle
import gzip
import numpy as np

try:
    from MMTK import Configuration
    MMTK = True
except ImportError:
    MMTK = False


class ConfigurationLoader:
    """Loads and filters molecular configurations

    Uses context pattern - accepts BPMF instance with all state.
    """

    def __init__(self, context):
        """Initialize with BPMF context

        Parameters
        ----------
        context : BPMF instance
            The BPMF object containing args, top, system, data, log, etc.
        """
        self.ctx = context

    def get_confs_to_rescore(self,
                            nconfs=None,
                            site=False,
                            minimize=True,
                            sort=True):
        """Returns configurations to rescore and their corresponding energies

        Parameters
        ----------
        nconfs : int or None
          Number of configurations to keep. If it is smaller than the number
          of unique configurations, then the lowest energy configurations will
          be kept. If it is larger, then the lowest energy configuration will be
          duplicated. If it is None, then all unique configurations will be kept.
        site : bool
          If True, configurations that are outside of the binding site
          will be discarded.
        minimize : bool
          If True, the configurations will be minimized
        sort : bool
          If True, configurations and energies will be sorted by DECREASING energy.

        Returns
        -------
        confs : list of np.array
          Configurations
        energies : list of float
          Energies of the configurations
        """
        # Get configurations
        count = {'xtal': 0, 'dock6': 0, 'initial_CD': 0, 'duplicated': 0}

        # based on the score option
        if self.ctx.args.FNs['score'] == 'default':
          confs = [np.copy(self.ctx.data['CD'].confs['ligand'])]
          count['xtal'] = 1
          Es = {}
          if nconfs is None:
            nconfs = 1
        elif (self.ctx.args.FNs['score'] is None) or (not os.path.isfile(
            self.ctx.args.FNs['score'])):
          confs = []
          Es = {}
        elif self.ctx.args.FNs['score'].endswith('.mol2') or \
             self.ctx.args.FNs['score'].endswith('.mol2.gz'):
          import chimpss.algdock.IO
          IO_dock6_mol2 = chimpss.algdock.IO.dock6_mol2()
          (confs, Es) = IO_dock6_mol2.read(self.ctx.args.FNs['score'], \
            reorder=self.ctx.top.inv_prmtop_atom_order_L,
            multiplier=0.1) # to convert Angstroms to nanometers
          count['dock6'] = len(confs)
        elif self.ctx.args.FNs['score'].endswith('.mdcrd'):
          import chimpss.algdock.IO
          IO_crd = chimpss.algdock.IO.crd()
          lig_crds = IO_crd.read(self.ctx.args.FNs['score'], \
            multiplier=0.1) # to convert Angstroms to nanometers
          # TODO: Add OpenMM support
          if MMTK:
            natoms = self.ctx.top.universe.numberOfAtoms()
          else:
            natoms = self.ctx.top.numberOfAtoms()
          confs = np.array_split(lig_crds, lig_crds.shape[0] / natoms)
          confs = [conf[self.ctx.top.inv_prmtop_atom_order_L, :] for conf in confs]
          Es = {}
        elif self.ctx.args.FNs['score'].endswith('.nc'):
          from netCDF4 import Dataset
          dock6_nc = Dataset(self.ctx.args.FNs['score'], 'r')
          confs = [
            dock6_nc.variables['confs'][n][self.ctx.top.inv_prmtop_atom_order_L, :]
            for n in range(dock6_nc.variables['confs'].shape[0])
          ]
          Es = dict([(key, dock6_nc.variables[key][:])
                     for key in dock6_nc.variables.keys() if key != 'confs'])
          dock6_nc.close()
          count['dock6'] = len(confs)
        elif self.ctx.args.FNs['score'].endswith('.pkl.gz'):
          F = gzip.open(self.ctx.args.FNs['score'], 'r')
          confs = pickle.load(F)
          F.close()
          if not isinstance(confs, list):
            confs = [confs]
          Es = {}
        else:
          raise Exception('Input configuration format not recognized')

        # based on the seeds
        # TODO: Use CD seeds for BC
        if (self.ctx.data['CD'].confs['seeds'] is not None) and \
           (self.ctx.args.params['CD']['pose']==-1):
          confs = confs + self.ctx.data['CD'].confs['seeds']
          Es = {}
          count['initial_CD'] = len(self.ctx.data['CD'].confs['seeds'])

        if len(confs) == 0:
          return ([], {})

        # Site filtering: keep only configurations in the binding site
        if site:
          # Filters out configurations not in the binding site
          confs_in_site = []
          Es_in_site = dict([(label, []) for label in Es.keys()])

          if MMTK:
            old_eval = None
            if (None, None, None) in self.ctx.top.universe._evaluator.keys():
              old_eval = self.ctx.top.universe._evaluator[(None, None, None)]
            self.ctx.system.setParams({'site': True, 'T': self.ctx.T_TARGET})
            for n in range(len(confs)):
              self.ctx.top.universe.setConfiguration(
                Configuration(self.ctx.top.universe, confs[n]))
              if self.ctx.top.universe.energy() < 1.:
                confs_in_site.append(confs[n])
                for label in Es.keys():
                  Es_in_site[label].append(Es[label][n])
            if old_eval is not None:
              self.ctx.top.universe._evaluator[(None, None, None)] = old_eval
          else:
            # OpenMM implementation
            # Save current system state before temporarily switching to site-only force field
            saved_system = self.ctx.top.OMM_system
            saved_simulation = self.ctx.top.OMM_simulation
            saved_context = self.ctx.top.context

            # Temporarily set system with site force field for filtering
            self.ctx.system.setParams({'site': True, 'T': self.ctx.T_TARGET})

            # Get sphere force information for debugging
            sphere_ff = self.ctx.system._forceFields.get('site')

            print(f'  Sphere filtering: center={self.ctx.args.params["CD"]["site_center"]}, R={self.ctx.args.params["CD"]["site_max_R"]}')
            print(f'  Checking {len(confs)} configurations...')

            for n in range(len(confs)):
              self.ctx.top.setConfiguration(confs[n])

              # Calculate COM and distance from sphere center
              com = self.ctx.top.centerOfMass()
              distance_from_center = ((com[0] - sphere_ff.center[0])**2 +
                                      (com[1] - sphere_ff.center[1])**2 +
                                      (com[2] - sphere_ff.center[2])**2)**0.5

              # Get total energy
              total_energy = self.ctx.top.energy()

              # Print details for first 3 configs or configs that fail
              if n < 3 or total_energy >= 1.:
                # Get energy breakdown by force
                print(f'    Config {n}: COM=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f}), dist={distance_from_center:.3f} nm, E_total={total_energy:.1f} kcal/mol')
                import openmm.unit as unit
                for force_idx in range(self.ctx.top.OMM_system.getNumForces()):
                    force = self.ctx.top.OMM_system.getForce(force_idx)
                    force_group = force.getForceGroup()
                    state_force = self.ctx.top.context.getState(getEnergy=True, groups={force_group})
                    energy_kcal = state_force.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 4.184
                    force_name = force.getName() if hasattr(force, 'getName') else force.__class__.__name__
                    print(f'      {force_name} (group {force_group}): {energy_kcal:.1f} kcal/mol')

              if total_energy < 1.:
                confs_in_site.append(confs[n])
                for label in Es.keys():
                  Es_in_site[label].append(Es[label][n])

            print(f'  Result: {len(confs_in_site)}/{len(confs)} configurations passed sphere filter')

            # Restore original system state (with MM and grid forces)
            self.ctx.top.OMM_system = saved_system
            self.ctx.top.OMM_simulation = saved_simulation
            self.ctx.top.context = saved_context

          confs = confs_in_site
          Es = Es_in_site

        # Check if energy calculation works
        if MMTK:
          try:
            self.ctx.top.universe.energy()
          except ValueError:
            return (confs, {})
        else:
          try:
            self.ctx.top.energy()
          except Exception:
            return (confs, {})

        if minimize:
          # Save trajectory before minimization for debugging
          if os.environ.get('ALGDOCK_SAVE_TRAJECTORIES', 'false').lower() == 'true' and len(confs) > 0:
            self.ctx._save_debug_trajectory(confs, 'CD_reinit_before_minimize.pdb', process='CD')

          Es = {}
          (confs, energies) = self.ctx._checkedMinimizer(confs)

          # Save trajectory after minimization for debugging
          if os.environ.get('ALGDOCK_SAVE_TRAJECTORIES', 'false').lower() == 'true' and len(confs) > 0:
            self.ctx._save_debug_trajectory(confs, 'CD_reinit_after_minimize.pdb', process='CD')
        else:
          # Evaluate energies
          energies = []
          for conf in confs:
            if MMTK:
              self.ctx.top.universe.setConfiguration(
                Configuration(self.ctx.top.universe, conf))
              energies.append(self.ctx.top.universe.energy())
            else:
              self.ctx.top.setConfiguration(conf)
              energies.append(self.ctx.top.energy())

        if sort and len(confs) > 0:
          # Sort configurations by DECREASING energy
          energies, confs = (list(l) for l in zip(*sorted(zip(energies, confs), \
            key=lambda p:p[0], reverse=True)))

        # Shrink or extend configuration and energy array
        if nconfs is not None:
          confs = confs[-nconfs:]
          energies = energies[-nconfs:]
          while len(confs) < nconfs:
            confs.append(confs[-1])
            energies.append(energies[-1])
            count['duplicated'] += 1
          count['nconfs'] = nconfs
        else:
          count['nconfs'] = len(confs)
        count['minimized'] = {True: ' minimized', False: ''}[minimize]
        Es['total'] = np.array(energies)

        self.ctx.log.tee(
          "  keeping {nconfs}{minimized} configurations out of\n  {xtal} from xtal, {dock6} from dock6, {initial_CD} from initial CD, and {duplicated} duplicated"
          .format(**count))
        return (confs, Es)
