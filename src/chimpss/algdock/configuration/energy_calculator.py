"""Configuration energy calculation

Calculates energies for molecular configurations using various methods
(MM, OBC, grids, external programs).
"""

import os
import time
import numpy as np
from chimpss.algdock.IO import load_pkl_gz, write_pkl_gz, HMStime

try:
    from MMTK import Configuration
    MMTK = True
except ImportError:
    MMTK = False

# DEBUG flag from BindingPMF
DEBUG = False


class ConfigurationEnergyCalculator:
    """Calculates energies for molecular configurations

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

    def configuration_energies(self, minimize=False, max_confs=None):
        """
        Calculates the energy for configurations from self.args.FNs['score']

        Parameters
        ----------
        minimize : bool
            If True, configurations will be minimized before energy calculation
        max_confs : int or None
            Maximum number of configurations to evaluate

        Returns
        -------
        confs : list
            List of configurations
        Es : dict
            Dictionary of energy terms for each configuration
        """
        # Determine the name of the file
        prefix = 'xtal' if self.ctx.args.FNs['score']=='default' else \
          os.path.basename(self.ctx.args.FNs['score']).split('.')[0]
        if minimize:
          prefix = 'min_' + prefix
        energyFN = os.path.join(self.ctx.args.dir['CD'], prefix + '.pkl.gz')

        # Set the force field to fully interacting
        params_full = self.ctx.system.paramsFromAlpha(1.0, 'CD')
        self.ctx.system.setParams(params_full)

        # Load the configurations
        if os.path.isfile(energyFN):
          (confs, Es) = load_pkl_gz(energyFN)
        else:
          (confs, Es) = self.ctx._get_confs_to_rescore(site=False, \
            minimize=minimize, sort=False)

        self.ctx.log.set_lock('CD')
        self.ctx.log.tee("\n>>> Calculating energies for %d configurations, "%len(confs) + \
          "starting at " + \
          time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "\n")
        self.ctx.log.recordStart('configuration_energies')

        updated = False
        # Calculate MM and OBC energies
        if not 'MM' in Es.keys():
          Es = self.ctx.system.energyTerms(confs, Es)
          solvation_o = self.ctx.args.params['CD']['solvation']
          self.ctx.args.params['CD']['solvation'] = 'Full'
          if self.ctx.system.isForce('OBC'):
            del self.ctx._forceFields['OBC']
          self.ctx.system.clear_evaluators()
          self.ctx.system.setParams(params_full)
          Es = self.ctx.system.energyTerms(confs, Es)
          self.ctx.args.params['CD']['solvation'] = solvation_o
          updated = True

        # Direct electrostatic energy
        FN = os.path.join(os.path.dirname(self.ctx.args.FNs['grids']['ELE']),
                          'direct_ele.nc')
        if not 'direct_ELE' in Es.keys() and os.path.isfile(FN):
          key = 'direct_ELE'
          Es[key] = np.zeros(len(confs))

          if MMTK:
            from chimpss.algdock.ForceFields.Grid.Interpolation import InterpolationForceField
            FF = InterpolationForceField(FN, \
              scaling_property='scaling_factor_electrostatic')
            self.ctx.top.universe.setForceField(FF)
            for c in range(len(confs)):
              self.ctx.top.universe.setConfiguration(
                Configuration(self.ctx.top.universe, confs[c]))
              Es[key][c] = self.ctx.top.universe.energy()
          else:
            # OpenMM implementation
            from chimpss.algdock.ForceFields.Grid.GridForceOpenMM import GridForceOpenMM
            import openmm
            import openmm.unit as unit

            # Get scaling factors for electrostatic
            scaling_factors = self.ctx._get_scaling_factors('scaling_factor_electrostatic')

            # Create grid force
            gf = GridForceOpenMM(FN, name='direct_ELE', strength=1.0,
                                 scaling_property='scaling_factor_electrostatic')

            # Add to a temporary system to calculate energies
            # We need to recreate the system for each grid force
            for c in range(len(confs)):
              # Create a fresh system with just the grid force
              temp_system = openmm.System()
              for i in range(self.ctx.top.numberOfAtoms()):
                mass = self.ctx.top.OMM_system.getParticleMass(i)
                temp_system.addParticle(mass)

              # Add grid force
              gf.add_to_system(temp_system, self.ctx.top.molecule, scaling_factors)

              # Create temporary context
              integrator = openmm.LangevinIntegrator(300 * unit.kelvin,
                                                      1 / unit.picosecond,
                                                      0.002 * unit.picoseconds)
              context = openmm.Context(temp_system, integrator)
              context.setPositions(confs[c] * unit.nanometer)

              # Get energy
              state = context.getState(getEnergy=True)
              Es[key][c] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

              del context, integrator

          updated = True

        # Calculate symmetry-corrected RMSD
        if not 'rmsd' in Es.keys() and (self.ctx.args.params['CD']['rmsd'] is
                                        not False):
          Es['rmsd'] = self.ctx.get_rmsds(confs)
          updated = True

        if updated:
          self.ctx.log.tee("\nElapsed time for ligand MM, OBC, and grid energies: " + \
            HMStime(self.ctx.log.timeSince('configuration_energies')), \
            process='CD')
        self.ctx.log.clear_lock('CD')

        # Reduce the number of conformations
        if max_confs is not None:
          confs = confs[:max_confs]

        # Implicit solvent energies
        self.ctx.data['CD'].confs['starting_poses'] = None
        from chimpss.algdock.postprocessing import Postprocessing
        pp_complete = Postprocessing(self.ctx.args, self.ctx.log, self.ctx.top, self.ctx.top_RL, self.ctx.system, self.ctx.data, self.ctx.save).run([('original', 0, 0, 'R')])

        for phase in self.ctx.args.params['CD']['phases']:
          if not 'R' + phase in Es.keys():
            Es['R' + phase] = self.ctx.args.params['CD']['receptor_' + phase]

        toClear = []
        for phase in self.ctx.args.params['CD']['phases']:
          for moiety in ['L', 'RL']:
            if not moiety + phase in Es.keys():
              outputname = os.path.join(self.ctx.args.dir['CD'],
                                        '%s.%s%s' % (prefix, moiety, phase))
              if phase.startswith('NAMD'):
                traj_FN = os.path.join(self.ctx.args.dir['CD'],
                                       '%s.%s.dcd' % (prefix, moiety))
                self.ctx._write_traj(traj_FN, confs, moiety)
              elif phase.startswith('sander'):
                traj_FN = os.path.join(self.ctx.args.dir['CD'],
                                       '%s.%s.mdcrd' % (prefix, moiety))
                self.ctx._write_traj(traj_FN, confs, moiety)
              elif phase.startswith('gbnsr6'):
                traj_FN = os.path.join(self.ctx.args.dir['CD'], \
                  '%s.%s%s'%(prefix,moiety,phase),'in.crd')
              elif phase.startswith('OpenMM'):
                traj_FN = None
              elif phase in ['APBS_PBSA']:
                traj_FN = os.path.join(self.ctx.args.dir['CD'],
                                       '%s.%s.pqr' % (prefix, moiety))
              else:
                raise Exception('Unknown phase!')
              if not traj_FN in toClear:
                toClear.append(traj_FN)
              for program in ['NAMD', 'sander', 'gbnsr6', 'OpenMM', 'APBS']:
                if phase.startswith(program):
                  # TODO: Mechanism to do partial calculation
                  Es[moiety+phase] = getattr(self.ctx,'_%s_Energy'%program)(confs, \
                    moiety, phase, traj_FN, outputname, debug=DEBUG)
                  updated = True
                  # Get any data added since the calculation started
                  if os.path.isfile(energyFN):
                    (confs_o, Es_o) = load_pkl_gz(energyFN)
                    for key in Es_o.keys():
                      if key not in Es.keys():
                        Es[key] = Es_o[key]
                  # Store the data
                  self.ctx.log.tee(write_pkl_gz(energyFN, (confs, Es)))
                  break
        for FN in toClear:
          if (FN is not None) and os.path.isfile(FN):
            os.remove(FN)

        for key in Es.keys():
          Es[key] = np.array(Es[key])
        self.ctx._combine_MM_and_solvent(Es)

        if updated:
          self.ctx.log.set_lock('CD')
          self.ctx.log.tee("\nElapsed time for energies: " + \
            HMStime(self.ctx.log.timeSince('configuration_energies')), \
            process='CD')
          self.ctx.log.clear_lock('CD')

          # Get any data added since the calculation started
          if os.path.isfile(energyFN):
            (confs_o, Es_o) = load_pkl_gz(energyFN)
            for key in Es_o.keys():
              if key not in Es.keys():
                Es[key] = Es_o[key]

          # Store the data
          self.ctx.log.tee(write_pkl_gz(energyFN, (confs, Es)))
        return (confs, Es)
