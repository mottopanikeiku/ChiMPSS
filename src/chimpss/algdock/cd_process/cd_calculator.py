"""CD thermodynamic path calculator

Calculates binding PMF for the CD replica exchange ladder.

CD cools while turning on receptor-ligand interactions from high temperature
with grids OFF (e.g., 600K, state C) to target temperature with grids ON
(e.g., 300K, state D). Alpha (0.0 to 1.0) controls:
  - Temperature cooling: T_HIGH to T_TARGET
  - Grid coupling: OFF to ON via alpha_g sigmoid
  - Solvation (mode-dependent): OBC scaling varies by solvation mode

Purpose: Calculate binding free energy by gradually introducing receptor interactions.
"""

import os
import time
import copy
import numpy as np
from chimpss.algdock.IO import load_pkl_gz, write_pkl_gz, HMStime

# Physical constant
R = 8.314472 / 1000.0  # Gas constant in kJ/(mol*K)


class CDCalculator:
    """Calculates CD (C→D state) binding free energies

    Uses context pattern - accepts BPMF instance with all state.
    """

    def __init__(self, context):
        """Initialize with BPMF context

        Parameters
        ----------
        context : BPMF instance
            The BPMF object containing args, log, data, top, system, etc.
        """
        self.ctx = context

    def initial_CD(self, randomOnly=False):
        """
        Docks the ligand into the receptor

        Intermediate thermodynamic states are chosen such that
        thermodynamic length intervals are approximately constant.
        Configurations from each state are subsampled to seed the next simulation.

        Extracted from BindingPMF.initial_CD (lines 538-560)
        """

        if (len(self.ctx.data['CD'].protocol) >
            0) and (self.ctx.data['CD'].protocol[-1]['crossed']):
          return  # Initial CD already complete

        from chimpss.algdock.ligand_preparation import LigandPreparation
        seeds = LigandPreparation(self.ctx.args, self.ctx.log, self.ctx.top, self.ctx.system,
                                  self.ctx._get_confs_to_rescore, self.ctx.iterator,
                                  self.ctx.data).run('CD')

        from chimpss.algdock.initialization import Initialization
        Initialization(self.ctx.args, self.ctx.log, self.ctx.top, self.ctx.system,
                      self.ctx.iterator, self.ctx.data, self.ctx.save, self.ctx._u_kln).run('CD', seeds)

        return True

    def SIRS(self, process):
        """Sampling importance resampling

        Extracted from BindingPMF.SIRS (lines 1013-1058)

        Parameters
        ----------
        process : str
            'BC' or 'CD'
        """
        # The code below is only for sampling importance resampling
        if not self.ctx.args.params[process]['sampling_importance_resampling']:
          return

        # Calculate appropriate free energy
        if process == 'BC':
          self.ctx.calc_f_L(do_solvation=False)
          f_k = self.ctx.f_L['BC_MBAR'][-1]
        elif process == 'CD':
          self.calc_f_RL(do_solvation=False)
          f_k = self.ctx.f_RL['grid_MBAR'][-1]

        # Get weights for sampling importance resampling
        # MBAR weights for replica exchange configurations

        protocol = self.ctx.data[process].protocol
        Es_repX = [[copy.deepcopy(self.ctx.data[process].Es[k][-1])] for k in range(len(protocol))]
        (u_kln, N_k) = self.ctx._u_kln(Es_repX, protocol)

        # This is a more direct way to get the weights
        from pymbar.utils import kln_to_kn
        u_kn = kln_to_kn(u_kln, N_k=N_k)

        from pymbar.utils import logsumexp
        log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
        logW = f_k - u_kn.T - log_denominator_n[:, np.newaxis]
        W_nl = np.exp(logW)
        for k in range(len(protocol)):
          W_nl[:, k] = W_nl[:, k] / np.sum(W_nl[:, k])

        # This is for conversion to 2 indicies: state and snapshot
        cum_N_state = np.cumsum([0] + list(N_k))

        def linear_index_to_snapshot_index(ind):
          state_index = list(ind < cum_N_state).index(True) - 1
          nis_index = ind - cum_N_state[state_index]
          return (state_index, nis_index)

        # Selects new replica exchange snapshots
        confs_repX = self.ctx.data[process].confs['last_repX']
        self.ctx.data[process].confs['replicas'] = []
        for k in range(len(protocol)):
          (s,n) = linear_index_to_snapshot_index(\
            np.random.choice(range(W_nl.shape[0]), size = 1, p = W_nl[:,k])[0])
          self.ctx.data[process].confs['replicas'].append(np.copy(confs_repX[s][n]))

    def calc_f_RL(self, readOnly=False, do_solvation=True, redo=False):
        """
        Calculates the binding potential of mean force
        redo recalculates f_RL and B except grid_MBAR

        Extracted from BindingPMF.calc_f_RL (lines 562-857)
        """
        if self.ctx.data['CD'].protocol == []:
          return  # Initial CD is incomplete

        # Initialize variables as empty lists or by loading data
        if self.ctx.args.params['CD']['pose'] == -1:
          f_RL_FN = os.path.join(self.ctx.args.dir['CD'], 'f_RL.pkl.gz')
        else:
          f_RL_FN = os.path.join(self.ctx.args.dir['CD'], \
            'f_RL_pose%03d.pkl.gz'%self.ctx.args.params['CD']['pose'])

        dat = load_pkl_gz(f_RL_FN)
        if (dat is not None):
          (self.ctx.f_L, self.ctx.stats_RL, self.ctx.f_RL, self.ctx.B) = dat
        else:
          self.ctx._clear_f_RL()
        if readOnly:
          return True

        if redo:
          for key in self.ctx.f_RL.keys():
            if key != 'grid_MBAR':
              self.ctx.f_RL[key] = []
          self.ctx.B = {'MMTK_MBAR': []}
          for phase in self.ctx.args.params['CD']['phases']:
            for method in ['min_Psi', 'mean_Psi', 'EXP', 'MBAR']:
              self.ctx.B[phase + '_' + method] = []

        # Make sure all the energies are available
        for c in range(self.ctx.data['CD'].cycle):
          if len(self.ctx.data['CD'].Es[-1][c].keys()) == 0:
            self.ctx.log.tee("  skipping the binding PMF calculation")
            return
        if not hasattr(self.ctx, 'f_L'):
          self.ctx.log.tee("  skipping the binding PMF calculation")
          return

        start_string = "\n>>> Complex free energy calculations, starting at " + \
          time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "\n"
        self.ctx.log.recordStart('BPMF')

        updated = False

        def set_updated_to_True(updated, start_string, quiet=False):
          if (updated is False):
            self.ctx.log.set_lock('CD')
            if not quiet:
              self.ctx.log.tee(start_string)
          return True

        K = len(self.ctx.data['CD'].protocol)

        # Store stats_RL
        # Internal energies
        self.ctx.stats_RL['u_K_sampled'] = \
          [self.ctx._u_kln([self.ctx.data['CD'].Es[-1][c]],[self.ctx.data['CD'].protocol[-1]]) \
            for c in range(self.ctx.data['CD'].cycle)]
        self.ctx.stats_RL['u_KK'] = \
          [np.hstack([self.ctx._u_kln([self.ctx.data['CD'].Es[k][c]],[self.ctx.data['CD'].protocol[k]]) \
            for k in range(len(self.ctx.data['CD'].protocol))]) \
              for c in range(self.ctx.data['CD'].cycle)]

        # Interaction energies
        for c in range(len(self.ctx.stats_RL['Psi_grid']), self.ctx.data['CD'].cycle):
          self.ctx.stats_RL['Psi_grid'].append(
              (self.ctx.data['CD'].Es[-1][c]['LJr'] + \
               self.ctx.data['CD'].Es[-1][c]['LJa'] + \
               self.ctx.data['CD'].Es[-1][c]['ELE'])/(R*self.ctx.T_TARGET))
          updated = set_updated_to_True(updated,
                                        start_string,
                                        quiet=not do_solvation)

        # Estimate cycle at which simulation has equilibrated
        eqc_o = self.ctx.stats_RL['equilibrated_cycle']
        self.ctx.stats_RL['equilibrated_cycle'] = self.ctx._get_equilibrated_cycle('CD')
        if self.ctx.stats_RL['equilibrated_cycle'] != eqc_o:
          updated = set_updated_to_True(updated,
                                        start_string,
                                        quiet=not do_solvation)

        # Store rmsd values
        if (self.ctx.args.params['CD']['rmsd'] is not False):
          k = len(self.ctx.data['CD'].protocol) - 1
          for c in range(self.ctx.data['CD'].cycle):
            if not 'rmsd' in self.ctx.data['CD'].Es[k][c].keys():
              confs = [conf for conf in self.ctx.data['CD'].confs['samples'][k][c]]
              self.ctx.data['CD'].Es[k][c]['rmsd'] = self.ctx.get_rmsds(confs)
        self.ctx.stats_RL['rmsd'] = [(np.hstack([self.ctx.data['CD'].Es[k][c]['rmsd']
          if 'rmsd' in self.ctx.data['CD'].Es[k][c].keys() else [] \
            for c in range(self.ctx.stats_RL['equilibrated_cycle'][-1], \
                           self.ctx.data['CD'].cycle)])) \
              for k in range(len(self.ctx.data['CD'].protocol))]

        # Calculate CD free energies that have not already been calculated
        while len(self.ctx.f_RL['grid_MBAR']) < self.ctx.data['CD'].cycle:
          self.ctx.f_RL['grid_MBAR'].append([])
        while len(self.ctx.stats_RL['mean_acc']) < self.ctx.data['CD'].cycle:
          self.ctx.stats_RL['mean_acc'].append([])

        for c in range(self.ctx.data['CD'].cycle):
          # If solvation free energies are not being calculated,
          # only calculate the grid free energy for the current cycle
          if (not do_solvation) and c < (self.ctx.data['CD'].cycle - 1):
            continue
          # Check if this cycle's free energy has already been calculated
          if isinstance(self.ctx.f_RL['grid_MBAR'][c], (list, np.ndarray)) and len(self.ctx.f_RL['grid_MBAR'][c]) > 0:
            continue

          fromCycle = self.ctx.stats_RL['equilibrated_cycle'][c]
          extractCycles = range(fromCycle, c + 1)

          # Extract relevant energies
          CD_Es = [Es[fromCycle:c+1] \
            for Es in self.ctx.data['CD'].Es]

          # Use MBAR for the grid scaling free energy estimate
          (u_kln, N_k) = self.ctx._u_kln(CD_Es, self.ctx.data['CD'].protocol)

          MBAR = self.ctx.run_MBAR(u_kln, N_k)[0]
          self.ctx.f_RL['grid_MBAR'][c] = MBAR
          updated = set_updated_to_True(updated,
                                        start_string,
                                        quiet=not do_solvation)

          self.ctx.log.tee("  calculated grid scaling free energy of %.2f RT "%(\
                      self.ctx.f_RL['grid_MBAR'][c][-1])+\
                   "using cycles %d to %d"%(fromCycle, c))

          # Average acceptance probabilities
          mean_acc = np.zeros(K - 1)
          for k in range(0, K - 1):
            (u_kln, N_k) = self.ctx._u_kln(CD_Es[k:k + 2],
                                       self.ctx.data['CD'].protocol[k:k + 2])
            N = min(N_k)
            acc = np.exp(-u_kln[0, 1, :N] - u_kln[1, 0, :N] + u_kln[0, 0, :N] +
                         u_kln[1, 1, :N])
            mean_acc[k] = np.mean(np.minimum(acc, np.ones(acc.shape)))
          self.ctx.stats_RL['mean_acc'][c] = mean_acc

        if not do_solvation:
          if updated:
            if not self.ctx.log.run_type.startswith('timed'):
              self.ctx.log.tee(write_pkl_gz(f_RL_FN, \
                (self.ctx.f_L, self.ctx.stats_RL, self.ctx.f_RL, self.ctx.B)))
            self.ctx.log.clear_lock('CD')
          return True

        # Make sure postprocessing is complete
        from chimpss.algdock.postprocessing import Postprocessing
        pp_complete = Postprocessing(self.ctx.args, self.ctx.log, self.ctx.top, self.ctx.top_RL, self.ctx.system, self.ctx.data, self.ctx.save).run()
        if not pp_complete:
          return False
        self.ctx.calc_f_L()

        # Make sure all the phase energies are available
        for c in range(self.ctx.data['CD'].cycle):
          for phase in self.ctx.args.params['CD']['phases']:
            for prefix in ['L', 'RL']:
              if not prefix + phase in self.ctx.data['CD'].Es[-1][c].keys():
                self.ctx.log.tee("  postprocessed energies for %s unavailable" % phase)
                return

        # Store stats_RL internal energies for phases
        for phase in self.ctx.args.params['CD']['phases']:
          self.ctx.stats_RL['u_K_'+phase] = \
            [self.ctx.data['CD'].Es[-1][c]['RL'+phase][:,-1]/(R*self.ctx.T_TARGET) \
              for c in range(self.ctx.data['CD'].cycle)]

        # Interaction energies
        for phase in self.ctx.args.params['CD']['phases']:
          if (not 'Psi_' + phase in self.ctx.stats_RL):
            self.ctx.stats_RL['Psi_' + phase] = []
          for c in range(len(self.ctx.stats_RL['Psi_' + phase]),
                         self.ctx.data['CD'].cycle):
            self.ctx.stats_RL['Psi_'+phase].append(
              (self.ctx.data['CD'].Es[-1][c]['RL'+phase][:,-1] - \
               self.ctx.data['CD'].Es[-1][c]['L'+phase][:,-1] - \
               self.ctx.args.original_Es[0][0]['R'+phase][:,-1])/(R*self.ctx.T_TARGET))

        # Predict native pose
        if self.ctx.args.params['CD']['pose'] == -1:
          (self.ctx.stats_RL['pose_inds'], self.ctx.stats_RL['scores']) = \
            self.ctx._get_pose_prediction()

        # BPMF assuming receptor and complex solvation cancel
        self.ctx.B['MMTK_MBAR'] = [-self.ctx.f_L['BC_MBAR'][-1][-1] + \
          self.ctx.f_RL['grid_MBAR'][c][-1] for c in range(len(self.ctx.f_RL['grid_MBAR']))]

        # BPMFs
        for phase in self.ctx.args.params['CD']['phases']:
          for key in [phase + '_solv']:
            if not key in self.ctx.f_RL:
              self.ctx.f_RL[key] = []
          for method in ['min_Psi', 'mean_Psi', 'EXP', 'MBAR']:
            if not phase + '_' + method in self.ctx.B:
              self.ctx.B[phase + '_' + method] = []

          # Receptor solvation
          f_R_solv = self.ctx.args.original_Es[0][0]['R' + phase][:, -1].mean() / (
            R * self.ctx.T_TARGET)

          for c in range(len(self.ctx.B[phase + '_MBAR']), self.ctx.data['CD'].cycle):
            updated = set_updated_to_True(updated, start_string)
            extractCycles = range(self.ctx.stats_RL['equilibrated_cycle'][c], c + 1)

            # From the full grid to the fully bound complex in phase
            u_RL = np.concatenate([\
              self.ctx.data['CD'].Es[-1][c]['RL'+phase][:,-1]/(R*self.ctx.T_TARGET) \
              for c in extractCycles])
            u_sampled = np.concatenate([\
              self.ctx.stats_RL['u_K_sampled'][c] for c in extractCycles])

            du = u_RL - u_sampled
            min_du = min(du)
            weights = np.exp(-du + min_du)

            # Filter outliers
            if self.ctx.args.params['CD']['pose'] > -1:
              toKeep = du > (np.mean(du) - 3 * np.std(du))
              du = du[toKeep]
              weights[~toKeep] = 0.

            weights = weights / sum(weights)

            # Exponential average
            f_RL_solv = -np.log(np.exp(-du + min_du).mean()) + min_du - f_R_solv

            # DEBUG: Print details if extreme
            if abs(f_RL_solv) > 1000:
              print(f"\n  DEBUG: Extreme f_RL_solv = {f_RL_solv:.2f} RT for cycle {c}")
              print(f"    u_RL (postprocessed complex): mean={np.mean(u_RL):.2f}, min/max={u_RL.min():.2f}/{u_RL.max():.2f} RT")
              print(f"    u_sampled (CD sampled): mean={np.mean(u_sampled):.2f}, min/max={u_sampled.min():.2f}/{u_sampled.max():.2f} RT")
              print(f"    du = u_RL - u_sampled: mean={np.mean(du):.2f}, min/max={du.min():.2f}/{du.max():.2f} RT")
              print(f"    min_du = {min_du:.2f} RT")
              print(f"    f_R_solv = {f_R_solv:.2f} RT")
              print(f"    exp(-du+min_du) mean = {np.exp(-du + min_du).mean():.6e}")
              print(f"    -log(exp_mean) = {-np.log(np.exp(-du + min_du).mean()):.2f}")
              print(f"    Final: -log(exp_mean) + min_du - f_R_solv = {f_RL_solv:.2f}\n")

            # Interaction energies
            Psi = np.concatenate([self.ctx.stats_RL['Psi_'+phase][c] \
              for c in extractCycles])
            min_Psi = min(Psi)
            max_Psi = max(Psi)

            # Complex solvation
            self.ctx.f_RL[phase + '_solv'].append(f_RL_solv)

            # Various BPMF estimates
            self.ctx.B[phase + '_min_Psi'].append(min_Psi)
            self.ctx.B[phase + '_mean_Psi'].append(np.sum(weights * Psi))
            # Avoid log(0) by using maximum with small epsilon
            exp_sum = sum(weights*np.exp(Psi-max_Psi))
            self.ctx.B[phase+'_EXP'].append(\
              np.log(np.maximum(exp_sum, 1e-300)) + max_Psi)

            # Calculate BPMF with component breakdown
            phase_solv = self.ctx.f_L[phase+'_solv'][-1]
            bc_mbar = self.ctx.f_L['BC_MBAR'][-1][-1]
            grid_mbar = self.ctx.f_RL['grid_MBAR'][-1][-1]

            bpmf = - phase_solv - bc_mbar + grid_mbar + f_RL_solv
            self.ctx.B[phase+'_MBAR'].append(bpmf)

            # Debug: print components if BPMF is extreme
            bpmf_value = float(bpmf.item() if hasattr(bpmf, 'item') else bpmf)
            if abs(bpmf_value) > 1000:
              # Convert all components to scalars for printing
              phase_solv_val = float(phase_solv.item() if hasattr(phase_solv, 'item') else phase_solv)
              bc_mbar_val = float(bc_mbar.item() if hasattr(bc_mbar, 'item') else bc_mbar)
              grid_mbar_val = float(grid_mbar.item() if hasattr(grid_mbar, 'item') else grid_mbar)
              f_RL_solv_val = float(f_RL_solv.item() if hasattr(f_RL_solv, 'item') else f_RL_solv)

              print(f"\n  WARNING: Extreme BPMF value ({bpmf_value:.2f} RT) for {phase}, cycle {c}")
              print(f"    Components:")
              print(f"      -f_L[{phase}_solv] = -{phase_solv_val:.2f}")
              print(f"      -f_L[BC_MBAR]     = -{bc_mbar_val:.2f}")
              print(f"      +f_RL[grid_MBAR]  = +{grid_mbar_val:.2f}")
              print(f"      +f_RL_solv        = +{f_RL_solv_val:.2f}")
              print(f"      TOTAL             = {bpmf_value:.2f} RT\n")

            self.ctx.log.tee("  calculated %s binding PMF of %.5g RT with cycles %d to %d"%(\
              phase, bpmf_value, \
              self.ctx.stats_RL['equilibrated_cycle'][c], c))

        if updated:
          self.ctx.log.tee(
            write_pkl_gz(f_RL_FN, (self.ctx.f_L, self.ctx.stats_RL, self.ctx.f_RL, self.ctx.B)))
          self.ctx.log.tee("\nElapsed time for binding PMF estimation: " + \
            HMStime(self.ctx.log.timeSince('BPMF')))
        self.ctx.log.clear_lock('CD')
