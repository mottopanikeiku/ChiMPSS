"""BC thermodynamic path free energy calculator

Calculates free energies for the BC replica exchange ladder.

BC warms the unbound ligand from target temperature (e.g., 300K, state B)
to high temperature (e.g., 600K, state C). Alpha (1.0 to 0.0) controls:
  - Temperature warming: T_TARGET to T_HIGH
  - Solvation scaling (mode-dependent): OBC may scale from 1.0 to 0.0

Purpose: Enhanced sampling and optional desolvation for ligand free energy.
"""

import os
import time
import numpy as np
from chimpss.algdock.IO import load_pkl_gz, write_pkl_gz, HMStime

# Physical constant
R = 8.314472 / 1000.0  # Gas constant in kJ/(mol*K)


class BCCalculator:
    """Calculates BC (B→C state) free energies

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

    def initial_BC(self):
        """
        Warms the ligand from self.T_TARGET to self.T_HIGH

        Intermediate thermodynamic states are chosen such that
        thermodynamic length intervals are approximately constant.
        Configurations from each state are subsampled to seed the next simulation.

        Extracted from BindingPMF.initial_BC (lines 515-540)
        """

        if (len(self.ctx.data['BC'].protocol) >
            0) and (self.ctx.data['BC'].protocol[-1]['crossed']):
          return  # Initial BC is already complete

        self.ctx.log.recordStart('BC')

        from chimpss.algdock.ligand_preparation import LigandPreparation
        seeds = LigandPreparation(self.ctx.args, self.ctx.log, self.ctx.top,
                                  self.ctx.system, self.ctx._get_confs_to_rescore,
                                  self.ctx.iterator, self.ctx.data).run('BC')


        from chimpss.algdock.initialization import Initialization
        Initialization(self.ctx.args, self.ctx.log, self.ctx.top, self.ctx.system,
                      self.ctx.iterator, self.ctx.data, self.ctx.save,
                      self.ctx._u_kln).run('BC', seeds)

        return True

    def calc_f_L(self, readOnly=False, do_solvation=True, redo=False):
        """
        Calculates ligand-specific free energies:
        1. Reduced free energy for BC path (cooling ligand from T_HIGH to T_TARGET)
        2. Solvation free energy of the ligand using single-step free energy perturbation

        Parameters
        ----------
        readOnly : bool
            If True, only read existing data without recalculation
        do_solvation : bool
            If True, calculate solvation free energy
        redo : bool
            Debugging option (currently unused)
        """
        # Initialize variables as empty lists or by loading data
        f_L_FN = os.path.join(self.ctx.args.dir['BC'], 'f_L.pkl.gz')
        dat = load_pkl_gz(f_L_FN)
        if dat is not None:
          (self.ctx.stats_L, self.ctx.f_L) = dat
        else:
          self.ctx.stats_L = dict(\
            [(item,[]) for item in ['equilibrated_cycle','mean_acc']])
          self.ctx.stats_L['protocol'] = self.ctx.data['BC'].protocol
          self.ctx.f_L = dict([(key,[]) for key in ['BC_MBAR'] + \
            [phase+'_solv' for phase in self.ctx.args.params['BC']['phases']]])
        if readOnly or self.ctx.data['BC'].protocol == []:
          return

        K = len(self.ctx.data['BC'].protocol)

        # Make sure all the energies are available
        for c in range(self.ctx.data['BC'].cycle):
          if len(self.ctx.data['BC'].Es[-1][c].keys()) == 0:
            self.ctx.log.tee("  skipping the BC free energy calculation")
            return

        start_string = "\n>>> Ligand free energy calculations, starting at " + \
          time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "\n"
        self.ctx.log.recordStart('free energy')

        # Store stats_L internal energies
        self.ctx.stats_L['u_K_sampled'] = \
          [self.ctx._u_kln([self.ctx.data['BC'].Es[-1][c]],[self.ctx.data['BC'].protocol[-1]]) \
            for c in range(self.ctx.data['BC'].cycle)]
        self.ctx.stats_L['u_KK'] = \
          [np.sum([self.ctx._u_kln([self.ctx.data['BC'].Es[k][c]],[self.ctx.data['BC'].protocol[k]]) \
            for k in range(len(self.ctx.data['BC'].protocol))],0) \
              for c in range(self.ctx.data['BC'].cycle)]

        self.ctx.stats_L['equilibrated_cycle'] = self.ctx._get_equilibrated_cycle('BC')

        # Calculate BC free energies that have not already been calculated,
        # in units of RT
        updated = False
        for c in range(len(self.ctx.f_L['BC_MBAR']), self.ctx.data['BC'].cycle):
          if not updated:
            self.ctx.log.set_lock('BC')
            if do_solvation:
              self.ctx.log.tee(start_string)
            updated = True

          fromCycle = self.ctx.stats_L['equilibrated_cycle'][c]
          toCycle = c + 1

          # BC free energy
          BC_Es = []
          for BC_Es_state in self.ctx.data['BC'].Es:
            BC_Es.append(BC_Es_state[fromCycle:toCycle])
          (u_kln, N_k) = self.ctx._u_kln(BC_Es, self.ctx.data['BC'].protocol)
          MBAR = self.ctx.run_MBAR(u_kln, N_k)[0]
          self.ctx.f_L['BC_MBAR'].append(MBAR)

          # Average acceptance probabilities
          BC_mean_acc = np.zeros(K - 1)
          for k in range(0, K - 1):
            (u_kln, N_k) = self.ctx._u_kln(BC_Es[k:k + 2],
                                           self.ctx.data['BC'].protocol[k:k + 2])
            N = min(N_k)
            acc = np.exp(-u_kln[0, 1, :N] - u_kln[1, 0, :N] + u_kln[0, 0, :N] +
                         u_kln[1, 1, :N])
            BC_mean_acc[k] = np.mean(np.minimum(acc, np.ones(acc.shape)))
          self.ctx.stats_L['mean_acc'].append(BC_mean_acc)

          self.ctx.log.tee("  calculated BC free energy of %.2f RT "%(\
                      self.ctx.f_L['BC_MBAR'][-1][-1])+\
                   "using cycles %d to %d"%(fromCycle, c))

        if not do_solvation:
          if updated:
            if not self.ctx.log.run_type.startswith('timed'):
              write_pkl_gz(f_L_FN, (self.ctx.stats_L, self.ctx.f_L))
            self.ctx.log.clear_lock('BC')
          return True

        # Make sure postprocessing is complete
        from chimpss.algdock.postprocessing import Postprocessing
        pp_complete = Postprocessing(self.ctx.args, self.ctx.log, self.ctx.top,
                                     self.ctx.top_RL, self.ctx.system, self.ctx.data,
                                     self.ctx.save).run([('BC', -1, -1, 'L')])
        if not pp_complete:
          return False

        # Store stats_L internal energies
        for phase in self.ctx.args.params['BC']['phases']:
          self.ctx.stats_L['u_K_'+phase] = \
            [self.ctx.data['BC'].Es[-1][c]['L'+phase][:,-1]/(R*self.ctx.T_TARGET) \
              for c in range(self.ctx.data['BC'].cycle)]

        # Calculate solvation free energies that have not already been calculated,
        # in units of RT
        for phase in self.ctx.args.params['BC']['phases']:
          if not phase + '_solv' in self.ctx.f_L:
            self.ctx.f_L[phase + '_solv'] = []
          if not 'mean_' + phase in self.ctx.f_L:
            self.ctx.f_L['mean_' + phase] = []

          for c in range(len(self.ctx.f_L[phase + '_solv']), self.ctx.data['BC'].cycle):
            if not updated:
              self.ctx.log.set_lock('BC')
              self.ctx.log.tee(start_string)
              updated = True

            fromCycle = self.ctx.stats_L['equilibrated_cycle'][c]
            toCycle = c + 1

            if not ('L' + phase) in self.ctx.data['BC'].Es[-1][c].keys():
              raise Exception('L%s energies not found in cycle %d' % (phase, c))

            # Arbitrarily, solvation is the
            # 'forward' direction and desolvation the 'reverse'
            u_L = np.concatenate([self.ctx.data['BC'].Es[-1][n]['L'+phase] \
              for n in range(fromCycle,toCycle)])/(R*self.ctx.T_TARGET)

            # u_sampled is the energy from BC sampling using protocol[-1]
            # This includes MM + OBC (since protocol[-1] has OBC=1.0)
            # We reconstruct this using _u_kln to match the old working code
            u_sampled = np.concatenate([
                self.ctx._u_kln([self.ctx.data['BC'].Es[-1][c]],
                                [self.ctx.data['BC'].protocol[-1]])
                for c in range(fromCycle, toCycle)
            ])

            # DEBUG: Print actual values (disabled - too verbose for tests)
            DEBUG_CALC_F_L = True
            if DEBUG_CALC_F_L:
                self.ctx.log.tee(f"\n{'='*80}")
                self.ctx.log.tee(f"BC protocol[-1] keys: {self.ctx.data['BC'].protocol[-1].keys()}")
                self.ctx.log.tee(f"BC Es[-1][{fromCycle}] keys: {self.ctx.data['BC'].Es[-1][fromCycle].keys()}")
                self.ctx.log.tee(f"DEBUG calc_f_L solvation FE for {phase}, cycles {fromCycle} to {toCycle-1}")
                self.ctx.log.tee(f"{'='*80}")
                self.ctx.log.tee(f"u_L shape: {u_L.shape}")
                self.ctx.log.tee(f"u_sampled shape: {u_sampled.shape}")
                self.ctx.log.tee(f"u_L[:, 0] (MM energy from {phase}):")
                self.ctx.log.tee(f"  First 5: {u_L[:5, 0]}")
                self.ctx.log.tee(f"  Mean: {np.mean(u_L[:, 0]):.2f} RT")
                self.ctx.log.tee(f"  Min/Max: {np.min(u_L[:, 0]):.2f} / {np.max(u_L[:, 0]):.2f} RT")
                self.ctx.log.tee(f"u_L[:, -1] (total energy from {phase}):")
                self.ctx.log.tee(f"  First 5: {u_L[:5, -1]}")
                self.ctx.log.tee(f"  Mean: {np.mean(u_L[:, -1]):.2f} RT")
                self.ctx.log.tee(f"  Min/Max: {np.min(u_L[:, -1]):.2f} / {np.max(u_L[:, -1]):.2f} RT")
                self.ctx.log.tee(f"u_sampled (using MM column):")
                self.ctx.log.tee(f"  First 5: {u_sampled[:5]}")
                self.ctx.log.tee(f"  Mean: {np.mean(u_sampled):.2f} RT")
                self.ctx.log.tee(f"  Min/Max: {np.min(u_sampled):.2f} / {np.max(u_sampled):.2f} RT")

            du_F = (u_L[:, -1] - u_sampled)
            if DEBUG_CALC_F_L:
                self.ctx.log.tee(f"du_F = u_L[:,-1] - u_sampled:")
                self.ctx.log.tee(f"  First 5: {du_F[:5]}")
                self.ctx.log.tee(f"  Mean: {np.mean(du_F):.2f} RT")
                self.ctx.log.tee(f"  Min/Max: {np.min(du_F):.2f} / {np.max(du_F):.2f} RT")

            min_du_F = min(du_F)
            w_L = np.exp(-du_F + min_du_F)
            if DEBUG_CALC_F_L:
                self.ctx.log.tee(f"Exponential weights:")
                self.ctx.log.tee(f"  w_L mean: {np.mean(w_L):.6e}")
                self.ctx.log.tee(f"  w_L min/max: {np.min(w_L):.6e} / {np.max(w_L):.6e}")

            f_L_solv = -np.log(np.mean(w_L)) + min_du_F
            mean_u_phase = np.sum(u_L[:, -1] * w_L) / np.sum(w_L)
            if DEBUG_CALC_F_L:
                self.ctx.log.tee(f"Result: f_L_solv = {f_L_solv:.2f} RT")
                self.ctx.log.tee(f"{'='*80}\n")

            self.ctx.f_L[phase + '_solv'].append(f_L_solv)
            self.ctx.f_L['mean_' + phase].append(mean_u_phase)
            self.ctx.log.tee("  calculated " + phase + " solvation free energy of " + \
                     "%.5g RT "%(f_L_solv) + \
                     "using cycles %d to %d"%(fromCycle, toCycle-1))

        if updated:
          self.ctx.log.tee(write_pkl_gz(f_L_FN, (self.ctx.stats_L, self.ctx.f_L)))
          self.ctx.log.tee("\nElapsed time for free energy calculation: " + \
            HMStime(self.ctx.log.timeSince('free energy')))
          self.ctx.log.clear_lock('BC')
        return True
