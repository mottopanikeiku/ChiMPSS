"""CD State Management

Handles thermodynamic state insertion logic for the C→D process.
"""

import os
import numpy as np
from chimpss.algdock.IO import write_pkl_gz

# Physical constants used by state manager
scalables = ['OBC', 'sLJr', 'sELE', 'LJr', 'LJa', 'ELE']


class CDStateManager:
    """Manages CD thermodynamic state insertion and manipulation

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

    def clear_f_RL(self):
        """Clear and reinitialize CD free energy storage structures

        Initializes empty data structures for:
        - stats_RL: Statistical data (internal energies, acceptance rates, etc.)
        - f_RL: Free energy components
        - B: Binding PMF estimates
        """
        # stats_RL will include internal energies, interaction energies,
        # the cycle by which the bound state is equilibrated,
        # the mean acceptance probability between replica exchange neighbors,
        # and the rmsd, if applicable
        phase_f_RL_keys = \
          [phase+'_solv' for phase in self.ctx.args.params['CD']['phases']]

        # Initialize variables as empty lists
        stats_RL = [('u_K_'+FF,[]) \
          for FF in ['ligand','sampled']+self.ctx.args.params['CD']['phases']]
        stats_RL += [('Psi_'+FF,[]) \
          for FF in ['grid']+self.ctx.args.params['CD']['phases']]
        stats_RL += [(item,[]) \
          for item in ['equilibrated_cycle','cum_Nclusters','mean_acc','rmsd']]
        self.ctx.stats_RL = dict(stats_RL)
        self.ctx.stats_RL['protocol'] = self.ctx.data['CD'].protocol
        # Free energy components
        self.ctx.f_RL = dict([(key,[]) \
          for key in ['grid_MBAR'] + phase_f_RL_keys])
        # Binding PMF estimates
        self.ctx.B = {'MMTK_MBAR': []}
        for phase in self.ctx.args.params['CD']['phases']:
          for method in ['min_Psi', 'mean_Psi', 'EXP', 'MBAR']:
            self.ctx.B[phase + '_' + method] = []

        # Store empty list
        if self.ctx.args.params['CD']['pose'] == -1:
          f_RL_FN = os.path.join(self.ctx.args.dir['CD'], 'f_RL.pkl.gz')
        else:
          f_RL_FN = os.path.join(self.ctx.args.dir['CD'], \
            'f_RL_pose%03d.pkl.gz'%self.ctx.args.params['CD']['pose'])
        if hasattr(self.ctx, 'run_type') and (not self.ctx.log.run_type.startswith('timed')):
          self.ctx.log.tee(
            write_pkl_gz(f_RL_FN, (self.ctx.f_L, self.ctx.stats_RL, self.ctx.f_RL, self.ctx.B)))

    def insert_state(self, alpha, clear=True):
        """
        Inserts a new thermodynamic state into the CD protocol.
        Samples for previous cycles are added by sampling importance resampling.
        Clears grid_MBAR.

        Parameters
        ----------
        alpha : float
            The coupling parameter for the new state
        clear : bool
            Whether to clear f_RL after insertion (default True)
        """
        # Defines a new thermodynamic state based on the neighboring state
        neighbor_ind = [alpha < p['alpha']
                        for p in self.ctx.data['CD'].protocol].index(True) - 1
        params_n = self.ctx.system.paramsFromAlpha(
          alpha, params_o=self.ctx.data['CD'].protocol[neighbor_ind])

        # For sampling importance resampling,
        # prepare an augmented matrix for pymbar calculations
        # with a new thermodynamic state
        (u_kln_s, N_k) = self.ctx._u_kln(self.ctx.data['CD'].Es, self.ctx.data['CD'].protocol)
        (K, L, N) = u_kln_s.shape

        u_kln_n = self.ctx._u_kln(self.ctx.data['CD'].Es, [params_n])[0]
        L += 1
        N_k = np.append(N_k, [0])

        u_kln = np.zeros([K, L, N])
        u_kln[:, :-1, :] = u_kln_s
        for k in range(K):
          u_kln[k, -1, :] = u_kln_n[k, 0, :]

        # Determine SIR weights
        weights = self.ctx.run_MBAR(u_kln, N_k, augmented=True)[1][:, -1]
        weights = weights / sum(weights)

        # Resampling
        # Convert linear indices to 3 indicies: state, cycle, and snapshot
        cum_N_state = np.cumsum([0] + list(N_k))
        cum_N_cycle = [np.cumsum([0] + [self.ctx.data['CD'].Es[k][c]['MM'].shape[0] \
          for c in range(len(self.ctx.data['CD'].Es[k]))]) for k in range(len(self.ctx.data['CD'].Es))]

        def linear_index_to_snapshot_index(ind):
          state_index = list(ind < cum_N_state).index(True) - 1
          nis_index = ind - cum_N_state[state_index]
          cycle_index = list(nis_index < cum_N_cycle[state_index]).index(True) - 1
          nic_index = nis_index - cum_N_cycle[state_index][cycle_index]
          return (state_index, cycle_index, nic_index)

        def snapshot_index_to_linear_index(state_index, cycle_index, nic_index):
          return cum_N_state[state_index] + cum_N_cycle[state_index][
            cycle_index] + nic_index

        # Terms to copy
        if self.ctx.args.params['CD']['pose'] > -1:
          # Pose BPMF
          terms = ['MM',\
            'k_angular_ext','k_spatial_ext','k_angular_int'] + scalables
        else:
          # BPMF
          terms = ['MM', 'site'] + scalables

        CD_Es_s = []
        confs_s = []
        for c in range(len(self.ctx.data['CD'].Es[0])):
          CD_Es_c = dict([(term, []) for term in terms])
          confs_c = []
          for n_in_c in range(len(self.ctx.data['CD'].Es[-1][c]['MM'])):
            if (cum_N_cycle[-1][c] == 0):
              (snapshot_s,snapshot_c,snapshot_n) = linear_index_to_snapshot_index(\
               np.random.choice(range(len(weights)), size = 1, p = weights)[0])
            else:
              snapshot_c = np.inf
              while (snapshot_c > c):
                (snapshot_s,snapshot_c,snapshot_n) = linear_index_to_snapshot_index(\
                 np.random.choice(range(len(weights)), size = 1, p = weights)[0])
            for term in terms:
              CD_Es_c[term].append(\
                np.copy(self.ctx.data['CD'].Es[snapshot_s][snapshot_c][term][snapshot_n]))
            if self.ctx.args.params['CD']['keep_intermediate']:
              # Has not been tested:
              confs_c.append(\
                np.copy(self.ctx.data['CD'].confs['samples'][snapshot_s][snapshot_c]))
          for term in terms:
            CD_Es_c[term] = np.array(CD_Es_c[term])
          CD_Es_s.append(CD_Es_c)
          confs_s.append(confs_c)

        # Insert resampled values
        self.ctx.data['CD'].protocol.insert(neighbor_ind + 1, params_n)
        self.ctx.data['CD'].Es.insert(neighbor_ind + 1, CD_Es_s)
        self.ctx.data['CD'].confs['samples'].insert(neighbor_ind + 1, confs_s)
        self.ctx.data['CD'].confs['replicas'].insert(neighbor_ind+1, \
          np.copy(self.ctx.data['CD'].confs['replicas'][neighbor_ind]))

        if clear:
          self.clear_f_RL()

    def insert_states_between_low_acc(self):
        """Insert thermodynamic states between those with low acceptance probabilities

        Iterates through CD protocol and inserts new states between pairs
        with acceptance probability < 0.4, repeating until all pairs are above threshold.
        """
        # Insert thermodynamic states between those with low acceptance probabilities
        eq_c = self.ctx._get_equilibrated_cycle('CD')[-1]

        def calc_mean_acc(k):
          CD_Es = [Es[eq_c:self.ctx.data['CD'].cycle] for Es in self.ctx.data['CD'].Es]
          (u_kln,N_k) = self.ctx._u_kln(CD_Es[k:k+2],\
                                    self.ctx.data['CD'].protocol[k:k+2])
          N = min(N_k)
          acc = np.exp(-u_kln[0, 1, :N] - u_kln[1, 0, :N] + u_kln[0, 0, :N] +
                       u_kln[1, 1, :N])
          return np.mean(np.minimum(acc, np.ones(acc.shape)))

        updated = False
        k = 0
        while k < len(self.ctx.data['CD'].protocol) - 1:
          mean_acc = calc_mean_acc(k)
          # print k, self.ctx.data['CD'].protocol[k]['alpha'], self.ctx.data['CD'].protocol[k+1]['alpha'], mean_acc
          while mean_acc < 0.4:
            if not updated:
              updated = True
              self.ctx.log.set_lock('CD')
            alpha_k = self.ctx.data['CD'].protocol[k]['alpha']
            alpha_kp = self.ctx.data['CD'].protocol[k + 1]['alpha']
            alpha_n = (alpha_k + alpha_kp) / 2.
            report = '  inserted state'
            report += ' between %.5g and %.5g at %.5g\n' % (alpha_k, alpha_kp, alpha_n)
            report += '  to improve acceptance rate from %.5g ' % mean_acc
            self.insert_state(alpha_n, clear=False)
            mean_acc = calc_mean_acc(k)
            report += 'to %.5g' % mean_acc
            # print k, self.ctx.data['CD'].protocol[k]['alpha'], self.ctx.data['CD'].protocol[k+1]['alpha'], mean_acc
            self.ctx.log.tee(report)
          k += 1
        if updated:
          self.clear_f_RL()
          self.ctx.save('CD')
          self.ctx.log.tee("")
          self.ctx.log.clear_lock('CD')
