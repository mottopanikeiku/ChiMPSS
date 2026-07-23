"""Clustering and pose prediction algorithms

These methods handle RMSD calculation, hierarchical clustering,
and pose prediction from CD sampling data.

Extracted from BindingPMF.py - these methods require access to
simulation state via a context object.
"""

import numpy as np
import scipy.cluster.hierarchy


class PoseAnalyzer:
    """Analyzes binding poses through clustering and scoring

    Instance methods that require access to simulation data and state.
    """

    def __init__(self, context):
        """Initialize with BPMF context

        Parameters
        ----------
        context : BPMF instance
            The BindingPMF instance with access to:
            - self.data (simulation data)
            - self.stats_RL (CD statistics)
            - self.get_rmsds (RMSD calculator)
            - self.args (simulation arguments)
        """
        self.ctx = context

    def get_rmsd_matrix(self):
        """Calculate pairwise RMSD matrix for equilibrated CD samples

        Returns
        -------
        np.array
            Condensed distance matrix (1D array) of pairwise RMSDs

        Extracted from BindingPMF._get_rmsd_matrix (lines 1108-1151)
        """
        process = 'CD'
        equilibrated_cycle = self.ctx.stats_RL['equilibrated_cycle'][-1]

        # Gather snapshots
        for k in range(equilibrated_cycle, self.ctx.data[process].cycle):
          if not isinstance(self.ctx.data[process].confs['samples'][-1][k], list):
            self.ctx.data[process].confs['samples'][-1][k] = [
              self.ctx.data[process].confs['samples'][-1][k]
            ]
        import itertools
        confs = np.array([conf for conf in itertools.chain.from_iterable(\
          [self.ctx.data[process].confs['samples'][-1][c] \
            for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])])

        cum_Nk = np.cumsum([0] + [len(self.ctx.data['CD'].confs['samples'][-1][c]) \
          for c in range(self.ctx.data['CD'].cycle)])
        nsamples = cum_Nk[-1]

        # Obtain a full rmsd matrix
        # TODO: Check this
        if ('rmsd_matrix' in self.ctx.stats_RL.keys()) and \
            (len(self.ctx.stats_RL['rmsd_matrix'])==(nsamples*(nsamples-1)/2)):
          rmsd_matrix = self.ctx.stats_RL['rmsd_matrix']
        else:
          # Create a new matrix
          rmsd_matrix = []
          for c in range(len(confs)):
            rmsd_matrix.extend(self.ctx.get_rmsds(confs[c + 1:], confs[c]))
          rmsd_matrix = np.clip(rmsd_matrix, 0., None)
          self.ctx.stats_RL['rmsd_matrix'] = rmsd_matrix

        # TODO: Write code to extend previous matrix
        # Extend a previous matrix
        # rmsd_matrix = self.ctx.stats_RL['rmsd_matrix']
        # from scipy.spatial.distance import squareform
        # rmsd_matrix_sq = squareform(rmsd_matrix)
        #
        # for c in range(len(confs)):
        #   rmsd_matrix.extend(self.ctx.get_rmsds(confs[c+1:], confs[c]))
        # rmsd_matrix = np.clip(rmsd_matrix, 0., None)
        # self.ctx.stats_RL['rmsd_matrix'] = rmsd_matrix

        return rmsd_matrix

    @staticmethod
    def cluster_samples(rmsd_matrix):
        """Perform hierarchical clustering on RMSD matrix

        Parameters
        ----------
        rmsd_matrix : np.array
            Condensed distance matrix from get_rmsd_matrix()

        Returns
        -------
        list
            Cluster assignments for each sample (reindexed by order of appearance)

        Extracted from BindingPMF._cluster_samples (lines 1153-1168)
        """
        # Clustering
        Z = scipy.cluster.hierarchy.linkage(rmsd_matrix, method='complete')
        assignments = np.array(\
          scipy.cluster.hierarchy.fcluster(Z, 0.1, criterion='distance'))

        # Reindexes the assignments in order of appearance
        new_index = 0
        mapping_to_new_index = {}
        for assignment in assignments:
          if not assignment in mapping_to_new_index.keys():
            mapping_to_new_index[assignment] = new_index
            new_index += 1
        assignments = [mapping_to_new_index[a] for a in assignments]
        return assignments

    def get_pose_prediction(self, representative='medoid'):
        """Predict binding poses using clustering and scoring

        Parameters
        ----------
        representative : str
            Method for selecting cluster representative:
            - 'medoid': configuration with minimum mean RMSD to cluster members
            - phase name (e.g. 'OpenMM_Gas'): minimum interaction energy in that phase

        Returns
        -------
        tuple
            (pose_inds, scores) where:
            - pose_inds: list of (cycle, n) tuples for representative configurations
            - scores: dict of scoring metrics for each cluster

        Extracted from BindingPMF._get_pose_prediction (lines 1170-1281)
        """
        process = 'CD'
        equilibrated_cycle = self.ctx.stats_RL['equilibrated_cycle'][-1]
        stats = self.ctx.stats_RL

        rmsd_matrix = self.get_rmsd_matrix()
        assignments = self.cluster_samples(rmsd_matrix)

        cum_Nk = np.cumsum([0] + [len(self.ctx.data[process].confs['samples'][-1][c]) \
          for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])

        def linear_index_to_pair(ind):
          cycle = list(ind < cum_Nk).index(True) - 1
          n = ind - cum_Nk[cycle]
          return (cycle + equilibrated_cycle, n)

        # Select a representative of each cluster
        pose_inds = []
        scores = {}

        if representative == 'medoid':
          # based on the medoid
          from scipy.spatial.distance import squareform
          rmsd_matrix_sq = squareform(rmsd_matrix)
          for n in range(max(assignments) + 1):
            inds = [i for i in range(len(assignments)) if assignments[i] == n]
            rmsd_matrix_n = rmsd_matrix_sq[inds][:, inds]
            (cycle,
             n) = linear_index_to_pair(inds[np.argmin(np.mean(rmsd_matrix_n, 0))])
            pose_inds.append((cycle, n))
        else:
          if 'Psi_' + representative in stats.keys():
            # based on the lowest interaction energy in specified phase
            phase = representative
            Psi_n = np.concatenate([stats['Psi_'+phase][c] \
                      for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])
            for n in range(max(assignments) + 1):
              inds = [i for i in range(len(assignments)) if assignments[i] == n]
              (cycle, n) = linear_index_to_pair(inds[np.argmin(Psi_n[inds])])
              pose_inds.append((cycle, n))

        # If relevant, store the rmsd of the representatives
        if self.ctx.args.params['CD']['rmsd']:
          scores['rmsd'] = []
          for (cycle, n) in pose_inds:
            scores['rmsd'].append(self.ctx.data['CD'].Es[-1][cycle]['rmsd'][n])

        # Score clusters based on total energy
        uo = np.concatenate([stats['u_K_sampled'][c] \
          for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])
        for phase in (['grid'] + self.ctx.args.params[process]['phases']):
          if phase != 'grid':
            un = np.concatenate([stats['u_K_'+phase][c] \
              for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])
            du = un - uo
            min_du = min(du)
            weights = np.exp(-du + min_du)
          else:
            un = uo
            weights = np.ones(len(assignments))
          cluster_counts = np.histogram(assignments, \
            bins=np.arange(len(set(assignments))+1)-0.5,
            weights=weights)[0]
          # by free energy
          # Avoid log(0) by adding small epsilon to zero counts
          cluster_fe = -np.log(np.maximum(cluster_counts, 1e-300))
          cluster_fe -= np.min(cluster_fe)
          scores[phase + '_fe_u'] = cluster_fe
          # by minimum and mean energy
          scores[phase + '_min_u'] = []
          scores[phase + '_mean_u'] = []
          for n in range(max(assignments) + 1):
            un_n = [un[i] for i in range(len(assignments)) if assignments[i] == n]
            scores[phase + '_min_u'].append(np.min(un_n))
            scores[phase + '_mean_u'].append(np.mean(un_n))

        if process == 'CD':
          # Score clusters based on interaction energy
          Psi_o = np.concatenate([stats['Psi_grid'][c] \
            for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])
          for phase in (['grid'] + self.ctx.args.params[process]['phases']):
            if phase != 'grid':
              Psi_n = np.concatenate([stats['Psi_'+phase][c] \
                for c in range(equilibrated_cycle,self.ctx.data[process].cycle)])
              dPsi = Psi_n - Psi_o
              min_dPsi = min(dPsi)
              weights = np.exp(-dPsi + min_dPsi)
            else:
              Psi_n = Psi_o
              weights = np.ones(len(assignments))
            cluster_counts = np.histogram(assignments, \
              bins=np.arange(len(set(assignments))+1)-0.5,
              weights=weights)[0]
            # by free energy
            # Avoid log(0) by adding small epsilon to zero counts
            cluster_fe = -np.log(np.maximum(cluster_counts, 1e-300))
            cluster_fe -= np.min(cluster_fe)
            scores[phase + '_fe_Psi'] = cluster_fe
            # by minimum and mean energy
            scores[phase + '_min_Psi'] = []
            scores[phase + '_mean_Psi'] = []
            for n in range(max(assignments) + 1):
              Psi_n_n = [
                Psi_n[i] for i in range(len(assignments)) if assignments[i] == n
              ]
              scores[phase + '_min_Psi'].append(np.min(Psi_n_n))
              scores[phase + '_mean_Psi'].append(np.mean(Psi_n_n))

        for key in scores.keys():
          scores[key] = np.array(scores[key])

        return (pose_inds, scores)
