"""Core free energy calculation algorithms

These are pure computational methods extracted from BindingPMF.py
with no state dependencies - safe to extract first.

All methods are static - they operate only on input/output data
with no side effects.
"""

import numpy as np
import pymbar
from concurrent.futures import ThreadPoolExecutor
from chimpss.algdock.parallel_config import get_optimal_worker_count

# Constants
R = 8.3144621e-3  # kJ/mol/K (gas constant for OpenMM)
scalables = ['OBC', 'sLJr', 'sELE', 'LJr', 'LJa', 'ELE']


class FreeEnergyCalculator:
    """Pure free energy calculation methods

    All methods are static - no state dependencies.
    Input/output only, no side effects.
    """

    @staticmethod
    def _compute_u_kl_for_state_k(k, eTs_k, protocol, scalables, addMM, addSite, noBeta, R):
        """
        Compute u_kl (energies at all l states) for a single sampled state k.

        This is the core computation extracted for parallelization.

        Parameters
        ----------
        k : int
            Sampled state index
        eTs_k : list
            Energy terms for state k: list (over cycles) of dicts (of energy arrays)
        protocol : list
            Thermodynamic state parameters
        scalables : list
            List of scalable energy terms
        addMM : bool
            Whether to include MM energy
        addSite : bool
            Whether to include site energy
        noBeta : bool
            If True, don't divide by RT
        R : float
            Gas constant

        Returns
        -------
        tuple : (k, u_kl, N_k)
            k : state index
            u_kl : np.array of shape [L, N_k] - energies for this k at all l states
            N_k : int - number of samples for this k
        """
        L = len(protocol)
        C = len(eTs_k)

        # Count samples
        probe_keys = ['MM','k_angular_ext','k_spatial_ext','k_angular_int'] + scalables
        probe_key = [key for key in eTs_k[0].keys() if key in probe_keys][0]
        N_k = sum(len(eTs_k[c][probe_key]) for c in range(C))

        # Build base energy
        E_base = 0.0
        if addMM:
            E_base += np.concatenate([eTs_k[c]['MM'] for c in range(C)])
        if addSite:
            E_base += np.concatenate([eTs_k[c]['site'] for c in range(C)])

        # Allocate output for this k
        u_kl = np.zeros([L, N_k], dtype=np.float64)

        # Compute energy at each evaluation state l
        for l in range(L):
            E = 1. * E_base
            for scalable in scalables:
                if scalable in protocol[l].keys():
                    full_strength_term = np.concatenate([eTs_k[c][scalable] for c in range(C)])
                    E += protocol[l][scalable] * full_strength_term

            for key in ['k_angular_ext', 'k_spatial_ext', 'k_angular_int']:
                if key in protocol[l].keys():
                    E += protocol[l][key] * np.concatenate([eTs_k[c][key] for c in range(C)])

            if noBeta:
                u_kl[l, :] = E
            else:
                u_kl[l, :] = E / (R * protocol[l]['T'])

        return (k, u_kl, N_k)

    @staticmethod
    def u_kln(eTs, protocol, noBeta=False):
        """
        Computes a reduced potential energy matrix.
        
        k is the sampled state, l is the state for which energies are evaluated.

        Extracted from BindingPMF._u_kln (lines 2213-2373)

        Parameters
        ----------
        eTs : dict or list
            Energy terms dictionary/list structure:
            - dict (of mapped energy terms) of numpy arrays (over states), or
            - list (over states) of dictionaries (of mapped energy terms) of numpy arrays (over configurations), or
            - list (over states) of lists (over cycles) of dictionaries (of mapped energy terms) of numpy arrays (over configurations)
        protocol : list
            List of thermodynamic states (dicts with keys like 'T', 'MM', 'site', scalables, etc.)
        noBeta : bool
            If True, energy will not be divided by RT

        Returns
        -------
        u_kln : np.array or tuple
            If (K==1 and L==1): returns flattened array
            Otherwise: returns (u_kln, N_k) where:
                u_kln : reduced potential energy matrix [K x L x N_max]
                N_k : array of sample sizes [K]
        """
        L = len(protocol)

        addMM = ('MM' in protocol[0].keys()) and (protocol[0]['MM'])
        addSite = ('site' in protocol[0].keys()) and (protocol[0]['site'])
        probe_keys = ['MM','k_angular_ext','k_spatial_ext','k_angular_int'] + \
          scalables
        probe_key = [key for key in protocol[0].keys() if key in probe_keys][0]

        if isinstance(eTs, dict):
          # There is one configuration per state
          K = len(eTs[probe_key])
          N_k = np.ones(K, dtype=int)
          u_kln = []
          E_base = np.zeros(K)
          if addMM:
            E_base += eTs['MM']
          if addSite:
            E_base += eTs['site']
          for l in range(L):
            E = 1. * E_base
            for scalable in scalables:
              if scalable in protocol[l].keys():
                E += protocol[l][scalable] * eTs[scalable]
            for key in ['k_angular_ext', 'k_spatial_ext', 'k_angular_int']:
              if key in protocol[l].keys():
                E += protocol[l][key] * eTs[key]
            if noBeta:
              u_kln.append(E)
            else:
              u_kln.append(E / (R * protocol[l]['T']))
        elif isinstance(eTs[0], dict):
          K = len(eTs)
          N_k = np.array([len(eTs[k][probe_key]) for k in range(K)])
          u_kln = np.zeros([K, L, N_k.max()], np.float64)

          for k in range(K):
            E_base = 0.0
            if addMM:
              E_base += eTs[k]['MM']
            if addSite:
              E_base += eTs[k]['site']
            for l in range(L):
              E = 1. * E_base
              for scalable in scalables:
                if scalable in protocol[l].keys():
                  E += protocol[l][scalable] * eTs[k][scalable]
              for key in ['k_angular_ext', 'k_spatial_ext', 'k_angular_int']:
                if key in protocol[l].keys():
                  E += protocol[l][key] * eTs[k][key]
              if noBeta:
                u_kln[k, l, :N_k[k]] = E
              else:
                u_kln[k, l, :N_k[k]] = E / (R * protocol[l]['T'])
        elif isinstance(eTs[0], list):
          K = len(eTs)
          N_k = np.zeros(K, dtype=int)

          for k in range(K):
            for c in range(len(eTs[k])):
              N_k[k] += len(eTs[k][c][probe_key])
          u_kln = np.zeros([K, L, N_k.max()], np.float64)

          # DEBUG: Print first few states' energies and protocol
          debug_u_kln = False  # Disabled - too verbose
          if debug_u_kln:
            print("\n" + "="*80)
            print("DEBUG _u_kln")
            print("="*80)
            print(f"K (sampled states) = {K}, L (evaluated states) = {L}")
            print(f"N_k (samples per state) = {N_k[:min(5, K)]}")

            # Show what energy keys are available
            print(f"\nEnergy keys available in eTs[0][0]: {list(eTs[0][0].keys())}")

            # Show protocol for first and last few states
            print("\nProtocol (first 3 and last 3 states):")
            for l in [0, 1, 2, L-3, L-2, L-1]:
              if 0 <= l < L:
                grid_params = {s: protocol[l].get(s, 0) for s in ['sLJr', 'sELE', 'LJr', 'LJa', 'ELE']}
                alpha = protocol[l].get('alpha', '?')
                T = protocol[l].get('T', '?')
                print(f"  State {l}: alpha={alpha:.4f}, T={T:.1f}, grids={grid_params}")

            # Check if grid energies are non-zero
            print(f"\nGrid energies check (state 0, cycle 0, first sample):")
            for grid_key in ['sLJr', 'sELE', 'LJr', 'LJa', 'ELE']:
              if grid_key in eTs[0][0]:
                vals = eTs[0][0][grid_key]
                if len(vals) > 0:
                  print(f"  {grid_key}: {vals[0]:.2f} kJ/mol (mean={vals.mean():.2f}, nonzero={np.count_nonzero(vals)}/{len(vals)})")
              else:
                print(f"  {grid_key}: NOT IN eTs")

            # Check if protocol has grid terms
            print(f"\nProtocol grid params (state 0 and state {L-1}):")
            for state_idx in [0, L-1]:
              if state_idx < L:
                print(f"  State {state_idx}:")
                for grid_key in ['sLJr', 'sELE', 'LJr', 'LJa', 'ELE']:
                  val = protocol[state_idx].get(grid_key, "MISSING")
                  print(f"    {grid_key}: {val}")

          # Parallelize over k (sampled states) - each state independent
          n_workers = get_optimal_worker_count()

          # Prepare tasks - each task computes u_kl for one state k
          tasks = [(k, eTs[k], protocol, scalables, addMM, addSite, noBeta, R)
                   for k in range(K)]

          # Execute in parallel
          with ThreadPoolExecutor(max_workers=n_workers) as executor:
              results = list(executor.map(
                  lambda task: FreeEnergyCalculator._compute_u_kl_for_state_k(*task),
                  tasks
              ))

          # Assemble results into u_kln matrix
          for k, u_kl, N_k_result in results:
              u_kln[k, :, :N_k[k]] = u_kl

        if (K == 1) and (L == 1):
          return np.array(u_kln).ravel()
        else:
          return (u_kln, N_k)

    @staticmethod
    def run_MBAR(u_kln, N_k, augmented=False):
        """
        Estimates the free energy of a transition using BAR and MBAR
        
        Extracted from BindingPMF.run_MBAR (lines 2139-2211)

        Parameters
        ----------
        u_kln : np.array
            Reduced potential energy matrix [K x L x N_max]
        N_k : np.array
            Number of samples per state [K]
        augmented : bool
            Whether to use augmented states (for state insertion)

        Returns
        -------
        f_k_MBAR : np.array
            Free energies for each state [K]
        W_nl : np.array or None
            Configuration weights [N_total x L] (None if MBAR fails)
        """
        K = len(N_k) - 1 if augmented else len(N_k)
        f_k_FEPF = np.zeros(K)
        f_k_BAR = np.zeros(K)
        W_nl = None
        for k in range(K - 1):
          w_F = u_kln[k, k + 1, :N_k[k]] - u_kln[k, k, :N_k[k]]
          min_w_F = min(w_F)
          w_R = u_kln[k + 1, k, :N_k[k + 1]] - u_kln[k + 1, k + 1, :N_k[k + 1]]
          min_w_R = min(w_R)
          f_k_FEPF[k + 1] = -np.log(np.mean(np.exp(-w_F + min_w_F))) + min_w_F
          try:
            # Try pymbar 4.x API (lowercase, returns dict)
            bar_result = pymbar.bar(w_F, w_R, \
                           relative_tolerance=1.0E-5, \
                           verbose=False, \
                           compute_uncertainty=False)
            f_k_BAR[k+1] = bar_result['Delta_f']
          except AttributeError:
            # Fall back to pymbar 3.x API (uppercase, returns scalar)
            try:
              f_k_BAR[k+1] = pymbar.BAR(w_F, w_R, \
                             relative_tolerance=1.0E-5, \
                             verbose=False, \
                             compute_uncertainty=False)
            except Exception as e:
              f_k_BAR[k + 1] = f_k_FEPF[k + 1]
              print('Error with BAR. Using FEP.')
              print(f'BAR exception details: {type(e).__name__}: {e}')
              import traceback
              traceback.print_exc()
          except Exception as e:
            f_k_BAR[k + 1] = f_k_FEPF[k + 1]
            print('Error with BAR. Using FEP.')
            print(f'BAR exception details: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
        f_k_FEPF = np.cumsum(f_k_FEPF)
        f_k_BAR = np.cumsum(f_k_BAR)
        try:
          if augmented:
            f_k_BAR = np.append(f_k_BAR, [0])
          f_k_pyMBAR = pymbar.MBAR(u_kln, N_k, \
            relative_tolerance=1.0E-5, \
            verbose = False, \
            initial_f_k = f_k_BAR, \
            maximum_iterations = 20)
          f_k_MBAR = f_k_pyMBAR.f_k
          # Check if MBAR converged
          if hasattr(f_k_pyMBAR, 'n_iterations'):
            if f_k_pyMBAR.n_iterations >= 20:
              print(f'WARNING: MBAR may not have converged (reached max iterations: {f_k_pyMBAR.n_iterations})')
              print(f'  Consider increasing maximum_iterations or improving state overlap')
          # Handle API change in pymbar 4.x: getWeights() -> weights()
          if hasattr(f_k_pyMBAR, 'getWeights'):
            W_nl = f_k_pyMBAR.getWeights()
          else:
            W_nl = f_k_pyMBAR.weights()
        except Exception as e:
          print(N_k, f_k_BAR)
          f_k_MBAR = f_k_BAR
          print('Error with MBAR. Using BAR.')
          print(f'MBAR exception details: {type(e).__name__}: {e}')
          import traceback
          traceback.print_exc()
        if np.isnan(f_k_MBAR).any():
          f_k_MBAR = f_k_BAR
          print('Error with MBAR. Using BAR.')
        return (f_k_MBAR, W_nl)

    @staticmethod
    def get_equilibrated_cycle(data_process, stats_dict, process_name):
        """
        Detect equilibration point using pymbar
        
        Extracted from BindingPMF._get_equilibrated_cycle (lines 1086-1131)

        Parameters
        ----------
        data_process : SimulationData
            Simulation data object for the process (BC or CD)
        stats_dict : dict
            Statistics dictionary (stats_L or stats_RL)
        process_name : str
            'BC' or 'CD'

        Returns
        -------
        equilibrated_cycle : list
            Array of equilibration points per state
        """
        # Get previous results, if any
        if ('equilibrated_cycle' in stats_dict.keys()) and \
            stats_dict['equilibrated_cycle']!=[]:
          equilibrated_cycle = stats_dict['equilibrated_cycle']
        else:
          equilibrated_cycle = [0]

        # Estimate equilibrated cycle
        for last_c in range(len(equilibrated_cycle), data_process.cycle):
          # PyMBAR 4.x uses integrated_autocorrelation_time (with underscores)
          # PyMBAR 3.x used integratedAutocorrelationTime (camelCase)
          try:
            # Try PyMBAR 4.x API first
            correlation_times = [np.inf] + [\
              pymbar.timeseries.integrated_autocorrelation_time(\
                np.concatenate([data_process.Es[0][c]['mean_energies'] \
                  for c in range(start_c,len(data_process.Es[0])) \
                  if 'mean_energies' in data_process.Es[0][c].keys()])) \
                     for start_c in range(1,last_c)]
          except AttributeError:
            # Fall back to PyMBAR 3.x API
            correlation_times = [np.inf] + [\
              pymbar.timeseries.integratedAutocorrelationTime(\
                np.concatenate([data_process.Es[0][c]['mean_energies'] \
                  for c in range(start_c,len(data_process.Es[0])) \
                  if 'mean_energies' in data_process.Es[0][c].keys()])) \
                     for start_c in range(1,last_c)]
          g = 2 * np.array(correlation_times) + 1
          nsamples_tot = [n for n in reversed(np.cumsum([len(data_process.Es[0][c]['MM']) \
            for c in reversed(range(last_c))]))]
          nsamples_ind = nsamples_tot / g
          equilibrated_cycle_last_c = max(np.argmax(nsamples_ind), 1)
          equilibrated_cycle.append(equilibrated_cycle_last_c)

        return equilibrated_cycle
