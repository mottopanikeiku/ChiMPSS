from chimpss.algdock.BindingPMF import HMStime

import time
import numpy as np

import copy

try:
  import MMTK
  import MMTK.Units
  from MMTK.ParticleProperties import Configuration
except ImportError:
  MMTK = None

try:
  import openmm
  import openmm.unit as unit
except ImportError:
  openmm = None

from chimpss.algdock.BindingPMF import R
from chimpss.algdock.BindingPMF import scalables

from chimpss.algdock.parallel_config import configure_thread_limits, ParallelExecutor
from chimpss.algdock.simulation_iterator import set_worker_iterator, _iteration_worker_func

class ReplicaExchange():
  """Runs replica exchange

    Attributes
    ----------
    args : chimpss.algdock.simulation_arguments.SimulationArguments
      Simulation arguments
    log : chimpss.algdock.logger.Logger
      Simulation log
    top : chimpss.algdock.topology.TopologyMMTK
      Topology of the ligand
    system : chimpss.algdock.system.System
      Simulation system
    _get_confs_to_rescore : function
      Returns the configurations to rescore
    iterator : chimpss.algdock.simulation_iterator.SimulationIterator
      Performs an iteration on one thermodynamic state
    data : chimpss.algdock.simulation_data.SimulationData
      Stores results from the simulation
  """
  def __init__(self, args, log, top, system, iterator, data, save, _u_kln):
    """Initializes the class

      Parameters
      ----------
      args : chimpss.algdock.simulation_arguments.SimulationArguments
        Simulation arguments
      log : chimpss.algdock.logger.Logger
        Simulation log
      top : chimpss.algdock.topology.TopologyMMTK
        Topology of the ligand
      system : chimpss.algdock.system.System
        Simulation system
      iterator : chimpss.algdock.simulation_iterator.SimulationIterator
        Performs an iteration on one thermodynamic state
      data : chimpss.algdock.simulation_data.SimulationData
        Stores results from the simulation
      save : chimpss.algdock.BindingPMF.save
        Saves the data
      _u_kln : chimpss.algdock.BindingPMF._u_kln
        Evaluates energies in different thermodynamic states
    """
    self.args = args
    self.log = log
    self.top = top
    self.system = system
    self.iterator = iterator
    self.data = data
    self.save = save
    self._u_kln = _u_kln

    # Configure thread limits to prevent overprovisioning (M workers x N threads)
    threads_per_worker = configure_thread_limits(self.args.cores)
    self.log.tee("  Thread coordination: %s workers, %s threads/worker" %
                 (self.args.cores, threads_per_worker))

    # Set global iterator for multiprocessing workers
    # The iterator contains OpenMM objects that can't be pickled,
    # but by setting it globally before ProcessPoolExecutor forks, each worker
    # process gets its own independent copy with its own OpenMM context
    set_worker_iterator(iterator)

    # Create persistent thread pool executor
    self.executor = ParallelExecutor(
        n_workers=self.args.cores,
        name_prefix='replica'
    )

  def run(self, process):
    """
    Performs a cycle of replica exchange
    """
    if not process in ['CD', 'BC']:
      raise Exception('Process must be CD or BC')

    # Initialize diagnostic logger for replica exchange
    from chimpss.algdock.diagnostic_logger import DiagnosticLogger
    diag = DiagnosticLogger(self.args.dir[process], implementation='openmm')
    self.log.tee("  Diagnostic logging for %s RepX: %s" % (process, diag.filename))

# GMC

    def gMC_initial_setup():
      """
      Initialize BAT converter object.
      Decide which internal coord to crossover. Here, only the soft torsions will be crossovered.
      Produce a list of replica (state) index pairs to be swaped. Only Neighbor pairs will be swaped.
      Assume that self.top.universe, self.top.molecule and K (number of states) exist
      as global variables when the function is called.
      """
      from chimpss.algdock.rigid_bodies import identifier
      import itertools
      BAT_converter = identifier(self.top.universe, self.top.molecule)
      BAT = BAT_converter.BAT(extended=True)
      # this assumes that the torsional angles are stored in the tail of BAT
      softTorsionId = [
        i + len(BAT) - BAT_converter.ntorsions
        for i in BAT_converter._softTorsionInd
      ]
      torsions_to_crossover = []
      for i in range(1, len(softTorsionId)):
        combinations = itertools.combinations(softTorsionId, i)
        for c in combinations:
          torsions_to_crossover.append(list(c))
      #
      BAT_converter.BAT_to_crossover = torsions_to_crossover
      if len(BAT_converter.BAT_to_crossover) == 0:
        self.log.tee('  GMC No BAT to crossover')
      state_indices = range(K)
      state_indices_to_swap = zip( state_indices[0::2], state_indices[1::2] ) + \
                      zip( state_indices[1::2], state_indices[2::2] )
      #
      return BAT_converter, state_indices_to_swap

    #
    def do_gMC(nr_attempts, BAT_converter, state_indices_to_swap,
               torsion_threshold):
      """
      Assume self.top.universe, confs, protocol, state_inds, inv_state_inds exist as global variables
      when the function is called.
      If at least one of the torsions in the combination chosen for an crossover attempt
      changes more than torsion_threshold, the crossover will be attempted.
      The function will update confs.
      It returns the number of attempts and the number of accepted moves.
      """
      if nr_attempts < 0:
        raise Exception('Number of attempts must be nonnegative!')
      if torsion_threshold < 0.:
        raise Exception('Torsion threshold must be nonnegative!')
      #
      if len(BAT_converter.BAT_to_crossover) == 0:
        return 0., 0.
      #
      from random import randrange
      # get reduced energies and BAT for all configurations in confs
      BATs = []
      energies = np.zeros(K, dtype=float)
      for c_ind in range(K):
        s_ind = state_inds[c_ind]
        if MMTK:
          self.top.universe.setConfiguration(
            Configuration(self.top.universe, confs[c_ind]))
        else:
          self.top.setConfiguration(confs[c_ind])
        BATs.append(np.array(BAT_converter.BAT(extended=True), dtype=float))
        self.system.setParams(protocol[s_ind])
        if MMTK:
          reduced_e = self.top.universe.energy() / (R * protocol[s_ind]['T'])
        else:
          reduced_e = self.top.energy() / (R * protocol[s_ind]['T'])
        energies[c_ind] = reduced_e
      #
      nr_sets_of_torsions = len(BAT_converter.BAT_to_crossover)
      #
      attempt_count, acc_count = 0, 0
      sweep_count = 0
      while True:
        sweep_count += 1
        if (sweep_count * K) > (1000 * nr_attempts):
          self.log.tee(
            '  GMC Sweep too many times, but few attempted. Consider reducing torsion_threshold.'
          )
          return attempt_count, acc_count
        #
        for state_pair in state_indices_to_swap:
          conf_ind_k0 = inv_state_inds[state_pair[0]]
          conf_ind_k1 = inv_state_inds[state_pair[1]]
          # check if it should attempt for this pair of states
          ran_set_torsions = BAT_converter.BAT_to_crossover[randrange(
            nr_sets_of_torsions)]
          do_crossover = np.any(
            np.abs(BATs[conf_ind_k0][ran_set_torsions] -
                   BATs[conf_ind_k1][ran_set_torsions]) >= torsion_threshold)
          if do_crossover:
            attempt_count += 1
            # BAT and reduced energies before crossover
            BAT_k0_be = copy.deepcopy(BATs[conf_ind_k0])
            BAT_k1_be = copy.deepcopy(BATs[conf_ind_k1])
            e_k0_be = energies[conf_ind_k0]
            e_k1_be = energies[conf_ind_k1]
            # BAT after crossover
            BAT_k0_af = copy.deepcopy(BAT_k0_be)
            BAT_k1_af = copy.deepcopy(BAT_k1_be)
            for index in ran_set_torsions:
              tmp = BAT_k0_af[index]
              BAT_k0_af[index] = BAT_k1_af[index]
              BAT_k1_af[index] = tmp
            # Cartesian coord and reduced energies after crossover.
            BAT_converter.Cartesian(BAT_k0_af)
            self.system.setParams(protocol[state_pair[0]])
            if MMTK:
              e_k0_af = self.top.universe.energy() / (
                R * protocol[state_pair[0]]['T'])
              conf_k0_af = copy.deepcopy(self.top.universe.configuration().array)
            else:
              e_k0_af = self.top.energy() / (
                R * protocol[state_pair[0]]['T'])
              conf_k0_af = copy.deepcopy(self.top.configuration().array)
            #
            BAT_converter.Cartesian(BAT_k1_af)
            self.system.setParams(protocol[state_pair[1]])
            if MMTK:
              e_k1_af = self.top.universe.energy() / (
                R * protocol[state_pair[1]]['T'])
              conf_k1_af = copy.deepcopy(self.top.universe.configuration().array)
            else:
              e_k1_af = self.top.energy() / (
                R * protocol[state_pair[1]]['T'])
              conf_k1_af = copy.deepcopy(self.top.configuration().array)
            #
            de = (e_k0_be - e_k0_af) + (e_k1_be - e_k1_af)
            # update confs, energies, BATS
            if (de > 0) or (np.random.uniform() < np.exp(de)):
              acc_count += 1
              confs[conf_ind_k0] = conf_k0_af
              confs[conf_ind_k1] = conf_k1_af
              #
              energies[conf_ind_k0] = e_k0_af
              energies[conf_ind_k1] = e_k1_af
              #
              BATs[conf_ind_k0] = BAT_k0_af
              BATs[conf_ind_k1] = BAT_k1_af
            #
            if attempt_count == nr_attempts:
              return attempt_count, acc_count

    #
    self.log.set_lock(process)

    confs = self.data[process].confs['replicas']
    protocol = self.data[process].protocol

    terms = ['MM']
    if process == 'BC':
      terms += ['OBC']
    elif process == 'CD':
      if self.args.params['CD']['pose'] > -1:
        # Pose BPMF
        terms += ['k_angular_ext', 'k_spatial_ext', 'k_angular_int']
      else:
        terms += ['site']
      terms += scalables

    # A list of pairs of replica indicies
    K = len(protocol)
    pairs_to_swap = []
    for interval in range(1, min(5, K)):
      lower_inds = []
      for lowest_index in range(interval):
        lower_inds += range(lowest_index, K - interval, interval)
      upper_inds = np.array(lower_inds) + interval
      pairs_to_swap += zip(lower_inds, upper_inds)

    from chimpss.algdock.repX import attempt_swaps

    # Setting the force field will load grids
    # before multiple processes are spawned
    for k in range(K):
      self.system.setParams(protocol[k])

    # If it has not been set up, set up Smart Darting
    self.iterator.initializeSmartDartingConfigurations(
      self.data[process].confs['SmartDarting'], process, self.log, self.data)

    # storage[key][sweep_index][state_index] will contain data
    # from the replica exchange sweeps
    storage = {}
    for var in ['confs', 'state_inds', 'energies']:
      storage[var] = []

    self.log.recordStart('repX cycle')

    # GMC
    do_gMC = self.args.params[process]['GMC_attempts'] > 0
    if do_gMC:
      self.log.tee('  Using GMC for %s' % process)
      nr_gMC_attempts = K * self.args.params[process]['GMC_attempts']
      torsion_threshold = self.args.params[process]['GMC_tors_threshold']
      gMC_attempt_count = 0
      gMC_acc_count = 0
      time_gMC = 0.0
      BAT_converter, state_indices_to_swap = gMC_initial_setup()

    # MC move statistics
    acc = {}
    att = {}
    for move_type in ['ExternalMC', 'SmartDarting', 'Sampler']:
      acc[move_type] = np.zeros(K, dtype=int)
      att[move_type] = np.zeros(K, dtype=int)
      self.log.timings[move_type] = 0.
    self.log.timings['repX'] = 0.

    mean_energies = []

    # Do replica exchange
    state_inds = list(range(K))
    inv_state_inds = list(range(K))
    nsweeps = self.args.params[process]['sweeps_per_cycle']
    nsnaps = nsweeps // self.args.params[process]['snaps_per_cycle']  # Integer division for Python 3
    for sweep in range(nsweeps):
      E = {}
      for term in terms:
        E[term] = np.zeros(K, dtype=float)
      # Sample within each state using process pool executor
      # Use module-level worker function instead of bound method for pickling
      tasks = [(confs[k], process, protocol[state_inds[k]], False, k) for k in range(K)]
      results = self.executor.map(_iteration_worker_func, tasks)

      # GMC
      if do_gMC:
        time_start_gMC = time.time()
        att_count, acc_count = do_gMC(nr_gMC_attempts, BAT_converter,
                                      state_indices_to_swap, torsion_threshold)
        gMC_attempt_count += att_count
        gMC_acc_count += acc_count
        time_gMC = +(time.time() - time_start_gMC)

      # Store energies
      for k in range(K):
        confs[k] = results[k]['confs']
      mean_energies.append(np.mean([results[k]['Etot'] for k in range(K)]))
      E = self.system.energyTerms(confs, E, process=process)

      # DEBUG: Show unscaled energies and per-state scaling factors
      if sweep == 0 and self.data[process].cycle == 0:
        print("\n" + "="*80)
        print("DEBUG: Energy Analysis at Cycle 0, Sweep 0")
        print("="*80)
        print("\nUnscaled energies (from energyTerms with all grids at alpha=1.0):")
        for key in sorted(E.keys()):
          if key in scalables or key in ['MM', 'OBC', 'site']:
            print("  {:8s}: mean={:10.2f} kJ/mol, min={:10.2f}, max={:10.2f}".format(
              key, np.mean(E[key]), np.min(E[key]), np.max(E[key])))

        print("\nPer-state scaling factors (applied in _u_kln):")
        for k in range(K):
          state = protocol[state_inds[k]]
          print("\n  State {:2d}: alpha={:.4f}, T={:.1f}K".format(
            k, state.get('alpha', -1), state['T']))
          if process == 'CD':
            for scalable in scalables:
              if scalable in state:
                unscaled_energy = np.mean(E[scalable])
                scaled_energy = state[scalable] * unscaled_energy
                print("    {:8s}: scale={:.4f}, unscaled={:10.2f}, scaled={:10.2f} kJ/mol".format(
                  scalable, state[scalable], unscaled_energy, scaled_energy))
        print("="*80 + "\n")

      # Store MC move statistics
      for k in range(K):
        for move_type in ['ExternalMC', 'SmartDarting', 'Sampler']:
          key = 'acc_' + move_type
          if key in results[k].keys():
            acc[move_type][state_inds[k]] += results[k][key]
            att[move_type][state_inds[k]] += results[k]['att_' + move_type]
            self.log.timings[move_type] += results[k]['time_' + move_type]

      # Calculate u_ij (i is the replica, and j is the configuration),
      #    a list of arrays
      (u_ij, N_k) = self._u_kln(E, [protocol[state_inds[c]] for c in range(K)])
      # Do the replica exchange
      repX_start_time = time.time()
      # Track state assignments before swap to detect which replicas had accepted swaps
      state_inds_before_swap = list(state_inds)
      (state_inds, inv_state_inds) = \
        attempt_swaps(state_inds, inv_state_inds, u_ij, pairs_to_swap, \
          self.args.params[process]['attempts_per_sweep'])
      self.log.timings['repX'] += (time.time() - repX_start_time)

      # Determine which replicas had swaps accepted this sweep
      replicas_that_swapped = []
      for replica_idx in range(K):
        if state_inds[replica_idx] != state_inds_before_swap[replica_idx]:
          replicas_that_swapped.append(replica_idx)

      # Store data in local variables
      if (sweep + 1) % self.args.params[process]['snaps_per_cycle'] == 0:
        storage['confs'].append(list(confs))
        storage['state_inds'].append(list(state_inds))
        storage['energies'].append(copy.deepcopy(E))

      # LOG: Every 10th sweep to track replica exchange progress
      if sweep % 10 == 0:
        # Calculate mean energy per state
        inv_state_inds_now = np.zeros(K, dtype=int)
        for k in range(K):
          inv_state_inds_now[state_inds[k]] = k

        mean_E_per_state = np.zeros(K)
        for k in range(K):
          replica_idx = inv_state_inds_now[k]
          mean_E_per_state[k] = np.mean(E['MM'][replica_idx])

        # Pass context and protocol for enhanced diagnostics
        context = None
        if hasattr(self.top, 'context'):
          context = self.top.context

        diag.log_repx_cycle(
          process=process,
          cycle=self.data[process].cycle,
          state_inds=state_inds,
          mean_energies_per_state=mean_E_per_state,
          context=context,
          protocol=self.data[process].protocol,
          energy_threshold=10000.0,  # kJ/mol - log detailed diagnostics above this
          confs=confs,  # Pass all replica configurations
          inv_state_inds_now=inv_state_inds_now,  # Pass state->replica mapping
          system=self.system,  # Pass system to set parameters and positions
          replicas_that_swapped=replicas_that_swapped  # Track which replicas had accepted swaps
        )

    # GMC
    if do_gMC:
      self.log.tee('  {0}/{1} crossover attempts ({2:.3g}) accepted in {3}'.format(\
        gMC_acc_count, gMC_attempt_count, \
        float(gMC_acc_count)/float(gMC_attempt_count) \
          if gMC_attempt_count > 0 else 0, \
        HMStime(time_gMC)))

    # Report
    self.log.tee("  completed cycle %d in %s"%(self.data[process].cycle, \
      HMStime(self.log.timeSince('repX cycle'))))
    MC_report = " "
    for move_type in ['ExternalMC', 'SmartDarting', 'Sampler']:
      total_acc = np.sum(acc[move_type])
      total_att = np.sum(att[move_type])
      if total_att > 0:
        MC_report += " %s %d/%d=%.2f (%.1f s);"%(move_type, \
          total_acc, total_att, float(total_acc)/total_att, \
          self.log.timings[move_type])
    MC_report += " repX t %.1f s" % self.log.timings['repX']
    self.log.tee(MC_report)

    # Adapt HamiltonianMonteCarlo parameters
    if self.args.params[process]['sampler'] == 'HMC':
      # Use OpenMM units for time (femtoseconds)
      fs = unit.femtoseconds if openmm else 1.0
      acc_rates = np.array(acc['Sampler'], dtype=np.float64) / att['Sampler']
      for k in range(K):
        acc_rate = acc_rates[k]
        if acc_rate > 0.8:
          protocol[k]['delta_t'] += 0.125 * fs
          protocol[k]['steps_per_trial'] = min(protocol[k]['steps_per_trial']*2,\
            self.args.params[process]['steps_per_sweep'])
        elif acc_rate < 0.4:
          if protocol[k]['delta_t'] < 2.0 * fs:
            protocol[k]['steps_per_trial'] = max(
              int(protocol[k]['steps_per_trial'] / 2.), 1)
          protocol[k]['delta_t'] -= 0.25 * fs
          if acc_rate < 0.1:
            protocol[k]['delta_t'] -= 0.25 * fs
        if protocol[k]['delta_t'] < 0.1 * fs:
          protocol[k]['delta_t'] = 0.1 * fs

    # Get indicies for sorting by thermodynamic state, not replica
    inv_state_inds = np.zeros((nsnaps, K), dtype=int)
    for snap in range(nsnaps):
      state_inds = storage['state_inds'][snap]
      for k in range(K):
        inv_state_inds[snap][state_inds[k]] = k

    # Sort energies and conformations by thermodynamic state
    # and store in global variables
    #   self.data[process].Es and self.data[process].confs['samples']
    # and also local variable confs_repX
    for k in range(K):
      E_k = {}
      if k == 0:
        E_k['acc'] = acc
        E_k['att'] = att
        E_k['mean_energies'] = mean_energies
      for term in terms:
        E_term = np.array([storage['energies'][snap][term][\
          inv_state_inds[snap][k]] for snap in range(nsnaps)])
        E_k[term] = E_term
      self.data[process].Es[k].append(E_k)

    confs_repX = []
    for k in range(K):
      confs_k = [storage['confs'][snap][inv_state_inds[snap][k]] \
        for snap in range(nsnaps)]
      if self.args.params[process]['keep_intermediate'] or \
          ((process=='BC') and (k==0)) or (k==(K-1)):
        self.data[process].confs['samples'][k].append(confs_k)
      confs_repX.append(confs_k)
    self.data[process].confs['last_repX'] = confs_repX

    # Store final conformation of each replica
    self.data[process].confs['replicas'] = \
      [np.copy(storage['confs'][-1][inv_state_inds[-1][k]]) \
       for k in range(K)]

    if self.args.params[process]['darts_per_sweep'] > 0:
      self.system.setParams(self.data[process].protocol[-1])
      new_confs = [np.copy(conf) \
        for conf in self.data[process].confs['samples'][k][-1]]
      self.iterator.addSmartDartingConfigurations(new_confs, process,
                                                  self.log, self.data)

    self.data[process].cycle += 1
    self.save(process)
    self.log.tee("  Diagnostic summary for cycle %d:" % (self.data[process].cycle - 1))
    diag.summarize()
    self.log.tee("")
    self.log.clear_lock(process)
