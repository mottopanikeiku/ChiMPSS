import numpy as np
import time
try:
  import MMTK
  import MMTK.Units
  from MMTK.ParticleProperties import Configuration
except ImportError:
  MMTK = None

try:
  import openmm
  import openmm.unit as unit
  from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff
  from openmm import *
except ImportError:
  OpenMM = None

# Global iterator for multiprocessing workers
# When using ProcessPoolExecutor with fork mode (Linux default), each worker
# process gets an independent copy of this iterator with its own OpenMM context.
# This avoids pickling issues while ensuring each worker has isolated state.
_worker_iterator = None

def set_worker_iterator(iterator):
  """
  Set the global iterator for worker processes

  Must be called BEFORE creating ProcessPoolExecutor. When workers fork,
  each gets an independent copy of the iterator with its own OpenMM context.

  Parameters
  ----------
  iterator : SimulationIterator
      The iterator instance to use
  """
  global _worker_iterator
  _worker_iterator = iterator

def _iteration_worker_func(seed, process, params_k, initialize, reference):
  """
  Module-level worker function for ProcessPoolExecutor

  This function is picklable because it's defined at module level.
  Each worker process has its own independent copy of _worker_iterator
  (inherited via fork), so there are no race conditions.

  Parameters
  ----------
  seed : np.array
      Starting configuration
  process : str
      Process name ('BC' or 'CD')
  params_k : dict
      Thermodynamic parameters for this state
  initialize : bool
      Whether to initialize
  reference : int
      Reference index for this replica

  Returns
  -------
  dict
      Results dictionary from iteration
  """
  global _worker_iterator

  if _worker_iterator is None:
    raise RuntimeError("Worker iterator not set - call set_worker_iterator() before creating ProcessPoolExecutor")

  return _worker_iterator.iteration(seed, process, params_k, initialize, reference)

class SimulationIterator:
  """SimulationIterators take a molecular configuration and generate a new one.

    ...

    Attributes
    ----------
    args : chimpss.algdock.simulation_arguments.SimulationArguments
      Simulation arguments
    top : chimpss.algdock.topology.TopologyMMTK
      Topology with ligand
    system : chimpss.algdock.system.System
      System
    _sampler : dict
  """
  def __init__(self, args, top, system):
    """Initializes the class

      Parameters
      ----------
      args : chimpss.algdock.simulation_arguments.SimulationArguments
        Simulation arguments
      top : chimpss.algdock.topology.TopologyMMTK
        Topology with ligand
      system : chimpss.algdock.system.System
        System
    """
    self.args = args
    self.top = top
    self.system = system

    self._samplers = {}
    # Only import SmartDarting if it will be used
    if (self.args.params['BC']['darts_per_seed'] > 0 or
        self.args.params['BC']['darts_per_sweep'] > 0 or
        self.args.params['CD']['darts_per_seed'] > 0 or
        self.args.params['CD']['darts_per_sweep'] > 0):
      # Uses cython class
      # from SmartDarting import SmartDartingIntegrator # @UnresolvedImport
      # Uses python class
      from chimpss.algdock.Integrators.SmartDarting.SmartDarting \
        import SmartDartingIntegrator # @UnresolvedImport
      if MMTK:
        self._samplers['BC_SmartDarting'] = SmartDartingIntegrator(\
        self.top.universe, self.top.molecule, False)
        self._samplers['CD_SmartDarting'] = SmartDartingIntegrator(\
        self.top.universe, self.top.molecule, True)
      else:
        self._samplers['BC_SmartDarting'] = SmartDartingIntegrator(\
        self.top.OMM_simulation, self.top.molecule, False)
      self._samplers['CD_SmartDarting'] = SmartDartingIntegrator(\
      self.top.OMM_simulation, self.top.molecule, True)

    # Import and create ExternalMC sampler
    # Note: MMTK version always created this, so we do too for consistency
    if MMTK:
      from chimpss.algdock.Integrators.ExternalMC.ExternalMC import ExternalMCIntegrator
      self._samplers['ExternalMC'] = ExternalMCIntegrator(\
        self.top.universe, self.top.molecule, step_size=0.25*MMTK.Units.Ang)
    else:
      from chimpss.algdock.Integrators.ExternalMC.ExternalMC import ExternalMCIntegratorOpenMM
      self._samplers['ExternalMC'] = ExternalMCIntegratorOpenMM(\
        self.top.OMM_simulation, self.top.molecule, step_size=0.25, topology=self.top)

    for p in ['BC', 'CD']:
      if self.args.params[p]['sampler'] == 'HMC':
        if MMTK:
          from chimpss.algdock.Integrators.HamiltonianMonteCarlo.HamiltonianMonteCarlo \
            import HamiltonianMonteCarloIntegrator
          self._samplers[p] = HamiltonianMonteCarloIntegrator(self.top.universe)
        else:
          from chimpss.algdock.Integrators.HamiltonianMonteCarlo.HamiltonianMonteCarlo \
            import HamiltonianMonteCarloIntegratorUsingOpenMM
          self._samplers[p] = HamiltonianMonteCarloIntegratorUsingOpenMM(
            self.top.molecule, self.top, self.top.OMM_system)
      elif self.args.params[p]['sampler'] == 'NUTS':
        from NUTS import NUTSIntegrator  # @UnresolvedImport
        self._samplers[p] = NUTSIntegrator(self.top.universe)
      elif self.args.params[p]['sampler'] == 'VV':
        from chimpss.algdock.Integrators.VelocityVerlet.VelocityVerlet \
          import VelocityVerletIntegrator
        self._samplers[p] = VelocityVerletIntegrator(self.top.universe)
      else:
        raise Exception('Unrecognized sampler!')

  def iteration(self, seed, process, params_k, \
      initialize=False, reference=0, skip_setParams=False):
    """Performs an iteration for a single thermodynamic state

    Parameters
    ----------
    seed : np.array
      Starting configuration
    process : str
      Process, either 'BC' or 'CD'
    params_k : dict of float
      Parameters describing a thermodynamic state
    skip_setParams : bool
      If True, skip calling setParams (assumes it was already called)
    """
    # PROFILING: Track time breakdown in iteration
    profile_iteration = initialize and not hasattr(self, '_iteration_profiled')
    accumulate_timing = initialize and hasattr(self, '_iteration_timing')
    if profile_iteration:
      import time as time_module
      self._iteration_profiled = True
      self._iteration_timing = {'setParams': 0.0, 'setPositions': 0.0, 'sampler': 0.0, 'copy_results': 0.0}
      self._iteration_count = 0
      t_iter_start = time_module.time()
    elif accumulate_timing:
      import time as time_module
      t_iter_start = time_module.time()

    # Set parameters first (this may restore a cached context)
    if profile_iteration or accumulate_timing:
      t_setparams = time_module.time()
    if not skip_setParams:
      self.system.setParams(params_k)
    if profile_iteration or accumulate_timing:
      self._iteration_timing['setParams'] += time_module.time() - t_setparams

    # Then set positions (must be after setParams to use correct context)
    if profile_iteration or accumulate_timing:
      t_setpos = time_module.time()
    if MMTK:
      self.top.universe.setConfiguration(Configuration(self.top.universe, seed))
    else:
      #TODO: CHECK seed atom orders
      self.top.setConfiguration(seed)
    if profile_iteration or accumulate_timing:
      self._iteration_timing['setPositions'] += time_module.time() - t_setpos
    if 'delta_t' in params_k.keys():
      delta_t = params_k['delta_t']
    else:
      raise Exception('No time step specified')
    if 'steps_per_trial' in params_k.keys():
      steps_per_trial = params_k['steps_per_trial']
    else:
      steps_per_trial = self.args.params[process]['steps_per_sweep']

    if initialize:
      steps = self.args.params[process]['steps_per_seed']
      ndarts = self.args.params[process]['darts_per_seed']
    else:
      steps = self.args.params[process]['steps_per_sweep']
      ndarts = self.args.params[process]['darts_per_sweep']

    random_seed = (reference * reference) + int(abs(seed[0][0] * 10000))
    if self.args.random_seed > 0:
      random_seed += self.args.random_seed
    else:
      random_seed += int(time.time() * 1000)

    random_seed = random_seed%32767 #This is the bug has been fixed
    results = {}


    # Execute external MCMC moves
    if (process == 'CD') and (self.args.params['CD']['MCMC_moves']>0) \
        and (params_k['alpha'] < 0.1) and (self.args.params['CD']['pose']==-1):
      time_start_ExternalMC = time.time()
      dat = self._samplers['ExternalMC'](ntrials=5, T=params_k['T'])
      results['acc_ExternalMC'] = dat[2]
      results['att_ExternalMC'] = dat[3]
      results['time_ExternalMC'] = (time.time() - time_start_ExternalMC)

    # Execute dynamics sampler
    time_start_sampler = time.time()
    if profile_iteration or accumulate_timing:
      t_sampler = time_module.time()
    # Pass state_id for trajectory filename uniqueness (alpha for CD, T for BC)
    state_id = params_k.get('alpha', params_k.get('T', 0))
    dat = self._samplers[process](\
      steps=steps, steps_per_trial=steps_per_trial, \
      T=params_k['T'], delta_t=delta_t, \
      normalize=(process=='BC'), adapt=initialize, random_seed=random_seed, \
      seed_index=reference, state_id=state_id)
    if profile_iteration or accumulate_timing:
      self._iteration_timing['sampler'] += time_module.time() - t_sampler
    results['acc_Sampler'] = dat[2]
    results['att_Sampler'] = dat[3]
    results['delta_t'] = dat[4]
    results['time_Sampler'] = (time.time() - time_start_sampler)

    # Execute smart darting
    if (ndarts > 0) and not ((process == 'CD') and (params_k['alpha'] < 0.1)):
      time_start_SmartDarting = time.time()
      dat = self._samplers[process+'_SmartDarting'](\
        ntrials=ndarts, T=params_k['T'], random_seed=random_seed+5)
      results['acc_SmartDarting'] = dat[2]
      results['att_SmartDarting'] = dat[3]
      results['time_SmartDarting'] = (time.time() - time_start_SmartDarting)

    # Store and return results
    if profile_iteration or accumulate_timing:
      t_copy = time_module.time()
    results['confs'] = np.copy(dat[0][-1])
    results['Etot'] = dat[1][-1]
    results['reference'] = reference
    if profile_iteration or accumulate_timing:
      self._iteration_timing['copy_results'] += time_module.time() - t_copy
      self._iteration_count += 1

    return results

  def iteration_worker(self, input, output):
    """Executes an iteration from a multiprocessing queue

    Parameters
    ----------
    input : multiprocessing.Queue
      Tasks to complete
    output : multiprocessing.Queue
      Completed tasks
    """
    for args in iter(input.get, 'STOP'):
      result = self.iteration(*args)
      output.put(result)

  def initializeSmartDartingConfigurations(self, seeds, process, log, data):
    """Initializes the configurations for Smart Darting

    Parameters
    ----------
    seeds : list of np.array
      Starting configurations
    process : str
      Process, either 'BC' or 'CD'
    log : chimpss.algdock.logger.Logger
      Logger that includes tee function
    data : chimpss.algdock.simulation_data.SimulationData
      Location for minimized configurations
    """
    if self.args.params[process]['darts_per_seed'] > 0:
      outstr = self._samplers[process + '_SmartDarting'].set_confs(seeds)
      data[process].confs['SmartDarting'] = \
        self._samplers[process+'_SmartDarting'].confs
      log.tee(outstr)

  def addSmartDartingConfigurations(self, new_confs, process, log, data):
    """Adds new configurations for Smart Darting

    Parameters
    ----------
    new_confs : list of np.array
      New configurations
    process : str
      Process, either 'BC' or 'CD'
    data : chimpss.algdock.simulation_data.SimulationData object
      Location for minimized configurations
    """
    if self.args.params[process]['darts_per_seed'] > 0:
      confs_SmartDarting = [np.copy(conf) \
        for conf in data[process].confs['samples'][k][-1]]
      outstr = self._samplers[process+'_SmartDarting'].set_confs(\
        new_confs + data[process].confs['SmartDarting'])
      data[process].confs['SmartDarting'] = \
        self._samplers[process+'_SmartDarting'].confs
      log.tee(outstr)

  def clearSmartDartingConfigurations(self, process):
    """Clears the list of configurations for Smart Darting

    Parameters
    ----------
    seeds : list of np.array
      Starting configurations
    process : str
      Process, either 'BC' or 'CD'
    """
    if self.args.params[process]['darts_per_seed'] > 0:
      self._samplers[process + '_SmartDarting'].confs = []
