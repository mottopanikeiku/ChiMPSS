#!/usr/bin/env python

# TODO: Free energy of external confinement for poseBPMFs

# Configure JAX platform to match OpenMM platform BEFORE importing pymbar
# This must come before any imports that trigger jax/pymbar loading
from chimpss.algdock.parallel_config import configure_jax_platform
# PyMBAR stays on CPU; OpenMM uses CUDA/OpenCL (set below via JAX_PLATFORMS)
configure_jax_platform('CPU')

import os
import pickle
import gzip
import copy
import sys


import chimpss.algdock.IO 
from chimpss.algdock.IO import load_pkl_gz
from chimpss.algdock.IO import write_pkl_gz
from chimpss.algdock.logger import NullDevice

import time
import numpy as np

from collections import OrderedDict

from chimpss.algdock import dictionary_tools
from chimpss.algdock.free_energy import FreeEnergyCalculator
from chimpss.algdock.analysis import PoseAnalyzer
from chimpss.algdock.configuration import ConfigurationManager, ConfigurationLoader, ConfigurationEnergyCalculator
from chimpss.algdock.bc_process import BCCalculator
from chimpss.algdock.cd_process import CDStateManager, CDCalculator

try:
  import MMTK
  import MMTK.Units
  from MMTK.ParticleProperties import Configuration
  from MMTK.ForceFields import ForceField
except ImportError:
  MMTK = None

try:
  import openmm
  import openmm.unit as unit
  from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff
  from openmm import *
  import parmed as pmd # need to install ParmEd

except ImportError:
  openmm = None

try:
  import Scientific
  try:
    from Scientific._vector import Vector
  except:
    from Scientific.Geometry.VectorModule import Vector
except ImportError:
  Scientific = None
  Vector = None

# Keep PyMBAR/JAX on CPU — GPU is reserved for OpenMM MD steps
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pymbar.timeseries

import multiprocessing
from multiprocessing import Process
try:
  import arguments
except ImportError:
  # arguments module may not be available in all contexts
  arguments = None
# For profiling. Unnecessary for normal execution.
# from memory_profiler import profile

#############
# Constants #
#############

if MMTK:
  R = 8.3144621 * MMTK.Units.J / MMTK.Units.mol / MMTK.Units.K
else:
  # For OpenMM, energies are in kJ/mol, so R is in kJ/mol/K
  import openmm.unit as unit
  R = 8.3144621 * unit.joule / unit.mole / unit.kelvin
  R = R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

scalables = ['OBC', 'sLJr', 'sELE', 'LJr', 'LJa', 'ELE']

# In APBS, minimum ratio of PB grid length to maximum dimension of solute
LFILLRATIO = 4.0  # For the ligand
RFILLRATIO = 2.0  # For the receptor/complex

DEBUG = False
DEBUG_ENERGY = False  # Energy extraction details
DEBUG_HMC = False     # HMC sampling details
DEBUG_INIT = False    # Initialization details

def debug_print(message, category='general', file=None):
    """
    Print debug message based on category

    Parameters
    ----------
    message : str
        Message to print
    category : str
        Debug category: 'general', 'energy', 'hmc', 'init'
    file : file-like
        Output file (default: sys.stderr)
    """
    import sys
    if file is None:
        file = sys.stderr

    enabled = {
        'general': DEBUG,
        'energy': DEBUG_ENERGY or DEBUG,
        'hmc': DEBUG_HMC or DEBUG,
        'init': DEBUG_INIT or DEBUG
    }

    if enabled.get(category, DEBUG):
        file.write(f"DEBUG [{category}]: {message}\n")
        file.flush()


def HMStime(s):
  """
  Given the time in seconds, an appropriately formatted string.
  """
  if s < 60.:
    return '%.2f s' % s
  elif s < 3600.:
    return '%d:%.2f' % (int(s / 60 % 60), s % 60)
  else:
    return '%d:%d:%.2f' % (int(s / 3600), int(s / 60 % 60), s % 60)


##############
# Main Class #
##############


class BPMF:
  def __init__(self, **kwargs):
    """Parses the input arguments and runs the requested calculation"""

    #         mod_path = os.path.join(os.path.dirname(a.__file__), 'BindingPMF.py')
    #         print """###########
    # # AlGDock #
    # ###########
    # Molecular docking with adaptively scaled alchemical interaction grids
    #
    # in {0}
    # last modified {1}
    #     """.format(mod_path, time.ctime(os.path.getmtime(mod_path)))

    from chimpss.algdock.argument_parser import SimulationArguments
    self.args = SimulationArguments(**kwargs)

    from chimpss.algdock.simulation_data import SimulationData
    self.data = {}
    self.data['BC'] = SimulationData(self.args.dir['BC'], 'BC', \
      self.args.params['CD']['pose'])
    self.data['CD'] = SimulationData(self.args.dir['CD'], 'CD', \
      self.args.params['CD']['pose'])

    if not 'max_time' in kwargs.keys():
      kwargs['max_time'] = None
    if not 'run_type' in kwargs.keys():
      kwargs['run_type'] = None

    from chimpss.algdock.logger import Logger
    self.log = Logger(self.args, \
      max_time=kwargs['max_time'], run_type=kwargs['run_type'])

    self.T_HIGH = self.args.params['BC']['T_HIGH']
    self.T_TARGET = self.args.params['BC']['T_TARGET']

    self._setup()

    print('\n*** Simulation parameters and constants ***')
    for p in ['BC', 'CD']:
      print('\nfor %s:' % p)
      print(dictionary_tools.dict_view(self.args.params[p])[:-1])

    self.run(kwargs['run_type'])

  def _setup(self):
    """Creates an MMTK InfiniteUniverse and adds the ligand"""
    if MMTK:
      from chimpss.algdock.topology import TopologyMMTK
      self.top = TopologyMMTK(self.args)
      self.top_RL = TopologyMMTK(self.args, includeReceptor=True)
    else:
      from chimpss.algdock.topology import TopologyUsingOpenMM
      self.top = TopologyUsingOpenMM(self.args)
      self.top_RL = TopologyUsingOpenMM(self.args, includeReceptor=True)
    # Initialize rmsd calculation function
    from chimpss.algdock.RMSD import hRMSD
    self.get_rmsds = hRMSD(self.args.FNs['prmtop']['L'], \
      self.top.inv_prmtop_atom_order_L)

    # Obtain reference pose
    if self.data['CD'].pose > -1:
      if ('starting_poses' in self.data['CD'].confs.keys()) and \
         (self.data['CD'].confs['starting_poses'] is not None):
        starting_pose = np.copy(self.data['CD'].confs['starting_poses'][0])
      else:
        (confs, Es) = self._get_confs_to_rescore(site=False, \
          minimize=False, sort=False)
        if self.args.params['CD']['pose'] < len(confs):
          starting_pose = np.copy(confs[self.args.params['CD']['pose']])
          self.data['CD'].confs['starting_poses'] = [np.copy(starting_pose)]
        else:
          self._clear('CD')
          self._store_infinite_f_RL()
          raise Exception('Pose index greater than number of poses')
    else:
      starting_pose = None
    
    from chimpss.algdock.system import System
    self.system = System(self.args,
                         self.log,
                         self.top,
                         self.top_RL,
                         starting_pose=starting_pose)

    # Measure the binding site
    if (self.args.params['CD']['site'] == 'Measure'):
      self.args.params['CD']['site'] = 'Sphere'
      if self.args.params['CD']['site_measured'] is not None:
        (self.args.params['CD']['site_max_R'],self.args.params['CD']['site_center']) = \
          self.args.params['CD']['site_measured']
      else:
        print('\n*** Measuring the binding site ***')
        self.system.setParams(
          self.system.paramsFromAlpha(1.0, 'CD', site=False))
        (confs, Es) = self._get_confs_to_rescore(site=False, minimize=True)
        if len(confs) > 0:
          # Use the center of mass for configurations
          # within 20 RT of the lowest energy
          cutoffE = Es['total'][-1] + 20 * (R * self.T_TARGET)
          coms = []
          for (conf, E) in reversed(list(zip(confs, Es['total']))):
            if E <= cutoffE:
              if MMTK:
                self.top.universe.setConfiguration(
                Configuration(self.top.universe, conf))
                coms.append(np.array(self.top.universe.centerOfMass()))

              else:
                self.top.setConfiguration(conf)
                coms.append(self.top.centerOfMass())

            else:
              break
          print('  %d configurations fit in the binding site' % len(coms))
          coms = np.array(coms)
          center = (np.min(coms, 0) + np.max(coms, 0)) / 2
          max_R = max(
            np.ceil(np.max(np.sqrt(np.sum(
              (coms - center)**2, 1))) * 10.) / 10., 0.6)
          self.args.params['CD']['site_max_R'] = max_R
          self.args.params['CD']['site_center'] = center
          if MMTK:
            self.top.universe.setConfiguration(
            Configuration(self.top.universe, confs[-1]))
          else:
            self.top.setConfiguration(confs[-1])

        if ((self.args.params['CD']['site_max_R'] is None) or \
            (self.args.params['CD']['site_center'] is None)):
          raise Exception('No binding site parameters!')
        else:
          self.args.params['CD']['site_measured'] = \
            (self.args.params['CD']['site_max_R'], \
             self.args.params['CD']['site_center'])

      # Print the binding site parameters for debugging
      print('  Binding site parameters:')
      print('    site_center: %.5f %.5f %.5f' % tuple(self.args.params['CD']['site_center']))
      print('    site_max_R: %.2f' % self.args.params['CD']['site_max_R'])

    # Read the reference ligand and receptor coordinates
    import chimpss.algdock.IO
    IO_crd = chimpss.algdock.IO.crd()
    if self.args.FNs['inpcrd']['R'] is not None:
      if os.path.isfile(self.args.FNs['inpcrd']['L']):
        lig_crd = IO_crd.read(self.args.FNs['inpcrd']['L'], multiplier=0.1)
      self.data['CD'].confs['receptor'] = IO_crd.read(\
        self.args.FNs['inpcrd']['R'], multiplier=0.1)
    elif self.args.FNs['inpcrd']['RL'] is not None:
      complex_crd = IO_crd.read(self.args.FNs['inpcrd']['RL'], multiplier=0.1)
      if MMTK:
        lig_crd = complex_crd[self.top_RL.L_first_atom:self.top_RL.L_first_atom + \
        self.top.universe.numberOfAtoms(),:]
        self.data['CD'].confs['receptor'] = np.vstack(\
        (complex_crd[:self.top_RL.L_first_atom,:],\
         complex_crd[self.top_RL.L_first_atom + self.top.universe.numberOfAtoms():,:]))
      else:
        natoms = self.top.numberOfAtoms()
        lig_crd = complex_crd[self.top_RL.L_first_atom:self.top_RL.L_first_atom + natoms,:]
        self.data['CD'].confs['receptor'] = np.vstack(\
        (complex_crd[:self.top_RL.L_first_atom,:],\
         complex_crd[self.top_RL.L_first_atom + natoms:,:]))

    elif self.args.FNs['inpcrd']['L'] is not None:
      self.data['CD'].confs['receptor'] = None
      if os.path.isfile(self.args.FNs['inpcrd']['L']):
        lig_crd = IO_crd.read(self.args.FNs['inpcrd']['L'], multiplier=0.1)
    else:
      lig_crd = None

    if lig_crd is not None:
      if MMTK:
        self.data['CD'].confs['ligand'] = lig_crd[self.top.inv_prmtop_atom_order_L, :]
        self.top.universe.setConfiguration(\
        Configuration(self.top.universe,self.data['CD'].confs['ligand']))
      else:
        self.data['CD'].confs['ligand'] = lig_crd
        self.top.setConfiguration(self.data['CD'].confs['ligand'])
      if MMTK:
        if self.top_RL.universe is not None and self.data['CD'].confs['receptor'] is not None:
          self.top_RL.universe.setConfiguration(\
          Configuration(self.top_RL.universe, \
          np.vstack((self.data['CD'].confs['receptor'],self.data['CD'].confs['ligand']))))
      else:
        if self.top_RL.OMM_simulation is not None and self.data['CD'].confs['receptor'] is not None:
          # Always vstack receptor+ligand (like MMTK version) to ensure correct ordering
          vstacked = np.vstack((self.data['CD'].confs['receptor'],self.data['CD'].confs['ligand']))
          self.top_RL.setConfiguration(vstacked)

    if self.args.params['CD']['rmsd'] is not False:
      if self.args.params['CD']['rmsd'] is True:
        if lig_crd is not None:
          rmsd_crd = lig_crd[self.top.inv_prmtop_atom_order_L, :]
        else:
          raise Exception('Reference structure for rmsd calculations unknown')
      else:
        if MMTK:
          rmsd_crd = IO_crd.read(self.args.params['CD']['rmsd'], \
          natoms=self.top.universe.numberOfAtoms(), multiplier=0.1)
          rmsd_crd = rmsd_crd[self.top.inv_prmtop_atom_order_L, :]
        else:
          n_atoms = self.top.numberOfAtoms()
          rmsd_crd = IO_crd.read(self.args.params['CD']['rmsd'], natoms=n_atoms, multiplier=0.1)
      self.data['CD'].confs['rmsd'] = rmsd_crd

      self.get_rmsds.set_ref_configuration(self.data['CD'].confs['rmsd'])

    # If configurations are being rescored, start with a docked structure
    (confs, Es) = self._get_confs_to_rescore(site=False, minimize=False)
    if len(confs) > 0:
      if MMTK:
        self.top.universe.setConfiguration(
        Configuration(self.top.universe, confs[-1]))
      else:
        self.top.setConfiguration(confs[-1])

    from chimpss.algdock.simulation_iterator import SimulationIterator
    self.iterator = SimulationIterator(self.args, self.top, self.system)

    # Load progress
    from chimpss.algdock.postprocessing import Postprocessing
    Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run(readOnly=True)

    self.calc_f_L(readOnly=True)
    self.calc_f_RL(readOnly=True)

    if self.args.random_seed > 0:
      np.random.seed(self.args.random_seed)

  def _save_debug_trajectory(self, confs, filename, process='CD'):
    """Delegate to ConfigurationManager.save_debug_trajectory"""
    if not hasattr(self, '_config_manager'):
      self._config_manager = ConfigurationManager(self)
    return self._config_manager.save_debug_trajectory(confs, filename, process)

  def _get_scaling_factors(self, scaling_property):
    """Delegate to ConfigurationManager.get_scaling_factors"""
    if not hasattr(self, '_config_manager'):
      self._config_manager = ConfigurationManager(self)
    return self._config_manager.get_scaling_factors(scaling_property)

  def run(self, run_type):
    from chimpss.algdock.postprocessing import Postprocessing

    self.log.recordStart('run')
    self.log.run_type = run_type
    if run_type=='configuration_energies' or \
       run_type=='minimized_configuration_energies':
      self.configuration_energies(\
        minimize = (run_type=='minimized_configuration_energies'), \
        max_confs = 50)
    elif run_type == 'store_params':
      self.save('BC', keys=['progress'])
      self.save('CD', keys=['progress'])
    elif run_type == 'initial_BC':
      self.initial_BC()
    elif run_type == 'BC':  # Sample the BC process
      self.sim_process('BC')
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run([('BC', -1, -1, 'L')])
      self.calc_f_L()
    elif run_type == 'initial_CD':
      self.initial_CD()
    elif run_type == 'CD':  # Sample the CD process
      self.sim_process('CD')
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run()
      self.calc_f_RL()
      # self.targeted_FEP()
    elif run_type == 'timed':  # Timed replica exchange sampling
      BC_complete = self.sim_process('BC')
      if BC_complete:
        pp_complete = Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run([('BC', -1, -1, 'L')])
        if pp_complete:
          self.calc_f_L()
          CD_complete = self.sim_process('CD')
          if CD_complete:
            pp_complete = Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run()
            if pp_complete:
              self.calc_f_RL()
              # self.targeted_FEP()
    elif run_type == 'timed_BC':  # Timed BC only
      BC_complete = self.sim_process('BC')
      if BC_complete:
        pp_complete = Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run([('BC', -1, -1, 'L')])
        if pp_complete:
          self.calc_f_L()
    elif run_type == 'timed_CD':  # Timed CD only
      CD_complete = self.sim_process('CD')
      if CD_complete:
        pp_complete = Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run()
        if pp_complete:
          self.calc_f_RL()
          # self.targeted_FEP()
    elif run_type == 'postprocess':  # Postprocessing
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run()
    elif run_type == 'redo_postprocess':
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run(redo_CD=True)
    elif run_type == 'redo_pose_prediction':
      self.calc_f_RL(readOnly=True)
      # Predict native pose
      if self.args.params['CD']['pose'] == -1:
        (self.stats_RL['pose_inds'], self.stats_RL['scores']) = \
          self._get_pose_prediction()
        f_RL_FN = os.path.join(self.args.dir['CD'], 'f_RL.pkl.gz')
        self.log.tee(
          write_pkl_gz(f_RL_FN, (self.f_L, self.stats_RL, self.f_RL, self.B)))
      # self.targeted_FEP()
    elif (run_type == 'free_energies') or (run_type == 'redo_free_energies'):
      self.calc_f_L(redo=(run_type == 'redo_free_energies'))
      self.calc_f_RL(redo=(run_type == 'redo_free_energies'))
      # self.targeted_FEP()
    elif run_type == 'all':
      self.sim_process('BC')
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run([('BC', -1, -1, 'L')])
      self.calc_f_L()
      self.sim_process('CD')
      Postprocessing(self.args, self.log, self.top, self.top_RL, self.system, self.data, self.save).run()
      self.calc_f_RL()
      # self.targeted_FEP()
    elif run_type == 'render_docked':
      # For 4 figures
      # 1002*4/600. = 6.68 in at 600 dpi
      #  996*4/600. = 6.64 in at 600 dpi
      view_args = {'axes_off':True, 'size':[996,996], 'scale_by':0.80, \
                   'render':'TachyonInternal'}
      if hasattr(self, '_view_args_rotate_matrix'):
        view_args['rotate_matrix'] = getattr(self, '_view_args_rotate_matrix')
      self.show_samples(prefix='docked', \
        show_ref_ligand=True, show_starting_pose=True, \
        show_receptor=True, save_image=True, execute=True, quit=True, \
        view_args=view_args)
      if self.args.params['CD']['pose'] == -1:
        (self.stats_RL['pose_inds'], self.stats_RL['scores']) = \
          self._get_pose_prediction()
        self.show_pose_prediction(score='grid_fe_u',
          show_ref_ligand=True, show_starting_pose=False, \
          show_receptor=True, save_image=True, execute=True, quit=True, \
          view_args=view_args)
        self.show_pose_prediction(score='OpenMM_OBC2_fe_u',
          show_ref_ligand=True, show_starting_pose=False, \
          show_receptor=True, save_image=True, execute=True, quit=True, \
          view_args=view_args)
    elif run_type == 'render_intermediates':
      view_args = {'axes_off':True, 'size':[996,996], 'scale_by':0.80, \
                   'render':'TachyonInternal'}
      if hasattr(self, '_view_args_rotate_matrix'):
        view_args['rotate_matrix'] = getattr(self, '_view_args_rotate_matrix')
#      self.render_intermediates(\
#        movie_name=os.path.join(self.args.dir['CD'],'CD-intermediates.gif'), \
#        view_args=view_args)
      self.render_intermediates(nframes=8, view_args=view_args)
    elif run_type == 'clear_intermediates':
      for process in ['BC', 'CD']:
        print('Clearing intermediates for ' + process)
        for state_ind in range(1,
                               len(self.data[process].confs['samples']) - 1):
          for cycle_ind in range(
              len(self.data[process].confs['samples'][state_ind])):
            self.data[process].confs['samples'][state_ind][cycle_ind] = []
        self.save(process)
    if run_type is not None:
      print("\nElapsed time for execution of %s: %s" % (
        run_type, HMStime(self.log.timeSince('run'))))

  ###########
  # BC #
  ###########
  def initial_BC(self):
    """Delegate to BCCalculator.initial_BC"""
    if not hasattr(self, '_bc_calculator'):
      self._bc_calculator = BCCalculator(self)
    return self._bc_calculator.initial_BC()

  def calc_f_L(self, readOnly=False, do_solvation=True, redo=False):
    """Delegate to BCCalculator.calc_f_L"""
    if not hasattr(self, '_bc_calculator'):
      self._bc_calculator = BCCalculator(self)
    return self._bc_calculator.calc_f_L(readOnly, do_solvation, redo)

  ###########
  # Docking #
  ###########
  def initial_CD(self, randomOnly=False):
    """Delegate to CDCalculator.initial_CD"""
    if not hasattr(self, '_cd_calculator'):
      self._cd_calculator = CDCalculator(self)
    return self._cd_calculator.initial_CD(randomOnly)

  def calc_f_RL(self, readOnly=False, do_solvation=True, redo=False):
    """Delegate to CDCalculator.calc_f_RL"""
    if not hasattr(self, '_cd_calculator'):
      self._cd_calculator = CDCalculator(self)
    return self._cd_calculator.calc_f_RL(readOnly, do_solvation, redo)

  def _store_infinite_f_RL(self):
    if self.args.params['CD']['pose'] == -1:
      f_RL_FN = os.path.join(self.args.dir['CD'], 'f_RL.pkl.gz')
    else:
      f_RL_FN = os.path.join(self.args.dir['CD'],\
        'f_RL_pose%03d.pkl.gz'%self.args.params['CD']['pose'])
    self.log.tee(write_pkl_gz(f_RL_FN, (self.f_L, [], np.inf, np.inf)))

  def _get_equilibrated_cycle(self, process):
    """Delegate to FreeEnergyCalculator.get_equilibrated_cycle"""
    # Get the appropriate stats dict based on process
    if process == 'BC':
      stats_dict = self.stats_L if hasattr(self, 'stats_L') else {}
    elif process == 'CD':
      stats_dict = self.stats_RL if hasattr(self, 'stats_RL') else {}
    else:
      stats_dict = {}

    return FreeEnergyCalculator.get_equilibrated_cycle(
      self.data[process], stats_dict, process)

  def _get_rmsd_matrix(self):
    """Delegate to PoseAnalyzer.get_rmsd_matrix"""
    if not hasattr(self, '_pose_analyzer'):
      self._pose_analyzer = PoseAnalyzer(self)
    return self._pose_analyzer.get_rmsd_matrix()

  def _cluster_samples(self, rmsd_matrix):
    """Delegate to PoseAnalyzer.cluster_samples"""
    return PoseAnalyzer.cluster_samples(rmsd_matrix)

  def _get_pose_prediction(self, representative='medoid'):
    """Delegate to PoseAnalyzer.get_pose_prediction"""
    if not hasattr(self, '_pose_analyzer'):
      self._pose_analyzer = PoseAnalyzer(self)
    return self._pose_analyzer.get_pose_prediction(representative)

  def configuration_energies(self, minimize=False, max_confs=None):
    """Delegate to ConfigurationEnergyCalculator.configuration_energies"""
    if not hasattr(self, '_energy_calculator'):
      self._energy_calculator = ConfigurationEnergyCalculator(self)
    return self._energy_calculator.configuration_energies(minimize, max_confs)

  ######################
  # Internal Functions #
  ######################

  def sim_process(self, process):
    """
    Simulate and analyze a BC or CD process.

    As necessary, first conduct an initial BC or CD
    and then run a desired number of replica exchange cycles.
    """
    # Reset random seed for reproducibility
    # This ensures BC and CD phases start with deterministic random state
    if self.args.random_seed > 0:
      np.random.seed(self.args.random_seed)
      self.log.tee(f"  Reset random seed to {self.args.random_seed} for {process} simulation")

    if (self.data[process].protocol==[]) or \
       (not self.data[process].protocol[-1]['crossed']):
      time_left = getattr(self, 'initial_' + process)()
      if not time_left:
        return False

    # Main loop for replica exchange
    if (self.args.params[process]['repX_cycles'] is not None) and \
       ((self.data[process].cycle < \
         self.args.params[process]['repX_cycles'])):

      # Load configurations to score from another program
      if (process=='CD') and (self.data['CD'].cycle==1) and \
         (self.args.params['CD']['pose'] == -1) and \
         (self.args.FNs['score'] is not None) and \
         (self.args.FNs['score']!='default'):
        self.log.set_lock('CD')
        self.log.tee("\n>>> Reinitializing replica exchange configurations")
        self.system.setParams(self.system.paramsFromAlpha(1.0, 'CD'))
        confs = self._get_confs_to_rescore(\
          nconfs=len(self.data['CD'].protocol), site=True, minimize=True)[0]
        self.log.clear_lock('CD')
        if len(confs) > 0:
          self.data['CD'].confs['replicas'] = confs

      self.log.tee("\n>>> Replica exchange for {0}, starting at {1}\n".format(\
        process, time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())), \
        process=process)
      self.log.recordStart(process + '_repX_start')
      start_cycle = self.data[process].cycle
      cycle_times = []
      while (self.data[process].cycle <
             self.args.params[process]['repX_cycles']):
        from chimpss.algdock.replica_exchange import ReplicaExchange
        ReplicaExchange(self.args, self.log, self.top, self.system,
                      self.iterator, self.data, self.save, self._u_kln).run(process)
        self.SIRS(process)
        cycle_times.append(self.log.timeSince('repX cycle'))
        if process == 'CD':
          # Skip adaptive state insertion in testing mode for reproducibility
          if not self.args.params[process].get('test_disable_adaptive_insertion', False):
            self._insert_CD_state_between_low_acc()
          else:
            self.log.tee("  TEST-ONLY: Skipping adaptive CD state insertion for reproducibility")
        if not self.log.isTimeForTask(cycle_times):
          return False
      self.log.tee("Elapsed time for %d cycles of replica exchange: %s"%(\
         (self.data[process].cycle - start_cycle), \
          HMStime(self.log.timeSince(process+'_repX_start'))), \
          process=process)

    # If there are insufficient configurations,
    #   do additional replica exchange on the BC process
    if (process == 'BC'):
      E_MM = []
      for k in range(len(self.data['BC'].Es[0])):
        E_MM += list(self.data['BC'].Es[0][k]['MM'])
      while len(E_MM) < self.args.params['CD']['seeds_per_state']:
        self.log.tee(
          "More samples from high temperature ligand simulation needed",
          process='BC')
        from chimpss.algdock.replica_exchange import ReplicaExchange
        ReplicaExchange(self.args, self.log, self.top, self.system,
                      self.iterator, self.data, self.save, self._u_kln).run('BC')
        self.SIRS(process)
        cycle_times.append(self.log.timeSince('repX cycle'))
        if not self.log.isTimeForTask(cycle_times):
          return False
        E_MM = []
        for k in range(len(self.data['BC'].Es[0])):
          E_MM += list(self.data['BC'].Es[0][k]['MM'])

    # Clear evaluators to save memory
    self.system.clear_evaluators()

    return True  # The process has completed

  def SIRS(self, process):
    """Delegate to CDCalculator.SIRS"""
    if not hasattr(self, '_cd_calculator'):
      self._cd_calculator = CDCalculator(self)
    return self._cd_calculator.SIRS(process)

  def _insert_CD_state(self, alpha, clear=True):
    """Delegate to CDStateManager.insert_state"""
    if not hasattr(self, '_cd_state_manager'):
      self._cd_state_manager = CDStateManager(self)
    return self._cd_state_manager.insert_state(alpha, clear)

  def _insert_CD_state_between_low_acc(self):
    """Delegate to CDStateManager.insert_states_between_low_acc"""
    if not hasattr(self, '_cd_state_manager'):
      self._cd_state_manager = CDStateManager(self)
    return self._cd_state_manager.insert_states_between_low_acc()

  def _get_confs_to_rescore(self,
                            nconfs=None,
                            site=False,
                            minimize=True,
                            sort=True):
    """Delegate to ConfigurationLoader.get_confs_to_rescore"""
    if not hasattr(self, '_config_loader'):
      self._config_loader = ConfigurationLoader(self)
    return self._config_loader.get_confs_to_rescore(nconfs, site, minimize, sort)

  def _checkedMinimizer(self, confs):
    """Delegate to ConfigurationManager.checkedMinimizer"""
    if not hasattr(self, '_config_manager'):
      self._config_manager = ConfigurationManager(self)
    return self._config_manager.checkedMinimizer(confs)

  def run_MBAR(self, u_kln, N_k, augmented=False):
    """Delegate to FreeEnergyCalculator.run_MBAR"""
    return FreeEnergyCalculator.run_MBAR(u_kln, N_k, augmented)

  def _u_kln(self, eTs, protocol, noBeta=False):
    """Delegate to FreeEnergyCalculator.u_kln"""
    return FreeEnergyCalculator.u_kln(eTs, protocol, noBeta)

  def _clear_f_RL(self):
    """Delegate to CDStateManager.clear_f_RL"""
    if not hasattr(self, '_cd_state_manager'):
      self._cd_state_manager = CDStateManager(self)
    return self._cd_state_manager.clear_f_RL()

  def save(self, p, keys=['progress', 'data']):
    """Saves results

    Parameters
    ----------
    p : str
      The process, either 'BC' or 'CD'
    keys : list of str
      Save the progress, the data, or both
    """
    if 'progress' in keys:
      self.log.tee(self.args.save_pkl_gz(p, self.data[p]))
    if 'data' in keys:
      self.log.tee(self.data[p].save_pkl_gz())

  def __del__(self):
    if (not DEBUG) and len(self.args.toClear) > 0:
      print("\n>>> Clearing files")
      for FN in self.args.toClear:
        if os.path.isfile(FN):
          os.remove(FN)
          print('  removed ' + os.path.relpath(FN, self.args.dir['start']))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    description=
    'Molecular docking with adaptively scaled alchemical interaction grids')

  for key in arguments.args.keys():
    parser.add_argument('--' + key, **arguments.args[key])
  args = parser.parse_args()

  if args.run_type in ['render_docked', 'render_intermediates']:
    from chimpss.algdock.BindingPMF_plots import BPMF_plots
    self = BPMF_plots(**vars(args))
  else:
    self = BPMF(**vars(args))
