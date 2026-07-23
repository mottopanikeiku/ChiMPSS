import os, sys
import numpy as np
import copy

try:
  import MMTK
  from MMTK.ParticleProperties import Configuration
  from MMTK.ForceFields import ForceField
  from MMTK.ForceFields import Amber12SBForceField
except ImportError:
  MMTK = None

try:
  import openmm
  import openmm.unit as unit
  from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, NoCutoff
except ImportError:
  openmm = None

from chimpss.algdock.BindingPMF import scalables
from chimpss.algdock.BindingPMF import HMStime
import chimpss.algdock.IO
prmtop_IO = chimpss.algdock.IO.prmtop()
varnames = ['POINTERS','TITLE','ATOM_NAME','AMBER_ATOM_TYPE','CHARGE','MASS',\
            'NONBONDED_PARM_INDEX','LENNARD_JONES_ACOEF','LENNARD_JONES_BCOEF',\
            'ATOM_TYPE_INDEX','BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN',\
            'RADII','SCREEN']

# DEBUG FLAGS - Set to True to enable verbose debug output
DEBUG_ALPHA_G = False
DEBUG_GRID_PARAMS = False
DEBUG_ENERGYTERMS_BC = False
DEBUG_ENERGYTERMS_CD = False
DEBUG_SYSTEM_CACHE = False
DEBUG_ATOM_COUNT = False
DEBUG_SITE_FORCE = False

term_map = {
  # MMTK force names
  'cosine dihedral angle': 'MM',
  'electrostatic/pair sum': 'MM',
  'harmonic bond': 'MM',
  'harmonic bond angle': 'MM',
  'Lennard-Jones': 'MM',
  'OpenMM': 'MM',
  'OBC': 'OBC',
  'OBC_desolv': 'OBC',
  'site': 'site',
  'sLJr': 'sLJr',
  'sELE': 'sELE',
  'sLJa': 'sLJa',
  'LJr': 'LJr',
  'LJa': 'LJa',
  'ELE': 'ELE',
  'pose dihedral angle': 'k_angular_int',
  'pose external dihedral': 'k_angular_ext',
  'pose external distance': 'k_spatial_ext',
  'pose external angle': 'k_angular_ext',
  # Identity mappings for energyTerms() output keys (already aggregated category names)
  'MM': 'MM',
  'k_angular_int': 'k_angular_int',
  'k_angular_ext': 'k_angular_ext',
  'k_spatial_ext': 'k_spatial_ext',
  # OpenMM force class names
  'PoseInternalRestraint': 'k_angular_int',
  'PoseSpatialRestraint': 'k_spatial_ext',
  'PoseOrientationRestraint': 'k_angular_ext',
  'HarmonicBondForce': 'MM',
  'HarmonicAngleForce': 'MM',
  'PeriodicTorsionForce': 'MM',
  'NonbondedForce': 'MM',
  'CustomNonbondedForce': 'MM',
  'CustomBondForce': 'MM',
  'CustomAngleForce': 'MM',
  'CustomTorsionForce': 'MM',
  'CustomExternalForce': 'MM',
  'CustomCentroidBondForce': 'site',
  'CustomGBForce': 'OBC',
  'GBSAOBCForce': 'OBC'
}

class System:
  """Forces and masses

  ...

  Attributes
  ----------
  _evaluators : dict
  _forceFields : dict
  T : float
    Current temperature
  args : chimpss.algdock.simulation_arguments.SimulationArguments
    Simulation arguments
  log : chimpss.algdock.logger.Logger
    Simulation log
  top : chimpss.algdock.topology.Topology
    Topology of the ligand
  top_RL : chimpss.algdock.topology.Topology
    Topology of the complex
  starting_pose : numpy.array
    Starting pose used for simulations with an external restraint
  """
  def __init__(self, args, log, top, top_RL=None, starting_pose=None):
    """Initializes the class

    Parameters
    ----------
    args : chimpss.algdock.simulation_arguments.SimulationArguments
      Simulation arguments
    log : chimpss.algdock.logger.Logger
      Simulation log
    top : chimpss.algdock.topology.Topology
      Topology of the ligand
    top_RL : chimpss.algdock.topology.Topology
      Topology of the complex
    starting_pose : numpy.array
      Starting pose used for simulations with an external restraint
    """
    self.args = args
    self.log = log
    self.top = top   # old-> chimpss.algdock.topology.TopologyMMTK, now: chimpss.algdock.topology.TopologyUsingOpenMM
    self.top_RL = top_RL
    self.starting_pose = starting_pose

    self._evaluators = {}
    self._forceFields = {}

    # Molecular mechanics force fields
    if MMTK:
      self._forceFields['gaff'] = Amber12SBForceField(
      parameter_file=self.args.FNs['forcefield'],
      mod_files=self.args.FNs['frcmodList'])
    else:
      self._forceFields['gaff'] = None # Use ambertools to generate topology files, which include GAFF parameters.

  def _get_scaling_factors_openmm(self, scaling_property):
    """Calculate scaling factors from prmtop for OpenMM.

    Uses the same formulas as Pipeline/prmtop2database.py to calculate
    scaling factors directly from the prmtop file.

    Parameters
    ----------
    scaling_property : str
      The scaling property to extract (e.g., 'scaling_factor_electrostatic',
      'scaling_factor_LJr', 'scaling_factor_LJa')

    Returns
    -------
    np.array
      Per-atom scaling factors in prmtop order (for complex topology,
      includes zeros for receptor atoms)
    """
    ligand_prmtop = self.args.FNs['prmtop']['L']
    prmtop = prmtop_IO.read(ligand_prmtop, varnames)

    NATOM = prmtop['POINTERS'][0]
    NTYPES = prmtop['POINTERS'][1]

    if scaling_property == 'scaling_factor_electrostatic':
      # FIX: OpenMM grids are already in kJ/mol, so no 4.184 factor needed
      # MMTK grids stay in kcal/mol, so MMTK needs the 4.184 factor
      # AMBER prmtop files multiply the actual charge by 18.2223
      ligand_scaling_factors = np.array([charge / 18.2223
                                   for charge in prmtop['CHARGE']])

    elif scaling_property in ['scaling_factor_LJr', 'scaling_factor_LJa']:
      # Extract Lennard-Jones well depth and radii for each atom type
      LJ_radius = np.zeros(NTYPES, dtype=float)
      LJ_depth = np.zeros(NTYPES, dtype=float)
      for i in range(NTYPES):
        LJ_index = prmtop['NONBONDED_PARM_INDEX'][NTYPES * i + i] - 1
        if prmtop['LENNARD_JONES_ACOEF'][LJ_index] < 1.0e-6:
          LJ_radius[i] = 0
          LJ_depth[i] = 0
        else:
          factor = 2 * prmtop['LENNARD_JONES_ACOEF'][LJ_index] / prmtop['LENNARD_JONES_BCOEF'][LJ_index]
          LJ_radius[i] = pow(factor, 1.0 / 6.0) * 0.5
          LJ_depth[i] = prmtop['LENNARD_JONES_BCOEF'][LJ_index] / 2 / factor

      root_LJ_depth = np.sqrt(LJ_depth)
      LJ_diameter = LJ_radius * 2
      atom_type_indices = np.array([prmtop['ATOM_TYPE_INDEX'][atom_index] - 1
                                     for atom_index in range(NATOM)])

      if scaling_property == 'scaling_factor_LJr':
        # FIX: OpenMM grids are already converted to kJ/mol in GridForceOpenMM.__init__
        # MMTK grids stay in kcal/mol, so MMTK needs the 4.184 factor
        # OpenMM should NOT have the 4.184 factor to avoid double conversion
        # sqrt(epsilon) * (2*R)^6  (no 4.184 for OpenMM)
        ligand_scaling_factors = np.array([root_LJ_depth[type_idx] * (LJ_diameter[type_idx] ** 6)
                                     for type_idx in atom_type_indices])
      else:  # scaling_factor_LJa
        # FIX: Same for LJa
        # sqrt(epsilon) * (2*R)^3  (no 4.184 for OpenMM)
        ligand_scaling_factors = np.array([root_LJ_depth[type_idx] * (LJ_diameter[type_idx] ** 3)
                                     for type_idx in atom_type_indices])

    else:
      # Unknown scaling property, return all ones
      return np.ones(self.top.numberOfAtoms())

    # If we're using a complex topology, pad with zeros for receptor atoms
    # Grid forces should only apply to ligand atoms
    total_atoms = self.top.numberOfAtoms()
    if total_atoms > NATOM:
      # Complex topology - need to pad with zeros for receptor atoms
      complex_scaling_factors = np.zeros(total_atoms)
      ligand_start = self.top.L_first_atom
      ligand_end = ligand_start + NATOM
      complex_scaling_factors[ligand_start:ligand_end] = ligand_scaling_factors
      # self.log.tee(f"  DEBUG _get_scaling_factors: total_atoms={total_atoms}, ligand_atoms={NATOM}, ligand_range=[{ligand_start}:{ligand_end}]")
      # self.log.tee(f"  DEBUG _get_scaling_factors: non-zero scaling factors: {(complex_scaling_factors != 0).sum()}")
      return complex_scaling_factors
    else:
      # Ligand-only topology
      # self.log.tee(f"  DEBUG _get_scaling_factors: ligand-only topology with {NATOM} atoms")
      return ligand_scaling_factors

  def setParams(self, params):
    """Sets the universe evaluator to values appropriate for the parameters.

    Parameters
    ----------
    params : dict
      The elements in the dictionary params can be:
        MM - True, to turn on the Generalized AMBER force field
        site - True, to turn on the binding site
        sLJr - scaling of the soft Lennard-Jones repulsive grid
        sLJa - scaling of the soft Lennard-Jones attractive grid
        sELE - scaling of the soft electrostatic grid
        LJr - scaling of the Lennard-Jones repulsive grid
        LJa - scaling of the Lennard-Jones attractive grid
        ELE - scaling of the electrostatic grid
        k_angular_int - spring constant of flat-bottom wells for angular internal degrees of freedom (kJ/nm)
        k_spatial_ext - spring constant of flat-bottom wells for spatial external degrees of freedom (kJ/nm)
        k_angular_ext - spring constant of flat-bottom wells for angular external degrees of freedom (kJ/nm)
        T - the temperature in K
    """

    self.T = params['T']

    # Store current params for debug output in energyTerms
    self._current_params = params.copy()

    # Reuse evaluators that have been stored
    # Include topology ID to distinguish between ligand/complex/receptor systems
    if MMTK:
      topology_id = id(self.top.universe)
    else:
      # For OpenMM, use the number of atoms in the topology to distinguish topologies
      # NOTE: Don't use OMM_system.getNumParticles() because virtual particles from
      # force fields (like sphere sites) can increase the particle count beyond atom count
      topology_id = self.top.numberOfAtoms()

    # Store params for access in energyTerms
    self.params = params

    evaluator_key = str(topology_id) + ',' + ','.join(['%s:%s'%(k,params[k]) \
      for k in sorted(params.keys())])

    if MMTK:
      if evaluator_key in self._evaluators.keys():
        self.top.universe._evaluator[(None,None,None)] = \
          self._evaluators[evaluator_key]
    # In MMTK, these evaluators were associated with the universe (equal to OpenMM system)
        return
    else:
      # OpenMM: Cache systems by topology, update parameters in-place
      import openmm.unit as unit
      import numpy as np

      # Build cache key based on which forces are PRESENT, not their strengths
      topology_key = str(topology_id) + ',' + ','.join([
          'MM' if params.get('MM') else '',
          'OBC' if 'OBC' in params else '',
          'site' if params.get('site') else '',
          'grids' if any(k in params for k in ['LJr','LJa','ELE','sLJr','sELE','sLJa']) else ''
      ])

      if topology_key in self._evaluators.keys():
        # Restore cached system and context
        cached_data = self._evaluators[topology_key]

        # CRITICAL: Check if we're switching contexts (not just reusing current one)
        switching_contexts = (self.top.context is not cached_data['context'])

        if switching_contexts:
          # CRITICAL FIX: Save current positions BEFORE switching contexts
          # Without this, restoring a cached context reverts to old positions from
          # when that context was last used, breaking MCMC detailed balance
          current_state = self.top.context.getState(getPositions=True)
          current_positions = current_state.getPositions()

        # Restore cached objects
        self.top.OMM_system = cached_data['system']
        self.top.OMM_simulation = cached_data['simulation']
        self.top.context = cached_data['context']

        if switching_contexts:
          # Apply current positions to the restored cached context
          self.top.context.setPositions(current_positions)

        # Update grid force strengths in-place (don't rebuild!)
        if 'grid_forces' in cached_data:
            for scalable, grid_wrapper in cached_data['grid_forces'].items():
                if scalable in params:
                    grid_wrapper.update_strength_in_context(
                        grid_wrapper.gridforce_ref,
                        self.top.context,
                        params[scalable],
                        grid_wrapper.base_scaling_factors
                    )

        # Update OBC strength if present
        if 'OBC' in self._forceFields and 'OBC' in params:
            self._forceFields['OBC'].context = self.top.context
            self._forceFields['OBC'].set_strength(params['OBC'])

        return

    # Otherwise create a new evaluator
    fflist = []
    grid_forces_dict = {}  # Store grid force wrappers for caching
    site_force = None  # Store site force for caching
    if ('MM' in params.keys()) and params['MM']:
      fflist.append(self._forceFields['gaff'])
    if ('site' in params.keys()) and params['site']:
      # Set up the binding site in the force field
      append_site = True  # Whether to append the binding site to the force field
      if not 'site' in self._forceFields.keys():
        print(self.args.params['CD']['site'], self.args.params['CD']['site_center'], self.args.params['CD']['site_max_R'])
        if (self.args.params['CD']['site']=='Sphere') and \
           (self.args.params['CD']['site_center'] is not None) and \
           (self.args.params['CD']['site_max_R'] is not None):
          if MMTK:
            from chimpss.algdock.ForceFields.Sphere.Sphere import SphereForceField
            self._forceFields['site'] = SphereForceField(
              center=self.args.params['CD']['site_center'],
              max_R=self.args.params['CD']['site_max_R'],
              name='site')
          else:
            from chimpss.algdock.ForceFields.Sphere.SphereForceOpenMM import SphereForceOpenMM
            self._forceFields['site'] = SphereForceOpenMM(
              center=self.args.params['CD']['site_center'],
              max_R=self.args.params['CD']['site_max_R'],
              name='site')
        elif (self.args.params['CD']['site']=='Cylinder') and \
             (self.args.params['CD']['site_center'] is not None) and \
             (self.args.params['CD']['site_direction'] is not None):
          if MMTK:
            from chimpss.algdock.ForceFields.Cylinder.Cylinder import CylinderForceField
            self._forceFields['site'] = CylinderForceField(
              origin=self.args.params['CD']['site_center'],
              direction=self.args.params['CD']['site_direction'],
              max_Z=self.args.params['CD']['site_max_Z'],
              max_R=self.args.params['CD']['site_max_R'],
              name='site')
          else:
            # Cylinder not yet implemented for OpenMM
            print('Cylinder binding site not yet implemented for OpenMM!')
            append_site = False
        else:
          # Do not append the site if it is not defined
          print('Binding site not defined!')
          append_site = False
      if append_site:
        site_force = self._forceFields['site']
        fflist.append(site_force)

    # Add scalable terms
    for scalable in scalables:
      # Always add all scalable forces to maintain consistent force indices
      # Strength parameter controls activity (can be 0)
      should_add = scalable in params.keys()

      if should_add:
        # Load the force field if not already loaded
        should_load = not scalable in self._forceFields.keys()
        if should_load:
          if scalable == 'OBC':
            if MMTK:
              from chimpss.algdock.ForceFields.OBC.OBC import OBCForceField
              if self.args.params['CD']['solvation']=='Fractional' and \
                  ('ELE' in params.keys()):
                self.log.recordStart('grid_loading')
                self._forceFields['OBC'] = OBCForceField(\
                  desolvationGridFN=self.args.FNs['grids']['desolv'])
                self.log.tee('  %s grid loaded from %s in %s'%(scalable, \
                  os.path.basename(self.args.FNs['grids']['desolv']), \
                  HMStime(self.log.timeSince('grid_loading'))))
              else:
                self._forceFields['OBC'] = OBCForceField()
            else:
              # For OpenMM, use OpenMM's native OBC implementation
              from chimpss.algdock.ForceFields.OBC.OBCForceOpenMM import OBCForceOpenMM
              self._forceFields['OBC'] = OBCForceOpenMM(
                topology=self.top.molecule,
                system=self.top.OMM_system)
          else:  # Grids
            self.log.recordStart('grid_loading')
            grid_FN = self.args.FNs['grids'][{
              'sLJr': 'LJr',
              'sLJa': 'LJa',
              'sELE': 'ELE',
              'LJr': 'LJr',
              'LJa': 'LJa',
              'ELE': 'ELE'
            }[scalable]]
            grid_scaling_factor = 'scaling_factor_' + \
              {'sLJr':'LJr','sLJa':'LJa','sELE':'electrostatic', \
               'LJr':'LJr','LJa':'LJa','ELE':'electrostatic'}[scalable]

            # Determine the grid threshold
            if scalable == 'sLJr':
              grid_thresh = 10.0
            elif scalable == 'sELE':
              # The maximum value is set so that the electrostatic energy
              # less than or equal to the Lennard-Jones repulsive energy
              # for every heavy atom at every grid point

              if MMTK:
                scaling_factors_ELE = np.array([ \
                self.top.molecule.getAtomProperty(a, 'scaling_factor_electrostatic') \
                  for a in self.top.molecule.atomList()],dtype=float)
                scaling_factors_LJr = np.array([ \
                self.top.molecule.getAtomProperty(a, 'scaling_factor_LJr') \
                  for a in self.top.molecule.atomList()],dtype=float)
              else:
                """
                get the scaling_factors_ELE and scaling_factors_LJr using openmm instead of MMTK.
                Note that MMTK may provide the values in a different atom order compared to OpenMM.
                """
                from openmm.app import NoCutoff
                scaling_factors_ELE = []
                scaling_factors_LJr = []
                ligand_prmtop = self.args.FNs['prmtop']['L']
                prmtop = AmberPrmtopFile(ligand_prmtop)
                topology = prmtop.topology

                prmtop_alg = prmtop_IO.read(ligand_prmtop, varnames)
                NATOM = prmtop_alg['POINTERS'][0]
                NTYPES = prmtop_alg['POINTERS'][1]
                LJ_radius = np.ndarray(shape=(NTYPES), dtype=float)
                LJ_depth = np.ndarray(shape=(NTYPES), dtype=float)
                for i in range(NTYPES):
                  LJ_index = prmtop_alg['NONBONDED_PARM_INDEX'][NTYPES * i + i] - 1
                  if prmtop_alg['LENNARD_JONES_ACOEF'][LJ_index] < 1.0e-6:
                    LJ_radius[i] = 0
                    LJ_depth[i] = 0
                  else:
                    factor = 2 * prmtop_alg['LENNARD_JONES_ACOEF'][LJ_index] / prmtop_alg['LENNARD_JONES_BCOEF'][
                      LJ_index]
                    LJ_radius[i] = pow(factor, 1.0 / 6.0) * 0.5
                    LJ_depth[i] = prmtop_alg['LENNARD_JONES_BCOEF'][LJ_index] / 2 / factor
                root_LJ_depth = np.sqrt(LJ_depth)
                LJ_diameter = LJ_radius * 2
                atom_type_indicies = [prmtop_alg['ATOM_TYPE_INDEX'][atom_index] - 1 for atom_index in range(NATOM)]
                scaling_factor_LJr_dict = dict()
                for (name, type_index) in zip(prmtop_alg['ATOM_NAME'], atom_type_indicies):
                  # Scaling factors used to calculate sELE grid threshold
                  # Grids are already in kJ/mol, so no 4.184 conversion needed
                  scaling_factor_LJr_dict[name.strip()] = round(
                    root_LJ_depth[type_index] * (LJ_diameter[type_index] ** 6), 6)

                atoms_name = [a.name for a in topology.atoms()]
                atoms_elements = dict()
                for a in topology.atoms():
                  atoms_elements[a.name] = a.element.symbol
                system = prmtop.createSystem(nonbondedMethod=NoCutoff,
                                             constraints=None)
                force = system.getForce(3)
                assert force.getName() == 'NonbondedForce'
                for i in range(system.getNumParticles()):
                  atom_name = atoms_name[i]
                  charge, sigma, epsilon = force.getParticleParameters(i)
                  charge_val = charge.value_in_unit(unit.elementary_charge)
                  epsilon = epsilon.value_in_unit(unit.kilojoule / unit.mole)
                  if epsilon == 0:
                    continue
                  # Scaling factors used to calculate sELE grid threshold
                  # Grids are already in kJ/mol, so no 4.184 conversion needed
                  scaling_factors_ELE.append(round(charge_val, 6))
                  scaling_factors_LJr.append(scaling_factor_LJr_dict[atom_name])
                scaling_factors_ELE = np.array(scaling_factors_ELE)
                scaling_factors_LJr = np.array(scaling_factors_LJr)

              #---------
              toKeep = np.logical_and(scaling_factors_LJr > 10.,
                                      abs(scaling_factors_ELE) > 0.1)

              scaling_factors_ELE = scaling_factors_ELE[toKeep]
              scaling_factors_LJr = scaling_factors_LJr[toKeep]

              grid_thresh = min(
                abs(scaling_factors_LJr * 10.0 / scaling_factors_ELE))
            else:
              grid_thresh = -1  # There is no threshold for grid points

            if MMTK:
              from chimpss.algdock.ForceFields.Grid.Interpolation \
                import InterpolationForceField
              self._forceFields[scalable] = InterpolationForceField(grid_FN, \
                name=scalable, interpolation_type='Trilinear', \
                strength=params[scalable], scaling_property=grid_scaling_factor,
                inv_power=4 if scalable=='LJr' else None, \
                grid_thresh=grid_thresh)
            else:
              # OpenMM implementation
              from chimpss.algdock.ForceFields.Grid.GridForceOpenMM import GridForceOpenMM
              self._forceFields[scalable] = GridForceOpenMM(grid_FN, \
                name=scalable, \
                strength=params[scalable], scaling_property=grid_scaling_factor,
                inv_power=4 if scalable=='LJr' else None, \
                grid_thresh=grid_thresh)
              # Store grid force wrapper for caching
              grid_forces_dict[scalable] = self._forceFields[scalable]
            self.log.tee('  %s grid loaded from %s in %s'%(scalable, \
              os.path.basename(grid_FN), \
              HMStime(self.log.timeSince('grid_loading'))))

        # Set the force field strength to the desired value
        self._forceFields[scalable].set_strength(params[scalable])
        fflist.append(self._forceFields[scalable])

    if ('k_angular_int' in params.keys()) or \
       ('k_spatial_ext' in params.keys()) or \
       ('k_angular_ext' in params.keys()):

      # Load the force field if it has not been loaded
      if MMTK:
        if not ('ExternalRestraint' in self._forceFields.keys()):
          Xo = np.copy(self.top.universe.configuration().array)
          self.top.universe.setConfiguration(
            Configuration(self.top.universe, self.starting_pose))
          import chimpss.algdock.rigid_bodies
          rb = chimpss.algdock.rigid_bodies.identifier(self.top.universe,
                                               self.top.molecule)
          (TorsionRestraintSpecs, ExternalRestraintSpecs) = rb.poseInp()
          self.top.universe.setConfiguration(Configuration(
            self.top.universe, Xo))

          # Create force fields
          from chimpss.algdock.ForceFields.Pose.PoseFF import InternalRestraintForceField
          self._forceFields['InternalRestraint'] = \
            InternalRestraintForceField(TorsionRestraintSpecs)
          from chimpss.algdock.ForceFields.Pose.PoseFF import ExternalRestraintForceField
          self._forceFields['ExternalRestraint'] = \
            ExternalRestraintForceField(*ExternalRestraintSpecs)
      else:
        # OpenMM pose restraints
        if not ('ExternalRestraint' in self._forceFields.keys()):
          # Use rigid_bodies_openmm to analyze structure
          import chimpss.algdock.rigid_bodies_openmm as rigid_bodies_openmm
          rb = rigid_bodies_openmm.RigidBodyIdentifier(
            self.top.molecule, self.starting_pose)
          (TorsionRestraintSpecs, ExternalRestraintSpecs) = rb.poseInp()

          # Create pose restraint force wrappers
          # Forces will be added to system during OpenMM system recreation
          from chimpss.algdock.ForceFields.Pose.PoseFFOpenMM import InternalRestraintForceOpenMM
          self._forceFields['InternalRestraint'] = \
            InternalRestraintForceOpenMM(TorsionRestraintSpecs, k=1.0)

          from chimpss.algdock.ForceFields.Pose.PoseFFOpenMM import ExternalRestraintForceOpenMM_PathA
          self._forceFields['ExternalRestraint'] = \
            ExternalRestraintForceOpenMM_PathA(
              self.top.molecule, ExternalRestraintSpecs,
              k_spatial=1.0, k_angular=1.0)

      # Set parameter values
      if ('k_angular_int' in params.keys()):
        self._forceFields['InternalRestraint'].set_k(\
          params['k_angular_int'])
        fflist.append(self._forceFields['InternalRestraint'])

      if ('k_spatial_ext' in params.keys()):
        self._forceFields['ExternalRestraint'].set_k_spatial(\
          params['k_spatial_ext'])
        fflist.append(self._forceFields['ExternalRestraint'])

      if ('k_angular_ext' in params.keys()):
        self._forceFields['ExternalRestraint'].set_k_angular(\
          params['k_angular_ext'])

    if MMTK:
      compoundFF = fflist[0]
      for ff in fflist[1:]:
        compoundFF += ff
      self.top.universe.setForceField(compoundFF)

      if self.top_RL.universe is not None:
        if 'OBC_RL' in params.keys():
          if not 'OBC_RL' in self._forceFields.keys():
            from chimpss.algdock.ForceFields.OBC.OBC import OBCForceField
            self._forceFields['OBC_RL'] = OBCForceField()
          self._forceFields['OBC_RL'].set_strength(params['OBC_RL'])
          if (params['OBC_RL'] > 0):
            self.top_RL.universe.setForceField(self._forceFields['OBC_RL'])

      eval = ForceField.EnergyEvaluator(\
        self.top.universe, self.top.universe._forcefield, None, None, None, None)
      eval.key = evaluator_key
      self.top.universe._evaluator[(None, None, None)] = eval
      self._evaluators[evaluator_key] = eval
    else:
      # OpenMM force field management
      # For OpenMM, we need to rebuild the system with the requested forces
      # Save current positions (only the real atoms, not dummy particles)
      old_positions = self.top.context.getState(getPositions=True).getPositions()
      # Extract only positions for the real atoms (first numberOfAtoms positions)
      real_atom_positions = old_positions[:self.top.numberOfAtoms()]

      # Create a new system with the base forces (from prmtop)
      from openmm.app import NoCutoff, OBC2
      prmtop = AmberPrmtopFile(self.args.FNs['prmtop']['L'])

      # Recreate system with MM forces if requested
      if ('MM' in params.keys()) and params['MM']:
        # Determine if we're in BC or CD phase by checking for CD-specific grid keys
        is_cd_phase = any(key in params for key in ['sLJr', 'LJr', 'LJa', 'ELE', 'sELE'])

        # Create system with OBC to maintain consistent force indices
        # Use set_strength() to control OBC activity (0 = inactive, >0 = active)
        # This eliminates force index shifts between minimization and sampling
        use_obc = 'OBC' in params

        if use_obc:
          # Use NoCutoff for all forces (most accurate, no cutoff artifacts)
          self.top.OMM_system = prmtop.createSystem(
            nonbondedMethod=NoCutoff,
            constraints=None,
            implicitSolvent=OBC2)

          # Update OBC wrapper's force reference after system recreation
          if 'OBC' in self._forceFields:
            obc_wrapper = self._forceFields['OBC']
            # Find the new GBSAOBCForce in the recreated system
            for i in range(self.top.OMM_system.getNumForces()):
              f = self.top.OMM_system.getForce(i)
              if isinstance(f, openmm.GBSAOBCForce):
                obc_wrapper.force = f
                obc_wrapper.force_index = i
                obc_wrapper.system = self.top.OMM_system
                # Clear cached parameters since the force has changed
                if hasattr(obc_wrapper, '_original_charges'):
                  delattr(obc_wrapper, '_original_charges')
                  delattr(obc_wrapper, '_original_radii')
                  delattr(obc_wrapper, '_original_scale_factors')
                # Note: set_strength() will be called after context is created
                break
        else:
          self.top.OMM_system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None)
      else:
        # Create minimal system without MM forces
        self.top.OMM_system = openmm.System()
        for i in range(self.top.numberOfAtoms()):
          mass = prmtop.topology._chains[0]._residues[0]._atoms[i].element.mass
          self.top.OMM_system.addParticle(mass)

      # Add grid forces and other custom forces
      # Track any dummy particles added by forces (e.g., SphereForce)
      dummy_particle_positions = []
      for ff in fflist:
        if hasattr(ff, 'add_to_system'):
          # Different force types have different add_to_system signatures
          if hasattr(ff, 'scaling_property'):
            # GridForceOpenMM - needs scaling factors
            scaling_factors = self._get_scaling_factors_openmm(ff.scaling_property)
            # If topology is ligand-only but scaling_factors are padded for complex,
            # extract just the ligand portion
            n_topology_atoms = self.top.molecule.getNumAtoms()
            if len(scaling_factors) > n_topology_atoms:
              # Complex scaling factors, extract ligand portion
              ligand_start = self.top.L_first_atom if hasattr(self.top, 'L_first_atom') else 0
              ligand_end = ligand_start + n_topology_atoms
              scaling_factors = scaling_factors[ligand_start:ligand_end]
            force_idx = ff.add_to_system(self.top.OMM_system, self.top.molecule, scaling_factors)
            # DEBUG: Print force index for grid forces
            # if hasattr(ff, 'name'):
            #   self.log.tee(f"DEBUG setParams: Added grid force '{ff.name}' at index {force_idx}")
            #   self.log.tee(f"  scaling_property: {ff.scaling_property}")
            #   self.log.tee(f"  scaling_factors (first 5): {scaling_factors[:5]}")
            #   self.log.tee(f"  scaling_factors (min/max/mean): {scaling_factors.min():.1f}/{scaling_factors.max():.1f}/{scaling_factors.mean():.1f}")
            #   self.log.tee(f"  _final_scaling: {ff._final_scaling}")
            #   self.log.tee(f"  strength: {ff.strength}")
          else:
            # SphereForceOpenMM and other forces - don't need scaling factors
            ff.add_to_system(self.top.OMM_system, self.top.molecule)
            # Check if this force added a dummy particle (e.g., SphereForce)
            if hasattr(ff, 'dummy_particle_index') and hasattr(ff, 'center'):
              dummy_particle_positions.append(ff.center * unit.nanometer)


      # Recreate the context with the new system
      integrator = openmm.LangevinIntegrator(params['T'] * unit.kelvin,
                                             1 / unit.picosecond,
                                             0.002 * unit.picoseconds)
      from openmm.app import Simulation, Topology, Element

      # Set force groups BEFORE creating context
      num_forces = self.top.OMM_system.getNumForces()
      for i in range(num_forces):
        self.top.OMM_system.getForce(i).setForceGroup(i)

      # Create topology with dummy particles if needed
      if self.top.OMM_system.getNumParticles() > self.top.molecule.getNumAtoms():
        # System has more particles than topology (due to dummy particles)
        # Create a new topology that includes the extra particles
        modified_topology = Topology()
        # Copy existing topology
        for chain in self.top.molecule.chains():
          new_chain = modified_topology.addChain(chain.id)
          for residue in chain.residues():
            new_residue = modified_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
              modified_topology.addAtom(atom.name, atom.element, new_residue)

        # Add dummy particles
        dummy_chain = modified_topology.addChain('X')
        dummy_residue = modified_topology.addResidue('DUM', dummy_chain)
        num_dummies = self.top.OMM_system.getNumParticles() - self.top.molecule.getNumAtoms()
        for i in range(num_dummies):
          modified_topology.addAtom(f'D{i}', Element.getBySymbol('H'), dummy_residue)

        topology = modified_topology
      else:
        topology = self.top.molecule

      self.top.OMM_simulation = Simulation(topology,
                                           self.top.OMM_system,
                                           integrator)

      # Store context reference in OBC wrapper for parameter updates
      if 'OBC' in self._forceFields:
        self._forceFields['OBC'].context = self.top.OMM_simulation.context
        # Now apply the OBC strength setting with the new context
        if 'OBC' in params:
          self._forceFields['OBC'].set_strength(params['OBC'])

      # Update pose restraint parameters in context if present
      if 'InternalRestraint' in self._forceFields and 'k_angular_int' in params:
        self._forceFields['InternalRestraint'].updateParametersInContext(
          self.top.OMM_simulation.context)
      if 'ExternalRestraint' in self._forceFields:
        if 'k_spatial_ext' in params or 'k_angular_ext' in params:
          self._forceFields['ExternalRestraint'].updateParametersInContext(
            self.top.OMM_simulation.context)

      # Set positions: real atoms + any dummy particles
      all_positions = list(real_atom_positions) + dummy_particle_positions

      self.top.OMM_simulation.context.setPositions(all_positions)
      self.top.context = self.top.OMM_simulation.context

      # Cache the system and context to avoid expensive recreation
      # Use topology_key (not evaluator_key) so we can reuse systems across parameter changes
      if DEBUG_SYSTEM_CACHE:
          print(f"DEBUG: Caching system with site_force={site_force}")
          if site_force:
              print(f"  site_force.center={site_force.center}")
      self._evaluators[topology_key] = {
        'system': self.top.OMM_system,
        'simulation': self.top.OMM_simulation,
        'context': self.top.context,
        'grid_forces': grid_forces_dict,  # Store grid force wrappers for parameter updates
        'site_force': site_force  # Store site force for dummy particle position
      }

  def energyTerms(self, confs, E=None, process='CD'):
    """Calculates energy terms for a series of configurations

    Units are kJ/mol.

    Parameters
    ----------
    confs : list of np.array
      Configurations
    E : dict of np.array
      Dictionary to add to
    process : str
      Process, either 'BC' or 'CD'

    Returns
    -------
    E : dict of np.array
      Dictionary of energy terms
    """
    # PROFILING: Add timing for energyTerms
    import time
    time_start_energyTerms = time.time()
    profile_energyTerms = not hasattr(self, '_energyTerms_profiled')
    if profile_energyTerms:
      self._energyTerms_profiled = True
      timings = {
        'setup': 0,
        'setPositions': [],
        'getState_per_force': [],
        'total_per_conf': []
      }
      time_setup_start = time.time()

    if E is None:
      E = {}

    # Save current context/simulation state before energyTerms modifies it
    saved_context = self.top.context if hasattr(self.top, 'context') else None
    saved_simulation = self.top.OMM_simulation if hasattr(self.top, 'OMM_simulation') else None

    # Save current params to restore after energyTerms completes
    # This ensures debug output in subsequent calls shows actual sampling params
    saved_params = self._current_params.copy() if hasattr(self, '_current_params') else None

    # Save current params for debug output (before we override with params_full)
    # This allows debug output to show scaled energies rather than full-strength
    if hasattr(self, '_current_params'):
      self._params_before_energyTerms = self._current_params.copy()
    else:
      self._params_before_energyTerms = {}

    params_full = self.paramsFromAlpha(alpha=1.0,
                                               process=process,
                                               site=(process == 'CD'))
    if process == 'CD':
      for scalable in scalables:
        params_full[scalable] = 1
    self.setParams(params_full)

    # Molecular mechanics and grid interaction energies
    E['MM'] = np.zeros(len(confs), dtype=float)
    if process == 'BC':
      if 'OBC' in params_full.keys():
        E['OBC'] = np.zeros(len(confs), dtype=float)
    if process == 'CD':
      for term in (scalables):
        E[term] = np.zeros(len(confs), dtype=float)
      if self.isForce('site'):
        E['site'] = np.zeros(len(confs), dtype=float)
      if self.isForce('InternalRestraint'):
        E['k_angular_int'] = np.zeros(len(confs), dtype=float)
      if self.isForce('ExternalRestraint'):
        E['k_angular_ext'] = np.zeros(len(confs), dtype=float)
        E['k_spatial_ext'] = np.zeros(len(confs), dtype=float)

    if profile_energyTerms:
      timings['setup'] = time.time() - time_setup_start

    if MMTK:
      for c in range(len(confs)):
        self.top.universe.setConfiguration(
          Configuration(self.top.universe, confs[c]))
        eT = self.top.universe.energyTerms()
        for (key, value) in eT.items():
          if key == 'electrostatic':
            pass  # For some reason, MMTK double-counts electrostatic energies
          elif key.startswith('pose'):
            # For pose restraints, the energy is per spring constant unit
            E[term_map[key]][c] += value / params_full[term_map[key]]
          else:
            try:
              E[term_map[key]][c] += value
            except KeyError:
              print(key)
              print('Keys in eT', eT.keys())
              print('Keys in term map', term_map.keys())
              print('Keys in E', E.keys())
              raise Exception('key not found in term map or E')
      return E
    else:
      # Force groups are already set in setParams() for performance
      force_groups = self.top.OMM_system.getNumForces()

      for c in range(len(confs)):
        if profile_energyTerms:
          time_conf_start = time.time()
          time_setPos_start = time.time()

        # Reorder atoms from MMTK order to prmtop order for OpenMM
        # confs are in MMTK order (from sampling), but OpenMM expects prmtop order
        # Handle virtual particles separately - they come after real atoms
        num_real_atoms = len(self.top.prmtop_atom_order_L)
        num_particles = confs[c].shape[0]

        if c == 0 and DEBUG_ATOM_COUNT:
            print(f"DEBUG atom counting: num_real_atoms={num_real_atoms}, num_particles={num_particles}")

        if num_particles > num_real_atoms:
          # System has virtual particles (e.g., centroid pseudo-atom)
          # Reorder real atoms, then append virtual particles unchanged
          conf_prmtop = np.vstack([
            confs[c][self.top.prmtop_atom_order_L, :],  # Reordered real atoms
            confs[c][num_real_atoms:, :]  # Virtual particles (unchanged)
          ])
        else:
          # No virtual particles, just reorder
          conf_prmtop = confs[c][self.top.prmtop_atom_order_L, :]

        # Append dummy particle position if site restraint exists
        # The dummy particle represents the fixed sphere center
        # CRITICAL: Use same topology_id calculation as setParams for cache consistency
        if MMTK:
            topology_id = id(self.top.universe)
        else:
            topology_id = self.top.numberOfAtoms()

        topology_key = str(topology_id) + ',' + ','.join([
            'MM' if self.params.get('MM') else '',
            'OBC' if 'OBC' in self.params else '',
            'site' if self.params.get('site') else '',
            'grids' if any(k in self.params for k in ['LJr','LJa','ELE','sLJr','sELE','sLJa']) else ''
        ])

        if c == 0 and DEBUG_SYSTEM_CACHE:
            print(f"DEBUG energyTerms: topology_key={topology_key}")
            print(f"  topology_key in _evaluators? {topology_key in self._evaluators}")
            if topology_key in self._evaluators:
                cached_data = self._evaluators[topology_key]
                print(f"  site_force in cached_data? {'site_force' in cached_data}")
                if 'site_force' in cached_data:
                    print(f"  site_force is not None? {cached_data['site_force'] is not None}")

        if topology_key in self._evaluators:
            cached_data = self._evaluators[topology_key]
            if 'site_force' in cached_data and cached_data['site_force'] is not None:
                # Check if dummy particle needs to be appended
                # The system has num_real_atoms + 1 dummy particle
                expected_particles = num_real_atoms + 1
                current_particles = conf_prmtop.shape[0]

                if c == 0 and DEBUG_ATOM_COUNT:
                    print(f"DEBUG site dummy check:")
                    print(f"  System expects: {expected_particles} particles ({num_real_atoms} real + 1 dummy)")
                    print(f"  Current conf has: {current_particles} particles")

                if current_particles == expected_particles:
                    # Configuration already has dummy (normal case when sampled with site force)
                    pass
                elif current_particles == num_real_atoms:
                    # Configuration only has real atoms (e.g., from initialization without site force)
                    # Need to append the dummy
                    conf_prmtop = np.vstack([
                        conf_prmtop,
                        cached_data['site_force'].center.reshape(1, 3)
                    ])
                else:
                    # Unexpected number of particles - this IS a bug!
                    raise Exception(
                        f"BUG: Configuration has {current_particles} particles, "
                        f"but expected either {num_real_atoms} (real atoms only) "
                        f"or {expected_particles} (real + dummy). "
                        f"This suggests a mismatch in system configuration!"
                    )

        self.top.OMM_simulation.context.setPositions(conf_prmtop)

        if profile_energyTerms:
          timings['setPositions'].append(time.time() - time_setPos_start)

        for i in range(force_groups):
          if profile_energyTerms:
            time_getState_start = time.time()

          # Get force first to query its actual force group
          # Forces can be reordered, so force at index i might not be in group i
          force = self.top.OMM_system.getForce(i)
          actual_force_group = force.getForceGroup()

          # Check dummy particle position and COM for site force
          if DEBUG_SITE_FORCE and c < 3 and i == 6:  # Force 6 is site force
            print(f"[DEBUG] Config {c}, Force {i}: {force.__class__.__name__}")

          if DEBUG_SITE_FORCE and force.__class__.__name__ == 'CustomCentroidBondForce' and c < 3:
            state_with_pos = self.top.OMM_simulation.context.getState(getEnergy=True, getPositions=True, groups={actual_force_group})
            positions = state_with_pos.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

            # Calculate COM of real atoms (assuming last particle is dummy)
            real_atoms_pos = positions[:-1]
            dummy_pos = positions[-1]

            # Get masses for COM calculation
            masses = []
            for atom_idx in range(len(real_atoms_pos)):
              mass = self.top.OMM_system.getParticleMass(atom_idx).value_in_unit(unit.dalton)
              masses.append(mass)
            masses = np.array(masses)

            com = np.average(real_atoms_pos, axis=0, weights=masses)
            distance = np.linalg.norm(com - dummy_pos)

            print(f"\n[DEBUG SITE FORCE] Config {c}:")
            print(f"  COM (real atoms): [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] nm")
            print(f"  Dummy position: [{dummy_pos[0]:.4f}, {dummy_pos[1]:.4f}, {dummy_pos[2]:.4f}] nm")
            print(f"  Distance: {distance:.4f} nm")
            print(f"  Max_R: 1.5 nm")
            print(f"  Expected energy if outside: {0.5 * 10000 * max(0, distance - 1.5)**2:.2f} kJ/mol")

          group_state = self.top.OMM_simulation.context.getState(getEnergy=True, groups={actual_force_group})

          if profile_energyTerms:
            timings['getState_per_force'].append(time.time() - time_getState_start)

          group_energy = group_state.getPotentialEnergy()
          force_name = force.__class__.__name__

          # DEBUG: Print all forces for first and last config only
          # For BC, this reduces output from N configs to just 2 per energyTerms call
          debug_this_config = (c == 0 or c == len(confs) - 1)
          if DEBUG_ENERGYTERMS_BC and debug_this_config and process == 'BC':
            import openmm.unit as unit
            energy_val = group_energy.value_in_unit(unit.kilojoule_per_mole) if hasattr(group_energy, 'value_in_unit') else group_energy
            force_group = force.getForceGroup()
            print(f"[energyTerms BC c={c}/{len(confs)-1}] Force {i}: {force_name}, forceGroup={force_group}, energy={energy_val:.2f} kJ/mol")
          elif DEBUG_ENERGYTERMS_CD and c == 0 and process == 'CD':
            # For CD, print scaled energy if this is a grid force
            import openmm.unit as unit
            energy_val = group_energy.value_in_unit(unit.kilojoule_per_mole) if hasattr(group_energy, 'value_in_unit') else group_energy
            force_group = force.getForceGroup()

            # Get the force name early to check if it's a grid force
            temp_force_name = force.__class__.__name__
            if temp_force_name == 'Force':
              temp_force_name = force.getName()

            # Apply scaling factor if this is a grid force
            scaling_factor = None
            if hasattr(self, '_params_before_energyTerms') and temp_force_name in self._params_before_energyTerms:
              scaling_factor = self._params_before_energyTerms[temp_force_name]
              scaled_energy = energy_val * scaling_factor
              print(f"[energyTerms CD] Force {i}: {temp_force_name}, forceGroup={force_group}, energy={energy_val:.2f} kJ/mol (full), {scaled_energy:.2f} kJ/mol (scaled by {scaling_factor:.4f})")
            else:
              print(f"[energyTerms CD] Force {i}: {temp_force_name}, forceGroup={force_group}, energy={energy_val:.2f} kJ/mol")

          # For generic "Force" objects (like grid forces from GridForce plugin),
          # we can identify them by their custom name set via setName()
          if force_name == 'Force':
            # Get the custom name we set in GridForceOpenMM.add_to_system()
            custom_name = force.getName()
            # if debug_forces:
            #   self.log.tee(f"  Force {i} is generic 'Force' with name '{custom_name}'")

            # Use the custom name to identify which grid this is
            # Names are: 'LJr', 'LJa', 'ELE', 'sLJr', 'sLJa', 'sELE'
            if custom_name in ['LJr', 'LJa', 'ELE', 'sLJr', 'sLJa', 'sELE']:
              force_name = custom_name
              # Make sure it's in term_map and E
              if force_name not in term_map:
                term_map[force_name] = force_name
              if force_name not in E:
                E[force_name] = np.zeros(len(confs))
            else:
              # Fallback to generic 'grid' if name doesn't match expected patterns
              force_name = 'grid'
              if 'grid' not in term_map:
                term_map['grid'] = 'grid'
              if 'grid' not in E:
                E['grid'] = np.zeros(len(confs))

          # CustomCentroidBondForce (site restraint) also has a custom name
          elif force_name == 'CustomCentroidBondForce':
            custom_name = force.getName()
            if custom_name:  # Use custom name if set (e.g., 'Sphere', 'site')
              force_name = 'site'  # Map to 'site' term
              # Make sure 'site' is in term_map and E
              if 'site' not in term_map:
                term_map['site'] = 'site'
              if 'site' not in E:
                E['site'] = np.zeros(len(confs))

          # Skip forces that don't contribute to energy (like CMMotionRemover)
          if force_name == 'CMMotionRemover':
            continue

          try:
            # Convert OpenMM Quantity to float (kJ/mol)
            if hasattr(group_energy, 'value_in_unit'):
              import openmm.unit as unit
              energy_val = group_energy.value_in_unit(unit.kilojoule_per_mole)
              E[term_map[force_name]][c] += energy_val
              # if debug_forces:
              #   self.log.tee(f"  Force {i} ({force_name} -> {term_map[force_name]}): {energy_val:.2f} kJ/mol")
            else:
              E[term_map[force_name]][c] += group_energy
              # if debug_forces:
              #   self.log.tee(f"  Force {i} ({force_name} -> {term_map[force_name]}): {group_energy} (no units)")
          except KeyError:
            print(force_name)
            print('Keys in term map', term_map.keys())
            print('Keys in E', E.keys())
            raise Exception('key not found in term map or E')

        if profile_energyTerms:
          timings['total_per_conf'].append(time.time() - time_conf_start)

      # Write profiling results
      # if profile_energyTerms:
      #   total_time = time.time() - time_start_energyTerms
      #   with open('energyTerms_profiling.txt', 'w') as f:
      #     f.write(f"\n{'='*70}\n")
      #     f.write(f"energyTerms() PROFILING (first call, {len(confs)} configurations)\n")
      #     f.write(f"{'='*70}\n")
      #     f.write(f"  setup time                : {timings['setup']*1000:8.1f} ms\n")
      #     f.write(f"  setPositions (per conf)   : avg={np.mean(timings['setPositions'])*1000:7.3f} ms, total={np.sum(timings['setPositions'])*1000:8.1f} ms\n")
      #     f.write(f"  getState calls            : {len(timings['getState_per_force'])} total calls\n")
      #     f.write(f"  getState (per call)       : avg={np.mean(timings['getState_per_force'])*1000:7.3f} ms, total={np.sum(timings['getState_per_force'])*1000:8.1f} ms\n")
      #     f.write(f"  total per configuration   : avg={np.mean(timings['total_per_conf'])*1000:7.1f} ms\n")
      #     f.write(f"  TOTAL TIME                : {total_time*1000:8.1f} ms ({total_time:.2f} s)\n")
      #     f.write(f"{'='*70}\n\n")
      #   print(f"\n*** energyTerms() profiling results written to energyTerms_profiling.txt ***\n")

      # Restore the original params state
      # This ensures subsequent energyTerms calls have correct debug output
      if saved_params is not None:
        self._current_params = saved_params

      # Restore the original context/simulation state
      if saved_context is not None:
        self.top.context = saved_context
      if saved_simulation is not None:
        self.top.OMM_simulation = saved_simulation

      return E

  def paramsFromAlpha(self,
                       alpha,
                       process='CD',
                       params_o=None,
                       site=True,
                       crossed=False):
    """Creates a parameter dictionary for a given alpha value

    Parameters
    ----------
    alpha : float
      Progress variable for the thermodynamic state.
      For CD: alpha=0.0 -> grids OFF (decoupled), alpha=1.0 -> grids ON (coupled).
      Hard grids (LJr/LJa/ELE) scale 0->1 with alpha_g(alpha).
      Soft grids (sLJr/sELE) peak at alpha=0.5 via alpha_sg(alpha).
    process : str
      Process, either 'BC' or 'CD'
    params_o : dict of float
      Parameter dictionary to modify
    site : bool
      If True, the ligand is restricted to the binding site
    crossed : bool
      If True, the initial protocol is complete

    Returns
    -------
    params : dict of float
      Parameter dictionary that corresponds to the given alpha
    """
    if (params_o is not None):
      params = copy.deepcopy(params_o)
      if 'steps_per_trial' not in params_o.keys():
        params[
          'steps_per_trial'] = 1 * self.args.params[process]['steps_per_sweep']
    else:
      params = {}
      params[
        'steps_per_trial'] = 1 * self.args.params[process]['steps_per_sweep']

    params['MM'] = True
    if crossed is not None:
      params['crossed'] = crossed

    if process == 'CD':
      # Grid force scaling factor: sigmoid function that goes 0->1 as alpha goes 0->1
      alpha_g = 4. * (alpha - 0.5)**2 / (1 + np.exp(-100 * (alpha - 0.5)))
      if alpha_g < 1E-10:
        alpha_g = 0

      # DEBUG: Print alpha and alpha_g for first few states only
      if DEBUG_ALPHA_G:
        if not hasattr(self, '_alpha_debug_count'):
          self._alpha_debug_count = 0
        if self._alpha_debug_count < 5:
          print(f"[CD DEBUG] alpha={alpha:.4f}, alpha_g={alpha_g:.6f}, grid_strength={alpha_g*100:.2f}%")
          self._alpha_debug_count += 1

      if self.args.params['CD']['solvation']=='Desolvated' or \
         self.args.params['CD']['solvation']=='Reduced':
        params['OBC'] = 0
      elif self.args.params['CD']['solvation'] == 'Fractional':
        params['OBC'] = alpha_g  # Scales the solvent with the grid
      elif self.args.params['CD']['solvation'] == 'Full':
        params['OBC'] = 1.0
      if self.args.params['CD']['pose'] > -1:
        # Pose BPMF
        alpha_r = np.tanh(16 * alpha * alpha)
        params['alpha'] = alpha
        params['k_angular_int'] = self.args.params['CD']['k_pose'] * alpha_r
        params['k_angular_ext'] = self.args.params['CD']['k_pose']
        params['k_spatial_ext'] = self.args.params['CD']['k_pose']
        params['sLJr'] = alpha_g
        params['sLJa'] = alpha_g
        params['ELE'] = alpha_g
        params['T'] = alpha_r * (self.args.params['BC']['T_TARGET'] - self.args.params['BC']['T_HIGH']) + self.args.params['BC']['T_HIGH']
      else:
        # BPMF
        alpha_sg = 1. - 4. * (alpha - 0.5)**2
        params['alpha'] = alpha
        params['sLJr'] = alpha_sg
        if self.args.params['CD']['solvation'] != 'Reduced':
          params['sELE'] = alpha_sg
        params['LJr'] = alpha_g
        params['LJa'] = alpha_g
        params['ELE'] = alpha_g \
          if self.args.params['CD']['solvation']!='Reduced' else 0.2*alpha_g

        # DEBUG: Print detailed grid scaling parameters
        if DEBUG_GRID_PARAMS and hasattr(self, '_alpha_debug_count') and self._alpha_debug_count < 5:
          print(f"[CD GRID PARAMS] alpha_sg={alpha_sg:.6f}")
          print(f"[CD GRID PARAMS] sLJr={params['sLJr']:.6f}, sLJa={params.get('sLJa', 'N/A')}, sELE={params.get('sELE', 'N/A')}")
          print(f"[CD GRID PARAMS] LJr={params['LJr']:.6f}, LJa={params['LJa']:.6f}, ELE={params['ELE']:.6f}")
          print(f"[CD GRID PARAMS] OBC={params.get('OBC', 'N/A')}")

        if site is not None:
          params['site'] = site
        if self.args.params['CD']['temperature_scaling'] == 'Linear':
          params['T'] = alpha * (self.args.params['BC']['T_TARGET'] - self.args.params['BC']['T_HIGH']) + self.args.params['BC']['T_HIGH']
        elif self.args.params['CD']['temperature_scaling'] == 'Quadratic':
          params['T'] = alpha_g * (self.args.params['BC']['T_TARGET'] - self.args.params['BC']['T_HIGH']) + self.args.params['BC']['T_HIGH']
    elif process == 'BC':
      # If alpha = 0.0, T = T_HIGH. If alpha = 1.0, T = T_TARGET.
      params['alpha'] = alpha
      params['T'] = self.args.params['BC']['T_HIGH'] - alpha * (self.args.params['BC']['T_HIGH'] - self.args.params['BC']['T_TARGET'])
      if self.args.params['BC']['solvation'] == 'Desolvated':
        params['OBC'] = alpha
      elif self.args.params['BC']['solvation'] == 'Reduced':
        params['OBC'] = alpha
      elif self.args.params['BC']['solvation'] == 'Fractional':
        params['OBC'] = alpha
      elif self.args.params['BC']['solvation'] == 'Full':
        params['OBC'] = 1.0
    else:
      raise Exception("Unknown process!")

    return params

  def clear_evaluators(self):
    """Deletes the stored evaluators and grids to save memory
    """
    self._evaluators = {}
    for scalable in scalables:
      if (scalable in self._forceFields.keys()):
        del self._forceFields[scalable]

  def isForce(self, val):
    """Determines whether a force named 'val' is defined
    """
    return (val in self._forceFields.keys())

  def getGridParams(self):
    """Returns the counts, center, and spacing used for the electrostatic grid
    """
    self.setParams({'MM': True, 'ELE': 1})
    gd = self._forceFields['ELE'].grid_data
    dims = gd['counts']
    #TODO: what is this factor? Temporarily set it to 1.
    factor = 1
    center = factor * (gd['counts'] * gd['spacing'] / 2. + gd['origin'])
    spacing = factor * gd['spacing'][0]
    return (dims, center, spacing)
