# Package Imports
import os

import netCDF4 as nc
import numpy as np
import openmm.unit as unit
from openmm import *
from openmm.app import *
from openmmtools.states import SamplerState, ThermodynamicState

from chimpss.fultonmarket.analysis import FultonMarketAnalysis
from chimpss.fultonmarket.randolph import Randolph
from chimpss.fultonmarket.utils import *
from chimpss.shared.io import build_output_path, file_exists_skip, validate_name


class FultonMarket():
    """
    Unrestrained Parallel Tempering Replica Exchange.

    Parameters
    ----------
    input_pdb : str
        Path to input PDB file.
    input_system : str
        Path to OpenMM system XML file containing force field parameters.
    input_state : str, optional
        Path to OpenMM state XML file for initial context.
    sele_str : str, optional
        MDAnalysis-style selection string for atoms of interest.
    T_min : float
        Minimum temperature in Kelvin. Default 300.
    T_max : float
        Maximum temperature in Kelvin. Default 367.447.
    n_replicates : int
        Number of replicas across the temperature ladder. Default 12.
    """

    def __init__(self,
                 input_pdb: str,
                 input_system: str,
                 input_state: str = None,
                 sele_str: str = None,
                 T_min: float = 300,
                 T_max: float = 367.447,
                 n_replicates: int = 12,
                 protein_name: str = None,
                 ligand_name: str = None):

        printf('Welcome to FultonMarket.')

        self.temperatures = [temp * unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
        self.n_replicates = n_replicates
        self.sele_str = sele_str
        self.protein_name = validate_name(protein_name) if protein_name else None
        self.ligand_name = validate_name(ligand_name) if ligand_name else None

        self.input_pdb = input_pdb
        self.pdb = PDBFile(input_pdb)
        self._set_init_positions()
        printf(f'Found input_pdb: {input_pdb}')

        self.system = XmlSerializer.deserialize(open(input_system, 'r').read())
        self._set_init_box_vectors()
        printf(f'Found input_system: {input_system}')

        if input_state is not None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context
            printf(f'Found input_state: {input_state}')


    def _set_init_positions(self):
        """
        Extract atomic positions from the input PDB and replicate them across
        all replicas.

        Sets
        ----
        self.init_positions : list of Quantity
            One copy of the PDB positions per replica, each of shape
            (n_atoms, 3) in nanometers.
        self.n_atoms : int
            Total number of atoms in the system.
        """
        self.init_positions = self.pdb.getPositions(asNumpy=True)
        self.n_atoms = self.init_positions.shape[0]
        self.init_positions = [self.init_positions for _ in range(self.n_replicates)]


    def _set_init_box_vectors(self):
        """
        Read the default periodic box vectors from the OpenMM system and
        replicate them across all replicas.

        Sets
        ----
        self.init_box_vectors : list of tuple
            One copy of the system's default box vectors per replica.
        """
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        self.init_box_vectors = [self.init_box_vectors for _ in range(self.n_replicates)]


    def run(self,
            iter_length: float,
            dt: float = 2.0,
            sim_length: int = 25,
            total_sim_time: int = None,
            minimum_fraction: float = 0.25,
            init_overlap_thresh: float = 0.5,
            term_overlap_thresh: float = 0.35,
            output_dir: str = os.path.join(os.getcwd(), 'FultonMarket_output/'),
            n_resample: int = 1000,
            max_equil_fraction: float = 0.75,
            frobenius_thresh: float = 0.05,
            jsd_thresh: float = 0.10,
            getContacts_Info: dict = None):
        """
        Run parallel tempering replica exchange.

        Parameters
        ----------
        iter_length : float
            Time between replica swap attempts, in nanoseconds.
        dt : float
            Integration timestep in femtoseconds. Default 2.0.
        sim_length : int
            Duration of each sub-simulation in nanoseconds. Controls how often
            trajectory files are truncated, data is saved, resampling occurs,
            and convergence is evaluated.
            Recomended Value: int(10*timestep)
        total_sim_time : int, optional
            Total aggregate simulation time across all replicates, in
            nanoseconds. Used as a terminal stopping criterion if reached.
        init_overlap_thresh : float
            Minimum acceptance rate during the first sub-simulation before
            triggering a restart. Default 0.5.
        term_overlap_thresh : float
            Minimum acceptance rate at any point after the first sub-simulation
            before triggering a restart. Default 0.35.
        output_dir : str
            Path to output directory. Default: 'FultonMarket_output/' in cwd.
        n_resample : int
            Number of frames to importance resample per convergence check.
            Default 1000.
        max_equil_fraction : float
            Maximum fraction of the simulation that can be discarded as
            equilibration before convergence is considered unreliable.
            Default 0.75.
        frobenius_thresh : float
            Maximum normalised Frobenius norm between the current and any
            previous checkpoint distance matrix for convergence. Default 0.05.
        jsd_thresh : float
            Maximum Jensen-Shannon Divergence between the current and any
            previous checkpoint pairwise distance distribution for convergence.
            Default 0.10.
        getContacts_Info : dict, optional
            Keyword arguments forwarded to getContactDistanceMatrix.
            Required keys: ``getcontacts_script`` (path to
            get_dynamic_contacts.py), ``conda_env`` (conda env name
            containing getContacts). Optional keys: ``getcontacts_python``
            (explicit interpreter path), ``cores`` (CPU cores, default 10).
            If None, contact distance matrix convergence will raise an error
            when first attempted.
        """

        # Store run parameters
        self.total_sim_time = total_sim_time
        self.iter_length = iter_length
        self.dt = dt
        self.sim_length = sim_length
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh
        self.minimum_fraction4convergence = minimum_fraction
        self.n_resample = n_resample
        self.max_equil_fraction = max_equil_fraction
        self.frobenius_thresh = frobenius_thresh
        self.jsd_thresh = jsd_thresh
        self.getContacts_Info = getContacts_Info if getContacts_Info is not None else {}

        # Prepare output directories
        self.output_dir = output_dir
        self.name = output_dir.split('/')[-1]
        if self.protein_name and self.ligand_name:
            self.output_ncdf     = build_output_path(output_dir, self.protein_name, self.ligand_name, 'trajectory', 'ncdf')
            self.checkpoint_ncdf = build_output_path(output_dir, self.protein_name, self.ligand_name, 'checkpoint', 'ncdf')
        else:
            self.output_ncdf     = os.path.join(output_dir, 'output.ncdf')
            self.checkpoint_ncdf = os.path.join(output_dir, 'output_checkpoint.ncdf')
        self.save_dir = os.path.join(output_dir, 'saved_variables')
        os.makedirs(self.save_dir, exist_ok=True)

        # Resume support: only bail out when output.ncdf exists AND no
        # sub-simulations have been saved yet. If saved_variables/<N>/ exists,
        # `_configure_experiment_parameters` will set self.sim_no accordingly
        # and the main loop will resume from there.
        n_completed_subsims = len(os.listdir(self.save_dir))
        if n_completed_subsims == 0 and file_exists_skip(self.output_ncdf, 'FultonMarket trajectory'):
            print('Trajectory already exists with no saved sub-simulations. '
                  'Move or delete it to re-run.')
            return
        if n_completed_subsims > 0 and os.path.exists(self.output_ncdf):
            print(f'[resume] Found {n_completed_subsims} completed sub-simulation(s); '
                  f'will resume from sim_no={n_completed_subsims}.')

        printf(f'total_sim_time      : {self.total_sim_time} ns')
        printf(f'iter_length         : {self.iter_length} ns')
        printf(f'dt                  : {self.dt} fs')
        printf(f'sim_length          : {self.sim_length} ns')
        printf(f'n_replicates        : {self.n_replicates}')
        printf(f'init_overlap_thresh : {self.init_overlap_thresh}')
        printf(f'term_overlap_thresh : {self.term_overlap_thresh}')
        printf(f'output_dir          : {self.output_dir}')
        printf(f'temperatures        : {[np.round(T._value, 1) for T in self.temperatures]} K')
        printf(f'n_resample          : {self.n_resample}')
        printf(f'max_equil_fraction  : {self.max_equil_fraction}')
        printf(f'frobenius_thresh    : {self.frobenius_thresh}')
        printf(f'jsd_thresh          : {self.jsd_thresh}')
        printf(f'getContacts_Info    : {self.getContacts_Info}')

        self._configure_experiment_parameters()

        while not self.finished:
            if self.sim_no > 0:
                self._load_initial_args()

            self._build_states()
            self._set_parameters()

            self.simulation = Randolph(**self.params)
            self.simulation.main(init_overlap_thresh=init_overlap_thresh,
                                 term_overlap_thresh=term_overlap_thresh)

            self._save_sub_simulation()
            self.finished = self._evaluate_stopping_criterion(
                n_resample=self.n_resample,
                max_equil_fraction=self.max_equil_fraction,
                frobenius_thresh=self.frobenius_thresh,
                jsd_thresh=self.jsd_thresh,
                getContacts_Info=self.getContacts_Info,
            )
            self.sim_no += 1


    def _set_parameters(self):
        """
        Assemble the keyword-argument dictionary passed to the Randolph
        sampler constructor.

        Sets
        ----
        self.params : dict
            All simulation parameters required to instantiate a Randolph obj.
        """
        self.params = dict(sampler_states=self.sampler_states,
                           reference_thermo_state=self.thermodynamic_states[0],
                           sim_no=self.sim_no,
                           sim_time=self.sim_length,
                           temperatures=self.temperatures,
                           output_dir=self.output_dir,
                           output_ncdf=self.output_ncdf,
                           checkpoint_ncdf=self.checkpoint_ncdf,
                           iter_length=self.iter_length,
                           dt=self.dt)


    def _build_states(self):
        """
        Build both sampler and thermodynamic states required by Randolph.

        Delegates to `_build_sampler_states` and
        `_build_thermodynamic_states`.
        """
        self._build_sampler_states()
        self._build_thermodynamic_states()


    def _build_sampler_states(self):
        """
        Construct OpenMMTools SamplerState objects for each replica.

        On the first sub-simulation (sim_no == 0), positions and box vectors
        are read directly from the OpenMM Context (populated from the input
        state file). On subsequent sub-simulations they are loaded from saved
        numpy arrays and velocities are preserved.

        Sets
        ----
        self.sampler_states : list of SamplerState
            One SamplerState per replica, containing positions, box vectors,
            and (after sim_no > 0) velocities.
        """
        if self.sim_no == 0:
            printf('Setting initial positions via Context method.')
            self.sampler_states = [SamplerState(positions=self.init_positions,
                                                box_vectors=self.init_box_vectors).from_context(self.context)
                                   for _ in range(self.n_replicates)]
        else:
            printf('Setting initial positions via Velocity method.')
            self.sampler_states = build_sampler_states(self.n_replicates,
                                                       self.init_positions,
                                                       self.init_box_vectors,
                                                       self.init_velocities)


    def _build_thermodynamic_states(self):
        """
        Construct the OpenMMTools ThermodynamicState for the reference
        (lowest-temperature) replica, if not already initialised.

        The thermodynamic states for the remaining replicas are managed
        internally by Randolph using `self.temperatures`. This method only
        runs on the first call; subsequent calls are no-ops.

        Sets
        ----
        self.thermodynamic_states : list of ThermodynamicState
            Single-element list containing the NPT state at T_min and 1 bar.
        """
        if not hasattr(self, 'thermodynamic_states'):
            self.thermodynamic_states = [ThermodynamicState(system=self.system,
                                                            temperature=self.temperatures[0],
                                                            pressure=1.0 * unit.bar)]


    def _save_sub_simulation(self):
        """
        Persist the completed sub-simulation to disk via Randolph and update
        replica metadata.

        Calls `Randolph.save_simulation`, which writes positions, velocities,
        box vectors, state indices, and temperatures to the save directory,
        then updates `self.n_replicates` and `self.temperatures` to reflect
        any changes made during the run (e.g. replica insertion).

        Sets
        ----
        self.n_replicates : int
            Updated replica count after saving.
        self.temperatures : list of Quantity
            Updated temperature ladder after saving.
        """
        self.n_replicates, self.temperatures = self.simulation.save_simulation(self.save_dir)


    def _load_initial_args(self):
        """
        Load positions, velocities, box vectors, and state indices from the
        most recently completed sub-simulation, then reorder them from replica
        index to state (temperature) index ready for the next sub-simulation.

        Positions are loaded as a memory-mapped array if a direct `np.load`
        fails (e.g. large files). If numpy loading fails entirely, arguments
        are recovered directly from the NetCDF trajectory via
        `_recover_arguments`.

        Sets
        ----
        self.temperatures : list of Quantity
            Temperature ladder in Kelvin for the upcoming sub-simulation.
        self.n_replicates : int
            Number of replicas for the upcoming sub-simulation.
        self.init_positions : TrackedQuantity
            Positions ordered by state index, shape (n_replicates, n_atoms, 3)
            in nanometers.
        self.init_box_vectors : TrackedQuantity
            Box vectors ordered by state index, shape (n_replicates, 3, 3)
            in nanometers.
        self.init_velocities : TrackedQuantity or None
            Velocities ordered by state index in nm/ps, or None if not saved.
        """
        load_dir = os.path.join(self.save_dir, str(self.sim_no - 1))
        self.temperatures = np.load(os.path.join(load_dir, 'temperatures.npy'))
        self.temperatures = [t * unit.kelvin for t in self.temperatures]
        self.n_replicates = len(self.temperatures)
        printf(f'Loaded n_replicates: {self.n_replicates}')

        try:
            box_vectors = np.load(os.path.join(load_dir, 'box_vectors.npy'))
            n_frames = box_vectors.shape[0]
            init_box_vectors = box_vectors[-1]
            positions_path = os.path.join(load_dir, 'positions.npy')
            try:
                init_positions = np.load(positions_path)[-1]
            except Exception:
                init_positions = np.array(np.memmap(positions_path, mode='r', dtype='float32',
                                                    shape=(n_frames, self.n_replicates, self.n_atoms, 3))[-1])
            velocities_path = os.path.join(load_dir, 'velocities.npy')
            init_velocities = np.load(velocities_path) if os.path.exists(velocities_path) else None
            state_inds = np.load(os.path.join(load_dir, 'states.npy'))[-1]
        except Exception:
            init_velocities, init_positions, init_box_vectors, state_inds = self._recover_arguments()

        # Reorder arrays from replica index to state index
        reshaped_positions = np.empty_like(init_positions)
        reshaped_box_vectors = np.empty_like(init_box_vectors)
        for state in range(self.n_replicates):
            rep_ind = np.where(state_inds == state)[0]
            reshaped_positions[state] = init_positions[rep_ind]
            reshaped_box_vectors[state] = init_box_vectors[rep_ind]
        self.init_positions = convert_to_TrackedQuantity(reshaped_positions, unit.nanometer)
        self.init_box_vectors = convert_to_TrackedQuantity(reshaped_box_vectors, unit.nanometer)

        if init_velocities is not None:
            reshaped_velocities = np.empty_like(init_velocities)
            for state in range(self.n_replicates):
                rep_ind = np.where(state_inds == state)[0]
                reshaped_velocities[state] = init_velocities[rep_ind]
            self.init_velocities = convert_to_TrackedQuantity(reshaped_velocities, unit.nanometer / unit.picosecond)
        else:
            self.init_velocities = None


    def _configure_experiment_parameters(self):
        """
        Determine how many sub-simulations have already completed, validate
        the save directory, and set the initial stopping-criterion state.

        Raises
        ------
        RuntimeError
            If any sub-simulation save directory contains fewer than 5 files,
            indicating an incomplete or corrupted save.

        Sets
        ----
        self.sim_no : int
            Index of the next sub-simulation to run (0-based).
        self.total_n_sims : int
            Total number of sub-simulations required (only set when
            `total_sim_time` is not None).
        self.finished : bool
            Whether the stopping criterion is already satisfied before the
            main loop begins.
        self.converged : bool
            Initialised to False when resuming; updated by
            `_evaluate_stopping_criterion`.
        """
        incomplete = [d for d in os.listdir(self.save_dir) if len(os.listdir(os.path.join(self.save_dir, d))) < 5]
        if incomplete:
            raise RuntimeError(f'Incomplete save directories detected: {incomplete}. Remove or fix them before continuing.')

        self.sim_no = len(os.listdir(self.save_dir))
        printf(f'Resuming from sim_no: {self.sim_no}')
        if self.total_sim_time is not None:
            self.total_n_sims = int(np.ceil(self.total_sim_time / self.sim_length))
            printf(f'Total sub-simulations required: {self.total_n_sims}')
        self.finished = False
        if self.sim_no > 0:
            self.converged = False

    def _recover_arguments(self):
        """
        Fall-back loader that reads the final frame of each trajectory
        variable directly from the NetCDF output file.

        Used when the numpy save files are unavailable or corrupt. Reads
        from `self.output_ncdf`.

        Returns
        -------
        velocities : np.ndarray
            Velocities of the last frame, shape (n_replicates, n_atoms, 3).
        positions : np.ndarray
            Positions of the last frame, shape (n_replicates, n_atoms, 3).
        box_vectors : np.ndarray
            Box vectors of the last frame, shape (n_replicates, 3, 3).
        state_inds : np.ndarray
            State index for each replica in the last frame, shape
            (n_replicates,).
        """
        with nc.Dataset(self.output_ncdf, 'r') as ncfile:
            velocities = ncfile.variables['velocities'][-1].data
            positions = ncfile.variables['positions'][-1].data
            box_vectors = ncfile.variables['box_vectors'][-1].data
            state_inds = ncfile.variables['states'][-1].data
        return velocities, positions, box_vectors, state_inds


    def _evaluate_stopping_criterion(self, n_resample=1000, max_equil_fraction=0.75, frobenius_thresh=0.05, jsd_thresh=0.10, getContacts_Info=None):
        """
        Check whether the simulation should stop by evaluating a series of
        convergence criteria against the current and all previously saved
        resampled distance matrices.

        For each of the three distance matrix types (torsional, alpha-carbon,
        contact), two complementary metrics are computed against every previous
        checkpoint:

        - **Frobenius norm** — element-wise structural divergence between the
          full matrices. Sensitive to the magnitude of changes.
        - **Jensen-Shannon Divergence** — divergence between the distributions
          of pairwise distances (upper triangle). Sensitive to the shape of the
          conformational ensemble distribution, robust to outliers.

        Both metrics must be below their respective thresholds across all
        previous checkpoints for a matrix type to be declared converged.

        Parameters
        ----------
        n_resample : int
            Number of frames to importance resample. Default 1000.
        max_equil_fraction : float
            Maximum fraction of the simulation discardable as equilibration
            before convergence is considered unreliable. Default 0.75.
        frobenius_thresh : float
            Maximum normalised Frobenius norm between the current and any
            previous checkpoint matrix for convergence. Default 0.05.
        jsd_thresh : float
            Maximum Jensen-Shannon Divergence between the current and any
            previous checkpoint pairwise distance distribution for convergence.
            Bounded in [0, 1]. Default 0.10.
        getContacts_Info : dict, optional
            Keyword arguments forwarded directly to getContactDistanceMatrix.
            See run() docstring for accepted keys. Default None.

        Returns
        -------
        bool
            True if the simulation should stop.
        """
        # Time-only stopping path: when getContacts_Info is not configured the
        # downstream call to getContactDistanceMatrix raises, even when the
        # user has set --total_sim_time and is opting out of convergence-based
        # stopping. Short-circuit to a pure time-based decision in that case.
        if not getContacts_Info:
            if self.total_sim_time is None:
                printf("WARNING: no getContacts_Info AND no total_sim_time set "
                       "— simulation has no stopping criterion and will not stop.")
                return False
            at_max_time = self.sim_no >= self.total_n_sims
            done_or_in_progress = self.sim_no + 1
            if at_max_time:
                printf(f"Stopping: completed {done_or_in_progress} sub-simulations "
                       f"(>= total_n_sims={self.total_n_sims}); no getContacts → time-based stop.")
            else:
                printf(f"Continuing: completed {done_or_in_progress}/{self.total_n_sims} "
                       f"sub-simulations; no getContacts → skipping convergence check.")
            return at_max_time

        printf("Gathering convergence related data...")
        analyzer = FultonMarketAnalysis(input_dir=self.output_dir, pdb=self.input_pdb, sele_str=self.sele_str)

        # Equilibration
        ave_ener = analyzer.get_average_energy()
        t0 = detect_energy_equil(ave_ener)
        equil_fraction = t0 / ave_ener.shape[0]
        printf(f"Currently would discard {100*equil_fraction:.2f}% to energy equilibration")

        # Importance resampling and trajectory
        sim_dir = os.path.join(self.output_dir, 'saved_variables', str(self.sim_no))
        _ = analyzer.importance_resampling(n_samples=n_resample)
        traj = analyzer.write_resampled_traj(
            pdb_out=os.path.join(sim_dir, 'resampled_top.pdb'),
            dcd_out=os.path.join(sim_dir, 'resampled_trj.dcd'),
            weights_out=os.path.join(sim_dir, 'resampled_wghts.npy'),
            inds_out=os.path.join(sim_dir, 'resampled_indcs.npy'),
            return_traj=True,
        )

        # Compute current distance matrices
        torsional    = getTorsionalDistanceMatrix(traj, selection_string='protein or resname UNK')
        alpha_carbon = getAlphaCarbonDistanceMatrix(traj, selection_string='protein or resname UNK')
        contact_distance, _ = getContactDistanceMatrix(
            top_fn=os.path.join(sim_dir, 'resampled_top.pdb'),
            traj_fn=os.path.join(sim_dir, 'resampled_trj.dcd'),
            output_fn=os.path.join(sim_dir, 'resampled_contacts.tsv'),
            **(getContacts_Info if getContacts_Info is not None else {}),
        )
        current_matrices = {
            'torsion':      torsional,
            'alpha_carbon': alpha_carbon,
            'contact':      contact_distance,
        }

        # Save current matrices
        for name, matrix in current_matrices.items():
            np.save(os.path.join(sim_dir, f'resampled_{name}_matrix.npy'), matrix)

        # Effective post-equilibration fraction — whichever keeps more data
        min_post_equil_fraction = 1.0 - max_equil_fraction  # guaranteed minimum
        actual_post_equil_fraction = 1.0 - equil_fraction         # what MBAR detected
        effective_post_equil_fraction = max(actual_post_equil_fraction, min_post_equil_fraction)

        # Map the effective fraction back to the first valid sim_no
        first_valid_sim_no = int((1.0 - effective_post_equil_fraction) * self.sim_no)

        # Compare against previous checkpoints in the post-equilibration window only
        frob_results = {name: {} for name in current_matrices}
        jsd_results  = {name: {} for name in current_matrices}

        for prev_sim_no in range(first_valid_sim_no, self.sim_no):
            prev_dir = os.path.join(self.output_dir, 'saved_variables', str(prev_sim_no))
            for name, current_matrix in current_matrices.items():
                prev_path = os.path.join(prev_dir, f'resampled_{name}_matrix.npy')
                if not os.path.exists(prev_path):
                    continue
                prev_matrix = np.load(prev_path)
                frob_results[name][prev_sim_no] = frobenius_norm(current_matrix, prev_matrix)
                jsd_results[name][prev_sim_no]  = jsd_distance_matrices(current_matrix, prev_matrix)

        # A matrix type is converged if ALL previous comparisons are below BOTH thresholds
        matrix_converged = {
            name: (
                len(frob_results[name]) > 0
                and all(v < frobenius_thresh for v in frob_results[name].values())
                and all(v < jsd_thresh      for v in jsd_results[name].values())
            )
            for name in current_matrices
        }

        # Build checks
        at_max_time     = self.total_sim_time is not None and self.sim_no >= self.total_n_sims
        past_minimum    = self.sim_no >= (self.total_n_sims * self.minimum_fraction4convergence)
        equil_ok        = equil_fraction < max_equil_fraction
        torsion_ok      = matrix_converged['torsion']
        alpha_carbon_ok = matrix_converged['alpha_carbon']
        contact_ok      = matrix_converged['contact']

        checks = [
            ('Max simulation time reached',      at_max_time),
            ('Past minimum simulation fraction', past_minimum),
            ('Equilibration discard < 75%',      equil_ok),
            ('Torsion matrix converged',         torsion_ok),
            ('Alpha-carbon matrix converged',    alpha_carbon_ok),
            ('Contact matrix converged',         contact_ok),
        ]

        # Pretty print convergence report
        width = max(len(label) for label, _ in checks)
        printf("=" * (width + 12))
        printf(f"  {'Convergence Report':^{width}}")
        printf("=" * (width + 12))
        for label, result in checks:
            icon = "PASS" if result else "FAIL"
            printf(f"  [{icon}]  {label:<{width}}")
        printf("-" * (width + 12))

        # Per-matrix metric detail
        for name in current_matrices:
            frob_scores = frob_results[name]
            jsd_scores  = jsd_results[name]
            if frob_scores:
                frob_str = '  '.join(f'vs sim {k}: {v:.4f}' for k, v in sorted(frob_scores.items()))
                jsd_str  = '  '.join(f'vs sim {k}: {v:.4f}' for k, v in sorted(jsd_scores.items()))
                printf(f"  {name:>12} Frobenius -- {frob_str}  (thresh={frobenius_thresh})")
                printf(f"  {name:>12}       JSD -- {jsd_str}  (thresh={jsd_thresh})")
            else:
                printf(f"  {name:>12} -- no previous matrices to compare")
        printf("=" * (width + 12))

        # Stop if at max time, or if all convergence checks pass
        convergence_checks = [result for _, result in checks[1:]]
        if at_max_time or all(convergence_checks):
            return True
        return False
