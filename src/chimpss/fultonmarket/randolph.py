# Package Imports
import os

import mpiplus
import numpy as np
import openmm.unit as unit
from openmm import *
from openmm.app import *
from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, ParallelTemperingSampler

from chimpss.fultonmarket.utils import *
from chimpss.fultonmarket.utils import (
    _interpolate_new_positions,
    _interpolate_new_states,
    build_sampler_states,
    printf,
    truncate_ncdf,
)


class Randolph():
    """
    Parallel tempering replica exchange sampler, wrapping OpenMMTools
    ParallelTemperingSampler with adaptive temperature ladder management.

    Parameters
    ----------
    sampler_states : list of SamplerState
        Initial positions, box vectors, and (optionally) velocities for each
        replica.
    reference_thermo_state : ThermodynamicState
        Reference thermodynamic state.
    sim_no : int
        Index of the current sub-simulation (0-based).
    sim_time : float
        Total aggregate simulation time for this sub-simulation in nanoseconds.
    temperatures : np.ndarray
        1-D array of temperatures in Kelvin defining the replica ladder.
    output_dir : str
        Path to the directory where NetCDF output files are written.
    output_ncdf : str
        Path to the primary trajectory NetCDF file.
    checkpoint_ncdf : str
        Path to the checkpoint NetCDF file.
    iter_length : float
        Time between replica swap attempts in nanoseconds.
    dt : float
        Integration timestep in femtoseconds.
    """

    def __init__(self,
                 sampler_states,
                 reference_thermo_state,
                 sim_no: int,
                 sim_time: float,
                 temperatures: np.ndarray,
                 output_dir: str,
                 output_ncdf: str,
                 checkpoint_ncdf: str,
                 iter_length: float,
                 dt: float):

        self.sampler_states = sampler_states
        self.reference_state = reference_thermo_state
        self.sim_no = sim_no
        self.sim_time = sim_time
        self.output_dir = output_dir
        self.output_ncdf = output_ncdf
        self.checkpoint_ncdf = checkpoint_ncdf
        self.temperatures = temperatures.copy()
        self.n_replicates = len(self.temperatures)
        self.iter_length = iter_length
        self.dt = dt

        self._configure_simulation_parameters()
        self._build_simulation()


    def main(self, init_overlap_thresh: float, term_overlap_thresh: float):
        """
        Run the simulation until the target number of cycles is complete.
        """
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        self.current_cycle = 0
        while self.current_cycle < self.n_cycles:
            self._run_cycle()


    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def save_simulation(self, save_dir: str):
        """
        Persist trajectory data to disk and truncate the NetCDF files.
        """
        save_no_dir = os.path.join(save_dir, str(self.sim_no))
        os.makedirs(save_no_dir, exist_ok=True)

        ncdf_copy = os.path.join(self.output_dir, 'output_copy.ncdf')
        pos_memmap, velos, box_vectors, states, energies, temperatures = truncate_ncdf(
            self.output_ncdf, ncdf_copy, save_no_dir, self.reporter, False
        )
        np.save(os.path.join(save_no_dir, 'velocities.npy'), velos.data)
        del velos
        np.save(os.path.join(save_no_dir, 'box_vectors.npy'), box_vectors.data)
        del box_vectors
        np.save(os.path.join(save_no_dir, 'states.npy'), states.data)
        del states
        np.save(os.path.join(save_no_dir, 'energies.npy'), energies.data)
        del energies
        np.save(os.path.join(save_no_dir, 'temperatures.npy'), temperatures)

        checkpoint_copy = os.path.join(self.output_dir, 'output_checkpoint_copy.ncdf')
        truncate_ncdf(self.checkpoint_ncdf, checkpoint_copy, save_no_dir, self.reporter, True)

        os.replace(ncdf_copy, self.output_ncdf)
        os.replace(checkpoint_copy, self.checkpoint_ncdf)

        try:
            self.reporter.close()
        except Exception:
            pass

        return len(temperatures), [t * unit.kelvin for t in temperatures]


    def _configure_simulation_parameters(self):
        """
        Derive all iteration and cycle counts from the aggregate simulation
        time, number of replicates, swap interval, and timestep.
        """
        self.n_replicates = len(self.temperatures)
        sim_time_per_rep = self.sim_time / self.n_replicates
        printf(f'Simulation time per replicate : {np.round(sim_time_per_rep, 6)} ns')
        steps_per_rep = np.ceil(sim_time_per_rep * 1e6 / self.dt)
        printf(f'Steps per replicate           : {np.round(steps_per_rep, 0):.0f}')
        self.n_steps_per_iter = int(self.iter_length * 1e6 / self.dt)
        printf(f'Steps per iteration           : {self.n_steps_per_iter}')
        self.n_iters = int(np.ceil(steps_per_rep / self.n_steps_per_iter))
        printf(f'Total iterations              : {self.n_iters}')
        self.n_cycles = int(np.ceil(self.n_iters / 5))
        printf(f'Total cycles                  : {self.n_cycles}')
        self.n_iters_per_cycle = int(np.ceil(self.n_iters / self.n_cycles))
        printf(f'Iterations per cycle          : {self.n_iters_per_cycle}')
        self.checkpoint_interval = max(1, int(0.02 / self.iter_length))
        printf(f'Checkpoint interval           : {self.checkpoint_interval} iterations')
        printf(f'Temperature ladder ({self.n_replicates} reps) : '
               f'{[np.round(t._value, 1) for t in self.temperatures]} K')


    @mpiplus.on_single_node(0, sync_nodes=True)
    def _remove_ncdf(self):
        """Delete the primary output NetCDF file if it exists."""
        if os.path.exists(self.output_ncdf):
            printf(f'Removing {self.output_ncdf}')
            os.remove(self.output_ncdf)


    def _build_simulation(self):
        """Construct the OpenMMTools sampler, reporter, and simulation object."""
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond,
                                         collision_rate=1.0 / unit.picosecond,
                                         n_steps=self.n_steps_per_iter,
                                         reassign_velocities=False)

        self.simulation = ParallelTemperingSampler(mcmc_moves=move,
                                                   number_of_iterations=self.n_iters,
                                                   replica_mixing_scheme='swap-all')

        self.simulation._global_citation_silence = True

        self._remove_ncdf()

        atom_inds = tuple(range(self.reference_state.system.getNumParticles()))

        self.reporter = MultiStateReporter(self.output_ncdf,
                                           checkpoint_interval=self.checkpoint_interval,
                                           analysis_particle_indices=atom_inds)
        # Pass ONE SamplerState PER REPLICA. Passing a single SamplerState makes
        # openmmtools copy it to every replica, so each sub-simulation would
        # restart all replicas from the state-0 (T_min) configuration and throw
        # away the per-replica positions/velocities/boxes that
        # FultonMarket._load_initial_args carefully reorders. That regression
        # (`self.sampler_states[0]`, introduced in the Phase 4 port; the
        # original FultonMarket passed the full list) reset every replica every
        # sub-simulation for the entire 5-HT2B campaign.
        if len(self.sampler_states) != self.n_replicates:
            raise ValueError(
                f'{len(self.sampler_states)} sampler states for {self.n_replicates} '
                f'replicas; need exactly one per replica.'
            )
        self.simulation.create(thermodynamic_state=self.reference_state,
                               sampler_states=self.sampler_states,
                               storage=self.reporter,
                               temperatures=self.temperatures,
                               n_temperatures=self.n_replicates)


    def _run_cycle(self):
        """Advance the simulation by one cycle and evaluate exchange acceptance rates."""
        thresh = self.init_overlap_thresh if self.sim_no == 0 else self.term_overlap_thresh

        printf(f'Cycle {self.current_cycle}: advancing {self.n_iters_per_cycle} iterations.')
        if self.simulation.is_completed:
            self.simulation.extend(self.n_iters_per_cycle)
        else:
            self.simulation.run(self.n_iters_per_cycle)

        insert_inds = self._eval_acc_rates(thresh)

        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self._configure_simulation_parameters()
            self._build_simulation()
        else:
            self.current_cycle += 1


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _eval_acc_rates(self, acceptance_rate_thresh: float = 0.40) -> np.ndarray:
        """
        Read mixing statistics from the reporter and identify adjacent replica
        pairs with exchange rates below the given threshold.
        """
        temperatures = [float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]]
        accepted, proposed = self.reporter.read_mixing_statistics()
        acc_rates = np.nan_to_num(np.mean(accepted[1:] / proposed[1:], axis=0))

        insert_inds = []
        for state in range(len(acc_rates) - 1):
            rate = acc_rates[state, state + 1]
            printf(f'Exchange {np.round(temperatures[state], 2)} K <-> {np.round(temperatures[state + 1], 2)} K : {rate:.4f}')
            if rate < acceptance_rate_thresh:
                insert_inds.append(state + 1)
        return np.array(insert_inds)


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _read_temps_from_reporter(self) -> np.ndarray:
        """Read the current temperature ladder directly from the reporter."""
        return np.array([float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]])


    def _interpolate_states(self, insert_inds: np.ndarray):
        """Insert new intermediate replicas at the specified positions in the temperature ladder."""
        prev_temps = self._read_temps_from_reporter()
        self.temperatures, self.n_replicates = _interpolate_new_states(prev_temps, insert_inds)

        init_positions, init_box_vectors, init_velocities = self._load_inits()
        init_positions, init_box_vectors, init_velocities = _interpolate_new_positions(init_positions,
                                                                                       init_box_vectors,
                                                                                       init_velocities,
                                                                                       insert_inds,
                                                                                       self.n_replicates)

        self.sampler_states = build_sampler_states(self.n_replicates,
                                                   init_positions,
                                                   init_box_vectors,
                                                   init_velocities)


    def _load_inits(self):
        """Extract current positions, box vectors, and velocities from the sampler states."""
        n = len(self.sampler_states)
        init_positions  = np.array([self.sampler_states[i].positions._value.copy()   for i in range(n)])
        init_box_vectors = np.array([self.sampler_states[i].box_vectors._value.copy() for i in range(n)])

        if self.sim_no > 0 and self.sampler_states[0].velocities is not None:
            init_velocities = np.array([self.sampler_states[i].velocities._value.copy() for i in range(n)])
        else:
            init_velocities = None

        return init_positions, init_box_vectors, init_velocities
