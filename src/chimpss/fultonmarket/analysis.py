import glob
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import jax
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns
from openmm import *
from openmm.app import *

from chimpss.fultonmarket.utils import *

# Guard JAX environment logging — informational only, not functional
try:
    jax.print_environment_info()
    printf(f"Default JAX backend is {jax.default_backend()}")
except Exception:
    pass


class FultonMarketAnalysis():
    """
    Simulation analysis for replica exchange trajectories produced by FultonMarket.

    Loads energies, state indices, and (lazily) positions from the
    ``saved_variables`` sub-directories written by Randolph. Handles adaptive
    temperature ladders by backfilling energy matrices for sub-simulations
    that ran with fewer replicas than the final ladder, using MBAR importance
    resampling to estimate missing state energies.

    Parameters
    ----------
    input_dir : str
        Path to the FultonMarket output directory containing the
        ``saved_variables`` sub-directory.
    pdb : str
        Path to the reference PDB file used to construct the MDTraj topology.
    skip : int
        Number of frames to discard from the start of each sub-simulation
        when loading energies and state indices. Default 10.
    scheduling : str
        Temperature scheduling scheme identifier. Currently informational
        only. Default ``'Temperature'``.
    resSeqs : list of int, optional
        Residue sequence numbers (PDB numbering) to use for PCA and
        convergence analysis. If None, all protein atoms are used.
    sele_str : str, optional
        MDTraj atom-selection string for a ligand or other non-protein
        selection, applied when writing resampled trajectories.
    upper_limit : int, optional
        Maximum frame index (inclusive) to retain from the concatenated
        energy matrix. Useful for truncating analysis to a sub-interval.
    remove_harmonic : bool
        If True, subtract harmonic restraint energies from the loaded
        energies. Requires ``spring_centers``. Default False.
    spring_centers : np.ndarray, optional
        Spring centres required when ``remove_harmonic=True``.
    verbosity : str
        Controls how much is printed. ``'none'`` suppresses all output,
        ``'minimal'`` prints key milestones and errors (default),
        ``'all'`` prints everything.
    """

    def __init__(self,
                 input_dir: str,
                 pdb: str,
                 skip: int = 10,
                 scheduling: str = 'Temperature',
                 resSeqs: List[int] = None,
                 sele_str: str = None,
                 upper_limit: int = None,
                 remove_harmonic: bool = False,
                 spring_centers: np.ndarray = None,
                 verbosity: str = 'minimal'):

        assert verbosity in ('none', 'minimal', 'all'), \
            f"verbosity must be 'none', 'minimal', or 'all'; got {verbosity!r}"
        self.verbosity = verbosity

        self.input_dir = input_dir.rstrip('/')
        self.stor_dir = os.path.join(self.input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir), self.stor_dir
        self._printf(f'Found storage directory at {self.stor_dir}', level='minimal')
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'),
                                   key=lambda x: int(x.split('/')[-1]))
        self.pdb = pdb
        self.top = md.load_pdb(self.pdb).topology
        if resSeqs is not None:
            all_resSeqs = [self.top.residue(i).resSeq for i in range(self.top.n_residues)]
            self.resSeqs = [all_resSeqs.index(r) for r in resSeqs]
        else:
            self.resSeqs = None
        self.sele_str = sele_str
        self._printf(f'Ligand selection string: {sele_str}', level='all')
        self.skip = skip
        self.scheduling = scheduling
        self.temperatures_list = [np.round(np.load(os.path.join(d, 'temperatures.npy'), mmap_mode='r'), decimals=2)
                                  for d in self.storage_dirs]
        self.temperatures = self.temperatures_list[-1]
        self._printf(f'Temperature array shapes: {[(i, t.shape) for i, t in enumerate(self.temperatures_list)]}', level='all')

        # `skip` is a per-sub-simulation frame discard, so it only makes sense
        # relative to how many frames a sub-simulation actually holds. That
        # count is set by the exchange/save cadence, NOT by sim_length: at the
        # legacy 1 ps cadence a 25 ns segment held ~150 frames, but at the
        # production 17.5 ps cadence it holds only 9-11. The default skip=10
        # was calibrated for the former and silently empties every segment of
        # the latter, which then surfaces far downstream as a bogus "no frames"
        # error. Fail loudly and specifically here instead.
        seg_frames = [(d, int(np.load(os.path.join(d, 'states.npy'), mmap_mode='r').shape[0]))
                      for d in self.storage_dirs]
        starved = [(d, n) for d, n in seg_frames if n - skip < 1]
        if starved:
            worst = min(n for _, n in starved)
            raise ValueError(
                f'skip={skip} discards every frame of {len(starved)}/{len(seg_frames)} '
                f'sub-simulation(s) (smallest holds {worst} frames; '
                f'e.g. {starved[0][0]}).\n'
                f'The saved data is intact -- `skip` is simply too large for this '
                f'save cadence. Pass a smaller skip (skip=0 lets the statistical '
                f'equilibration detection in determine_equilibration()/t0 handle '
                f'the discard, which is the defensible choice at this cadence).'
            )

        self.state_inds = [np.load(os.path.join(d, 'states.npy'), mmap_mode='r')[skip:] for d in self.storage_dirs]
        self.unshaped_energies = [np.load(os.path.join(d, 'energies.npy'), mmap_mode='r')[skip:]
                                  for d in self.storage_dirs]
        self.energies = self._reshape_list(self.unshaped_energies)

        self._get_postions_map()

        self._backfill()

        if upper_limit is not None:
            self.energies = self.energies[:upper_limit + 1]
            self.map = self.map[:upper_limit + 1]
            self.upper_limit = upper_limit
        self._printf(f'Final energy matrix shape: {self.energies.shape}', level='minimal')

    def _printf(self, msg: str, level: str = 'all') -> None:
        """Print msg according to self.verbosity."""
        if self.verbosity == 'none':
            return
        if level == 'minimal' or self.verbosity == 'all':
            printf(msg)

    # ------------------------------------------------------------------
    # Energy access and plotting
    # ------------------------------------------------------------------
    def get_state_energies(self, state_index: int = 0) -> np.ndarray:
        """Return the time series of reduced energies for a single thermodynamic state."""
        return self.energies[:, state_index, state_index]


    def get_average_energy(self) -> np.ndarray:
        """Compute the mean reduced energy across all thermodynamic states at each frame."""
        n_frames, n_states = self.energies.shape[:2]
        state_energies = np.array([self.energies[:, state, state] for state in range(n_states)]).T
        return state_energies.mean(axis=1)


    def plot_average_energy(self, figsize: tuple = (6, 6), equilibration_method: str = 'energy'):
        """Plot the mean reduced energy over simulation time."""
        self.determine_equilibration(equilibration_method=equilibration_method)
        data = self.get_average_energy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(data.shape[0]), data, color='k')
        ax.vlines(self.t0, data.min(), data.max(), color='r', label='equilibration')
        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title('Mean energy')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.set_xlabel('Iterations')
        fig.tight_layout()
        return fig, ax


    def plot_energy_distributions(self, figsize: tuple = (8, 4), post_equil: bool = False):
        """Plot kernel density estimates of the energy distribution for each thermodynamic state."""
        if post_equil:
            if not hasattr(self, 't0'):
                self.determine_equilibration()
            state_energies = np.array([self.get_state_energies(state_index=s)[self.t0:] for s in range(self.energies.shape[1])]).T
        else:
            state_energies = np.array([self.get_state_energies(state_index=s) for s in range(self.energies.shape[1])]).T

        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(state_energies, ax=ax, legend=False)
        ax.set_xlabel('Energy (kJ/mol)')
        fig.tight_layout()
        return fig, ax


    # ------------------------------------------------------------------
    # Equilibration detection
    # ------------------------------------------------------------------
    def determine_equilibration(self, equilibration_method: str = 'energy', stride: int = 10):
        """Detect the equilibration point of the simulation and store it as ``self.t0``."""
        if hasattr(self, 't0'):
            return

        if equilibration_method == 'PCA':
            self.get_PCA(state_no=0, stride=stride, explained_variance_threshold=0.9)
            equil_times = np.array([detect_PC_equil(pc, self.reduced_cartesian) for pc in range(self.n_components)])
            self.t0 = int(np.sum(equil_times * (self.explained_variance / self.explained_variance.sum()))) * stride

        elif equilibration_method == 'energy':
            self.t0 = detect_energy_equil(self.get_average_energy())

        elif equilibration_method == 'None':
            self.t0 = 0

        else:
            raise ValueError(f"equilibration_method must be 'energy', 'PCA', or 'None', got '{equilibration_method}'")

    # ------------------------------------------------------------------
    # Importance resampling and weights
    # ------------------------------------------------------------------
    def importance_resampling(self, n_samples: int = -1, equilibration_method: str = 'energy',
                              specify_state: int = 0, replace: bool = True):
        """Compute MBAR weights and draw importance-resampled frames."""
        if not hasattr(self, 't0'):
            self.determine_equilibration(equilibration_method=equilibration_method)

        self.flat_inds = np.array([[state, ind] for ind in range(self.t0, self.energies.shape[0]) for state in range(self.energies.shape[1])])
        u_kln = np.array([self.energies[self.t0:, :, k].flatten() for k in range(self.energies.shape[2])])
        N_k = np.full(self.energies.shape[2], self.energies[self.t0:].shape[0])

        self.resampled_inds, self.weights, self.resampled_weights = resample_with_MBAR(objs=[self.flat_inds],
                                                                                       u_kln=u_kln,
                                                                                       N_k=N_k,
                                                                                       size=n_samples,
                                                                                       return_inds=False,
                                                                                       return_weights=True,
                                                                                       return_resampled_weights=True,
                                                                                       specify_state=specify_state,
                                                                                       replace=replace,
                                                                                       _printf=self._printf)

    def reshape_weights(self):
        """Distribute the flat MBAR weights into a 2-D array indexed by (temperature state, frame)."""
        self.reshaped_weights = np.zeros((len(self.temperatures), len(self.energies)))
        for (state, frame), weight in zip(self.flat_inds, self.weights[:, 0]):
            self.reshaped_weights[state, frame] = weight


    def plot_weights(self, figsize: tuple = (25, 10), savefig: str = None):
        """Visualise the MBAR weight matrix as a heatmap."""
        self.reshape_weights()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.reshaped_weights[:, ::10])
        ax.vlines(self.flat_inds[:, 1].min() / 10, 0, self.temperatures.shape[0],
                  color='red', label='equilibration')
        ax.set_yticks(np.arange(self.temperatures.shape[0])[::20], self.temperatures[::20])
        ax.set_xlabel('Replicate simulation time (ns)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Truncated MBAR weights')
        plt.legend(loc='upper left')
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
        return fig, ax


    # ------------------------------------------------------------------
    # Trajectory access and writing
    # ------------------------------------------------------------------

    def write_resampled_traj(self,
                              pdb_out: str,
                              dcd_out: str,
                              weights_out: str = None,
                              inds_out: str = None,
                              return_traj: bool = False,
                              correction: bool = True,
                              alt_load_positions_name: str = None):
        """Assemble and write the importance-resampled trajectory to disk."""
        if not hasattr(self, 'resampled_inds'):
            self.importance_resampling()

        if not hasattr(self, 'positions'):
            self._load_positions_box_vecs(positions_alt_name=alt_load_positions_name)

        n = len(self.resampled_inds)
        pos     = np.empty((n, self.positions[0].shape[2], 3))
        box_vec = np.empty((n, 3, 3))

        for i, (state, frame) in enumerate(self.resampled_inds):
            if i % max(1, n // 10) == 0:
                self._printf(f'{100 * i / n:.1f}% assembled', level='all')
            sim_no, sim_iter, sim_rep_ind = self.map[frame, state].astype(int)
            pos[i]     = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])

        self.traj = write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str, correction=correction)
        self.traj[0].save_pdb(pdb_out)
        self.traj.save_dcd(dcd_out)
        self._printf(f'{self.traj.n_frames} frames written to {pdb_out} and {dcd_out}', level='minimal')

        if weights_out is not None:
            np.save(weights_out, self.resampled_weights)
            self._printf(f'MBAR weights written to {weights_out}', level='all')

        if inds_out is not None:
            np.save(inds_out, self.resampled_inds)
            self._printf(f'Resampled indices written to {inds_out}', level='all')

        if return_traj:
            return self.traj


    def state_trajectory(self, state_no: int = 0, stride: int = 1) -> 'md.Trajectory':
        """Extract the continuous trajectory of a single thermodynamic state."""
        if not (hasattr(self, 'positions') or hasattr(self, 'box_vectors')):
            self._load_positions_box_vecs()

        inds = np.arange(0, self.energies.shape[0], stride)
        pos     = np.empty((len(inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(inds), 3, 3))

        for i, ind in enumerate(inds):
            sim_no, sim_iter, sim_rep_ind = self.map[ind, state_no].astype(int)
            pos[i]     = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])

        return write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str)


    # ------------------------------------------------------------------
    # PCA and reduced coordinates
    # ------------------------------------------------------------------

    def get_PCA(self,
                state_no: int = None,
                stride: int = 1,
                explained_variance_threshold: float = 0.9):
        """Project the trajectory onto its principal components."""
        if state_no is None:
            traj = deepcopy(self.traj)
        else:
            traj = self.state_trajectory(state_no, stride)
            if hasattr(self, 'upper_limit'):
                traj = traj[:self.upper_limit + 1]

        if self.resSeqs is not None:
            sele = traj.topology.select(
                f'resSeq {" ".join(str(r) for r in self.resSeqs)}'
            )
        else:
            sele = traj.topology.select('protein')
        traj = traj.atom_slice(sele)

        pca, self.reduced_cartesian, self.explained_variance, self.n_components = get_traj_PCA(
            traj, explained_variance_threshold=explained_variance_threshold
        )
        self._printf(f'Reduced Cartesian shape: {self.reduced_cartesian.shape}', level='all')


    def project_gpcr_pca(self,
                         pdb_code: str,
                         prefix: str,
                         resampled_dcd: str = None,
                         resampled_pdb: str = None,
                         chain: str = None,
                         resid_offset: int = 0,
                         stor_dir: str = './many_structures/',
                         out: str = None) -> np.ndarray:
        """
        Project the resampled trajectory onto a pre-computed GPCR PCA space.

        TODO (Phase 6): update this import once project_pca_gpcrs.py is migrated
        to chimpss.analysis.gpcr_pca and the sys.path hack below can be removed.
        """
        import json
        import tempfile

        import joblib
        import MDAnalysis as mda
        from MDAnalysis.coordinates.memory import MemoryReader

        # Locate the repo root (src/chimpss/fultonmarket/analysis.py -> go up 4 levels)
        _repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        from project_pca_gpcrs import (
            _imputation_setup,
            build_mobile_ag,
            fetch_bw_map,
            map_conserved_resids,
            project_trajectory,
        )

        if out is None:
            out = f"{pdb_code.upper()}_projections.npy"

        pca_path  = f"{prefix}_pca.joblib"
        meta_path = f"{prefix}_meta.json"
        ref_path  = f"{prefix}_ref.pdb"
        for path in (pca_path, meta_path, ref_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{path}' not found. "
                    f"Run generate_pca_gpcrs.py --prefix {prefix} first.")

        pca = joblib.load(pca_path)
        with open(meta_path, 'r') as fh:
            meta = json.load(fh)

        selection    = meta['selection']
        conserved_bw = meta['conserved_bw']
        expected_n   = meta['n_atoms']

        self._printf(
            f"GPCR PCA: {pca.n_components_} components, "
            f"trained on {len(meta['codes_retained'])} structures | "
            f"selection '{selection}' | expected atoms {expected_n}",
            level='minimal')

        ref_u  = mda.Universe(ref_path)
        ref_ag = ref_u.select_atoms('all')
        if ref_ag.n_atoms != expected_n:
            raise ValueError(
                f"Reference file '{ref_path}' has {ref_ag.n_atoms} atoms "
                f"but metadata says {expected_n}.  Regenerate the PCA with "
                f"the same --selection '{selection}'.")

        os.makedirs(stor_dir, exist_ok=True)
        self._printf(f"Fetching GPCRdb BW map for {pdb_code.upper()}...", level='minimal')
        _, bw_map = fetch_bw_map(pdb_code, stor_dir)

        if resampled_dcd is not None:
            top_path = resampled_pdb if resampled_pdb is not None else self.pdb
            u = mda.Universe(top_path, resampled_dcd)
            self._printf(
                f"Trajectory: {resampled_dcd} ({len(u.trajectory)} frames)",
                level='minimal')
        else:
            if not hasattr(self, 'traj'):
                raise RuntimeError(
                    "No trajectory available: call write_resampled_traj() "
                    "first, or supply resampled_dcd.")
            positions_aa = self.traj.xyz * 10.0
            tmp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False).name
            self.traj[0].save_pdb(tmp_pdb)
            u = mda.Universe(tmp_pdb, positions_aa, format=MemoryReader)
            os.unlink(tmp_pdb)
            self._printf(
                f"Trajectory: self.traj via MemoryReader ({len(u.trajectory)} frames)",
                level='minimal')

        if resid_offset:
            self._printf(f"Resid offset: {resid_offset:+d}", level='minimal')

        self._printf("Mapping conserved BW positions to trajectory residues...", level='all')
        resids, missing_info = map_conserved_resids(
            bw_map, conserved_bw, u,
            chain=chain,
            resid_offset=resid_offset)

        imputation = None
        if missing_info:
            n_missing = len(missing_info)
            if n_missing > 2:
                lines = "\n".join(f"  [{lbl}] {reason}"
                                  for lbl, _, reason in missing_info)
                raise Exception(
                    f"{n_missing} conserved BW positions are missing — "
                    f"too many for mean imputation (limit: 2).\n{lines}\n\n"
                    f"Retrain with generate_pca_gpcrs.py --bw_exclude "
                    f"to remove these positions.")

            self._printf(
                f"WARNING: {n_missing} residue(s) missing — mean-imputed "
                f"(zero contribution to all PCs):", level='minimal')
            for lbl, _, reason in missing_info:
                self._printf(f"  [{lbl}] {reason}", level='minimal')

            imputation = _imputation_setup(
                missing_info, conserved_bw, expected_n, pca)
            self._printf(
                f"  PC1 loading from imputed features: "
                f"{imputation['pc1_imputed_frac']:.1%} | "
                f"PC2: {imputation['pc2_imputed_frac']:.1%}",
                level='minimal')
            if (imputation['pc1_imputed_frac'] > 0.10
                    or imputation['pc2_imputed_frac'] > 0.10):
                self._printf(
                    "  WARNING: imputed features carry >10% of a PC's "
                    "loading — interpret projections with caution.",
                    level='minimal')
        else:
            self._printf(
                f"All {len(conserved_bw)} conserved positions mapped",
                level='all')

        mobile_ag = build_mobile_ag(u, resids, selection, chain=chain)
        expected_mobile = (expected_n if imputation is None
                           else len(imputation['present_feat_idx']) // 3)
        if mobile_ag.n_atoms != expected_mobile:
            raise ValueError(
                f"Selected {mobile_ag.n_atoms} atoms, expected {expected_mobile}.\n"
                f"  • Check chain (currently: {chain})\n"
                f"  • Check that all conserved residues are present\n"
                f"  • Training selection was '{selection}'")
        self._printf(
            f"Selected {mobile_ag.n_atoms} atoms "
            f"({len(conserved_bw) - len(missing_info)} present"
            + (f", {len(missing_info)} imputed" if missing_info else "")
            + ")",
            level='minimal')

        projections = project_trajectory(u, mobile_ag, ref_ag, pca,
                                         imputation=imputation)

        np.save(out, projections)
        self._printf(
            f"Projections saved → {out}  shape: {projections.shape}\n"
            f"  PC1 [{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}]  "
            f"PC2 [{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]",
            level='minimal')

        self.gpcr_pca_projections = projections
        return projections


    def get_weighted_reduced_cartesian(self,
                                        rc_upper_limit: int = None,
                                        return_weighted_rc: bool = False,
                                        use_state: bool = False,
                                        stride: int = 1):
        """Compute the MBAR-weighted mean reduced Cartesian coordinate and its uncertainty."""
        if rc_upper_limit is None:
            rc_upper_limit = np.inf

        if use_state:
            state_flat_inds = np.array([[0, ind] for ind in range(self.energies.shape[0])])[::stride]
            state_weights = np.ones(state_flat_inds.shape[0])
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = (
                calculate_weighted_rc(self.reduced_cartesian, state_flat_inds, rc_upper_limit,
                                      self.explained_variance, state_weights)
            )
        else:
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = (
                calculate_weighted_rc(self.reduced_cartesian, self.resampled_inds, rc_upper_limit,
                                      self.explained_variance, self.resampled_weights)
            )

        if return_weighted_rc:
            return self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err


    # ------------------------------------------------------------------
    # Trajectory truncation
    # ------------------------------------------------------------------

    def truncate(self):
        """Remove unused heavy atoms from saved position arrays to reduce disk usage."""
        self._load_positions_box_vecs(skip=0)

        keep_inds = get_truncation_atom_keep_inds(self.top)
        traj = md.load_pdb(self.pdb).atom_slice(keep_inds)
        self.top = deepcopy(traj.topology)

        for i, storage_dir in enumerate(self.storage_dirs[:-2]):
            top_fn = os.path.join(storage_dir, 'topology.pdb')
            if os.path.exists(top_fn):
                continue

            init_shape = self.positions[i].shape
            sim_pos = np.array(self.positions[i][:, :, keep_inds, :])
            self._printf(f'Truncated {storage_dir}: {init_shape} → {sim_pos.shape}', level='all')

            pos_fn = os.path.join(storage_dir, 'positions.npy')
            os.remove(pos_fn)
            mmap = np.memmap(pos_fn, mode='w+', dtype='float32', shape=sim_pos.shape)
            mmap[:] = sim_pos.copy()
            mmap.flush()
            traj[0].save_pdb(top_fn)

        self._load_positions_box_vecs()


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reshape_list(self, unshaped_list: List) -> List:
        """Reshape each energy array from replica order to state order."""
        return [
            self._reshape_array(unshaped_list[i], self.state_inds[i])
            for i in range(len(self.storage_dirs))
        ]


    def _reshape_array(self, unshaped_arr: np.ndarray, state_arr: np.ndarray) -> np.ndarray:
        """Reorder a single energy array so that axis-1 corresponds to thermodynamic state index."""
        reshaped_arr = np.empty(unshaped_arr.shape)
        for state in range(unshaped_arr.shape[1]):
            for iter_num in range(unshaped_arr.shape[0]):
                reshaped_arr[iter_num, state, :] = unshaped_arr[
                    iter_num, np.where(state_arr[iter_num] == state)[0], :
                ]
        return reshaped_arr


    def _get_postions_map(self):
        """Build a lookup map from (frame, state) to (sim_no, sim_iter, rep_ind)."""
        self.map = []
        for sim_no, sim_state_inds in enumerate(self.state_inds):
            n_iters, n_states = self.energies[sim_no].shape[:2]
            sim_map = np.empty((n_iters, n_states, 3), dtype=int)
            for sim_iter in range(n_iters):
                for sim_state in range(n_states):
                    rep_ind = np.where(sim_state_inds[sim_iter] == sim_state)[0][0]
                    sim_map[sim_iter, sim_state, :] = [sim_no, sim_iter, rep_ind]
            self.map.append(sim_map)


    def _determine_interpolation_inds(self):
        """Identify which thermodynamic state indices are missing from each sub-simulation."""
        final_set = self.temperatures
        self.interpolation_inds = []

        for set_i in self.temperatures_list:
            missing = [i for i, t in enumerate(final_set) if t not in set_i]
            assert len(missing) + len(set_i) == len(final_set), (
                f'Interpolation index mismatch: {len(missing)} missing + '
                f'{len(set_i)} present ≠ {len(final_set)} final'
            )
            self.interpolation_inds.append(missing)


    def _backfill(self):
        """Expand all sub-simulation energy matrices to the shape of the final temperature ladder."""
        self._determine_interpolation_inds()
        self._printf(f'Detected interpolation indices: {self.interpolation_inds}', level='all')

        filled_sim_inds = [
            i for i, inds in enumerate(self.interpolation_inds) if not inds
        ]

        interpolation_map = [
            np.arange(self.temperatures.shape[0])
            for _ in range(len(self.temperatures_list))
        ]
        for i, missing_inds in enumerate(self.interpolation_inds):
            for ind in missing_inds:
                interpolation_map[i] = interpolation_map[i][interpolation_map[i] != ind]

        backfilled_energies = []
        backfilled_map = []

        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            n_final = self.temperatures.shape[0]
            sim_energies = np.zeros((self.energies[sim_no].shape[0], n_final, n_final))
            sim_map      = np.zeros((self.map[sim_no].shape[0], n_final, 3))

            for i, ind in enumerate(interpolation_map[sim_no]):
                sim_energies[:, ind, interpolation_map[sim_no]] = self.energies[sim_no][:, i, :]
                sim_map[:, ind] = self.map[sim_no][:, i]

            for state_no in sim_interpolate_inds:
                filled_energies = np.concatenate([self.energies[s] for s in filled_sim_inds])
                filled_map      = np.concatenate([self.map[s]      for s in filled_sim_inds])

                state_energies = filled_energies[:, state_no]
                state_map      = filled_map[:, state_no]
                N_k = np.array([state_energies.shape[0]])

                res_energies, res_mappings, res_inds = resample_with_MBAR(
                    objs=[state_energies, state_map],
                    u_kln=np.array([state_energies[:, state_no]]),
                    N_k=N_k,
                    reshape_weights=state_energies.shape[0],
                    return_inds=True,
                    size=len(sim_energies),
                )

                sim_energies[:, state_no]    = res_energies.copy()
                sim_map[:, state_no]         = res_mappings.copy()
                sim_energies[:, :, state_no] = [
                    filled_energies[idx, :, state_no] for idx in res_inds
                ]

            backfilled_energies.append(sim_energies)
            backfilled_map.append(sim_map)

        self.energies  = np.concatenate(backfilled_energies, axis=0)
        self.map       = np.concatenate(backfilled_map, axis=0).astype(int)
        self.n_frames  = self.energies.shape[0]


    def _load_positions_box_vecs(self, positions_alt_name: str = None, skip: int = None):
        """Load position and box-vector arrays for all sub-simulations into memory-mapped numpy arrays."""
        if skip is None:
            skip = self.skip

        positions_fn = positions_alt_name if positions_alt_name is not None else 'positions.npy'
        truncated = False

        self.positions   = []
        self.box_vectors = []

        for i, storage_dir in enumerate(self.storage_dirs):
            pos_path = os.path.join(storage_dir, positions_fn)

            try:
                pos_i = np.load(pos_path, mmap_mode='r')[skip:]

            except Exception:
                top_fn = os.path.join(storage_dir, 'topology.pdb')
                if os.path.exists(top_fn):
                    truncated = True
                    top = md.load_pdb(top_fn).topology
                    self.top = deepcopy(top)
                else:
                    top = md.load_pdb(self.pdb).topology

                shape = (
                    self.unshaped_energies[i].shape[0] + self.skip,
                    self.unshaped_energies[i].shape[1],
                    top.n_atoms,
                    3,
                )
                try:
                    raw = np.memmap(pos_path, mode='r', dtype='float32', shape=shape)[skip:]
                    if truncated:
                        keep_inds = get_truncation_atom_keep_inds(top)
                        pos_i = raw[:, :, keep_inds]
                    else:
                        pos_i = raw
                except Exception as exc:
                    raise RuntimeError(
                        f'Cannot open {pos_path} as float32 with shape {shape}'
                    ) from exc

            # NB: do NOT advise deleting the directory here. This branch fires
            # whenever `skip` >= the segment's frame count, in which case the
            # data on disk is perfectly good and deleting it would throw away
            # completed production sampling. (It did read "delete the directory
            # and resume", which would have destroyed 44 intact lisuride
            # segments.) Genuine truncation is reported separately below.
            if pos_i.shape[0] == 0:
                on_disk = int(np.load(os.path.join(storage_dir, 'states.npy'),
                                      mmap_mode='r').shape[0])
                raise ValueError(
                    f'{storage_dir}: no frames survive skip={skip} '
                    f'(segment holds {on_disk} frames on disk). '
                    f'The saved data is intact -- lower `skip`; '
                    f'do not delete this directory.'
                )
            self.positions.append(pos_i)
            self._printf(f'Loaded positions from {storage_dir}: shape {pos_i.shape}', level='all')

            bv_path = os.path.join(storage_dir, 'box_vectors.npy')
            bv_i = np.load(bv_path, mmap_mode='r')[skip:]
            self.box_vectors.append(bv_i)
            self._printf(f'Loaded box vectors from {storage_dir}: shape {bv_i.shape}', level='all')



    # ==================================================================
    # Retro-analysis methods
    # ==================================================================

    def retro_analyze_all(
        self,
        n_resample: int = 100,
        sim_nos: List[int] = None,
        overwrite: bool = False,
        read_only: bool = False,
        output_cache_dir: str = None,
        getcontacts_script: str = None,
        conda_env: str = None,
        getcontacts_python: str = None,
        skip_contacts: bool = False,
    ) -> Dict[int, dict]:
        """Retroactively compute and save resampled distance matrices for all sub-simulations.

        Set `skip_contacts=True` to compute only the torsional and alpha-carbon
        matrices, which requires no getContacts installation.
        """
        import tempfile

        from chimpss.fultonmarket.retro_convergence import (
            compute_distance_matrices,
            load_matrices,
            log_mode,
            resolve_cache_dir,
            resolve_traj_paths,
            resolve_write_dir,
            save_matrices,
        )

        available = sorted(int(d.split('/')[-1]) for d in self.storage_dirs)
        targets   = sim_nos if sim_nos is not None else available
        skipped   = set(targets) - set(available)
        if skipped:
            self._printf(f'WARNING: sim_nos not found, skipping: {sorted(skipped)}', level='minimal')
        targets = [t for t in targets if t in available]

        log_mode(read_only, output_cache_dir, _printf=self._printf)
        self._printf(f'retro_analyze_all: processing {len(targets)} sub-simulations: {targets}', level='all')

        full_energies = self.energies
        full_map      = self.map
        frame_counts  = [e.shape[0] for e in self.unshaped_energies]

        all_matrices = {}

        for sim_no in targets:
            try:
                src_sim_dir   = self.storage_dirs[sim_no]
                write_sim_dir = resolve_write_dir(src_sim_dir, output_cache_dir, sim_no, read_only)

                cache_sim_dir = resolve_cache_dir(output_cache_dir, sim_no)
                existing = load_matrices(src_sim_dir, cache_sim_dir)
                n_expected = 2 if skip_contacts else 3
                if len(existing) >= n_expected and not overwrite:
                    self._printf(f'sim_no={sim_no}: all matrices present, skipping', level='all')
                    all_matrices[sim_no] = existing
                    continue

                self._printf(f'sim_no={sim_no}: computing matrices', level='minimal')

                upper = int(np.sum(frame_counts[:sim_no + 1])) - 1
                self.energies = full_energies[:upper + 1]
                self.map      = full_map[:upper + 1]

                for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                             'resampled_weights', 'positions', 'box_vectors', 'traj'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                self.importance_resampling(n_samples=n_resample)

                if read_only:
                    pdb_out      = '/dev/null'
                    dcd_out      = '/dev/null'
                    weights_out  = None
                    inds_out     = None
                    contacts_tsv = os.path.join(tempfile.mkdtemp(), 'contacts.tsv')
                else:
                    pdb_out, dcd_out, weights_out, inds_out = resolve_traj_paths(
                        src_sim_dir, write_sim_dir
                    )
                    contacts_tsv = os.path.join(
                        write_sim_dir or src_sim_dir, 'resampled_contacts.tsv'
                    )

                traj = self.write_resampled_traj(
                    pdb_out=pdb_out,
                    dcd_out=dcd_out,
                    weights_out=weights_out,
                    inds_out=inds_out,
                    return_traj=True,
                )

                matrices = compute_distance_matrices(
                    traj=traj,
                    pdb_out=pdb_out,
                    dcd_out=dcd_out,
                    contacts_tsv=contacts_tsv,
                    getcontacts_script=getcontacts_script,
                    conda_env=conda_env,
                    getcontacts_python=getcontacts_python,
                    skip_contacts=skip_contacts,
                    _printf=self._printf,
                )

                if not read_only:
                    save_matrices(matrices, write_sim_dir or src_sim_dir, sim_no, _printf=self._printf)
                else:
                    self._printf(f'sim_no={sim_no}: read_only=True — matrices in memory only', level='all')

                all_matrices[sim_no] = matrices

            except Exception as exc:
                self._printf(f'ERROR: sim_no={sim_no} failed: {exc}', level='minimal')
                import traceback
                traceback.print_exc()

        self.energies = full_energies
        self.map      = full_map
        for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                     'resampled_weights', 'positions', 'box_vectors', 'traj'):
            if hasattr(self, attr):
                delattr(self, attr)

        self._printf(f'retro_analyze_all complete: {len(all_matrices)} / {len(targets)} processed.', level='minimal')
        return all_matrices


    def retro_convergence_report(
        self,
        max_equil_fraction: float = 0.75,
        minimum_fraction: float = 0.25,
        frobenius_thresh: float = 0.05,
        jsd_thresh: float = 0.05,
        n_resample: int = 100,
        sim_nos: List[int] = None,
        read_only: bool = False,
        output_cache_dir: str = None,
        getcontacts_script: str = None,
        conda_env: str = None,
        getcontacts_python: str = None,
        skip_contacts: bool = False,
    ) -> Tuple[Dict[int, Dict[str, bool]], Dict[int, dict]]:
        """Evaluate convergence checks as a function of simulation progress.

        `skip_contacts` must match how the cached matrices were produced
        (retro_analyze_all(skip_contacts=True) caches torsion + alpha-carbon
        only). Without it, build_checks adds the two contact rows, which can
        never pass on absent scores, so STOP would be False at every checkpoint.
        """
        import tempfile

        from chimpss.fultonmarket.retro_convergence import (
            build_checks,
            compute_distance_matrices,
            evaluate_matrix_convergence,
            load_matrices,
            log_mode,
            print_sim_report,
            print_summary_table,
            resolve_cache_dir,
            resolve_traj_paths,
            resolve_write_dir,
            save_matrices,
        )

        total_n_sims = len(self.storage_dirs)
        available    = sorted(int(d.split('/')[-1]) for d in self.storage_dirs)
        targets      = sim_nos if sim_nos is not None else available
        targets      = [t for t in targets if t in available]

        log_mode(read_only, output_cache_dir, _printf=self._printf)
        self._printf(f'retro_convergence_report: {len(targets)} checkpoints, '
                     f'total_n_sims={total_n_sims} '
                     f'({100.0 * len(targets) / total_n_sims:.1f}% of simulation covered), '
                     f'frobenius_thresh={frobenius_thresh}, jsd_thresh={jsd_thresh}',
                     level='minimal')

        full_energies = self.energies
        full_map      = self.map
        frame_counts  = [e.shape[0] for e in self.unshaped_energies]

        matrix_cache: Dict[int, dict] = {}
        for sim_no in targets:
            src_sim_dir   = self.storage_dirs[sim_no]
            cache_sim_dir = resolve_cache_dir(output_cache_dir, sim_no)
            existing      = load_matrices(src_sim_dir, cache_sim_dir)
            if existing:
                matrix_cache[sim_no] = existing
                self._printf(f'sim_no={sim_no} ({100.0*(sim_no+1)/total_n_sims:.1f}%): '
                             f'pre-loaded {len(existing)} cached matrices from disk', level='all')

        report  = {}
        metrics = {}

        for sim_no in targets:
            progress_pct = 100.0 * (sim_no + 1) / total_n_sims
            self._printf(f'\n{"=" * 60}', level='all')
            self._printf(f'Evaluating checkpoint {progress_pct:.1f}% (sim_no={sim_no})', level='all')

            try:
                src_sim_dir   = self.storage_dirs[sim_no]
                write_sim_dir = resolve_write_dir(src_sim_dir, output_cache_dir, sim_no, read_only)

                upper = int(np.sum(frame_counts[:sim_no + 1])) - 1
                self.energies = full_energies[:upper + 1]
                self.map      = full_map[:upper + 1]

                for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                             'resampled_weights', 'positions', 'box_vectors', 'traj'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                self.determine_equilibration(equilibration_method='energy')
                equil_fraction = self.t0 / self.energies.shape[0]

                min_post_equil       = 1.0 - max_equil_fraction
                effective_post_equil = max(1.0 - equil_fraction, min_post_equil)
                first_valid_sim_no   = int((1.0 - effective_post_equil) * sim_no)

                if sim_no not in matrix_cache:
                    self._printf('  computing distance matrices on-the-fly', level='all')

                    self.importance_resampling(n_samples=n_resample)

                    if read_only:
                        pdb_out      = '/dev/null'
                        dcd_out      = '/dev/null'
                        weights_out  = None
                        inds_out     = None
                        contacts_tsv = os.path.join(tempfile.mkdtemp(), 'contacts.tsv')
                    else:
                        pdb_out, dcd_out, weights_out, inds_out = resolve_traj_paths(
                            src_sim_dir, write_sim_dir
                        )
                        contacts_tsv = os.path.join(
                            write_sim_dir or src_sim_dir, 'resampled_contacts.tsv'
                        )

                    traj = self.write_resampled_traj(
                        pdb_out=pdb_out,
                        dcd_out=dcd_out,
                        weights_out=weights_out,
                        inds_out=inds_out,
                        return_traj=True,
                    )

                    matrices = compute_distance_matrices(
                        traj=traj,
                        pdb_out=pdb_out,
                        dcd_out=dcd_out,
                        contacts_tsv=contacts_tsv,
                        getcontacts_script=getcontacts_script,
                        conda_env=conda_env,
                        getcontacts_python=getcontacts_python,
                        skip_contacts=skip_contacts,
                        _printf=self._printf,
                    )

                    if not read_only:
                        save_matrices(matrices, write_sim_dir or src_sim_dir, sim_no, _printf=self._printf)

                    matrix_cache[sim_no] = matrices

                current_matrices = matrix_cache[sim_no]

                frob_results, jsd_results = evaluate_matrix_convergence(
                    current_matrices=current_matrices,
                    matrix_cache=matrix_cache,
                    first_valid_sim_no=first_valid_sim_no,
                    current_sim_no=sim_no,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                )

                checks = build_checks(
                    sim_no=sim_no,
                    total_n_sims=total_n_sims,
                    minimum_fraction=minimum_fraction,
                    equil_fraction=equil_fraction,
                    max_equil_fraction=max_equil_fraction,
                    frob_results=frob_results,
                    jsd_results=jsd_results,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                    skip_contacts=skip_contacts,
                )

                print_sim_report(
                    sim_no=sim_no,
                    total_n_sims=total_n_sims,
                    checks=checks,
                    frob_results=frob_results,
                    jsd_results=jsd_results,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                    equil_fraction=equil_fraction,
                    effective_post_equil=effective_post_equil,
                    first_valid_sim_no=first_valid_sim_no,
                    _printf=self._printf,
                )

                report[sim_no] = checks

                comparison_sim_nos = sorted(
                    set().union(*(d.keys() for d in frob_results.values()))
                )
                metrics[sim_no] = {
                    'equil_fraction':      equil_fraction,
                    'first_valid_sim_no':  first_valid_sim_no,
                    'effective_post_equil': effective_post_equil,
                    'comparison_sim_nos':  comparison_sim_nos,
                    'frobenius': {
                        name: [frob_results[name].get(s) for s in comparison_sim_nos]
                        for name in ('torsion', 'alpha_carbon', 'contact')
                    },
                    'jsd': {
                        name: [jsd_results[name].get(s) for s in comparison_sim_nos]
                        for name in ('torsion', 'alpha_carbon', 'contact')
                    },
                }

            except Exception as exc:
                self._printf(f'ERROR: sim_no={sim_no} failed: {exc}', level='minimal')
                import traceback
                traceback.print_exc()

        self.energies = full_energies
        self.map      = full_map
        for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                     'resampled_weights', 'positions', 'box_vectors', 'traj'):
            if hasattr(self, attr):
                delattr(self, attr)

        print_summary_table(report, total_n_sims, _printf=self._printf)
        return report, metrics
