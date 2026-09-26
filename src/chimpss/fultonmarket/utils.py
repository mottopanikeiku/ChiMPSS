from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mdtraj as md
import netCDF4 as nc
import numpy as np
import openmm.unit as unit
import scipy.constants as cons
from openmm import *
from openmm.app import *
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.utils.utils import TrackedQuantity
from pymbar import MBAR
from pymbar.timeseries import detect_equilibration
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

from chimpss.fultonmarket.contact_network import ContactNetworkBuilder


def printf(my_string):
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)


def get_kT(temp):
    return temp * cons.gas_constant


def geometric_distribution(min_val, max_val, n_vals):
    return [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0)
            for i in range(n_vals)]


try:
    spring_constant_unit = (unit.joule)/(unit.angstrom*unit.angstrom*unit.mole)
except TypeError:
    spring_constant_unit = None

perms = jnp.array([x for x in itertools.product([-1, 0, 1], repeat=3)])


def jaxrmsd(a, b):
    return jnp.sqrt(jnp.mean(jnp.sum((b-a)**2, axis=-1), axis=-1))


def _jax_add_fn(a, b):
    return a + b


jax_add = jax.vmap(_jax_add_fn, in_axes=(0, None))
rmsd_j = jax.vmap(jaxrmsd, in_axes=(0, None))

def convert_to_TrackedQuantity(arr: np.array, u: openmm.unit):
    return TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=arr, mask=False, fill_value=1e+20), unit=u))


def _interpolate_new_states(prev_temps, insert_inds):

    # Add new states
    new_temps = [temp for temp in prev_temps]
    for displacement, ind in enumerate(insert_inds):
        temp_below = prev_temps[ind-1]
        temp_above = prev_temps[ind]
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state at', np.mean((temp_below, temp_above)), flush=True)
        new_temps.insert(ind + displacement, np.mean((temp_below, temp_above)))
    temperatures = [temp*unit.kelvin for temp in new_temps]
    n_replicates = len(temperatures)

    return temperatures, n_replicates


def _interpolate_new_positions(init_positions, init_box_vectors, init_velocities, insert_inds, n_replicates):

    # Add pos, box_vecs, velos for new temperatures
    init_positions = np.insert(init_positions, insert_inds, [init_positions[ind-1] for ind in insert_inds], axis=0)
    init_box_vectors = np.insert(init_box_vectors, insert_inds, [init_box_vectors[ind-1] for ind in insert_inds], axis=0)
    if init_velocities is not None:
        init_velocities = np.insert(init_velocities, insert_inds, [init_velocities[ind-1] for ind in insert_inds], axis=0)

    # Convert to quantities
    init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_positions, mask=False, fill_value=1e+20), unit=unit.nanometer))
    init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_box_vectors.reshape(n_replicates, 3, 3), mask=False, fill_value=1e+20), unit=unit.nanometer))

    if init_velocities is not None:
        init_velocities = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_velocities, mask=False, fill_value=1e+20), unit=(unit.nanometer / unit.picosecond)))

    return init_positions, init_box_vectors, init_velocities


def build_thermodynamic_states(self):

    # Build thermodynamic states
    printf(f'Creating {len(self.temperatures)} Thermodynamic States')
    self.thermodynamic_states = [ThermodynamicState(system=self.system, temperature=T) for T in self.temperatures]
    printf('Done Creating Thermodynamic States')
    printf(f'Assigning {len(self.spring_centers)} Restraints')
    assert len(self.temperatures) == len(self.spring_centers)


def build_sampler_states(n_replicates: int, pos: np.array, box_vec: np.array, velos: np.array=None):

    if velos is not None:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i], velocities=velos[i]) for i in range(n_replicates)]

    else:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i]) for i in range(n_replicates)]



def truncate_ncdf(ncdf_in, ncdf_out, out_dir, reporter, is_checkpoint: bool=False):
    print(f'Truncating {ncdf_in} to {ncdf_out}')

    src = nc.Dataset(ncdf_in, 'r')
    dest = nc.Dataset(ncdf_out, 'w')

    for name in src.ncattrs():
        dest.setncattr(name, src.getncattr(name))

    for dim_name, dim in src.dimensions.items():
        dest.createDimension(dim_name, (len(dim) if not dim.isunlimited() else None))

    for group_name, group in src.groups.items():
        dest.createGroup(group_name)
        for name, variable in src[group_name].variables.items():
            try:
                dest[group_name].createVariable(name, variable.datatype, variable.dimensions)
                dest[group_name][name][:] = src[group_name][name][:]
                dest[group_name][name].setncatts(src[group_name][name].__dict__)
            except Exception:
                print(group_name, name)
                pass

    for var_name, var in src.variables.items():
        var_out = dest.createVariable(var_name, var.datatype, var.dimensions)
        var_out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

        if not is_checkpoint:
            if var_name == 'positions':
                nframes, nstates, natoms, _ = var.shape
                pos_fn = os.path.join(out_dir, 'positions.npy')
                pos = np.memmap(pos_fn, dtype='float32', mode='w+', shape=var.shape)
                print(f'Saving positions to {pos_fn} with shape {var.shape}')
                for frame in range(nframes):
                    pos[frame] = var[frame].data
                pos.flush()
            elif var_name == 'box_vectors':
                box_vecs = var[:].copy()
            elif var_name == 'states':
                states = var[:].copy()
            elif var_name == 'energies':
                energies = var[:].copy().astype('float32')
            elif var_name == 'velocities':
                velocities = var[-1].copy().astype('float16')

        if var.dimensions[0] == 'iteration':
            if is_checkpoint:
                var_out[:] = var[-1:]
            else:
                var_out[:] = var[-10:]

        elif var_name == 'last_iteration':
            var_out[:] = var[:]
            if not is_checkpoint:
                mask_copy = var_out[:].copy()
                var_out[:] = np.ma.array(9, mask=mask_copy.mask, fill_value=mask_copy.fill_value)
                print(var_out)

        else:
            var_out[:] = var[:]

    dest.close()
    src.close()

    # Read reporter
    if not is_checkpoint:

        # Read temperatures
        temps = np.array([state.temperature._value for state in reporter.read_thermodynamic_states()[0]])


        # Close reporter
        reporter.close()

        return pos, velocities, box_vecs, states, energies, temps


def rmsd(a, b):
    if len(a.shape) == 1:
        return np.sqrt(np.mean((a - b)**2))
    else:
        return np.array([np.sqrt(np.mean((a[i] - b[i])**2)) for i in range(a.shape[0])])

def plot_MRC(domains, mean_weighted_rc, mean_weighted_rc_err, savefig: str=None):

    fig, ax = plt.subplots()

    ax.plot(domains, mean_weighted_rc, color='k')
    ax.errorbar(domains, np.abs( mean_weighted_rc - mean_weighted_rc[-1]), yerr=mean_weighted_rc_err, color='k', capsize=3)
    ax.set_ylabel('| MRC(t) - MRC(T) |')
    ax.set_xlabel('Total sim time (ns')
    plt.show()

    if savefig is not None:
        fig.savefig(savefig)


def PCA_convergence_detection(rc, rc_err):

    converged = np.array([False for i in range(len(rc)-1)])
    for i, (rc_i, rc_err_i) in enumerate(zip(rc[:-1], rc_err[:-1])):
        if (rc_i > rc[-1] and rc_i - rc_err_i <= rc[-1]) or (rc_i <= rc[-1] and rc_i + rc_err_i >= rc[-1]):
            converged[i] = True
        else:
            converged[:i+1] = False

    return converged


def write_traj_from_pos_boxvecs(pos, box_vec, top, sele_str, receptor_sele_str='chainid 0', correction: bool=True):

    # Create traj obj
    traj = md.Trajectory(xyz=pos.copy(),
                         topology=top,
                         time=np.arange(pos.shape[0]),
                         unitcell_lengths=box_vec.sum(axis=1),
                         unitcell_angles=np.repeat([90,90,90], pos.shape[0]).astype(np.float32).reshape(pos.shape[0], 3))

    # Correct periodic issues
    prot_sele = traj.topology.select(receptor_sele_str) # receptor should always be chainid 0
    if sele_str is not None and correction:
        if isinstance(sele_str, str):
            sele_str = [sele_str]
        lig_seles = []
        lig_coms = []
        for s in sele_str:
            try:
                lig_sele = traj.topology.select(s)
            except Exception:
                raise Exception(f'failed to parse {s}')
            lig_seles.append(lig_sele)
            lig_coms.append(md.compute_center_of_mass(traj.atom_slice(lig_sele)))
        prot_com = md.compute_center_of_mass(traj.atom_slice(prot_sele))
        for frame in range(traj.n_frames):
            for (s, lig_sele, lig_com) in zip(sele_str, lig_seles, lig_coms):
                best_trans, _ = best_translation_by_unitcell(traj.unitcell_lengths[frame], lig_com[frame], prot_com[frame])
                traj.xyz[frame, lig_sele, :] += best_trans


    # Align frames for viewing purposes
    if correction:
        traj = traj.superpose(traj, atom_indices=prot_sele, ref_atom_indices=prot_sele)

    return traj


def get_traj_PCA(traj, explained_variance_threshold: float=None):
    """
    """
    # PCA
    pca = PCA()
    reduced_cartesian = pca.fit_transform(traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3))
    explained_variance = np.array([np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(pca.n_components_)])

    if explained_variance_threshold is not None:
        n_components = int(np.where(explained_variance >= explained_variance_threshold)[0][0])

    return pca, reduced_cartesian[:,:n_components], explained_variance[:n_components], n_components


def calculate_weighted_rc(reduced_cartesian, resampled_inds, upper_limit, pca_weights, mbar_weights):
    assert reduced_cartesian.shape[0] == len(resampled_inds), f'{reduced_cartesian.shape}, {resampled_inds.shape}'
    assert reduced_cartesian.shape[0] == len(mbar_weights)
    assert reduced_cartesian.shape[1] == len(pca_weights)

    mean_weighted_rcs = []
    for (rc, frame_no, mbar_weight) in zip(reduced_cartesian, resampled_inds[:,1], mbar_weights):
        if frame_no <= upper_limit:
            mean_weighted_rcs.append(np.mean(rc * pca_weights) * pca_weights)

    return np.mean(mean_weighted_rcs), np.std(mean_weighted_rcs) / np.sqrt(reduced_cartesian.shape[0])


def resample_with_MBAR(objs: List, u_kln: np.array, N_k: np.array, size: int, reshape_weights: tuple=None, specify_state: int=0, return_inds: bool=False, return_weights: bool=False, return_resampled_weights: bool=False, replace: bool=True, _printf=None):

    # Get MBAR weights
    weights = compute_MBAR_weights(u_kln, N_k)

    # Reshape weights if specified
    if reshape_weights is not None:
        weights = weights.reshape(reshape_weights)

    # Get probabilities
    if len(weights.shape) == 1:
        probs = weights.copy()
    else:
        probs = weights[:, specify_state]


    _log = _printf if _printf is not None else printf
    # Resample
    if size == -1:
        resampled_inds = np.where(probs >= probs.max()*0.001)[0]
        _log(f'Top 99.9% of probability includes {len(resampled_inds)} no. of frames')
    else:
        probs /= np.nan_to_num(probs).sum() # Renormalize to avoid errors
        probs = np.nan_to_num(probs)
        try:
            resampled_inds = np.random.choice(range(len(probs)), size=size, replace=replace, p=probs)
        except Exception:
            resampled_inds = np.random.choice(range(len(probs)), size=len(np.where(probs > 0)[0]), replace=replace, p=probs)

    resampled_objs = []
    for obj in objs:
        resampled_objs.append(np.array([obj[resampled_ind] for resampled_ind in resampled_inds]))
    # Return resampled objects
    return_list = []
    if len(objs) == 1:
        return_list.append(resampled_objs[0])
    elif len(objs) > 1:
        for resampled_obj in resampled_objs:
            return_list.append(resampled_obj)

    # Optional returns
    if return_inds:
        return_list.append(resampled_inds)
    if return_weights:
        return_list.append(weights)
    if return_resampled_weights:
        resampled_weights = weights[resampled_inds, specify_state]
        return_list.append(resampled_weights)

    return return_list


def compute_MBAR_weights(u_kln, N_k):
    """
    Compute MBAR sample weights, tolerating a broken BAR initialiser.

    `initialize='BAR'` is only a warm start for the self-consistent solver, but
    in the installed pymbar it is a hard failure whenever any state pair lacks
    overlap: `MBAR._initialize_with_bar` wraps its `bar()` call in
    `except ConvergenceError:`, and `ConvergenceError` is never imported into
    `pymbar/mbar.py` (it lives in `pymbar/utils.py`). `bar()` legitimately
    raises `BoundsError`/`ConvergenceError` on poor overlap, and evaluating the
    except clause then dies with `NameError: name 'ConvergenceError' is not
    defined`. That killed 29 of 60 sub-simulations in the 2026-08-14 retro run.

    Poor overlap is expected at low sim_no, where only a few sub-simulations'
    worth of frames exist, so fall back to the default zeros initialisation
    rather than losing the sub-simulation. The converged weights are the same;
    only the number of solver iterations differs.
    """
    try:
        mbar = MBAR(u_kln, N_k, initialize='BAR')
    except Exception as exc:
        # NameError is the pymbar bug above; other failures here are equally
        # confined to the warm start, so retry without it before giving up.
        # Always emitted: FultonMarketAnalysis._printf defaults to level='all',
        # which verbosity='minimal' suppresses, and a fallback that changes the
        # solver path must stay visible in the log.
        printf(f"MBAR BAR-initialisation failed ({type(exc).__name__}: {exc}); "
             f"falling back to zeros initialisation.")
        mbar = MBAR(u_kln, N_k)

    return mbar.weights()

def detect_PC_equil(pc, reduced_cartesian):
    t0, _, _ = detect_equilibration(reduced_cartesian[:,pc])
    return t0

def detect_energy_equil(avg_energies):
    t0, _, _ = detect_equilibration(avg_energies)
    return t0

def get_truncation_atom_keep_inds(top):
    remove_atom_inds = top.select('resname HOH NA CL POP')
    return np.array([i for i in range(top.n_atoms) if i not in remove_atom_inds])

def best_translation_by_unitcell(cell_lengths, mobile_coords, target_coords):
    perms = np.array([x for x in itertools.product([-1, 0, 1], repeat=3)])
    translations = cell_lengths * perms
    permuted_positions = np.array([np.sum((translations[i], mobile_coords), axis=0) for i in range(translations.shape[0])])
    rmsds_of_permutations = np.array([rmsd(permuted_positions[i], target_coords) for i in range(permuted_positions.shape[0])])
    return translations[np.argmin(rmsds_of_permutations)], rmsds_of_permutations[np.argmin(rmsds_of_permutations)]

def best_translation_by_unitcell_jax(cell_lengths, mobile_coords, target_coords):
    translations = cell_lengths * perms
    permuted_positions = jax_add(translations, mobile_coords)
    rmsds_of_permutations = rmsd_j(permuted_positions, target_coords)
    return translations[jnp.argmin(rmsds_of_permutations)], rmsds_of_permutations[jnp.argmin(rmsds_of_permutations)]

best_translation_by_unitcell_jax = jax.vmap(best_translation_by_unitcell_jax, in_axes=(0, 0, None))

def torsional_distance_on_period(a, b, n):
    return (1/n) * (180 - jnp.abs((jnp.mod(jnp.abs(n * (a - b)), 360)-180)))

def get_angles_and_periods(traj):
    #Determine some torsions
    res_names = np.array([traj.top.atom(i).residue.name for i in range(traj.top.n_atoms)])
    phi_inds, phi = md.compute_phi(traj)
    psi_inds, psi = md.compute_psi(traj)
    chi1_inds, chi1 = md.compute_chi1(traj)
    chi2_inds, chi2 = md.compute_chi2(traj)

    chi2_residues = res_names[chi2_inds][:, 0] #All chi2 atoms are the same residue
    chi2_ns = jnp.array([2 if name in ["PHE", "TYR", "ASP", "LEU"] else 1 for name in chi2_residues])
    angles = jnp.concatenate([phi, psi, chi1, chi2], axis=1)
    periods = jnp.concatenate((jnp.ones(phi_inds.shape[0]),
                               jnp.ones(psi_inds.shape[0]),
                               jnp.ones(chi1_inds.shape[0]),
                               chi2_ns), axis=-1)

    return angles, periods


def getTorsionalDistanceMatrix(traj, selection_string=None):
    if selection_string:
        traj = traj.atom_slice(traj.top.select(selection_string))
    angles, periods = get_angles_and_periods(traj)
    distance_matrix = jnp.empty((angles.shape[0], angles.shape[0]))
    def row_op(angle_i, angle_is):
        return jax.vmap(torsional_distance_on_period, in_axes=(None, 0, None))(angle_i, angle_is, periods)
    #iterate over the rows, each pair is parallel across the GPU
    for i in range(angles.shape[0]):
        distance_matrix = distance_matrix.at[i, :].set(jnp.sqrt(jnp.mean((row_op(angles[i], angles))**2, axis=-1)))
    return distance_matrix


def getAlphaCarbonDistanceMatrix(traj, selection_string=None):
    import itertools

    if selection_string:
        traj = traj.atom_slice(traj.top.select(selection_string))
    CA_inds = traj.top.select('name CA')
    #The 'features' are each distance between pairs of alpha_carbons
    CA_pair_distances = md.compute_distances(traj, itertools.combinations(CA_inds, 2))
    distance_matrix = jnp.empty((CA_pair_distances.shape[0], CA_pair_distances.shape[0]))

    def ca_dist_func(a, b):
        return jnp.abs(b-a)

    def row_op(dist_i, dist_is):
        return jax.vmap(ca_dist_func, in_axes=(None, 0))(dist_i, dist_is)
    #iterate over the rows, each pair is parallel across the GPU
    for i in range(CA_pair_distances.shape[0]):
        distance_matrix = distance_matrix.at[i, :].set(jnp.sqrt(jnp.mean((row_op(CA_pair_distances[i], CA_pair_distances))**2, axis=-1)))
    return distance_matrix


def getContactDistanceMatrix(top_fn, traj_fn, output_fn, conda_env=None, getcontacts_script=None, getcontacts_python=None, cores: int = 10, _printf=None):
    """
    Run getContacts in a specified conda environment and compute a frame x frame
    contact distance matrix.

    Parameters
    ----------
    top_fn : str
        Path to topology PDB file.
    traj_fn : str
        Path to trajectory DCD file.
    output_fn : str
        Path for the getContacts TSV output file.
    conda_env : str, optional
        Name of the conda environment containing getContacts.
    getcontacts_script : str
        Path to the get_dynamic_contacts.py script. Required.
    getcontacts_python : str, optional
        Explicit path to the Python interpreter to use.
    cores : int
        Number of CPU cores passed to getContacts via --cores. Default 10.
    """
    if getcontacts_script is None:
        raise RuntimeError(
            "getcontacts_script must be provided. "
            "Pass the path to get_dynamic_contacts.py via getContacts_Info."
        )

    if getcontacts_python is not None:
        conda_python = getcontacts_python
    else:
        if conda_env is None:
            raise RuntimeError(
                "conda_env must be specified when getcontacts_python is not provided. "
                "Pass conda_env in getContacts_Info."
            )
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if not conda_prefix:
            raise RuntimeError(
                "CONDA_PREFIX is not set and getcontacts_python was not provided. "
                "Provide getcontacts_python explicitly via getContacts_Info."
            )
        _log = _printf if _printf is not None else printf
        current_env = os.path.basename(conda_prefix)
        _log(f"Detected current conda env: '{current_env}' — switching to: '{conda_env}'")
        conda_python = os.path.join(conda_prefix.replace(current_env, conda_env), 'bin', 'python')
        if not os.path.exists(conda_python):
            raise RuntimeError(
                f"Cannot locate Python at '{conda_python}'. "
                f"Resolved by replacing '{current_env}' with '{conda_env}' in CONDA_PREFIX. "
                "Provide getcontacts_python explicitly via getContacts_Info."
            )

    cmd = [conda_python, getcontacts_script,
           '--topology', top_fn,
           '--trajectory', traj_fn,
           '--output', output_fn,
           '--cores', str(cores),
           '--sele', 'protein or resname UNK',
           '--ligand', 'resname UNK',
           '--lipid', 'resname POP',
           '--itypes', 'all',
           '--stride', '1']

    #Run these on da line
    _log = _printf if _printf is not None else printf
    _log(f"Running Process: {chr(10).join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing commands. Return code: {e.returncode}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        sys.exit(e.returncode)
        raise Exception("Shouldn't handle gracefully...")

    X, features = ContactNetworkBuilder.from_tsv(tsv_fn=output_fn,
                                                 pdb_fn=top_fn,
                                                 token_mode='residue_atomclass').get_contact_vectors()
    X = X.astype(jnp.int8)
    def contact_dist_func(a, b):
        return jnp.mean(jnp.abs(b-a))  # All zero where equal, all one where not, return percentage

    def row_op(cont_i, cont_is):
        return jax.vmap(contact_dist_func, in_axes=(None, 0))(cont_i, cont_is)
    matrix = jax.vmap(row_op, in_axes=(0,None))(X, X)

    return matrix, features



def frobenius_norm(matrix_a: np.ndarray, matrix_b: np.ndarray, normalise: bool = True) -> float:
    """
    Compute the Frobenius norm of the difference between two distance matrices.

    Parameters
    ----------
    matrix_a : np.ndarray of shape (n, n)
    matrix_b : np.ndarray of shape (n, n)
    normalise : bool
        Divide by n*(n-1)/2 to make size-independent. Default True.
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(f'Matrices must have the same shape, got {matrix_a.shape} and {matrix_b.shape}.')
    raw_norm = np.sqrt(np.sum((matrix_a - matrix_b) ** 2))
    if normalise:
        n = matrix_a.shape[0]
        n_pairs = n * (n - 1) / 2
        return float(raw_norm / n_pairs)
    return float(raw_norm)


def jsd_distance_matrices(matrix_a: np.ndarray,
                           matrix_b: np.ndarray,
                           n_bins: int = 100) -> float:
    """
    Compute the Jensen-Shannon Divergence between the pairwise distance
    distributions of two distance matrices.
    """
    # Extract upper triangle (i > j, no diagonal)
    triu_inds = np.triu_indices_from(matrix_a, k=1)
    pairs_a = matrix_a[triu_inds]
    pairs_b = matrix_b[triu_inds]

    combined_min = min(pairs_a.min(), pairs_b.min())
    combined_max = max(pairs_a.max(), pairs_b.max())
    bins = np.linspace(combined_min, combined_max, n_bins + 1)

    hist_a, _ = np.histogram(pairs_a, bins=bins, density=True)
    hist_b, _ = np.histogram(pairs_b, bins=bins, density=True)

    eps = 1e-10
    hist_a = hist_a + eps
    hist_b = hist_b + eps

    return float(jensenshannon(hist_a, hist_b, base=2))
