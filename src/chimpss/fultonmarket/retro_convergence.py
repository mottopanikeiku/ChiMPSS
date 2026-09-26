"""
retro_convergence.py

Utility functions supporting FultonMarketAnalysis.retro_analyze_all and
FultonMarketAnalysis.retro_convergence_report. Not intended for direct use.
"""

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np


def printf(x):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)

MATRIX_NAMES = ('torsion', 'alpha_carbon', 'contact')


# ---------------------------------------------------------------------------
# Write directory resolution
# ---------------------------------------------------------------------------

def resolve_write_dir(
    src_sim_dir: str,
    output_cache_dir: Optional[str],
    sim_no: int,
    read_only: bool,
) -> Optional[str]:
    """
    Return the directory that derived files should be written to, creating
    it if necessary. Returns None in read-only mode.
    """
    if read_only:
        return None
    if output_cache_dir is not None:
        write_dir = os.path.join(output_cache_dir, 'saved_variables', str(sim_no))
        os.makedirs(write_dir, exist_ok=True)
        return write_dir
    return src_sim_dir


def resolve_cache_dir(output_cache_dir: Optional[str], sim_no: int) -> Optional[str]:
    """
    Return the cache directory path for a sim_no WITHOUT creating it. Used for
    reads only.
    """
    if output_cache_dir is None:
        return None
    return os.path.join(output_cache_dir, 'saved_variables', str(sim_no))


def resolve_traj_paths(
    src_sim_dir: str,
    write_sim_dir: Optional[str],
) -> Tuple[str, str, str, str]:
    """
    Resolve PDB/DCD/weights/indices paths for a sub-simulation.
    """
    write_dir = write_sim_dir or src_sim_dir

    def _pick(filename):
        if write_sim_dir:
            candidate = os.path.join(write_sim_dir, filename)
            if os.path.exists(candidate):
                return candidate
        src = os.path.join(src_sim_dir, filename)
        if os.path.exists(src):
            return src
        return os.path.join(write_dir, filename)

    return (
        _pick('resampled_top.pdb'),
        _pick('resampled_trj.dcd'),
        os.path.join(write_dir, 'resampled_wghts.npy'),
        os.path.join(write_dir, 'resampled_indcs.npy'),
    )


# ---------------------------------------------------------------------------
# Matrix I/O
# ---------------------------------------------------------------------------

def load_matrices(src_sim_dir: str, cache_sim_dir: Optional[str]) -> dict:
    """
    Load distance matrices, checking cache_sim_dir first then falling back
    to src_sim_dir.
    """
    matrices = {}
    for name in MATRIX_NAMES:
        filename = f'resampled_{name}_matrix.npy'
        if cache_sim_dir is not None:
            cache_path = os.path.join(cache_sim_dir, filename)
            if os.path.exists(cache_path):
                matrices[name] = np.load(cache_path)
                continue
        src_path = os.path.join(src_sim_dir, filename)
        if os.path.exists(src_path):
            matrices[name] = np.load(src_path)
    return matrices


def save_matrices(matrices: dict, write_dir: str, sim_no: int, _printf=None):
    """Save distance matrices to write_dir."""
    _log = _printf if _printf is not None else printf
    for name, matrix in matrices.items():
        out_path = os.path.join(write_dir, f'resampled_{name}_matrix.npy')
        np.save(out_path, matrix)
        _log(f'sim_no={sim_no}: saved {name} matrix -> {out_path}')


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def compute_distance_matrices(
    traj,
    pdb_out: str,
    dcd_out: str,
    contacts_tsv: str,
    getcontacts_script: str = None,
    conda_env: str = None,
    getcontacts_python: str = None,
    skip_contacts: bool = False,
    _printf=None,
) -> dict:
    """
    Compute torsional, alpha-carbon, and contact distance matrices from a
    resampled MDTraj trajectory.

    When `skip_contacts` is True the contact matrix is omitted and only the
    torsional and alpha-carbon matrices are returned. This mirrors the
    `skip_contacts` option on the live convergence path in
    `FultonMarket._evaluate_stopping_criterion`, and makes retroactive
    convergence analysis possible without a getContacts installation.
    """
    from chimpss.fultonmarket.utils import (
        getAlphaCarbonDistanceMatrix,
        getContactDistanceMatrix,
        getTorsionalDistanceMatrix,
    )

    torsional    = getTorsionalDistanceMatrix(traj, selection_string='protein or resname UNK')
    alpha_carbon = getAlphaCarbonDistanceMatrix(traj, selection_string='protein or resname UNK')

    if skip_contacts:
        _log = _printf if _printf is not None else printf
        _log('skip_contacts=True — contact distance matrix omitted.')
        return {
            'torsion':      torsional,
            'alpha_carbon': alpha_carbon,
        }

    contact_kwargs = dict(top_fn=pdb_out, traj_fn=dcd_out, output_fn=contacts_tsv)
    if getcontacts_script is not None:
        contact_kwargs['getcontacts_script'] = getcontacts_script
    if conda_env is not None:
        contact_kwargs['conda_env'] = conda_env
    if getcontacts_python is not None:
        contact_kwargs['getcontacts_python'] = getcontacts_python
    if _printf is not None:
        contact_kwargs['_printf'] = _printf

    contact_distance, _ = getContactDistanceMatrix(**contact_kwargs)

    return {
        'torsion':      torsional,
        'alpha_carbon': alpha_carbon,
        'contact':      contact_distance,
    }


# ---------------------------------------------------------------------------
# Convergence evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_matrix_convergence(
    current_matrices: dict,
    matrix_cache: Dict[int, dict],
    first_valid_sim_no: int,
    current_sim_no: int,
    frobenius_thresh: float,
    jsd_thresh: float,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Compare current_matrices against all cached previous checkpoints within
    the post-equilibration window [first_valid_sim_no, current_sim_no).
    """
    from chimpss.fultonmarket.utils import frobenius_norm, jsd_distance_matrices

    frob_results = {name: {} for name in MATRIX_NAMES}
    jsd_results  = {name: {} for name in MATRIX_NAMES}

    for prev_sim_no in range(first_valid_sim_no, current_sim_no):
        prev_matrices = matrix_cache.get(prev_sim_no, {})
        for name in MATRIX_NAMES:
            if name not in current_matrices or name not in prev_matrices:
                continue
            frob_results[name][prev_sim_no] = frobenius_norm(
                current_matrices[name], prev_matrices[name]
            )
            jsd_results[name][prev_sim_no] = jsd_distance_matrices(
                current_matrices[name], prev_matrices[name]
            )

    return frob_results, jsd_results


def build_checks(
    sim_no: int,
    total_n_sims: int,
    minimum_fraction: float,
    equil_fraction: float,
    max_equil_fraction: float,
    frob_results: dict,
    jsd_results: dict,
    frobenius_thresh: float,
    jsd_thresh: float,
    skip_contacts: bool = False,
) -> Dict[str, bool]:
    """
    Build the ordered checks dict for a single sim_no. Final key is 'STOP'.

    When `skip_contacts` is True the two contact rows are omitted entirely
    rather than recorded as failures. `_all_pass` returns False on an empty
    score dict, so leaving them in would pin 'STOP' to False forever whenever
    the contact matrix was never computed.
    """
    def _all_pass(scores, thresh):
        return bool(scores) and all(v < thresh for v in scores.values())

    checks = {
        'Past minimum simulation fraction':                   sim_no >= (total_n_sims * minimum_fraction),
        f'Equilibration discard < {max_equil_fraction:.0%}': equil_fraction < max_equil_fraction,
        'Torsion Frobenius converged':                        _all_pass(frob_results['torsion'],      frobenius_thresh),
        'Torsion JSD converged':                              _all_pass(jsd_results['torsion'],       jsd_thresh),
        'Alpha-carbon Frobenius converged':                   _all_pass(frob_results['alpha_carbon'], frobenius_thresh),
        'Alpha-carbon JSD converged':                         _all_pass(jsd_results['alpha_carbon'],  jsd_thresh),
    }
    if not skip_contacts:
        checks['Contact Frobenius converged'] = _all_pass(frob_results['contact'], frobenius_thresh)
        checks['Contact JSD converged']       = _all_pass(jsd_results['contact'],  jsd_thresh)
    checks['STOP'] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_sim_report(
    sim_no: int,
    total_n_sims: int,
    checks: Dict[str, bool],
    frob_results: dict,
    jsd_results: dict,
    frobenius_thresh: float,
    jsd_thresh: float,
    equil_fraction: float,
    effective_post_equil: float,
    first_valid_sim_no: int,
    _printf=None,
):
    progress_pct  = 100.0 * (sim_no + 1) / total_n_sims
    equil_pct     = 100.0 * equil_fraction
    post_equil_pct = 100.0 * effective_post_equil
    first_valid_pct = 100.0 * first_valid_sim_no / total_n_sims if total_n_sims > 0 else 0.0

    def _pct(k):
        return f'{100.0 * (k + 1) / total_n_sims:.1f}%'

    _log = _printf if _printf is not None else printf
    width = max(len(label) for label in checks)
    sep   = '=' * (width + 12)
    _log(sep)
    _log(f"  Convergence Report — {progress_pct:.1f}% of simulation complete  (sim_no={sim_no})")
    _log(sep)
    for label, result in checks.items():
        _log(f"  [{'PASS' if result else 'FAIL'}]  {label:<{width}}")
    _log('-' * (width + 12))
    _log(f"  Equilibration discards {equil_pct:.1f}% of data, "
         f"post-equil window covers {post_equil_pct:.1f}%, "
         f"comparing vs checkpoints from {first_valid_pct:.1f}%..{progress_pct:.1f}% of simulation")
    # Only report matrices that were actually part of the checks -- when
    # skip_contacts is on, the contact rows are absent from `checks` and
    # reporting them here would imply they were evaluated.
    reported = {'torsion', 'alpha_carbon'}
    if 'Contact Frobenius converged' in checks:
        reported.add('contact')

    for name in MATRIX_NAMES:
        if name not in reported:
            continue
        frob = frob_results[name]
        jsd  = jsd_results[name]
        if frob:
            _log(f"  {name:>12} Frobenius: {'  '.join(f'vs {_pct(k)}: {v:.4f}' for k, v in sorted(frob.items()))}  (thresh={frobenius_thresh})")
            _log(f"  {name:>12}       JSD: {'  '.join(f'vs {_pct(k)}: {v:.4f}' for k, v in sorted(jsd.items()))}  (thresh={jsd_thresh})")
        else:
            _log(f"  {name:>12}: no valid previous checkpoints in post-equil window")
    _log(sep)


def print_summary_table(report: Dict[int, Dict[str, bool]], total_n_sims: int, _printf=None):
    """
    Print a compact summary table of all checks as a function of simulation
    progress.
    """
    if not report:
        return

    sim_nos    = sorted(report.keys())
    all_labels = list(report[sim_nos[0]].keys())
    col_width  = max(len(lbl) for lbl in all_labels)
    sim_width  = 8
    sep        = '=' * (col_width + sim_width * len(sim_nos) + 4)

    headers = [f'{100.0 * (s + 1) / total_n_sims:.0f}%' for s in sim_nos]

    _log = _printf if _printf is not None else printf
    _log('\n' + sep)
    _log("  Summary Table — checks as a function of simulation progress")
    _log(sep)
    _log(f"  {'Check':<{col_width}}" + ''.join(f'{h:>{sim_width}}' for h in headers))
    _log('-' * (col_width + sim_width * len(sim_nos) + 4))
    for label in all_labels:
        row = f"  {label:<{col_width}}"
        for sim_no in sim_nos:
            result = report[sim_no].get(label)
            row += f"{'PASS':>{sim_width}}" if result else f"{'FAIL':>{sim_width}}"
        _log(row)
    _log(sep)


def log_mode(read_only: bool, output_cache_dir: Optional[str], _printf=None):
    _log = _printf if _printf is not None else printf
    if read_only:
        _log('Mode: read-only (in-memory only, nothing written)')
    elif output_cache_dir:
        _log(f'Mode: cache directory (writes -> {output_cache_dir})')
    else:
        _log('Mode: normal (reads and writes inside output_dir)')


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_RETRO_COLOR_MAP: Dict[str, str] = {
    'torsion': 'g',
    'alpha_carbon': 'c',
    'contact': 'darkorange',
}


def plot_convergence_metric(
    metric_dict: dict,
    func: str = 'jsd',
    distype: str = 'torsion',
    history: bool = True,
    show_all: bool = False,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Plot per-checkpoint convergence metric from ``retro_convergence_report``.

    Parameters
    ----------
    metric_dict : dict
        The ``metrics`` return value of
        ``FultonMarketAnalysis.retro_convergence_report()``.
    func : str
        Metric function; ``'jsd'`` or ``'frobenius'``.
    distype : str
        Distance type; one of ``'torsion'``, ``'alpha_carbon'``, ``'contact'``.
    history : bool
        If True, draw connecting lines from earlier checkpoints to the current one.
    show_all : bool
        If True, scatter every checkpoint (one point per sim_no). If False
        (default), only plot from the final checkpoint looking back.
    color_map : dict, optional
        Override the default ``_RETRO_COLOR_MAP``.
    """
    import matplotlib.pyplot as plt

    cmap = color_map if color_map is not None else _RETRO_COLOR_MAP
    n_sim_nos = len(metric_dict)
    if n_sim_nos == 0:
        return

    for sim_no, metrics in metric_dict.items():
        if show_all or sim_no == n_sim_nos - 1:
            dists = metrics.get(func, {}).get(distype, [])
            if dists:
                x_0, y_0 = sim_no, dists[-1]
                if history:
                    xs = [x_0 - i for i in range(len(dists))]
                    ys = list(reversed(dists))
                    plt.plot(
                        [(x + 1) / n_sim_nos for x in xs],
                        ys,
                        c=cmap[distype],
                    )
                    if show_all:
                        plt.scatter(
                            [(x + 1) / n_sim_nos for x in xs],
                            ys,
                            c=cmap[distype],
                            s=12,
                        )
                plt.scatter((x_0 + 1) / n_sim_nos, y_0, c=cmap[distype])


def add_equil_metric_to_plot(metric_dict: dict) -> None:
    """
    Overlay equilibration-fraction markers on an existing convergence plot.

    Markers are blue when the equilibration discard fraction is below 80 % and
    red when it meets or exceeds 80 %.

    Parameters
    ----------
    metric_dict : dict
        The ``metrics`` return value of
        ``FultonMarketAnalysis.retro_convergence_report()``.
    """
    import matplotlib.pyplot as plt

    n_sim_nos = len(metric_dict)
    if n_sim_nos == 0:
        return
    sim_nos = list(metric_dict.keys())
    equil_fractions = [metric_dict[k]['equil_fraction'] for k in sim_nos]
    colors = ['b' if frac < 0.80 else 'r' for frac in equil_fractions]
    plt.scatter(
        [(s + 1) / n_sim_nos for s in sim_nos],
        equil_fractions,
        c=colors,
        marker='*',
    )
