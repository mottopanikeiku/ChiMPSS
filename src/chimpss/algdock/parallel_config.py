"""Parallel execution configuration and thread coordination

This module manages thread limits and parallelization strategy to prevent
CPU thread overprovisioning when using multiprocessing or threading.

Also manages unified platform configuration between OpenMM and JAX/PyMBAR
to ensure consistent CPU/GPU usage.

Key insight: If using N parallel workers, each worker should use 1 thread
to avoid N workers x M threads = NxM total threads (overprovisioning).
"""

import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def configure_jax_platform(openmm_platform_name):
    """
    Configure JAX platform to match OpenMM platform choice

    This ensures PyMBAR (which uses JAX) runs on the same device type
    as OpenMM, avoiding resource conflicts and unnecessary transfers.

    Must be called BEFORE importing jax or pymbar.

    Parameters
    ----------
    openmm_platform_name : str
        OpenMM platform name: 'Reference', 'CPU', 'CUDA', 'OpenCL'

    Platform mapping:
    -----------------
    OpenMM Reference -> JAX cpu
    OpenMM CPU       -> JAX cpu
    OpenMM CUDA      -> JAX gpu
    OpenMM OpenCL    -> JAX gpu (JAX will use CUDA if available, else cpu)

    Examples
    --------
    >>> configure_jax_platform('Reference')  # Use CPU for JAX
    >>> configure_jax_platform('CUDA')        # Use GPU for JAX
    >>> configure_jax_platform('OpenCL')      # Try GPU for JAX
    """
    if openmm_platform_name in ['Reference', 'CPU']:
        jax_platform = 'cpu'
    elif openmm_platform_name in ['CUDA', 'OpenCL']:
        jax_platform = 'gpu'
    else:
        # Unknown platform, default to cpu
        jax_platform = 'cpu'

    # Set JAX platform via environment variable (must be set before import)
    os.environ['JAX_PLATFORMS'] = jax_platform

    # Also configure JAX to use 64-bit floats to match PyMBAR requirements
    os.environ['JAX_ENABLE_X64'] = '1'

    return jax_platform


def configure_thread_limits(n_workers):
    """
    Configure environment variables to limit threading per worker

    This prevents thread overprovisioning when using parallel workers.
    Must be called BEFORE importing NumPy, OpenMM, or other threaded libraries.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers (threads or processes)
        If 1, allow auto-detection of threads for single-worker case
        If >1, limit each worker to 1 thread

    Examples
    --------
    # Multi-worker case (8 workers, each with 1 thread = 8 total)
    >>> configure_thread_limits(8)

    # Single-worker case (1 worker with auto threads, e.g., 8)
    >>> configure_thread_limits(1)
    """
    if n_workers > 1:
        # Multi-worker: limit each worker to 1 thread
        threads_per_worker = '1'
    else:
        # Single-worker: use all available cores
        threads_per_worker = str(os.cpu_count() or 1)

    # OpenMP (used by NumPy, SciPy, OpenMM CPU platform)
    os.environ['OMP_NUM_THREADS'] = threads_per_worker

    # MKL (Intel Math Kernel Library)
    os.environ['MKL_NUM_THREADS'] = threads_per_worker

    # OpenBLAS
    os.environ['OPENBLAS_NUM_THREADS'] = threads_per_worker

    # OpenMM CPU platform specific
    os.environ['OPENMM_CPU_THREADS'] = threads_per_worker

    # JAX (if used) - use XLA_FLAGS only if not already set
    # Note: Some JAX versions don't support xla_cpu_parallelism_threads flag
    if n_workers > 1 and 'XLA_FLAGS' not in os.environ:
        # Try to set XLA flags, but this may not work on all JAX versions
        # If JAX complains about unknown flags, unset XLA_FLAGS in environment
        pass  # Let test environment handle XLA_FLAGS if needed

    return threads_per_worker


class ParallelExecutor:
    """
    Unified interface for parallel execution using ProcessPoolExecutor

    This replaces the old pattern of creating/destroying multiprocessing.Process
    objects every sweep with a persistent ProcessPoolExecutor.

    IMPORTANT: Uses ProcessPoolExecutor (not ThreadPoolExecutor) to avoid
    race conditions when multiple workers access OpenMM contexts. Each process
    gets its own isolated memory space and context, preventing corruption.

    Benefits over ThreadPoolExecutor:
    - No race conditions (isolated memory per process)
    - Each worker has its own OpenMM context (thread-safe)
    - Matches MMTK reference implementation conceptually

    Trade-offs:
    - Pickling overhead for task arguments
    - Higher memory usage (N contexts instead of 1 shared)
    - Slightly higher task startup cost

    Parameters
    ----------
    n_workers : int
        Number of parallel workers
        If 1, tasks execute sequentially (no process pool)
        If >1, creates ProcessPoolExecutor with n_workers processes
    name_prefix : str, optional
        Prefix for process names (for debugging)

    Examples
    --------
    # Create executor
    >>> executor = ParallelExecutor(n_workers=8, name_prefix='replica')

    # Execute tasks in parallel
    >>> tasks = [(conf, params, k) for k in range(50)]
    >>> results = executor.map(worker_func, tasks)

    # Cleanup (automatic on del, but can call explicitly)
    >>> executor.shutdown()
    """

    def __init__(self, n_workers, name_prefix='worker'):
        self.n_workers = n_workers

        if n_workers > 1:
            # Use ProcessPoolExecutor for true process isolation
            # Each process recreates OpenMM objects from scratch (see _iteration_worker_func)
            # This avoids pickling issues with SWIG-wrapped C++ objects
            self.executor = ProcessPoolExecutor(max_workers=n_workers)
        else:
            self.executor = None

    def map(self, func, tasks):
        """
        Execute func on each task in parallel

        Parameters
        ----------
        func : callable
            Function to execute, called as func(*task)
        tasks : list of tuples
            List of task arguments, where each task is a tuple of args

        Returns
        -------
        results : list
            Results in same order as tasks
        """
        if self.executor:
            # Parallel execution with ThreadPoolExecutor
            futures = [self.executor.submit(func, *task) for task in tasks]
            results = [f.result() for f in futures]
        else:
            # Sequential execution (single worker)
            results = [func(*task) for task in tasks]

        return results

    def shutdown(self, wait=True):
        """
        Shutdown the executor

        Parameters
        ----------
        wait : bool
            If True, wait for all tasks to complete before returning
        """
        if self.executor:
            self.executor.shutdown(wait=wait)
            self.executor = None

    def __del__(self):
        """Automatic cleanup on deletion"""
        self.shutdown(wait=True)

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown(wait=True)
        return False


def get_optimal_worker_count(requested_cores=None):
    """
    Determine optimal number of parallel workers

    Parameters
    ----------
    requested_cores : int or None
        Requested number of cores
        If None, use all available cores
        If > available, cap at available cores

    Returns
    -------
    n_workers : int
        Optimal number of workers (>= 1)
    """
    available_cores = os.cpu_count() or 1

    if requested_cores is None:
        return available_cores

    # Cap at available cores
    return max(1, min(requested_cores, available_cores))
