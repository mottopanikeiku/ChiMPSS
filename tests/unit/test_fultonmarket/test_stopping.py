"""Regression tests for fixed-duration FultonMarket stopping behavior."""

import pytest


pytest.importorskip("openmm", reason="openmm not installed")
pytest.importorskip("mdtraj", reason="mdtraj not installed")

from chimpss.fultonmarket import FultonMarket, Randolph
from openmm import unit


def _time_only_market(sim_no, total_n_sims=60):
    market = FultonMarket.__new__(FultonMarket)
    market.sim_no = sim_no
    market.total_sim_time = 1500
    market.total_n_sims = total_n_sims
    return market


def test_time_only_continues_before_target():
    market = _time_only_market(sim_no=58)

    finished = market._evaluate_stopping_criterion(
        getContacts_Info=None,
        skip_contacts=False,
    )

    assert finished is False


def test_time_only_stops_after_exact_target():
    # sim_no is zero-based and the check runs after saving, so sim_no=59 is
    # the 60th completed sub-simulation.
    market = _time_only_market(sim_no=59)

    finished = market._evaluate_stopping_criterion(
        getContacts_Info=None,
        skip_contacts=False,
    )

    assert finished is True


def test_time_only_does_not_run_sixty_first_segment():
    market = _time_only_market(sim_no=60)

    finished = market._evaluate_stopping_criterion(
        getContacts_Info=None,
        skip_contacts=False,
    )

    assert finished is True


def test_randolph_runs_exact_number_of_cycles():
    sampler = Randolph.__new__(Randolph)
    sampler.n_cycles = 3
    observed = []

    def run_cycle():
        observed.append(sampler.current_cycle)
        sampler.current_cycle += 1

    sampler._run_cycle = run_cycle
    sampler.main(init_overlap_thresh=0.5, term_overlap_thresh=0.35)

    assert observed == [0, 1, 2]


@pytest.mark.parametrize(
    ("timestep_fs", "iteration_ns"),
    [
        (2.0, 0.010),
        (3.5, 0.0175),
    ],
)
def test_iteration_length_preserves_five_thousand_steps(
    timestep_fs,
    iteration_ns,
):
    sampler = Randolph.__new__(Randolph)
    sampler.sim_time = 1.0
    sampler.temperatures = [300 * unit.kelvin, 310 * unit.kelvin]
    sampler.dt = timestep_fs
    sampler.iter_length = iteration_ns

    sampler._configure_simulation_parameters()

    assert sampler.n_steps_per_iter == 5_000
