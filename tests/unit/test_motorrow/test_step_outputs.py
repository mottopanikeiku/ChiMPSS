"""A MotorRow step must not look complete unless it is.

main() skips any step whose step_N.xml exists, so that file is the completion
marker. Per-cycle checkpoints used to overwrite it, so a crash mid-step (the
5-HT2B T140A NaN 2.5 ns into step 4) left a marker behind and a restart
silently skipped the rest of the step.
"""

import os

import pytest


pytest.importorskip("openmm", reason="openmm not installed")

from chimpss.motorrow import MotorRow


class _FakeSim:
    def __init__(self, fail_on_cycle=None):
        self.cycle, self.fail_on_cycle = 0, fail_on_cycle

    def step(self, n):
        self.cycle += 1
        if self.fail_on_cycle == self.cycle:
            raise RuntimeError('Particle coordinate is NaN')


def _motorrow(log):
    mr = MotorRow.__new__(MotorRow)
    mr._describe_state = lambda sim, name: None

    def write(kind):
        def _w(sim, fn):
            log.append((kind, os.path.basename(fn)))
            with open(fn, 'w') as f:
                f.write(kind)
        return _w
    mr._write_state, mr._write_structure = write('xml'), write('pdb')
    return mr


def test_crash_mid_step_leaves_no_completion_marker(tmp_path):
    xml, pdb = tmp_path / 'step_4.xml', tmp_path / 'step_4.pdb'
    mr = _motorrow([])
    with pytest.raises(RuntimeError, match='NaN'):
        mr._run_cycles(_FakeSim(fail_on_cycle=3), 4, range(1, 51), 50, 100, 5000, 3.5, str(xml), str(pdb))
    assert not xml.exists() and not pdb.exists()
    assert (tmp_path / 'step_4.partial.xml').exists()      # checkpoint kept for inspection


def test_completed_step_publishes_pdb_then_marker_and_drops_checkpoint(tmp_path):
    xml, pdb = tmp_path / 'step_4.xml', tmp_path / 'step_4.pdb'
    log = []
    _motorrow(log)._run_cycles(_FakeSim(), 4, range(1, 6), 5, 100, 500, 3.5, str(xml), str(pdb))
    assert xml.exists() and pdb.exists() and not (tmp_path / 'step_4.partial.xml').exists()
    assert log[-2:] == [('pdb', 'step_4.pdb'), ('xml', 'step_4.xml')]   # marker written last
    assert {name for kind, name in log[:-2]} == {'step_4.partial.xml'}


def test_partial_name():
    assert MotorRow._partial_name('/w/step_5.xml') == '/w/step_5.partial.xml'
