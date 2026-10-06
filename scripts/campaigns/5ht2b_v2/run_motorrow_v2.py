"""v2 MotorRow equilibration for one 5-HT2B system (prep1 env, as originally run).

Same 5-step ChiMPSS MotorRow protocol and dt as the original equil_hmr run
(/home/fcetin/MotorRow/RUN_CHIMPSS_MOTORROW.py), with one change: step 5 uses
the membrane barostat (XY isotropic, Z free, zero surface tension) so that
equilibration ends in the same ensemble production runs in.

Usage: python run_motorrow_v2.py <name> [--dt 3.5]
"""
import argparse
import os
import sys

sys.path.insert(0, '/home/fcetin/ChiMPSS/src')
V2 = '/expanse/lustre/projects/iit127/fcetin/5ht2b/v2'

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--dt', type=float, default=3.5)
p.add_argument('--step_5_nsteps', type=int, default=1250000)
a = p.parse_args()

pdb = f'{V2}/work_dir/systems/{a.name}.pdb'
xml = f'{V2}/work_dir/systems_hmr/{a.name}_FG_HMR.xml'
work = f'{V2}/equil_hmr/{a.name}'
for f in (pdb, xml):
    if not os.path.isfile(f):
        raise FileNotFoundError(f)
os.makedirs(work, exist_ok=True)
print(f'[run_motorrow_v2] {a.name}: pdb={pdb} xml={xml} work={work} dt={a.dt} fs '
      f'step5=membrane barostat, gamma=0', flush=True)

from chimpss.motorrow import MotorRow  # noqa: E402

mr = MotorRow(pdb_file=pdb, system_xml=xml, working_directory=work, lig_resname='UNK')
state_fn, pdb_fn = mr.main(pdb, step_5_nsteps=a.step_5_nsteps, dt=a.dt,
                           step_5_barostat='membrane', step_5_surface_tension=0.0)
print(f'[run_motorrow_v2] final state -> {state_fn}\n[run_motorrow_v2] final pdb   -> {pdb_fn}', flush=True)
