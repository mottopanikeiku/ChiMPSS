#!/usr/bin/env python3
"""
FultonMarket smoke test / general test case.

Runs a short parallel-tempering REMD on a MotorRow-equilibrated system.
Sub-simulation length and total simulation time are set to 1/10 of
production defaults so the test completes quickly on a single GPU.

Usage
-----
  python run_fultonmarket.py \\
    --pdb    path/to/PROTEIN_LIGAND.equil.pdb \\
    --system path/to/PROTEIN_LIGAND.system.xml \\
    --state  path/to/PROTEIN_LIGAND.state.xml \\
    --getcontacts-script /path/to/get_dynamic_contacts.py \\
    --getcontacts-env    getcontacts \\
    --out    /path/to/output_dir

  # HMR system (larger timestep / longer swap interval):
  python run_fultonmarket.py ... --hmr

  # Explicit opt-out of contact convergence:
  python run_fultonmarket.py ... --no-contacts

Typical MotorRow output layout (protein_name=NTSR1, ligand_name=ML-301):
  NTSR1_ML-301.equil.pdb   → --pdb
  NTSR1_ML-301.system.xml  → --system   (same XML that fed MotorRow)
  NTSR1_ML-301.state.xml   → --state
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--pdb',    required=True, help='Equilibrated PDB from MotorRow')
parser.add_argument('--system', required=True, help='System XML from Bridgeport')
parser.add_argument('--state',  required=True, help='State XML from MotorRow')
parser.add_argument('--out',
                    default='/media/volume/Josephs-Volume/ChiMPSS_Testing/fultonmarket_smoke',
                    help='Output directory (default: %(default)s)')
parser.add_argument('--n-replicates', type=int,   default=3,     help='Number of temperature replicas (default: 3)')
parser.add_argument('--T-min',        type=float, default=300.0, help='Minimum temperature in K (default: 300)')
parser.add_argument('--T-max',        type=float, default=310.0, help='Maximum temperature in K (default: 310)')
parser.add_argument('--sim-length',   type=float, default=0.01,  help='Sub-simulation length in ns (default: 0.01)')
parser.add_argument('--n-sims',       type=int,   default=1,     help='Number of sub-simulations (default: 1 — increase to test convergence machinery)')
parser.add_argument('--hmr', action='store_true',
                    help='HMR mode: sets dt=3.5 fs and iter_length=0.0175 ns (17.5 ps) '
                         'instead of the standard 2.0 fs / 0.010 ns (10 ps)')
parser.add_argument('--getcontacts-script', default=None, metavar='PATH',
                    help='Path to get_dynamic_contacts.py (required unless --no-contacts)')
parser.add_argument('--getcontacts-env', default=None, metavar='ENV',
                    help='Conda environment that has getContacts installed')
parser.add_argument('--no-contacts', action='store_true',
                    help='Explicitly skip contact distance matrix convergence checking. '
                         'Must be set to omit getContacts — contacts are used by default.')
args = parser.parse_args()

# --hmr automatically selects the appropriate timestep and swap interval
dt          = 3.5     if args.hmr else 2.0
iter_length = 0.0175  if args.hmr else 0.010

if not args.no_contacts and not args.getcontacts_script:
    parser.error(
        '--getcontacts-script is required unless you pass --no-contacts.\n'
        'Contact distance matrix convergence is enabled by default.'
    )

for label, path in [('--pdb', args.pdb), ('--system', args.system), ('--state', args.state)]:
    if not os.path.exists(path):
        sys.exit(f'ERROR: {label} not found: {path}')

os.makedirs(args.out, exist_ok=True)
RESULTS = os.path.join(args.out, 'fultonmarket_results.txt')


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(RESULTS, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# The stopping check runs after each save and counts the current zero-based
# sim_no, so n_sims * sim_length produces exactly n_sims saved segments.
total_sim_time = args.n_sims * args.sim_length

getContacts_Info = None
if not args.no_contacts:
    getContacts_Info = {'getcontacts_script': args.getcontacts_script}
    if args.getcontacts_env:
        getContacts_Info['conda_env'] = args.getcontacts_env

log('=' * 70)
log(f'FultonMarket smoke test  [{"HMR" if args.hmr else "standard"}]')
log(f'  pdb          : {args.pdb}')
log(f'  system       : {args.system}')
log(f'  state        : {args.state}')
log(f'  out          : {args.out}')
log(f'  n_replicates : {args.n_replicates}')
log(f'  T_min / T_max: {args.T_min} / {args.T_max} K')
log(f'  dt           : {dt} fs')
log(f'  iter_length  : {iter_length} ns  ({iter_length * 1e3:.2f} ps)')
log(f'  sim_length   : {args.sim_length} ns')
log(f'  n_sims       : {args.n_sims}')
log(f'  total_sim_time (derived): {total_sim_time} ns')
log(f'  skip_contacts: {args.no_contacts}')
log('=' * 70)

try:
    from chimpss.fultonmarket import FultonMarket

    market = FultonMarket(
        input_pdb=args.pdb,
        input_system=args.system,
        input_state=args.state,
        n_replicates=args.n_replicates,
        T_min=args.T_min,
        T_max=args.T_max,
    )

    market.run(
        iter_length=iter_length,
        dt=dt,
        sim_length=args.sim_length,
        total_sim_time=total_sim_time,
        output_dir=args.out,
        init_overlap_thresh=0.0,
        term_overlap_thresh=0.0,
        getContacts_Info=getContacts_Info,
        skip_contacts=args.no_contacts,
    )

    log('SUCCESS — FultonMarket smoke test completed.')
    log(f'Output files are in: {args.out}')

except Exception:
    log('FAILED — FultonMarket smoke test raised an exception.')
    tb = traceback.format_exc()
    print(tb, flush=True)
    with open(RESULTS, 'a', encoding='utf-8') as f:
        f.write(tb + '\n')
    sys.exit(1)
