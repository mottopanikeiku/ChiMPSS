import argparse, os

parser = argparse.ArgumentParser(
    description='Retroactively compute distance matrices and convergence report '
                'for a completed FultonMarket simulation.'
)

parser.add_argument('input_dir',  type=str,
                    help='Path to the FultonMarket output directory '
                         '(contains netcdfs and saved_variables dir).')
parser.add_argument('pdb',        type=str,
                    help='Path to the topology PDB used for the simulation.')

parser.add_argument('--output_cache_dir', default=None, type=str,
                    help='Directory to read/write cached distance matrices. '
                         'Default: <input_dir>/retro_cache/.')
parser.add_argument('--n_resample', default=1000, type=int,
                    help='Frames to importance-resample per sub-simulation. '
                         'Default 1000.')
parser.add_argument('--sele_str', default='resname UNK', type=str,
                    help="MDTraj selection for the ligand. Default 'resname UNK'.")
# Per-sub-simulation frame discard. FultonMarketAnalysis defaults to 10, which
# was calibrated for the legacy 1 ps exchange cadence (~150 frames/segment). At
# the production 17.5 ps cadence a 25 ns segment holds only 9-11 frames, so
# skip=10 discards 91-100% of the data and leaves MBAR underdetermined. Use 0
# and let determine_equilibration()/t0 do the discard statistically.
parser.add_argument('--skip', default=0, type=int,
                    help='Frames to discard from the start of each sub-simulation. '
                         'Default 0 (correct for the 17.5 ps production cadence); '
                         'the library default of 10 suits only the legacy 1 ps cadence.')
parser.add_argument('--sim_nos', default=None, type=str,
                    help='Comma-separated sub-simulation indices to analyse '
                         '(e.g. "16,17,18"). Default: all. Use this to exclude '
                         'segments recorded at a different exchange cadence.')

# Contact-matrix control. getContacts is an external tool and is frequently
# not installed; without it the retro path used to raise
# "getcontacts_script must be provided" and no convergence data could be
# produced at all. --skip_contacts computes the torsional and alpha-carbon
# matrices only, mirroring FultonMarket.run(skip_contacts=True).
parser.add_argument('--skip_contacts', action='store_true',
                    help='Skip the contact distance matrix (no getContacts needed).')
parser.add_argument('--getcontacts_script', default=None, type=str,
                    help='Path to getContacts get_dynamic_contacts.py.')
parser.add_argument('--conda_env', default=None, type=str,
                    help='Conda env containing getContacts.')
parser.add_argument('--getcontacts_python', default=None, type=str,
                    help='Explicit interpreter path for getContacts.')

args = parser.parse_args()

if not args.skip_contacts and args.getcontacts_script is None:
    parser.error('either --skip_contacts or --getcontacts_script is required')

# --- Resolve cache directory (default: <input_dir>/retro_cache/) ---
output_cache_dir = args.output_cache_dir or os.path.join(args.input_dir, 'retro_cache')
if not os.path.isdir(output_cache_dir):
    print(f"Creating cache directory at: {output_cache_dir}")
    os.makedirs(output_cache_dir, exist_ok=True)

sim_nos = None
if args.sim_nos:
    sim_nos = [int(s) for s in args.sim_nos.split(',') if s.strip()]

# --- Run ---
from FultonMarket.FultonMarketAnalysis import FultonMarketAnalysis

analyzer = FultonMarketAnalysis(input_dir=args.input_dir, pdb=args.pdb,
                                sele_str=args.sele_str, skip=args.skip)
matrices = analyzer.retro_analyze_all(
    n_resample=args.n_resample,
    sim_nos=sim_nos,
    output_cache_dir=output_cache_dir,
    skip_contacts=args.skip_contacts,
    getcontacts_script=args.getcontacts_script,
    conda_env=args.conda_env,
    getcontacts_python=args.getcontacts_python,
)

print(f"\nretro_analyze_all complete: {len(matrices)} sub-simulations processed.")
print(f"matrices cached under: {output_cache_dir}")
