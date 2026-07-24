"""
CLI entry point for FultonMarket parallel tempering REMD simulation.

Console script: chimpss-fultonmarket
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description='Run a FultonMarket parallel tempering replica exchange simulation.')

    # System inputs
    p.add_argument('input_pdb',    type=str,
                   help='Path to PDB file with starting coordinates')
    p.add_argument('input_system', type=str,
                   help='Path to XML file with the serialized system')
    p.add_argument('output_dir',   type=str,
                   help='Directory to store output')
    p.add_argument('--input_state', default=None, type=str,
                   help='Path to XML file of a recently written state')

    # Temperature ladder
    p.add_argument('--T_min',        default=300,  type=float,
                   help='Lowest temperature on the ladder (K). Default 300.')
    p.add_argument('--T_max',        default=367,  type=float,
                   help='Highest temperature on the ladder (K). Default 367.')
    p.add_argument('--n_replicates', default=68,   type=int,
                   help='Number of replica exchange states. Default 68.')

    # Simulation timing
    p.add_argument('--iter_length', default=0.010, type=float,
                   help='Time between replica swaps (ns). Default 0.010 (10 ps at 2 fs).')
    p.add_argument('--timestep',    default=2.0,   type=float,
                   help='Integration timestep (fs). Default 2.0.')
    p.add_argument('--sim_length',  default=25,    type=int,
                   help='Sub-simulation duration / checkpoint interval (ns). Default 25.')
    p.add_argument('--total_sim_time', default=1200, type=int,
                   help='Maximum aggregate simulation time (ns). Default 1200.')
    p.add_argument('--minimum_sim_fraction', default=0.35, type=float,
                   help='Minimum fraction of total_sim_time to run before auto-stop '
                        'is attempted. Default 0.35.')

    # Convergence thresholds
    p.add_argument('--n_resample', default=1000, type=int,
                   help='Frames to importance-resample per convergence check. Default 1000.')
    p.add_argument('--max_equil_fraction', default=0.75, type=float,
                   help='Max fraction of simulation discardable as equilibration '
                        'before convergence is considered unreliable. Default 0.75.')
    p.add_argument('--frobenius_thresh', default=0.05, type=float,
                   help='Normalised Frobenius norm threshold for distance matrix '
                        'convergence. Default 0.05.')
    p.add_argument('--jsd_thresh', default=0.10, type=float,
                   help='Jensen-Shannon divergence threshold for distance matrix '
                        'convergence. Default 0.10.')

    # getContacts
    p.add_argument('--getcontacts_script', default=None, type=str,
                   help='Path to get_dynamic_contacts.py. Required for contact '
                        'distance matrix convergence.')
    p.add_argument('--conda_env', default=None, type=str,
                   help='Name of the conda environment containing getContacts. '
                        'Required when --getcontacts_python is not given.')
    p.add_argument('--getcontacts_python', default=None, type=str,
                   help='Explicit path to the Python interpreter for getContacts. '
                        'When provided, --conda_env is ignored.')
    p.add_argument('--getcontacts_cores', default=10, type=int,
                   help='CPU cores passed to getContacts via --cores. Default 10.')

    p.add_argument('--protein-name', default=None, type=str,
                   help='Protein name for output file naming (no underscores or periods). '
                        'When combined with --ligand-name, outputs are named '
                        'PROTEIN_LIGAND.trajectory.ncdf / PROTEIN_LIGAND.checkpoint.ncdf.')
    p.add_argument('--ligand-name', default=None, type=str,
                   help='Ligand name for output file naming (no underscores or periods).')

    return p.parse_args()


def main():
    args = parse_args()

    from chimpss.fultonmarket import FultonMarket as FM

    market = FM(input_pdb=args.input_pdb,
                input_system=args.input_system,
                input_state=args.input_state,
                T_min=args.T_min,
                T_max=args.T_max,
                n_replicates=args.n_replicates,
                protein_name=args.protein_name,
                ligand_name=args.ligand_name)

    os.makedirs(args.output_dir, exist_ok=True)

    getContacts_Info = None
    if args.getcontacts_script is not None:
        getContacts_Info = {
            'getcontacts_script': args.getcontacts_script,
            'conda_env':          args.conda_env,
            'getcontacts_python': args.getcontacts_python,
            'cores':              args.getcontacts_cores,
        }

    market.run(iter_length=args.iter_length,
               dt=args.timestep,
               sim_length=args.sim_length,
               total_sim_time=args.total_sim_time,
               output_dir=args.output_dir,
               minimum_fraction=args.minimum_sim_fraction,
               n_resample=args.n_resample,
               max_equil_fraction=args.max_equil_fraction,
               frobenius_thresh=args.frobenius_thresh,
               jsd_thresh=args.jsd_thresh,
               getContacts_Info=getContacts_Info)


if __name__ == '__main__':
    main()
