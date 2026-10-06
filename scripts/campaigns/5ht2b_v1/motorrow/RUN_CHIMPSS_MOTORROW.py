"""
Run the ChiMPSS MotorRow 5-step equilibration on a single (pdb, system_xml) pair.

This is a thin wrapper around chimpss.motorrow.MotorRow that exposes --dt so we
can equilibrate HMR systems at 3.5 fs. The stock chimpss-motorrow console entry
point does not forward --dt, so we drive MotorRow.main(dt=...) ourselves here.

Usage:
    python RUN_CHIMPSS_MOTORROW.py <input_dir> <name> <hmr_xml_dir> <output_root> \
        [--lig_resname UNK] [--dt 3.5]

Inputs are read as:
    <input_dir>/<name>.pdb              # PDB from Bridgeport
    <hmr_xml_dir>/<name>_FG_HMR.xml     # HMR system XML (from generate_hmr_xmls.py)
Outputs go into:
    <output_root>/<name>/               # MotorRow's working directory
"""
import argparse
import os
import sys

# Use the ChiMPSS source tree directly (no pip install required)
sys.path.insert(0, '/home/fcetin/ChiMPSS/src')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input_dir',
                   help='Directory containing <name>.pdb')
    p.add_argument('name',
                   help='Base system name (without extension)')
    p.add_argument('hmr_xml_dir',
                   help='Directory containing <name>_FG_HMR.xml')
    p.add_argument('output_root',
                   help='Output root; MotorRow writes into <output_root>/<name>/')
    p.add_argument('--lig_resname', default='UNK',
                   help='Three-letter ligand resname (default: UNK)')
    p.add_argument('--dt', type=float, default=3.5,
                   help='Integration timestep in fs (default: 3.5 — assumes HMR system)')
    p.add_argument('--step_5_nsteps', type=int, default=1250000,
                   help='Number of steps for step 5 (default: 1250000)')
    args = p.parse_args()

    pdb_path = os.path.join(args.input_dir, args.name + '.pdb')
    xml_path = os.path.join(args.hmr_xml_dir, args.name + '_FG_HMR.xml')
    work_dir = os.path.join(args.output_root, args.name)

    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f'PDB not found: {pdb_path}')
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f'HMR system XML not found: {xml_path}')
    os.makedirs(work_dir, exist_ok=True)

    print(f'[RUN_CHIMPSS_MOTORROW] name        = {args.name}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] pdb         = {pdb_path}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] system_xml  = {xml_path}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] work_dir    = {work_dir}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] lig_resname = {args.lig_resname}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] dt          = {args.dt} fs', flush=True)

    from chimpss.motorrow import MotorRow
    mr = MotorRow(pdb_file=pdb_path,
                  system_xml=xml_path,
                  working_directory=work_dir,
                  lig_resname=args.lig_resname)
    state_fn, pdb_fn = mr.main(pdb_path,
                               step_5_nsteps=args.step_5_nsteps,
                               dt=args.dt)
    print(f'[RUN_CHIMPSS_MOTORROW] final state -> {state_fn}', flush=True)
    print(f'[RUN_CHIMPSS_MOTORROW] final pdb   -> {pdb_fn}', flush=True)


if __name__ == '__main__':
    main()
