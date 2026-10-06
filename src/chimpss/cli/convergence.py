"""
CLI entry point for noise-calibrated FultonMarket convergence assessment.

Console script: chimpss-convergence

Tests whether the T_min (state 0) ensemble is still drifting, at successive
checkpoints, by comparing the MBAR-reweighted first and second halves of the
post-equilibration run against a random contiguous-block-split null. See
chimpss.fultonmarket.convergence for the method.
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    p.add_argument('fultonmarket_dir', help='FultonMarket output dir (contains saved_variables/)')
    p.add_argument('topology_pdb', help='PDB with the same atom order (e.g. MotorRow step_5.pdb)')
    p.add_argument('--checkpoints', default=None, type=str,
                   help='Comma-separated sub-simulation counts to assess (default: every 10, plus the last)')
    p.add_argument('--ligand_resname', default='UNK')
    p.add_argument('--alpha', default=0.05, type=float, help='Significance level for drift. Default 0.05.')
    p.add_argument('--min_neff', default=50.0, type=float,
                   help='Minimum Kish effective state-0 samples in each half. Default 50.')
    p.add_argument('--min_indep', default=25.0, type=float,
                   help='Minimum independent samples per half, estimated from the block-split null. Default 25.')
    p.add_argument('--n_blocks', default=10, type=int, help='Contiguous time blocks (even). Default 10.')
    p.add_argument('--n_boot', default=500, type=int, help='Random block splits for the null. Default 500.')
    p.add_argument('--out', default=None, help='JSON report path (default: <fultonmarket_dir>/convergence.json)')
    return p.parse_args()


def main():
    args = parse_args()
    from chimpss.fultonmarket.convergence import convergence_report

    cps = [int(x) for x in args.checkpoints.split(',')] if args.checkpoints else None
    report = convergence_report(args.fultonmarket_dir, args.topology_pdb, checkpoints=cps,
                                ligand_resname=args.ligand_resname, alpha=args.alpha,
                                min_neff=args.min_neff, min_indep=args.min_indep, n_blocks=args.n_blocks, n_boot=args.n_boot,
                                _printf=lambda m: print(m, flush=True))
    out = args.out or os.path.join(args.fultonmarket_dir, 'convergence.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    final = report[-1] if report else None
    print(f'\nreport -> {out}')
    if final is not None:
        print(f'final ({final["n_segments"]} sub-simulations): converged={final["converged"]}')


if __name__ == '__main__':
    main()
