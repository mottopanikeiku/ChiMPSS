"""Entry point for chimpss-algdock — uses the vendored BPMF engine in chimpss.algdock."""

import sys


def main(argv=None):
    from chimpss.algdock import arguments
    from chimpss.algdock import BindingPMF as BPMF

    import argparse
    parser = argparse.ArgumentParser(
        prog='chimpss-algdock',
        description=(
            'Molecular docking with adaptively scaled alchemical interaction grids '
            '(BPMF engine). Accepts AMBER prmtop/inpcrd inputs natively. '
            'For OpenMM-XML → AMBER conversion, see chimpss.algdock.converter.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for key in arguments.args.keys():
        parser.add_argument('--' + key, **arguments.args[key])

    args = parser.parse_args(argv)

    if getattr(args, 'run_type', None) in ('render_docked', 'render_intermediates'):
        from chimpss.algdock.BindingPMF_plots import BPMF_plots
        BPMF_plots(**vars(args))
    else:
        BPMF(**vars(args))

    return 0


if __name__ == '__main__':
    sys.exit(main())
