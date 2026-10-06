"""
Generate force-group + HMR variants of Bridgeport system XMLs.

Reads each (<name>.pdb, <name>.xml) pair in --input_dir and writes a corresponding
<name>_FG_HMR.xml in --output_dir, using chimpss.bridgeport._utils.apply_force_groups
and apply_hmr. Total system mass is conserved (printed per system as a sanity check).

The resulting <name>_FG_HMR.xml is the system to hand to MotorRow / FultonMarket when
running at a 3.5 fs timestep.

Usage:
    python generate_hmr_xmls.py \
        --input_dir  /expanse/.../work_dir/systems \
        --output_dir /expanse/.../work_dir/systems_hmr \
        --names lisuride methylergonovine methysergide LSD LY266097 \
                lisuride_L362F_mutseq methylergonovine_T140A_mutseq \
                methysergide_A225G_mutseq LSD_L362F_mutseq
"""
import argparse
import os
import sys

from openmm import XmlSerializer
from openmm.app import PDBFile

# Import HMR utilities directly from the ChiMPSS source tree
sys.path.insert(0, '/home/fcetin/ChiMPSS/src')
from chimpss.bridgeport._utils import apply_force_groups, apply_hmr


def total_mass(system):
    return sum(
        system.getParticleMass(i).value_in_unit_system(
            __import__('openmm').unit.md_unit_system
        )
        for i in range(system.getNumParticles())
    )


def process_one(name, input_dir, output_dir):
    pdb_path = os.path.join(input_dir, name + '.pdb')
    xml_path = os.path.join(input_dir, name + '.xml')
    if not (os.path.isfile(pdb_path) and os.path.isfile(xml_path)):
        raise FileNotFoundError(f'Missing inputs for {name}: {pdb_path} or {xml_path}')

    topology = PDBFile(pdb_path).topology
    with open(xml_path) as f:
        system = XmlSerializer.deserialize(f.read())

    mass_before = total_mass(system)

    apply_force_groups(system)
    apply_hmr(system, topology)

    mass_after = total_mass(system)
    delta = mass_after - mass_before

    out_xml = os.path.join(output_dir, name + '_FG_HMR.xml')
    with open(out_xml, 'w') as f:
        f.write(XmlSerializer.serialize(system))

    print(f'[{name}] mass before={mass_before:.6f}  after={mass_after:.6f}  '
          f'delta={delta:.3e}  ->  {out_xml}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_dir', required=True,
                   help='Directory containing <name>.pdb and <name>.xml')
    p.add_argument('--output_dir', required=True,
                   help='Directory to write <name>_FG_HMR.xml into')
    p.add_argument('--names', nargs='+', required=True,
                   help='Base names (without extension) to process')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.names:
        process_one(name, args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
