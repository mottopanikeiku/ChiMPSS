"""Reproduce an original <name>_FG_HMR.xml from its plain system; compare exactly."""
import sys
import numpy as np
sys.path.insert(0, '/home/fcetin/ChiMPSS/src')
from openmm import XmlSerializer
from openmm.app import PDBFile
from chimpss.bridgeport._utils import apply_force_groups, apply_hmr
P = '/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir'
name = sys.argv[1]
s = XmlSerializer.deserialize(open(f'{P}/systems/{name}.xml').read())
apply_force_groups(s); apply_hmr(s, PDBFile(f'{P}/systems/{name}.pdb').topology)
ref = XmlSerializer.deserialize(open(f'{P}/systems_hmr/{name}_FG_HMR.xml').read())
m = np.array([s.getParticleMass(i)._value for i in range(s.getNumParticles())])
r = np.array([ref.getParticleMass(i)._value for i in range(ref.getNumParticles())])
fg = [(type(f).__name__, f.getForceGroup()) for f in s.getForces()]
fr = [(type(f).__name__, f.getForceGroup()) for f in ref.getForces()]
recip = lambda sy: [f.getReciprocalSpaceForceGroup() for f in sy.getForces() if type(f).__name__ == 'NonbondedForce']
print(f'{name}: masses identical={np.array_equal(m, r)} (max |d| {np.abs(m-r).max():.2e}) | '
      f'force groups identical={fg == fr} | PME recip group {recip(s)} vs {recip(ref)}')
