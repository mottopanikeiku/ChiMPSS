"""Small periodic water box to smoke-test the v2 production code on CPU."""
import sys
import openmm as mm
from openmm import app, unit
from openmmtools import testsystems
out = sys.argv[1]
ts = testsystems.WaterBox(box_edge=2.2 * unit.nanometer)
with open(f'{out}/smoke.pdb', 'w') as f:
    app.PDBFile.writeFile(ts.topology, ts.positions, f)
with open(f'{out}/smoke.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(ts.system))
ctx = mm.Context(ts.system, mm.LangevinMiddleIntegrator(300, 1, 0.002), mm.Platform.getPlatformByName('CPU'))
ctx.setPositions(ts.positions)
mm.LocalEnergyMinimizer.minimize(ctx)
ctx.setVelocitiesToTemperature(300)
with open(f'{out}/smoke_state.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(ctx.getState(getPositions=True, getVelocities=True)))
print('atoms', ts.system.getNumParticles())
