from __future__ import annotations

#MotorRow
import math
import os
import shutil
from datetime import datetime

import mdtraj as md
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

from chimpss.motorrow.utils import (
    get_positions_from_pdb,
    restrain_atoms,
    unpack_infiles,
)
from chimpss.shared.io import build_output_path, file_exists_skip, validate_name


class MotorRow():
    """
    A Class for Equilibration of Membrane Proteins.

    Follows a five-step protocol:

    - 0: Minimization
    - 1: NVT with restraints on Membrane Z and Protein XYZ (250 ps)
    - 2: NPT with restraints on Membrane Z and Protein XYZ (250 ps)
    - 3: NVT with no restraints (250 ps)
    - 4: NPT with no restraints, MonteCarloMembraneBarostat (2.5 ns)
    - 5: NPT with no restraints, MonteCarloBarostat (2.5 ns)

    Common settings: dt=2.0 fs, Temp=300 K, Platform=OpenCL.
    """

    def __init__(self, pdb_file, system_xml, working_directory, lig_resname: str='UNK', lig_chain: str=None,
                 protein_name: str=None, ligand_name: str=None):
        """
        Parse the xml into an openmm system; sets the self.system attribute from the xml file; sets the self.topology attribute from pdb_file

        Parameters:
            system_xml: string: path to the xml file that was generated using Bridgeport
            pdb_file: string: path to the pdb file that was generated using Bridgeport
            working directory: string: the directory to store simulation data in

        Returns:
            None
        """
        #If the working dir is absolute, leave it alone, otherwise make it abs
        if os.path.isabs(working_directory):
            self.abs_work_dir = working_directory
        else:
            self.abs_work_dir = os.path.join(os.getcwd(), working_directory)
        #Ensure that the working dir exists, and if not create it
        if not os.path.isdir(self.abs_work_dir):
            os.makedirs(self.abs_work_dir, exist_ok=True)
        #Get the system xml file (we want to create a system fresh from this every time)
        if system_xml is None or os.path.isabs(system_xml):
            pass
        else:
            shutil.copy(system_xml, os.path.join(self.abs_work_dir, system_xml))
            system_xml = os.path.join(self.abs_work_dir, system_xml)

        self.system_xml = system_xml

        # Copy the system XML into the working directory so it is always
        # co-located with the state XMLs (needed by downstream tools like
        # chimpss.algdock.converter which require both PDB and system XML).
        if system_xml is not None:
            dest = os.path.join(self.abs_work_dir, 'system.xml')
            if not os.path.exists(dest):
                shutil.copy(system_xml, dest)

        #Get the pdbfile, store the topology (and initial positions i guess)
        if os.path.isabs(pdb_file):
            pass
        else:
            shutil.copy(pdb_file, os.path.join(self.abs_work_dir, pdb_file))
            pdb_file = os.path.join(self.abs_work_dir, pdb_file)

        pdb = PDBFile(pdb_file)
        self.topology = pdb.topology
        self.lig_resname = lig_resname
        if lig_chain is not None:
            lig_chain = lig_chain.strip()
        self.lig_chain = lig_chain
        self.protein_name = validate_name(protein_name) if protein_name else None
        self.ligand_name = validate_name(ligand_name) if ligand_name else None


    def main(self, pdb_in, restrain_step_5: bool=False, step_5_nsteps: int=1250000, dt: float=2.0):
        """
        Run the standard five step equilibration
        0 - Minimization
        1 - NVT with restraints on Membrane Z and Protein XYZ 250ps
        2 - NPT with restraints on Membrane Z and Protein XYZ (MonteCarloMembraneBarostat) 250ps
        3 - NVT with no restraints 250 ps
        4 - NPT with no restraints (MonteCarloMembraneBarostat) 2.5ns
        5 - NPT with no restraints (MonteCarloBarostat) 2.5ns

        Parameters:
            pdb_in: string: path to the pdb file (same as init) - the initial structure

        Returns:
            state_fn: string: path to the XML serialized state file
            pdb_fn: string: path to the final structure of the equilibration simulation.
        """
        # Early exit if the fully-renamed final outputs already exist (e.g. restart after completion)
        if self.protein_name and self.ligand_name:
            final_pdb = build_output_path(self.abs_work_dir, self.protein_name, self.ligand_name, 'equil', 'pdb')
            final_xml = build_output_path(self.abs_work_dir, self.protein_name, self.ligand_name, 'state', 'xml')
            if os.path.exists(final_pdb) and os.path.exists(final_xml):
                print(f'[skip] MotorRow already complete — {final_pdb} exists')
                return final_xml, final_pdb

        if not os.path.isabs(pdb_in):
            shutil.copy(pdb_in, os.path.join(self.abs_work_dir, pdb_in))
            pdb_in = os.path.join(self.abs_work_dir, pdb_in)

        # Minimization (step 0) — skip if already done
        min_xml = os.path.join(self.abs_work_dir, 'minimized.xml')
        min_pdb = os.path.join(self.abs_work_dir, 'minimized.pdb')
        if file_exists_skip(min_xml, 'MotorRow minimization'):
            state_fn, pdb_fn = min_xml, min_pdb
        else:
            state_fn, pdb_fn = self._minimize(pdb_in, dt=dt)

        # Steps 1–5: (stepnum, nsteps, restrain_lig)
        step_configs = [
            (1, 125000,        True),           # NVT restrained     250 ps
            (2, 125000,        True),           # NPT restrained     250 ps
            (3, 125000,        True),           # NVT free           250 ps
            (4, 1250000,       True),           # NPT membrane barostat 2.5 ns
            (5, step_5_nsteps, restrain_step_5),# NPT isotropic     2.5 ns
        ]
        for stepnum, nsteps, restrain_lig in step_configs:
            step_xml = os.path.join(self.abs_work_dir, f'step_{stepnum}.xml')
            step_pdb = os.path.join(self.abs_work_dir, f'step_{stepnum}.pdb')
            if file_exists_skip(step_xml, f'MotorRow step {stepnum}'):
                state_fn, pdb_fn = step_xml, step_pdb
            else:
                state_fn, pdb_fn = self._run_step(state_fn, stepnum, nsteps=nsteps,
                                                   positions_from_pdb=pdb_fn,
                                                   restrain_lig=restrain_lig,
                                                   dt=dt)

        # If protein/ligand names were provided, rename final outputs with the new convention
        if self.protein_name and self.ligand_name:
            final_pdb = build_output_path(self.abs_work_dir, self.protein_name, self.ligand_name, 'equil', 'pdb')
            final_xml = build_output_path(self.abs_work_dir, self.protein_name, self.ligand_name, 'state', 'xml')
            os.rename(pdb_fn, final_pdb)
            os.rename(state_fn, final_xml)
            pdb_fn, state_fn = final_pdb, final_xml

        return state_fn, pdb_fn


    def _describe_state(self, sim: Simulation, name: str = "State"):
        """
        Report the energy of an openmm simulation

        Parameters:
            sim: Simulation: The OpenMM Simulation object to report the energy of
            name: string: Default="State" - An optional identifier to help distinguish what energy is being reported
        """
        state = sim.context.getState(getEnergy=True, getForces=True)
        self.PE = round(state.getPotentialEnergy()._value, 2)
        max_force = round(max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces()), 2)
        print(f"{name} has energy {self.PE} kJ/mol ", f"with maximum force {max_force} kJ/(mol nm)")


    def _write_state(self, sim: Simulation, xml_fn: str):
        """
        Serialize the State of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the State of
            xml_fn: string: The path to the xmlfile to write the serialized State to
        """
        state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        contents = XmlSerializer.serialize(state)
        with open(xml_fn, 'w') as f:
            f.write(contents)
        print(f'Wrote: {xml_fn}')


    def _write_system(self, sim: Simulation, xml_fn: str):
        """
        Serialize the System of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the System of
            xml_fn: string: The path to the xmlfile to write the serialized System to
        """
        with open(xml_fn, 'w') as f:
            f.write(XmlSerializer.serialize(sim.system))
        print(f'Wrote: {xml_fn}')


    def _write_structure(self, sim: Simulation, pdb_fn: str):
        """
        Write the structure of an OpenMM Simulation to a PDB file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to write the structure of
            pdb_fn: string: The path to the PDB file to write the structure to
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(), f, keepIds=True)
        print(f'Wrote: {pdb_fn}')

        top = md.Topology().from_openmm(sim.topology)
        traj = md.Trajectory(xyz=sim.context.getState(getPositions=True).getPositions(asNumpy=True),
                             topology=top,
                             unitcell_lengths=np.array([sim.context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)[i][i]._value for i in range(3)]),
                             unitcell_angles=np.array([90,90,90]))
        traj.image_molecules(inplace=True)
        wrapped_pdb_fn = os.path.join(os.path.dirname(pdb_fn), os.path.basename(pdb_fn).split('.')[0] + '_wrapped.pdb')
        traj.save_pdb(wrapped_pdb_fn)
        print(f'Wrote: {wrapped_pdb_fn}')



    def _minimize(self, pdb_in:str, pdb_out:str=None, state_xml_out:str=None, temp=300.0, dt=2.0, fc_pos: float=40.0):
        """
        Minimizes the structure of pdb_in

        Parameters:
            pdb_in - the structure to be minimized

        Returns:
            pdb_out - FilePath to the output structure
        """
        start = datetime.now()
        system, _, positions = unpack_infiles(self.system_xml, pdb_in)

        # Add restraint to ligand
        crds, prt_heavy_atoms, mem_heavy_atoms, lig_heavy_atoms = get_positions_from_pdb(pdb_in, lig_resname=self.lig_resname, lig_chain=self.lig_chain)
        if len(lig_heavy_atoms) > 0:
            system = restrain_atoms(system, crds, np.array(lig_heavy_atoms)[:,0], rst_name='lig_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))
        elif self.lig_resname is not None or self.lig_chain is not None:
            raise Exception('Could not find ligand...')

        # Protein Restraint
        system = restrain_atoms(system, crds, np.array(prt_heavy_atoms), rst_name='prot_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

        integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(self.topology, system, integrator, Platform.getPlatformByName('OpenCL'))
        simulation.context.setPositions(positions)
        self._describe_state(simulation, "Original state")
        simulation.minimizeEnergy()
        self._describe_state(simulation, "Minimized state")
        end = datetime.now() - start
        print(f'Minimization completed in {end}')

        if pdb_out is None:
            pdb_out = os.path.join(self.abs_work_dir, 'minimized.pdb')
        self._write_structure(simulation, pdb_out)

        if state_xml_out is None:
            state_xml_out = os.path.join(self.abs_work_dir, 'minimized.xml')
        self._write_state(simulation, state_xml_out)

        return state_xml_out, pdb_out


    def _run_step(self,
                  state_in:str,
                  stepnum:int,
                  state_xml_out:str=None,
                  pdb_out:str=None,
                  fc_pos:float=300.0,
                  nsteps=125000,
                  temp=300.0,
                  dt=2.0,
                  ncycles=50,
                  nstdout=1000,
                  fn_stdout=None,
                  ndcd=500,
                  append_dcd: bool=False,
                  fn_dcd=None,
                  press=1.0,
                  positions_from_pdb:str=None,
                  restrain_lig: bool=True):

        """
        Run different hard-coded Simulations based on the step number
        1 - NVT with restraints on Membrane Z and Protein XYZ 250ps
        2 - NPT with restraints on Membrane Z and Protein XYZ (MonteCarloMembraneBarostat) 250ps
        3 - NVT with no restraints 250 ps
        4 - NPT with no restraints (MonteCarloMembraneBarostat) 2.5ns
        5 - NPT with no restraints (MonteCarloBarostat) 2.5ns

        Parameters:
            state_in: string: path to a serialize OpenMM State as an xml file
            stepnum: int: Which step of the protocol to run.  This determines the addition of restraints/barostats
            state_xml_out: string: Default None - xml file path to write the output State of the step to
            pdb_out: string: Default None - pdb file path to write the last frame of the current step's simulation to
            fc_pos: float: Default 300.0 - Strength of the harmonic restraint holding heavy atoms in step 1 (units KJ/nm^2)
            nsteps: int: Default 125000 - Number of Simulation steps to take
            temp: float: Default 300 - Temperature of the Simulation (for setting initial velocities) (unit Kelvin)
            dt: float: Default 2.0 - Timestep of the simulation (unit femtosecond)
            ncycles: float: Default 50 - Number of cycles. state.xml is written out every cycle for resuming
            nstdout: int: Default 1000 - Number of steps to take between writing information to the State Data Reporter
            fn_stdout: string: Default None - If provided, will write the State Data Reporter data to this file name
            ndcd: int: Default 5000 - Number of steps to take between recording frames in the DCD trajectory file
            fn_dcd: string: Default None - If provided, will write the DCD trajectory file to this file name
            press: float: Default 1.0 - pressure for the Barostat during NPT simulations (unit bar)
            positions_from_pdb: string: Default None - For step one specifically, this must be provided to determine heavy atoms,
                                                        establish restraints, and set initial positions

        Returns:
            state_xml_out: string: Path to the written XML file containing the serialized State after the step has been run
            pdb_out: string: Path to the written PDB file containing the structure after the simulation has been run (last frame)
        """
        #Before ANY STEP
        start = datetime.now()
        #Establish State
        with open(self.system_xml) as f:
            system = XmlSerializer.deserialize(f.read())


        # Get atoms
        assert positions_from_pdb is not None
        crds, prt_heavy, mem_heavy, lig_heavy_atoms = get_positions_from_pdb(positions_from_pdb, lig_resname=self.lig_resname, lig_chain=self.lig_chain)

        # Ligand Restraint
        if restrain_lig:
            if len(lig_heavy_atoms) > 0:
                system = restrain_atoms(system, crds, np.array(lig_heavy_atoms)[:,0], rst_name='lig_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))
            elif self.lig_resname is not None or self.lig_chain is not None:
                raise Exception('Could not find ligand...')

        #STEP SPECIFIC ACTIONS
        if stepnum == 1:

            # Protein Restraint
            system = restrain_atoms(system, crds, np.array(prt_heavy), rst_name='prot_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

            #Membrane Restraint
            system = restrain_atoms(system, crds, np.array(mem_heavy), rst_name='mem_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

        elif stepnum == 2:

            # Protein Restraint
            system = restrain_atoms(system, crds, np.array(prt_heavy), rst_name='prot_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

            # Membrane Restraint
            system = restrain_atoms(system, crds, np.array(mem_heavy), rst_name='mem_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

            # Add MC Membrane Barostat
            system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
                                                       MonteCarloMembraneBarostat.XYIsotropic,
                                                       MonteCarloMembraneBarostat.ZFree, 100))

        elif stepnum == 3:
            # Protein Restraint
            system = restrain_atoms(system, crds, np.array(prt_heavy), rst_name='prot_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

        elif stepnum == 4:
            # Protein Restraint
            system = restrain_atoms(system, crds, np.array(prt_heavy), rst_name='prot_k', rst_strength=86.68*(joule)/(angstrom*angstrom*mole))

            # Add MC Membrane Barostat
            system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
                                                       MonteCarloMembraneBarostat.XYIsotropic,
                                                       MonteCarloMembraneBarostat.ZFree, 100))
        elif stepnum == 5:
            system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

        else:
            raise NotImplementedError('How did that happen?')

        # Any Step Establish Simulation
        integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(self.topology, system, integrator, Platform.getPlatformByName('OpenCL'))

        # Load Positions
        if stepnum == 1:
            assert positions_from_pdb is not None
            pdb = PDBFile(positions_from_pdb)
            simulation.context.setPositions(pdb.positions)
            simulation.context.setVelocitiesToTemperature(temp)
        else:
            simulation.loadState(state_in)

        if fn_stdout is None:
            fn_stdout = os.path.join(self.abs_work_dir, f'step_{stepnum}.log')

        if fn_dcd is None:
            fn_dcd = os.path.join(self.abs_work_dir, f'step_{stepnum}.dcd')

        SDR = StateDataReporter(fn_stdout, nstdout, step=True, time=True,
                                potentialEnergy=True, temperature=True, progress=False,
                                remainingTime=True, speed=True, volume=True,
                                totalSteps=nsteps, separator=' : ')

        simulation.reporters.append(SDR)
        DCDR = DCDReporter(file=fn_dcd, reportInterval=ndcd, append=append_dcd, enforcePeriodicBox=True)
        simulation.reporters.append(DCDR)
        print(f'Starting Step {stepnum} with forces {simulation.system.getForces()}')
        print(f'Starting Step {stepnum} with box_vectors {simulation.system.getDefaultPeriodicBoxVectors()}')

        # Write out state.xml
        if state_xml_out is None:
            state_xml_out = os.path.join(self.abs_work_dir, f'step_{stepnum}.xml')

        # Determine no. of steps per cycle
        steps_per_cycle = int(nsteps / ncycles)

        # Reconfigure steps needed to take if appending
        if append_dcd:
            steps_taken = float(simulation.context.getTime().value_in_unit(femtosecond) / dt)
            cycles_completed = math.floor(steps_taken / steps_per_cycle)
            print('steps_taken', steps_taken)
            print('cycles_complete', cycles_completed)
        else:
            cycles_completed = 0
            simulation.context.setTime(0)

        print('ncycles', ncycles)
        print('nsteps', nsteps)
        print('dt', dt)
        print('steps_per_cycle', steps_per_cycle)
        for cycle in range(cycles_completed+1, ncycles+1):
            print('Cycle', cycle, 'to', ((cycle/ncycles) * ((nsteps * dt) / 1e6)), 'ns')
            simulation.step(steps_per_cycle)
            self._describe_state(simulation, f'Step {stepnum}')
            self._write_state(simulation, state_xml_out)

        end = datetime.now() - start
        print(f'Step {stepnum} completed after {end}')
        print(f'Box Vectors after this step {simulation.system.getDefaultPeriodicBoxVectors()}')

        if pdb_out is None:
            pdb_out = os.path.join(self.abs_work_dir, f'step_{stepnum}.pdb')
        self._write_structure(simulation, pdb_out)

        for i in range(3):
            print('########################################################################################')
        return state_xml_out, pdb_out
