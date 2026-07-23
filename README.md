# ChiMPSS — Chicago Membrane Protein Simulation Suite

**ChiMPSS** is a three-stage Python pipeline for preparing and executing explicit-solvent,
explicit-membrane parallel-tempering replica-exchange molecular dynamics (PTRE) simulations
of membrane proteins, with a primary focus on class-A G-protein coupled receptors (GPCRs).

```
PDB / FASTA / SMILES
        │
        ▼
  ┌─────────────┐   system.xml + system.pdb
  │  Bridgeport │ ──────────────────────────►
  │ (construct) │
  └─────────────┘
        │
        ▼
  ┌─────────────┐   equil.pdb + equil.xml
  │  MotorRow   │ ──────────────────────────►
  │ (burn-in)   │
  └─────────────┘
        │
        ▼
  ┌─────────────┐   .nc trajectory files
  │ FultonMarket│ ──────────────────────────►
  │  (PTRE run) │
  └─────────────┘
```


---

## Background

GPCRs are the largest family of drug targets in the human genome (~800 genes), accounting
for roughly one-third of all FDA-approved small-molecule drugs and ~27% of global therapeutic
drug sales.  Rather than simple on/off switches, GPCRs exist as dynamic conformational
ensembles that continuously sample active, partially active, and inactive states.  Ligands
shift this equilibrium rather than flip a binary switch — a phenomenon known as *biased
agonism*, where different ligands stabilize distinct receptor conformations that preferentially
couple to different intracellular signal transducers.

Standard molecular dynamics simulations can observe these conformational transitions, but
results are sensitive to starting configuration, random-velocity seed, and hardware.
**Parallel-Tempering Replica Exchange** (PTRE) substantially improves reproducibility by
running multiple replicas at geometrically-spaced temperatures and periodically swapping
coordinates between replicas based on a Metropolis criterion — allowing the ensemble to
escape local minima and sample a broader conformational space.

ChiMPSS implements the complete PTRE workflow from raw crystal structure to converged
conformational ensemble.

---

## Installation

ChiMPSS depends on the OpenMM ecosystem, which is distributed via conda-forge rather than
PyPI. The full scientific stack must be installed with conda first; `pip` is only used to
register the package in editable mode and install dev tools.

### Recommended: one-command environment

```bash
conda env create -f conda-env.yml
conda activate chimpss
pip install -e ".[dev]"   # registers the package + installs pytest, ruff, black
```

### Manual conda install (alternative)

```bash
conda create -n chimpss python=3.10
conda activate chimpss
conda install -c conda-forge -c salilab \
    openmm openmmtools mdtraj mdanalysis pdbfixer \
    openff-toolkit pdb2pqr parmed pymbar mpiplus \
    rdkit scikit-learn netCDF4 seaborn jax \
    openbabel modeller py3Dmol vina
pip install -e ".[dev]"
```

### MPI support (for FultonMarket multi-node runs)

```bash
pip install -e ".[mpi]"   # installs mpi4py
```

### GPU-enabled JAX (required for FultonMarket on GPU)

The conda-forge `jax` package is CPU-only.  FultonMarket's PTRE engine requires a
GPU-accelerated JAX build, which must be installed separately via pip **after** the
conda environment is active.

**CUDA 12.x (most common on modern GPUs and HPC clusters):**

```bash
pip install -U "jax[cuda12]"
```

Verify GPU access after installation:

```python
import jax
print(jax.devices())   # should list one or more CudaDevice entries
```

> **Note:** the pip JAX wheel must match the CUDA toolkit version present on the
> machine.  On HPC systems (e.g. Expanse) load the appropriate CUDA module
> (`module load cuda/12.2` or similar) before running pip so the installer can
> detect the correct variant automatically.

### getcontacts (required for contact-distance convergence metrics)

getcontacts is not available on PyPI or conda-forge and must be installed directly
from its GitHub repository.

```bash
# clone anywhere on the machine
git clone https://github.com/getcontacts/getcontacts.git ~/getcontacts

# add the directory to PATH (add this line to ~/.bashrc or ~/.bash_profile)
export PATH="$HOME/getcontacts:$PATH"

# verify
get_dynamic_contacts.py --help
```

getcontacts also requires **VMD** (Visual Molecular Dynamics) for some contact
calculation backends.  Install VMD from https://www.ks.uiuc.edu/Research/vmd/ and
ensure the `vmd` binary is on your PATH.

After installation, confirm ChiMPSS can locate getcontacts:

```python
from chimpss.analysis.contacts import check_getcontacts
check_getcontacts()   # prints path if found, raises if not
```

---

## Stage 1 — Bridgeport: System Construction

```python
from chimpss.bridgeport import Bridgeport

bp = Bridgeport("input.json")
bp.run()
```

### Purpose

Bridgeport transforms a raw crystallographic or cryo-EM structure into a fully solvated,
membrane-embedded OpenMM system ready for simulation.  It handles every step from structure
retrieval to forcefield parameterization, including ligand replacement by analogue when
the desired ligand differs from the one present in the deposited structure.

### Inputs

| File | Description |
|---|---|
| `input.json` | JSON config specifying PDB ID, FASTA sequence, ligand SMILES, forcefield choices, and output paths |
| PDB structure | Raw crystal/cryo-EM structure from RCSB (or local file) |
| OPM reference | Membrane-orientation reference (downloaded automatically or provided locally) |

### Outputs

| File | Description |
|---|---|
| `systems/system.pdb` | Solvated, membrane-embedded coordinates |
| `systems/system.xml` | OpenMM serialized forcefield + topology |

### Step-by-step

1. **Align to membrane orientation.**
   The input structure is aligned to its counterpart in the Orientations of Proteins in
   Membranes (OPM) database, placing the protein correctly relative to the bilayer normal
   before the membrane is added.

2. **Separate ligand and protein.**
   The crystallographic ligand is removed from the protein chain.  If the desired
   simulation ligand matches the deposited one, it is retained for parameterization.  If a
   different compound is required (*analogue* mode), the deposited ligand is used only as a
   positional template.

3. **Repair missing loops.**
   Crystal structures frequently have unresolved loop regions.  Bridgeport uses
   **MODELLER 10.5** to predict three-dimensional coordinates for any missing segments,
   guided by the provided FASTA sequence.  If the MODELLER prediction contains obvious
   clashes or disordered regions, the AlphaFold structure is used as a secondary template.

4. **Protonate the protein.**
   The repaired protein is protonated with **PDB2PQR 3.6.1** at pH 7.0 to assign
   biologically relevant protonation states.

5. **Add solvent and membrane.**
   **PDBFixer 1.9.0** (an OpenMM extension) builds a POPC lipid bilayer around the
   protein and floods the remaining volume with explicit water, using a 2 nm padding in
   each dimension and neutralizing 150 mM NaCl.

6. **Parameterize protein, lipids, and water.**
   - Protein → **ff14SB** AMBER forcefield
   - POPC membrane → **Lipid17** AMBER forcefield
   - Water → **OPC3** model

7. **Prepare the ligand.**
   - *Direct preparation:* a ligand already present in the structure is protonated with
     RDKit at pH 7.0 and its geometry is validated.
   - *Analogue preparation:* a SMILES string for the desired compound is loaded with
     **RDKit 2023.09**.  3D conformers are generated and aligned to the crystallographic
     ligand by maximum common substructure (MCS); conformers are sampled until the MCS
     RMSD is below 2 Å.  Preparation then proceeds identically to the direct path.
   - Ligands are parameterized with the **OpenFF 2.1.0** force field via openff-toolkit.

8. **Assemble the OpenMM system.**
   The parameterized protein, membrane, water, and ligand topologies are joined into a
   single OpenMM `Topology` + `System` pair, then serialized to `system.pdb` and
   `system.xml`.

---

## Stage 2 — MotorRow: Equilibration

```python
from chimpss.motorrow import MotorRow

mr = MotorRow(
    pdb_file="systems/system.pdb",
    system_xml="systems/system.xml",
    working_directory="equilibration/",
    lig_resname="UNK",
)
mr.main(pdb_in="systems/system.pdb")
```

Or via CLI:

```bash
chimpss-motorrow --pdb systems/system.pdb \
                 --xml systems/system.xml \
                 --outdir equilibration/
```

### Purpose

Freshly constructed systems contain steric clashes and artificially large forces.  MotorRow
runs a five-step equilibration protocol that progressively relaxes the system from a
restrained, minimized state to a freely-fluctuating membrane ensemble, producing stable
starting coordinates for PTRE.

### Inputs

| Argument | Description |
|---|---|
| `pdb_file` | Output of Bridgeport (`system.pdb`) |
| `system_xml` | Output of Bridgeport (`system.xml`) |
| `working_directory` | Directory for DCD trajectories and checkpoint files |

### Outputs

| File | Description |
|---|---|
| `equil/step5_final.pdb` | Coordinates after the final equilibration step |
| `equil/step5_final.xml` | OpenMM state (velocities + box vectors) after equilibration |

### Five-step protocol

All steps use a 2 fs timestep, 300 K target temperature, and the OpenCL platform by default.
Heavy-atom restraints are applied via a harmonic flat-bottom potential.

| Step | Ensemble | Duration | Restraints | Barostat |
|---|---|---|---|---|
| 0 | — | until ΔE < tol | all heavy atoms | none |
| 1 | NVT | 250 ps | protein + ligand heavy atoms; membrane Z | none |
| 2 | NPT | 250 ps | protein + ligand heavy atoms; membrane Z | isotropic, 1 atm |
| 3 | NVT | 250 ps | none | none |
| 4 | NPT | 2.5 ns | none | Monte Carlo Membrane Barostat |
| 5 | NPT | 2.5 ns | none | Monte Carlo Barostat |

Steps 1–2 hold the protein and membrane in place while the solvent packs in.
Step 3 releases all restraints and allows initial thermal equilibration.
Steps 4–5 equilibrate box dimensions, first allowing membrane-specific xy/z coupling,
then switching to isotropic pressure coupling for the final ~3 ns.

---

## Stage 3 — FultonMarket: Parallel-Tempering REMD

```python
from chimpss.fultonmarket import FultonMarket

fm = FultonMarket(
    input_pdb="equilibration/step5_final.pdb",
    input_system="systems/system.xml",
    input_state="equilibration/step5_final.xml",
    T_min=310,
    T_max=367,
    n_replicates=12,
)
fm.run(total_sim_time=25, iter_length=1, output_dir="remd_leg1/")
```

Or via CLI:

```bash
chimpss-fultonmarket --pdb equilibration/step5_final.pdb \
                     --xml systems/system.xml \
                     --state equilibration/step5_final.xml \
                     --outdir remd_leg1/
```

MPI-parallel (multi-node HPC):

```bash
mpirun -n 12 chimpss-fultonmarket-mpi \
    --pdb equilibration/step5_final.pdb \
    --xml systems/system.xml \
    --outdir remd_leg1/
```

### Purpose

FultonMarket runs Parallel-Tempering Replica Exchange MD (PTRE) to generate a converged
conformational ensemble of the membrane protein.  By exchanging coordinates between replicas
at different temperatures, the simulation can overcome kinetic barriers and sample the full
range of biologically relevant receptor conformations — including active, inactive, and
intermediate states.

### Inputs

| Argument | Description |
|---|---|
| `input_pdb` | Equilibrated coordinates from MotorRow |
| `input_system` | OpenMM system XML from Bridgeport |
| `input_state` | (optional) OpenMM state XML from MotorRow (provides box vectors + velocities) |

### Outputs

| File | Description |
|---|---|
| `remd_legN/*.nc` | NetCDF4 trajectory files, one per replica |
| `remd_legN/reporter.log` | Per-iteration energies and exchange statistics |

### How it works

1. **Temperature ladder.**
   Replicas are distributed geometrically between `T_min` (default 310 K) and `T_max`
   (default 367 K).  The geometric spacing ensures equal acceptance ratios between adjacent
   replicas.  Starting with 12 replicas is typical for a GPCR-sized system.

2. **Propagation.**
   Each replica is integrated with the **Langevin Middle Integrator** at its assigned
   temperature (1 ps⁻¹ friction, 2 fs timestep).  After each iteration — defined as
   1 aggregate picosecond of propagation across all replicas — exchange moves are proposed
   between adjacent replicas using a Metropolis criterion.

3. **Adaptive state insertion (first leg only).**
   After the first 25 ns leg, the acceptance probability between every pair of adjacent
   replicas is checked.  If any pair exchanges less than 50% of the time, a new replica is
   inserted between them and the leg is restarted.  In subsequent legs the threshold is
   lowered to 35%.  This ensures adequate mixing without over-resolving the temperature ladder.

4. **Convergence assessment.**
   At the end of each 25 ns leg, the trajectory is scanned for equilibration using
   **pyMBAR 4.2** (`pymbar.timeseries.detect_equilibration`), then importance-resampled
   (n = 1000, with replacement) across the temperature ladder.  Simulation continues until
   two criteria are met simultaneously:
   - **Minimum length:** ≥ 1 µs aggregate simulation time across all replicas.
   - **Equilibration fraction:** ≥ 25% of the aggregate trajectory is classified as
     post-equilibration by MBAR.

5. **Distance-metric convergence tracking.**
   Three complementary metrics track how the conformational ensemble evolves between legs:
   - *Torsional distance* — root-mean-square periodic angular displacement across backbone
     and side-chain dihedrals.
   - *Alpha-carbon distance* — root-mean-square Cα translational displacement between frames.
   - *Contact distance* — fraction of intramolecular contacts (5–95% prevalence, from
     `getContacts`) whose binary on/off state differs between two frames (logical XOR).

---

## Analysis

The `chimpss.analysis` module provides tools for post-simulation analysis:

| Module | Description |
|---|---|
| `chimpss.analysis.gpcr_pca` | Project trajectories onto a PCA space built from 555 experimental GPCR structures with Ballesteros–Weinstein residue mapping |
| `chimpss.analysis.distance_matrix` | Pairwise distance matrices from torsional, Cα, and contact metrics |
| `chimpss.analysis.contacts` | Contact prevalence analysis via getContacts |
| `chimpss.analysis.pdb_fetch` | Batch retrieval of PDB structures by GPCR class |
| `chimpss.analysis.grid_potentials` | Electrostatic grid potential analysis |

### GPCR structural PCA

```python
from chimpss.analysis.gpcr_pca import Structure_Analyzer

sa = Structure_Analyzer(reference_pdbs="gpcr_reference_set/")
sa.fit()
sa.project(trajectory="remd_leg1/replica_0.nc", topology="equilibration/step5_final.pdb")
sa.plot()
```

The reference PCA was constructed from 555 human GPCR crystal structures sourced from
GPCRdb, filtered to the 203 Ballesteros–Weinstein positions present in ≥ 85% of structures,
yielding 812 backbone atoms per structure.  Projecting a simulation trajectory into this
space places each sampled conformation relative to the known active/inactive states of the
human GPCRome.

---

## Stage 4 — AlGDock: Binding PMF (BPMF)

The `chimpss.algdock` module is a vendored BPMF engine for computing absolute binding
free energies of small-molecule ligands.  No external AlGDock package is required — the
full engine is bundled under `src/chimpss/algdock/`.

```python
from chimpss.algdock import BindingPMF

bpmf = BindingPMF(
    ligand_tarball="ML-301_local.json",
    receptor_pdb="NTSR1.topology.pdb",
    receptor_inpcrd="NTSR1.topology.pdb",
    receptor_prmtop="NTSR1.prmtop",
    site="Measure",
    solvation="Full",
)
bpmf.run_BC()   # binding-site confinement (BC) phase
bpmf.run_CD()   # cool-down / alchemical decoupling (CD) phase
```

Or via CLI:

```bash
chimpss-algdock --ligand ML-301_local.json \
                --receptor-pdb NTSR1.topology.pdb \
                --receptor-prmtop NTSR1.prmtop \
                --site Measure --solvation Full
```

BPMF requires two external source packages compiled from local clones:

```bash
pip install -e /path/to/occupancy_fingerprinter
pip install -e /path/to/openmmgridforce   # requires CMake + OpenMM headers
```

---

## Neural Networks (`chimpss.neural_nets`)

The `chimpss.neural_nets` module provides JAX/Flax neural network models for
molecular representation learning, supporting the ML/CADD integration roadmap.

| Class | Description |
|---|---|
| `BatchNorm_VAE` | Fully-connected variational autoencoder with BatchNorm + Dropout (ported from Deep-MMS) |
| `WaveTransformVAE` | 3D convolutional VAE for wave-transformed molecular grids (Kuzminykh et al. 2018) |

```python
from chimpss.neural_nets.vae import BatchNorm_VAE, WaveTransformVAE
from chimpss.neural_nets.wave_transform import wave_transform_grid
```

Requires GPU-enabled JAX and Flax:

```bash
pip install -U "jax[cuda12]"
conda install -c conda-forge flax optax
```

---

## CLI reference

| Command | Description |
|---|---|
| `chimpss-motorrow` | Run MotorRow 5-step equilibration |
| `chimpss-fultonmarket` | Run FultonMarket PTRE (single process) |
| `chimpss-fultonmarket-mpi` | Run FultonMarket PTRE (MPI, multi-node) |
| `chimpss-retro-analysis` | Retrospective convergence analysis on completed legs |
| `chimpss-fetch-pdbs` | Batch-download PDB structures by accession list |
| `chimpss-generate-pca` | Build GPCR reference PCA from a set of PDB structures |
| `chimpss-project-pca` | Project new trajectories onto an existing PCA model |
| `chimpss-recovery` | Resume an interrupted FultonMarket run from the last checkpoint |
| `chimpss-algdock` | Run AlGDock BPMF free-energy calculation |

Pass `--help` to any command for full argument documentation.

---

## Testing

Unit tests (no MD stack required):

```bash
pytest tests/unit -m "not slow and not gpu"
```

Full verification on a Linux machine with the conda environment:

```bash
bash tests/VERIFY.sh
```

---

## Citation

If you use ChiMPSS in published work, please cite the three stage repositories:

- **Bridgeport** — system construction
- **MotorRow** — burn-in equilibration
- **FultonMarket** — parallel-tempering REMD

*(Formal citations will be added upon publication.)*
