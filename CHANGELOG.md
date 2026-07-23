# Changelog

All notable changes to ChiMPSS are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — AlGDock BPMF engine, neural nets, fixes, and repo cleanup

### Added
- `src/chimpss/algdock/`: full vendored AlGDock BPMF engine (no external package required).
  Includes `BindingPMF`, all ForceFields (Grid, OBC, Sphere, Cylinder, Pose, ElectricField,
  OpenMM), Integrators (HMC, ExternalMC, SmartDarting, VelocityVerlet), `BAT`, `IO`, `RMSD`,
  `HMR`, `NAMD`, and sub-packages `bc_process`, `cd_process`, `configuration`, `free_energy`,
  and `analysis`. Migrated from `simtk.openmm` → `openmm` namespace throughout.
- `src/chimpss/neural_nets/vae.py`: `BatchNorm_VAE` (fully-connected VAE, ported from
  Deep-MMS) and `WaveTransformVAE` (3D convolutional VAE for wave-transformed grids,
  Kuzminykh et al. 2018). Implemented in JAX/Flax.
- `src/chimpss/neural_nets/wave_transform.py`: 3D wave-transform preprocessing for
  molecular voxel grids.
- `src/chimpss/cli/algdock.py`: `chimpss-algdock` CLI entry point.
- `tests/integration/test_algdock_ntsr1.py`: first end-to-end BPMF integration test
  (ML-301 / NTSR1, protein-only, BC phase).
- `tests/integration/run_motorrow.py`, `tests/integration/run_fultonmarket.py`:
  integration runner scripts.
- `docs/`: Sphinx autodoc scaffold for ReadTheDocs, with `autodoc_mock_imports` covering
  the full heavy-dep stack (openmm, mdtraj, rdkit, jax, flax, …).

### Changed
- `MotorRow`: fixed platform selection (OpenCL → CUDA/CPU fallback) and restart-from-
  checkpoint logic; fix `rmsd` to use mean not sum before sqrt for 1-D arrays.
- `FultonMarket`: `FultonMarketAnalysis` shim module made callable; fix `parse_atom_inds`
  to avoid `np.empty` uninitialized garbage indices.
- `Bridgeport`: integrated `Adjust_System` notebook logic into `generate_systems`; guard
  `modeller` import with `try/except`; lazy-import `py3Dmol` in `Analogue.visualize_alignment`.
- `src/chimpss/__init__.py`: re-exports `BindingPMF` from `chimpss.algdock` at package level.
- `conda-env.yml`: expanded from ~10 packages to the full ~25-package dependency set,
  including `openmm`, `openmmtools`, `mdtraj`, `mdanalysis`, `pdbfixer`, `parmed`,
  `pdb2pqr`, `pymbar`, `flax`, `optax`, `matplotlib`, `nglview`, `requests`, and dev tools.

### Removed
- All tracked `.ipynb_checkpoints` files (17 files — Jupyter auto-saves, now gitignored).
- `Test_Things_Work.py`: WIP script with legacy flat imports; superseded by
  `tests/regression/test_fultonmarket_e2e.py`.
- `test_retro_conv.py`: Expanse-hardcoded paths; superseded by
  `tests/regression/test_retro_conv.py`.
- `recovery.py`: superseded by `chimpss-recovery` CLI entry point.
- Root-level `recovery.job`, `run_mpi.job`, `run_one.job`: duplicate of canonical
  `scripts/slurm/` versions.
- `test_0.1.0.txt`: old pip-install verification log.
- `_00_current_notes.txt`: dev scratch notes.

### Added (repo hygiene)
- `.gitignore`: covers `__pycache__`, `*.pyc`, `.ipynb_checkpoints`, `*.egg-info`,
  simulation outputs (`*.dcd`, `*.nc`, `*.chk`), scratch dirs (`NA/`, `Hoods/`),
  and ephemeral planning files.

## [0.1.2] — File naming, path handling, and existence checks

### Added
- `src/chimpss/shared/io.py`: three new utilities — `validate_name` (rejects names with `_` or `.`), `build_output_path` (constructs `PROTEIN_LIGAND.significance.ext` paths), `file_exists_skip` (returns True and logs if output already exists)
- `protein_name` / `ligand_name` optional args added to `MotorRow.__init__` and `FultonMarket.__init__`; CLI flags `--protein-name` / `--ligand-name` added to `chimpss-motorrow` and `chimpss-fultonmarket`
- `protein_name` required field added to Bridgeport JSON schema (`input_params['protein_name']`)

### Changed
- **Bridgeport**: `Protein.input_pdb_dir` + `Protein.input_pdb` merged into a single `Protein.input_pdb` full path. Old JSON files must be updated.
- **Bridgeport outputs**: system PDB renamed `PROTEIN_LIGAND.topology.pdb`; system XML renamed `PROTEIN_LIGAND.system.xml`. `run()` skips the full build if these outputs already exist.
- **MotorRow step naming**: `Step_{n}.xml/pdb` → `step_{n}.xml/pdb`, `step{n}.dcd` → `step_{n}.dcd`, `step{n}.stdout` → `step_{n}.log`. **Breaking for in-flight jobs** — rename files manually or re-run from scratch.
- **MotorRow final outputs**: when `protein_name`/`ligand_name` are provided, step 5 outputs are renamed to `PROTEIN_LIGAND.equil.pdb` and `PROTEIN_LIGAND.state.xml`. `main()` now loops over steps with per-step existence checks, enabling resume from a partial run.
- **FultonMarket outputs**: trajectory renamed `PROTEIN_LIGAND.trajectory.ncdf`; checkpoint renamed `PROTEIN_LIGAND.checkpoint.ncdf` (when names provided). `run()` skips if trajectory already exists.
- Example notebooks (`SMALL_MOLECULE_EXAMPLE`, `ANALOGUE_EXAMPLE`, `PEPTIDE_EXAMPLE`, `MUTATED_PEPTIDE_EXAMPLE`, `interactive`): updated JSON dict cells to use `protein_name` + single-path `Protein.input_pdb`
- Tutorial `07-System_Preparation.ipynb`: updated protein dict cell and visualization to use new schema

### Backwards compatibility
- `protein_name` / `ligand_name` are **optional** in MotorRow and FultonMarket — omitting them preserves old generic naming
- `protein_name` is **required** in Bridgeport JSON; old JSON files missing this key will raise `KeyError`
- `Protein.input_pdb_dir` key is **removed** from Bridgeport JSON

## [Unreleased]

### Added (Phase 8 — Test suite + CI)
- `tests/conftest.py`: shared fixtures — `tmp_dir`, `small_pdb_content`, `small_pdb`, `pdb_with_dummy`, `opm_pdb_path`
- `tests/unit/test_shared/test_logging.py`: pure-stdlib tests for `timestamp` and `printf`; mdtraj-gated test for `unique_residues` + `report_chain_information`
- `tests/unit/test_shared/test_io.py`: mdtraj-gated tests for `ensure_exists`, `write_FASTA`, `remove_dummy_atoms`
- `tests/unit/test_fultonmarket/test_retro_logic.py`: pure-numpy tests for `resolve_write_dir`, `resolve_cache_dir`, `build_checks`, `print_summary_table`, `log_mode` — runs on CI with no MD stack via direct submodule import (bypasses heavy package `__init__`)
- `tests/unit/test_fultonmarket/test_math.py`: openmm-gated tests for `geometric_distribution`, `frobenius_norm`, `jsd_distance_matrices`, `rmsd`
- `tests/unit/test_analysis/test_pdb_fetch.py`: requests-gated tests for `classify_method` (pure logic, no network)
- `tests/unit/test_analysis/test_gpcr_pca.py`: MDAnalysis-gated tests for `three_to_one`, `_normalize_resname`, `build_gpcrdb_sequence`, `sliding_window_align`, `naming_from_convention`
- `tests/regression/test_fultonmarket_e2e.py`: `@pytest.mark.slow @pytest.mark.gpu` rewrite of `Test_Things_Work.py`; uses `--sim-dir`/`--pdb` CLI options; skips when fixtures absent
- `tests/regression/test_retro_conv.py`: `@pytest.mark.slow` rewrite of root `test_retro_conv.py`; all Expanse-hardcoded paths replaced with `pytest.addoption` CLI args

### Changed (Phase 8)
- `tests/unit/test_fultonmarket/test_retro_plotting.py`: added `pytest.importorskip("openmm")` guard (was missing — would error on CI without openmm)
- `.github/workflows/ci.yml`: added `unit-tests` job across Python 3.8/3.10/3.12; installs `pytest numpy scipy matplotlib requests` and runs the three test files that need no MD stack



### Added
- `src/chimpss/shared/__init__.py`: new shared utilities package
- `src/chimpss/shared/io.py`: consolidated I/O helpers from `utils/utils.py` and `utility/` — `ensure_exists`, `write_FASTA` (canonical version, returns path), `cif2pdb`, `remove_dummy_atoms`, `isolate_chains`, `slice_select`, `change_resname`, `describe_system`, `describe_state`; Python 3.8 f-string fix in `isolate_chains` (nested quote removed)
- `src/chimpss/shared/logging.py`: consolidated logging helpers from `utility/Reporting.py` — `timestamp`, `printf`, `unique_residues`, `report_chain_information`; Python 3.8 f-string fix in `printf`
- `utils/ProteinPreparer.py`: shim re-export → `chimpss.bridgeport.protein_preparer.ProteinPreparer` (retained for one release cycle)

### Changed
- `src/chimpss/bridgeport/ligand.py`: three lazy imports in `_prepare_peptide` updated — `from utils.utils import write_FASTA` → `chimpss.shared.io`, `from utils.ProteinPreparer import ProteinPreparer` → `chimpss.bridgeport.protein_preparer`, `from RepairProtein.RepairProtein import RepairProtein` → `chimpss.bridgeport.repair_protein`; all inter-package `sys.path` hacks now eliminated from migrated bridgeport code

### Notes (Phase 5)
- `utils/utils.py` retained as-is: still imported by `Bridgeport/Bridgeport.py` (legacy) and several notebooks; will be shimmed in Phase 7 during notebook cleanup
- `utility/` directory retained as-is: `Build(BP).ipynb`, `ChiMPSS-Showcase.ipynb`, and `NewDistanceMatrix.ipynb` import from it; will be converted to shims in Phase 7

### Added (Phase 7 — Notebook audit, extraction, and reorganization)  commit 60c700d
- `src/chimpss/fultonmarket/retro_convergence.py`: added `plot_convergence_metric`, `add_equil_metric_to_plot`, and `_RETRO_COLOR_MAP` — extracted from `Retro_Analysis.ipynb`; exported from `chimpss.fultonmarket`
- `tests/unit/test_fultonmarket/test_retro_plotting.py`: smoke tests for the two new plotting helpers
- `notebooks/tutorials/`: 14 numbered tutorial notebooks (01–13) moved from flat `notebooks/`
- `notebooks/examples/`: 8 example notebooks moved from `interactive/` + `notebooks/`; `interactive_utils.py` co-located
- `notebooks/analysis/`: 7 analysis notebooks (PCA, Retro, Distance Matrix, Interaction Energy, MPI compare, Run_MR_FM, Analyze_MPI_3.5)
- `notebooks/exploratory/`: 10 exploratory notebooks (Minimize, Fieg2005, Water_Hole, Talha_Contacts, Static_Env, implicit_membrane, Adjust_System, Docking_Test, BOX_VISUALISATION, Ligand_test)

### Changed (Phase 7)
- 23 notebooks updated: `sys.path.append` hacks removed; old flat imports (`Bridgeport.Bridgeport`, `MotorRow`, `FultonMarket.*`, `utility.*`, `DistanceMatrix.*`, `RepairProtein.*`) replaced with `chimpss.*` package imports
- `notebooks/analysis/Retro_Analysis.ipynb`: function-definition cells replaced with `from chimpss.fultonmarket import plot_convergence_metric, add_equil_metric_to_plot`
- `notebooks/analysis/PCA_of_GPCRs.ipynb`: `Structure_Analyzer` class-definition cell replaced with `from chimpss.analysis.gpcr_pca import Structure_Analyzer`
- `notebooks/analysis/NewDistanceMatrix.ipynb`: note cell added documenting JAX backend relationship and planned Phase 7.5 CPU/GPU dispatch work

### Deleted (Phase 7)
- `DistanceMatrix/Untitled.ipynb`: empty 78-byte placeholder removed

### Added (Phase 6 — Analysis module + CLI entry points)
- `src/chimpss/analysis/__init__.py`: new analysis package
- `src/chimpss/analysis/contacts.py`: thin re-export of `ContactNetworkBuilder` from `chimpss.fultonmarket.contact_network`
- `src/chimpss/analysis/pdb_fetch.py`: library functions for querying GPCRdb — `get`, `fetch_family_proteins`, `list_families`, `classify_method`
- `src/chimpss/analysis/distance_matrix.py`: consolidated `DistanceMatrix.py` + `DistanceMatrixUtils.py` — `DistanceMatrix` class plus all `_calc_*` and `_compute_*` helpers; Python 3.8 f-string fix in `printf`
- `src/chimpss/analysis/grid_potentials.py`: library-only portion of root `grid_potentials.py` — `IO_Grid`, `select_whole_residues`, `parameterize_from_pdb`, `simulate_from_filepair`; module-level simulation script code (original lines 343–442) excluded; imports updated to `chimpss.bridgeport.*`
- `src/chimpss/analysis/gpcr_pca.py`: consolidated `gpcr_pca_utils.py` + library helpers from `generate_pca_gpcrs.py` + library helpers from `project_pca_gpcrs.py` — all BW-mapping helpers, `Structure_Analyzer`, `fetch_all_parallel`, PCA helper functions, projection helpers, verbose diagnostics
- `src/chimpss/cli/__init__.py`: CLI package
- `src/chimpss/cli/motorrow.py`: argparse wrapper → `chimpss-motorrow`
- `src/chimpss/cli/fultonmarket.py`: argparse wrapper → `chimpss-fultonmarket`
- `src/chimpss/cli/fultonmarket_mpi.py`: MPI GPU-assignment wrapper → `chimpss-fultonmarket-mpi`; delegates to `chimpss.cli.fultonmarket.main()` (no `runpy.run_path` dependency)
- `src/chimpss/cli/retro_analysis.py`: argparse wrapper → `chimpss-retro-analysis`; fixes `None` bug in original `RUN_RETRO_ANALYSIS.py` (`output_cache_dir` defaulted to `<input_dir>/retro_cache/`); adds `--sele_str`, `--getcontacts_script`, `--conda_env`, `--getcontacts_python` args (replacing hardcoded Expanse paths)
- `src/chimpss/cli/recovery.py`: argparse wrapper → `chimpss-recovery`; converts `sys.argv[1]` to proper argparse
- `src/chimpss/cli/fetch_pdbs.py`: CLI portion of `fetch_pdb_list.py` → `chimpss-fetch-pdbs`
- `src/chimpss/cli/generate_pca.py`: CLI portion of `generate_pca_gpcrs.py` → `chimpss-generate-pca`
- `src/chimpss/cli/project_pca.py`: CLI portion of `project_pca_gpcrs.py` → `chimpss-project-pca`
- `scripts/slurm/run_one.job`: Expanse single-GPU job script (updated `python RUN_FULTONMARKET.py` → `chimpss-fultonmarket`)
- `scripts/slurm/run_mpi.job`: Expanse 4-GPU MPI job script (updated `mpiexec.hydra -np 4 python RUN_FULTONMARKET_MPI.py` → `mpiexec.hydra -np 4 chimpss-fultonmarket-mpi`)
- `scripts/slurm/recovery.job`: recovery job script (updated `python recovery.py` → `chimpss-recovery`)

### Changed (Phase 6)
- `pyproject.toml`: `[project.scripts]` entries uncommented and active for all 8 entry points (`chimpss-bridgeport` deferred — no `RUN_BRIDGEPORT.py` source; Bridgeport is used interactively)

### Backward-compat shims added (Phase 6)
- `gpcr_pca_utils.py` → `from chimpss.analysis.gpcr_pca import *`
- `GetContactsHelper.py` → re-exports `ContactNetworkBuilder` from `chimpss.fultonmarket.contact_network`
- `DistanceMatrix/DistanceMatrix.py` → re-exports `DistanceMatrix` from `chimpss.analysis.distance_matrix`
- `DistanceMatrix/DistanceMatrixUtils.py` → re-exports all `_calc_*` / `_compute_*` helpers from `chimpss.analysis.distance_matrix`
- `fetch_pdb_list.py` → delegates to `chimpss.cli.fetch_pdbs.main()`
- `generate_pca_gpcrs.py` → delegates to `chimpss.cli.generate_pca.main()`
- `project_pca_gpcrs.py` → delegates to `chimpss.cli.project_pca.main()`


- `pyproject.toml`: package skeleton — `pip install -e .` now works; `requires-python = ">=3.8"`; loose dep pins; `[dev]`, `[mpi]`, `[docs]` extras
- `src/chimpss/__init__.py`: package entry point, `__version__ = "0.1.0"`
- `.pre-commit-config.yaml`: ruff + black + nbstripout hooks
- `.github/workflows/ci.yml`: lint + no-deps install smoke test across Python 3.8 / 3.10 / 3.12
- `CLAUDE.md`: agent context file — active reorganization status, resumption phrase, key preferences
- `src/chimpss/bridgeport/`: Bridgeport and all dependencies migrated from `Bridgeport/`, `utils/`, `RepairProtein/`, `Ligand/`, `ForceFields/`, `Minimizer/`; all `sys.path.append` hacks removed; cross-module imports rewritten to `chimpss.bridgeport.*`; IPython side effects guarded with try/except
- `Bridgeport/__init__.py`: shim re-export → `chimpss.bridgeport.Bridgeport` (retained for one release cycle)
- `tests/unit/test_bridgeport/test_imports.py`: smoke tests (skip-if-no-mdtraj)
- `src/chimpss/motorrow/`: MotorRow and MotorRow_utils migrated from repo root; `sys.path.append` removed; imports rewritten to `chimpss.motorrow.*`; wildcard `from MotorRow_utils import *` replaced with explicit named imports
- `MotorRow.py`: shim re-export → `chimpss.motorrow.MotorRow` (retained for one release cycle)
- `tests/unit/test_motorrow/test_imports.py`: smoke tests (skip-if-no-openmm)
- `src/chimpss/fultonmarket/`: FultonMarket stage migrated from `FultonMarket/`; all six submodules migrated (`fulton_market.py`, `randolph.py`, `analysis.py`, `utils.py`, `contact_network.py`, `retro_convergence.py`); module-level side effects (`faulthandler.enable`, `warnings.filterwarnings`, `np.seterr`) consolidated into `__init__.py`; `@staticmethod` decorators removed from module-level functions; JAX import-time side effects guarded with try/except; `sys.path` depth for `project_gpcr_pca` updated (2 → 4 levels); wildcard imports replaced with explicit named imports; Python 3.8 f-string backslash fixed with `chr(10)`.
- `FultonMarket/__init__.py`: shim re-export → `chimpss.fultonmarket` classes (retained for one release cycle)
- `FultonMarket/FultonMarketwithAnalyzer.py`, `FultonMarket/RandolphwithAnalyzer.py`, `FultonMarket/FultonMarketAnalysis.py`, `FultonMarket/FultonMarketUtils.py`, `FultonMarket/retro_convergence_utils.py`: shim re-exports (retained for one release cycle)
- `tests/unit/test_fultonmarket/test_imports.py`: smoke tests (skip-if-no-openmm/mdtraj)
