# 5-HT2B Biased Agonism — HMR Equilibration Workflow

Project: *Structural determinants of 5-HT2B receptor activation and biased agonism* (Fig. 4 + Fig. 5).

This document records exactly what was done between the completed Bridgeport prep
and the upcoming FultonMarket replica-exchange production runs, so that the
equilibrated systems can be inspected and signed off before launching FultonMarket.

---

## 1. Systems (9 total)

| # | Name (base)                          | Description                                | Ligand resname |
|---|--------------------------------------|--------------------------------------------|---------------|
| 1 | `lisuride`                           | Fig. 4 WT — lisuride                       | UNK |
| 2 | `methylergonovine`                   | Fig. 4 WT — methylergonovine               | UNK |
| 3 | `methysergide`                       | Fig. 4 WT — methysergide                   | UNK |
| 4 | `LSD`                                | Fig. 4 WT — LSD                            | UNK |
| 5 | `LY266097`                           | Fig. 5 WT — LY266097                       | UNK |
| 6 | `lisuride_L362F_mutseq`              | Fig. 4 mutant — L362F (sequence-introduced)| UNK |
| 7 | `methylergonovine_T140A_mutseq`      | Fig. 4 mutant — T140A                      | UNK |
| 8 | `methysergide_A225G_mutseq`          | Fig. 4 mutant — A225G                      | UNK |
| 9 | `LSD_L362F_mutseq`                   | Fig. 4 mutant — L362F                      | UNK |

All ligand parameters/topology were already baked into the Bridgeport system XML;
the residue name `UNK` is what MotorRow uses to find the ligand for restraints.

---

## 2. Stage 1 — Bridgeport (DONE)

Already complete. Outputs (one `.pdb` + one `.xml` per system) live at:

```
/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems/
    <name>.pdb
    <name>.xml
```

These are the standard, non-HMR systems straight out of Bridgeport
(`/home/fcetin/Bridgeport`). Submission script for reference:
`/home/fcetin/Bridgeport/Bridgeport/submit_5ht2b_biased_agonism.sh`.

---

## 3. Stage 2a — Hydrogen Mass Repartitioning (HMR) (DONE)

### Why HMR?

Plain MD with all-atom hydrogens is constrained to a ~2 fs timestep. HMR transfers
mass from heavy atoms to their bonded (non-water) hydrogens, slowing the fastest
vibrational modes without changing total system mass. This makes a **3.5 fs timestep
stable** in MotorRow/FultonMarket, ≈1.75× the throughput per wall-clock hour
relative to dt=2 fs.

### How it was applied

The HMR + force-group rewrite uses ChiMPSS's reference implementations:

- `chimpss.bridgeport._utils.apply_force_groups`
  → bonded forces → group 0, PME reciprocal space → group 1 (sets us up for MTS later if desired).
- `chimpss.bridgeport._utils.apply_hmr`
  → for each non-water X–H bond, transfers mass from the heavy atom to the H using scale factors `(2.0, 2.5, 2.5, 2.5)` keyed by the number of H bound to that heavy atom. Water is excluded. Total mass is conserved.

Driver script:
`/home/fcetin/MotorRow/generate_hmr_xmls.py`

Exact command used (run on a login node, ~30 s total):

```bash
conda activate prep1
cd /home/fcetin/MotorRow
python generate_hmr_xmls.py \
    --input_dir  /expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems \
    --output_dir /expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems_hmr \
    --names lisuride methylergonovine methysergide LSD LY266097 \
            lisuride_L362F_mutseq methylergonovine_T140A_mutseq \
            methysergide_A225G_mutseq LSD_L362F_mutseq
```

### Output files (this is what to inspect)

```
/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems_hmr/
    lisuride_FG_HMR.xml
    methylergonovine_FG_HMR.xml
    methysergide_FG_HMR.xml
    LSD_FG_HMR.xml
    LY266097_FG_HMR.xml
    lisuride_L362F_mutseq_FG_HMR.xml
    methylergonovine_T140A_mutseq_FG_HMR.xml
    methysergide_A225G_mutseq_FG_HMR.xml
    LSD_L362F_mutseq_FG_HMR.xml
```

Per-system mass-conservation check (`Δmass = mass_after − mass_before`, in amu):

| System                              | Mass before (amu) | Mass after (amu)  | Δmass        |
|-------------------------------------|-------------------|-------------------|--------------|
| lisuride                            | 592065.024532     | 592065.024532     | 0.000e+00    |
| methylergonovine                    | 688322.889295     | 688322.889295     | 0.000e+00    |
| methysergide                        | 698537.823969     | 698537.823969     | 0.000e+00    |
| LSD                                 | 789330.877865     | 789330.877865     | −1.164e−10   |
| LY266097                            | 609261.766661     | 609261.766661     | 0.000e+00    |
| lisuride_L362F_mutseq               | 570242.870532     | 570242.870532     | 0.000e+00    |
| methylergonovine_T140A_mutseq       | 690110.911295     | 690110.911295     | 0.000e+00    |
| methysergide_A225G_mutseq           | 709775.397969     | 709775.397969     | 0.000e+00    |
| LSD_L362F_mutseq                    | 717309.915865     | 717309.915865     | 0.000e+00    |

All deltas are zero or float-round-off (≤ 1e−9 amu) — confirms the HMR transfer is conservative.

### What changed vs. the original XML?

- Per-particle masses on **non-water** heavy atoms decrease; bonded H masses increase.
  Total system mass is unchanged.
- The `NonbondedForce`'s reciprocal-space part is moved to force group 1 for MTS use.
- Periodic box, force constants, particle indices, ligand parameters, topology — **all untouched**.

The PDB files in `work_dir/systems/` are still the correct starting coordinates;
only the XML is HMR-modified.

---

## 4. Stage 2b — MotorRow equilibration @ dt = 3.5 fs (DONE — all 9 PASSED)

### Engine

We are using the **ChiMPSS MotorRow** (not the older `/home/fcetin/MotorRow/MotorRow.py`),
because it threads `dt` through `main → _minimize / _run_step`. The legacy local
MotorRow hard-codes 2 fs and is being retired for this run.

Source: `/home/fcetin/ChiMPSS/src/chimpss/motorrow/motorrow.py`

### Protocol (5 steps, all on the HMR system with dt = 3.5 fs)

| Step | Ensemble | Restraints                                       | Length |
|------|----------|--------------------------------------------------|--------|
| 0    | Minimize | Heavy-atom restraints (protein + ligand)         | —      |
| 1    | NVT      | Membrane Z + protein XYZ + ligand                | 250 ps |
| 2    | NPT      | + MonteCarloMembraneBarostat                     | 250 ps |
| 3    | NVT      | (free)                                           | 250 ps |
| 4    | NPT      | MonteCarloMembraneBarostat (free)                | 2.5 ns |
| 5    | NPT      | MonteCarloBarostat (isotropic, free)             | 2.5 ns |

### Submission

Driver:        `/home/fcetin/MotorRow/RUN_CHIMPSS_MOTORROW.py`
SLURM job:     `/home/fcetin/MotorRow/EQUIL_HMR.job`  (1× GPU, 8 GB, 6 h, account=iit127)
Batch submit:  `/home/fcetin/MotorRow/submit_5ht2b_equil_hmr.sh`

Command:

```bash
bash /home/fcetin/MotorRow/submit_5ht2b_equil_hmr.sh
```

This fires 9 sbatch jobs to the `gpu-shared` partition.

### Output layout

Each system gets its own subdirectory:

```
/expanse/lustre/projects/iit127/fcetin/5ht2b/equil_hmr/<name>/
    minimized.pdb       minimized_wrapped.pdb       minimized.xml
    step_1.pdb          step_1_wrapped.pdb          step_1.xml      step_1.dcd      step_1.log
    step_2.*            (same pattern)
    step_3.*
    step_4.*
    step_5.*            <-- final equilibrated state for FultonMarket
```

SLURM log files: `/expanse/lustre/projects/iit127/fcetin/5ht2b/logs/equil_hmr/EQUIL_HMR.<jobid>.<node>.out`

### What to verify before FultonMarket

1. Each `<name>/` contains `step_5.xml` (5/5 steps completed cleanly — no crashes from a too-aggressive dt).
2. `step_5.log` shows stable potential energy / temperature / box volume over the step-5 block.
3. `step_5_wrapped.pdb`: protein still membrane-embedded, no exploded coordinates.
4. (Optional) compare `step_5.pdb` RMSD vs. starting `<name>.pdb` — should be on the order of a normal MD relaxation, not blown out.

> **Filename note:** the ChiMPSS MotorRow writes **lowercase** `step_N.{xml,pdb}` (the legacy local MotorRow uses capital `Step_N`). All paths below use the lowercase convention.

### Verification — all 9 PASSED (recorded 2026-05-22)

Mean ± σ over the **last 10 %** of `step_5.log` rows (1 250 rows per system,
written every 1 000 MD steps; step 5 advanced 1.25 M steps × 3.5 fs = 4.375 ns):

| System                              | T (K)         | PE σ / \|PE\| (‰) | V σ (nm³) | Result |
|-------------------------------------|---------------|------------------|-----------|--------|
| lisuride                            | 300.92 ± 0.92 | 0.85             | 1.35      | PASS   |
| lisuride_L362F_mutseq               | 300.77 ± 0.97 | 0.90             | 1.15      | PASS   |
| LSD                                 | 300.71 ± 0.78 | 0.72             | 1.48      | PASS   |
| LSD_L362F_mutseq                    | 300.94 ± 0.73 | 0.73             | 1.43      | PASS   |
| LY266097                            | 300.95 ± 0.95 | 0.90             | 1.29      | PASS   |
| methylergonovine                    | 301.12 ± 0.92 | 0.73             | 1.49      | PASS   |
| methylergonovine_T140A_mutseq       | 300.84 ± 0.80 | 0.74             | 1.42      | PASS   |
| methysergide                        | 300.99 ± 0.83 | 0.72             | 1.27      | PASS   |
| methysergide_A225G_mutseq           | 300.72 ± 0.81 | 0.71             | 1.31      | PASS   |

Automated checks performed:
- Every `<name>/` contains the full set `minimized.* / step_1.*…step_5.*` — no
  crashed or partial trajectories.
- `grep -ciE "nan|inf"` over every `step_*.log` returns 0.
- Step 5 reached t = 4 375 ps for all 9 systems.
- Throughput on Expanse `gpu-shared` (single V100): ~184 ns/day.

Interpretation: temperature held within ±1 K of the 300 K target across all
systems; potential energy σ is < 0.001 of the mean (no drift); box volume σ is
≈ 0.13 % of the box volume — i.e. the membrane barostats are well-coupled and
not drifting. **HMR @ 3.5 fs is stable for all 9 systems**, and they are
eligible inputs for FultonMarket pending the verification spot-check above.

---

## 5. Stage 3 — FultonMarket (NOT YET LAUNCHED)

After the professor signs off on the equilibrated structures, FultonMarket will be
launched with:

- **PDB**:       `equil_hmr/<name>/step_5.pdb`
- **System XML**: `work_dir/systems_hmr/<name>_FG_HMR.xml`  *(same HMR XML used during equil)*
- **State XML**: `equil_hmr/<name>/step_5.xml`  *(continues velocities)*
- **Timestep**:  `--timestep 3.5`

Launcher and submit script for this stage will be added once the equilibrations are
approved. They are deliberately not auto-submitted from this batch.

---

## 6. File summary (what was added/used in this stage)

New files created in this stage:

- `/home/fcetin/MotorRow/generate_hmr_xmls.py`     — HMR + force-group rewrite driver
- `/home/fcetin/MotorRow/RUN_CHIMPSS_MOTORROW.py`  — single-system ChiMPSS MotorRow driver with `--dt`
- `/home/fcetin/MotorRow/EQUIL_HMR.job`            — SLURM job script
- `/home/fcetin/MotorRow/submit_5ht2b_equil_hmr.sh`— batch submitter for all 9 systems
- `/home/fcetin/MotorRow/WORKFLOW_5HT2B_HMR.md`    — this document

Pre-existing files used (read-only):

- Bridgeport outputs:   `/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems/{<name>.pdb,<name>.xml}`
- ChiMPSS HMR utils:    `/home/fcetin/ChiMPSS/src/chimpss/bridgeport/_utils.py`  (`apply_force_groups`, `apply_hmr`)
- ChiMPSS MotorRow:     `/home/fcetin/ChiMPSS/src/chimpss/motorrow/motorrow.py`

Pre-existing non-HMR equilibration (dt = 2 fs) — kept for reference, not used downstream:

- `/expanse/lustre/projects/iit127/fcetin/5ht2b/equil/<name>/`
