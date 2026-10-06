# 5-HT2B FultonMarket campaign — v2 runbook

_Started 2026-10-06. Replaces the v1 production data in `../fultonmarket_hmr/`,
which is **not valid REMD** (replica-reset bug; see `../FULTONMARKET_HANDOFF.md`)._

## What changed from v1, and why

| | v1 | v2 | why |
|---|---|---|---|
| Replica continuity | every replica reset to the 300 K structure each sub-sim | each replica continues its own trajectory | `randolph.py` passed `sampler_states[0]`; fixed in ChiMPSS `24e6d44` |
| Ligand charge | neutral (all 9) | **+1**, basic amine protonated | standard bound state; salt bridge to D135 (D3.32) |
| System net charge | +7 to +9 e | **0** (8–10 extra Cl⁻ in bulk water) | PME net-charge artefact; ChiMPSS `0a3c324`, `9a59cce` |
| Production pressure coupling | isotropic | **membrane barostat**, XY isotropic / Z free, γ = 0 | let the bilayer relax area and thickness separately |
| MotorRow step 5 | isotropic barostat | membrane barostat, γ = 0 | end equilibration in the production ensemble |
| Convergence check | Frobenius/JSD (could not fail) | `chimpss-convergence` (noise-calibrated) | ChiMPSS `3f36d8a` |
| Resubmission | manual (lost 8 and 41 days) | **self-resubmitting** jobs | `scripts/RUN_FM_V2.job` |

**Unchanged on purpose:** protein model, membrane and water (the original
`work_dir/proteins/<name>_env.pdb` is reused verbatim and checked atom by
atom), force fields, HMR recipe (reproduced bit-for-bit), dt = 3.5 fs (PI
directive), 300–367 K ladder, 17.5 ps exchanges, 25 ns sub-sims, 1500 ns
aggregate target.

## Decisions for the PI to confirm

1. **Protonation sites.** Ergolines: N6, the ring-D tertiary amine bearing the
   N-methyl. LY266097: the tetrahydro-β-carboline secondary amine. Solution pKa
   of some ergolines is near 6.6–7.8, but the protonated form is the standard
   choice for the D3.32-bound state.
2. **γ = 0** (tensionless), as Amber Lipid17 is parameterized. MotorRow steps 2
   and 4 still use its original 300 bar·nm.
3. **Run length.** 1500 ns aggregate is still the target, about 8–10 ns per
   replica. Extending is cheap and lossless: raise `TOTAL_SIM_TIME` in
   `scripts/submit_fm_v2.sh` and resubmit. FultonMarket resumes from
   `saved_variables/`. Let `chimpss-convergence` decide whether to.

## Log

- **2026-10-06 ~06:16: canary FAILED on a wrong check, not on the simulation.**
  The check required box x == y, but XY-isotropic coupling keeps the x/y
  *ratio* constant; the real boxes are rectangular, and the CPU smoke test used
  a cube. Every continuity check passed. Fixed in `check_canary.py`; the re-run
  check on the same output passes
  (`logs/fultonmarket/CANARY_V2.54695077.recheck.log`: x/y constant to 3.4e-7,
  x/z varies 1.7%).
- **2026-10-06 ~11:55: production launched by hand** for the 8 equilibrated
  systems, since the launchers were blocked by the canary's failed exit code:
  FM_V2 54708710 (LSD), 54708711 (lisuride), 54708712 (methylergonovine),
  54708713 (methysergide), 54708714 (LY266097), 54708715 (lisuride_L362F),
  54708716 (LSD_L362F), 54708717 (methysergide_A225G). All carry
  `--membrane_barostat --surface_tension 0`; each later window self-resubmits.
- **methylergonovine_T140A_mutseq**: equilibration NaN'd 2.54 ns into MotorRow
  step 4 (the original v1 step; stable at ~300 K until then), consistent with
  stochastic 3.5 fs blow-ups. The partial step-4 files were moved to
  `equil_hmr/methylergonovine_T140A_mutseq/failed_attempt_1_step4_NaN/`; a plain
  resubmit would have *skipped* the rest of step 4 because MotorRow wrote its
  per-cycle checkpoint to `step_4.xml` (fixed in ChiMPSS; checkpoints now go to
  `step_N.partial.xml`). Rerun: EQUIL 54708724, then launcher 54708725 (afterok).

## Layout

    v2/json/<name>.json            Bridgeport input (protonated SMILES, v2 working_dir)
    v2/work_dir/systems/           neutralized systems (<name>.pdb/.xml)
    v2/work_dir/systems_hmr/       <name>_FG_HMR.xml  (what MotorRow/FultonMarket use)
    v2/equil_hmr/<name>/           MotorRow output; step_5.{pdb,xml} feed production
    v2/canary/LSD/                 3-sub-sim canary of the production code
    v2/fultonmarket/<name>/        PRODUCTION output (saved_variables/, convergence.json)
    v2/code/                       FROZEN code snapshot production runs (commit in code/COMMIT)
    v2/scripts/                    everything that built/runs v2
    v2/logs/{bridgeport,equil,fultonmarket,retro}/

## The job graph (runs unattended)

    BUILD_V2 (done, 9/9) -> VERIFY_V2 (done, 9/9 ALL PASS)
      -> EQUIL_V2_<name>            gpu-shared, ~1.5 h each
      -> CANARY_V2 (LSD only)       afterok EQUIL_V2_LSD; 4 replicas, 3 short sub-sims;
                                    checks replica continuity + membrane barostat; exits 1 on failure
      -> LAUNCH_<name>              afterok EQUIL_V2_<name> AND CANARY_V2
      -> FM_V2 (<name>)             48 h windows; at T-15 min each window submits its own
                                    successor (afterany on itself); stops at 60/60
      -> CONV_<name>                submitted by the window that reaches 60/60

**Pre-checked on CPU (2026-10-06, `logs/fultonmarket/SMOKE_V2.54695113.out`).** The
same frozen production code on a small water box, with membrane barostat and
3 sub-sims, passed every canary check: each replica starts exactly (0.0 nm)
where its state ended, replicas start from different structures (1.6 nm RMS),
x = y with a varying x/z ratio, and resume works. The GPU canary repeats this
on the real LSD system.

**If the canary fails, no production starts**: the launchers sit in
`DependencyNeverSatisfied`. Read `logs/fultonmarket/CANARY_V2.*.out`, fix the
problem, then resubmit the canary and launchers.

## Monitoring

    squeue -u fcetin -o "%.10i %.28j %.3t %.10M %R"
    for d in /expanse/lustre/projects/iit127/fcetin/5ht2b/v2/fultonmarket/*/; do
      echo "$(basename $d): $(ls $d/saved_variables 2>/dev/null | wc -l)/60"; done
    tail -3 logs/fultonmarket/FM_V2.<jobid>.*.out       # attempts, successors, JOB END

Each window logs `=== walltime near ... successor <jobid> (gen N) ===` before
its 48 h TIMEOUT. TIMEOUT is normal; read sub-sim counts, not SLURM states.

## When something stops

- **A system has < 60 and no job in `squeue`.** The chain broke (setup error,
  `FM_MAX_GEN` cap, or a failed sbatch at T-15 min). Read the last FM_V2 log,
  then `bash scripts/submit_fm_v2.sh <name>`. Never run two jobs on one system.
- **An equilibration failed.** Resubmit `sbatch -J EQUIL_V2_<name> scripts/EQUIL_V2.job <name>`.
  MotorRow skips completed steps. Then `bash scripts/submit_fm_v2.sh <name>`
  (its launcher will not fire).
- **"Project balance is not enough"** is GPU-reservation pressure, not a missing
  allocation. See the handoff doc. Wait ~30 min after cancels and retry.
- **NaNs** are handled by the retry loop inside each window. Do NOT lower dt.

## Convergence

`chimpss-convergence <fultonmarket_dir> <step_5.pdb>` (or `scripts/CONVERGENCE.job`)
writes `convergence.json` and a per-checkpoint table. A system is converged
when, for both protein conformation and ligand pose, there is no detectable
drift between the first and second halves of the run (p ≥ 0.05 against a
time-balanced block-split null), each half holds ≥ 25 independent and ≥ 50
effective state-0 samples. The method and its tests are in
`chimpss/fultonmarket/convergence.py`. Old v1 LY266097 scored 3–4
independent samples per half: clearly not converged.

## Code snapshot policy

Production imports only `v2/code/` (ChiMPSS `0a3c324`), never the live repo, so
working-tree edits cannot touch a running campaign. Commits since then change
no production-path file (checked). To change production code: commit, refresh
the snapshot **between** windows, and record the change here.
