# FultonMarket (stage 3) — status & handoff

_Last updated: 2026-10-06. Protein: 5-HT2B. 9 biased-agonism ligand systems._

> **TIMEOUT after 48 h is the NORMAL, HEALTHY end state of these jobs.** It is
> not a failure and it is not something to fix. A 1500 ns target takes ~4 × 48 h
> windows per system; every window that is not cut short by a NaN necessarily
> ends by hitting the wall clock. Read `saved_variables/` counts, never `sacct`.
> The 2026-08-05 round ended with all 9 jobs in TIMEOUT and was the single most
> productive round of the campaign (+129 sub-sims).

## ▶ 2026-10-06: v2 campaign launched — runbook in `v2/README.md`

The v1 data below (`fultonmarket_hmr/`) is not valid REMD. All nine systems were
rebuilt with protonated ligands and zero net charge, are equilibrating, and will
start production automatically, with a membrane barostat, the replica-reset fix
and self-resubmitting jobs. All of it lives under `v2/`; `v2/README.md` is the
runbook (job graph, monitoring, failure handling, decisions for the PI). The
sections below are kept for history.

## 🚨 2026-10-05: the production data is NOT valid REMD — replica-reset bug

**Production finished 2026-09-27: 540/540, all nine systems.** But every
sub-simulation of every system was run with a bug that restarted **all replicas
from one configuration** at the start of each sub-simulation. The data cannot
be repaired after the fact.

**The bug.** `src/chimpss/fultonmarket/randolph.py` passed
`sampler_states=self.sampler_states[0]` to `ParallelTemperingSampler.create`.
openmmtools copies a single SamplerState to every replica, so the per-replica
positions, velocities and boxes that `FultonMarket._load_initial_args` loads
and reorders were thrown away. The original FultonMarket
(`/home/fcetin/FultonMarket/FultonMarket/Randolph.py:198`) passed the full list.
The `[0]` came in with the ChiMPSS "Phase 4" port (4475d89, 2026-04-22), before
this campaign started. **Fixed in 24e6d44**, with a regression test
(`tests/unit/test_fultonmarket/test_sampler_states.py`) that fails on the old
code and passes on the fix.

**Evidence (protein Cα, Kabsch-aligned):**
- Frame 0 of every saved segment is identical across all replicas (≤ 0.002 nm,
  float noise). Checked in all nine systems, segments 1/20/40/59, plus lisuride
  1/5/14/16/30/59 (legacy cadence too).
- That common structure is exactly (0.0000 nm) the previous segment's final
  **300 K** configuration (methysergide seg 17→18: replica 146, state 0).
- 17.5 ps later every replica is 0.07–0.10 nm from it, the same as one
  in-segment step (0.08–0.12 nm). All sampled replicas trace back to the
  same single parent.
- By the end of a sub-sim, replicas have spread only 0.09–0.23 nm from the
  shared start.

**Consequence.** No replica ever ran more than one sub-simulation (~140–175 ps)
before being reset to the 300 K structure. The temperature ladder therefore
never performed REMD. Each system is effectively one chain of 60 restarts,
~8–10 ns in total, with short heated copies around it. High-T replicas are
300 K structures heated for ≤ 175 ps, not equilibrium high-T samples, so the
swaps also feed non-equilibrium configurations back down. The healthy-looking
acceptance (~0.7) and round-trip counts say nothing about sampling under these
conditions.

**What this means for the earlier sections.** "Exchange mixing is healthy",
"sampling health" and the 1500 ns framing should all be read in light of this.
The 2026-09-26 build issues (net charge, neutral ligands, isotropic barostat,
aggregate-vs-per-replica time) are still open too, so any rerun should settle
them in the same pass. **PI decision; nothing has been rerun.**

**Rough rerun cost (estimate).** 175 min per sub-sim on one V100 (measured,
~101k atoms) × 60 ≈ 175 GPU-h per system, plus about one 48 h window for the
sim-0 ladder build ⇒ roughly 2,000 GPU-h for nine systems, about 8–10 days of
wall time with nine GPUs kept busy. The allocation (TG-BIO250326, ~20,000 SU
left, expires 2027-05-01) covers that several times over. It scales up if the
per-replica time is increased.

### Convergence analysis — further problems found 2026-10-05

1. **The Frobenius check cannot fail.** It compares two 1000×1000 matrices of
   distances between randomly resampled frames, entry by entry, and row *i* in
   one checkpoint has no relation to row *i* in another. It is also normalized by
   n(n−1)/2 instead of √. Measured on methysergide: the SAME ensemble merely
   reordered scores 1.3e-4 (α-C), *more* than vs the previous checkpoint
   (1.2e-4) or vs checkpoint 1 (1.0e-4), all ~400× below the 0.05 threshold.
2. **The JSD check measures noise.** Adjacent checkpoints differ by 0.47–0.56
   (threshold 0.05), and the value vs an early checkpoint was *lower* than vs
   the previous one. It compares distributions of within-ensemble pair
   distances, so it is blind to the ensemble moving to a different basin.
3. **MBAR does not converge on this data.** It reports "No solution found" in
   22/23–27/34 sub-sims per system, and the "written" matrices mostly came from
   unconverged weights (near-uniform, i.e. pooled across temperatures). Not a
   JAX/float32 problem: plain pymbar 4.2.0 in float64 also fails (residual
   9.8e4, weights not normalized). Some failures also crash resampling with
   `probabilities do not sum to 1`. **Root cause: initialisation**, since the
   f_k span ~1e5 kT. **Fixed in 7f01f18**: `initialize='mean-reduced-potential'`
   converges on the same data in 4 s (residual 5e-11 kT, n_eff(300 K) 1132), and
   any solution that fails the MBAR self-consistency check now raises instead of
   being used.
4. `RUN_RETRO_ANALYSIS.py` only caches matrices; nothing ever called
   `retro_convergence_report`, the function that produces verdicts, and
   `skip_contacts` was not threaded into it (fixed 7fe532b).

**Net:** the convergence criterion needs redesigning before it means anything,
and it should be run on corrected data, not this data. The 2026-09-26 retro
caches were produced from both the reset-bug data and unconverged MBAR. Treat
them as invalid.

## 🔔 2026-09-26 session — status, one correction, two fixed blockers

**Production (as of 09-26): 525/540. All 540 completed 2026-09-27; see the 10-05 section above.**

| System | sub-sims | state |
|---|---:|---|
| lisuride | 60/60 | ✅ complete |
| lisuride_L362F_mutseq | 60/60 | ✅ complete |
| LY266097 | 60/60 | ✅ complete |
| methylergonovine | 60/60 | ✅ complete |
| methylergonovine_T140A_mutseq | 60/60 | ✅ complete |
| methysergide | 60/60 | ✅ complete |
| LSD_L362F_mutseq | 57/60 | running 54460273 |
| methysergide_A225G_mutseq | 57/60 | running 54460274 |
| LSD | 51/60 | running 54460270 → 54460271 |

Only **15 sub-sims** remain. All three were submitted 2026-09-26 and verified
one-job-per-system via `sacct --format=SubmitLine`. LSD is the long pole (9
segments ≈ 27 h) and carries a 2-deep `afterany` chain.

### ✅ CORRECTION: the allocation did NOT expire. There is no time pressure.

The section below titled "Allocation clock" says the allocation expires
**2026-08-28** and that finishing all nine systems is unlikely. **That is
obsolete.** `expanse-client project iit127` on 2026-09-26:

| | |
|---|---|
| TG project | **TG-BIO250326** |
| Total allocation | **27,091** |
| Total spent | 7,091 (fcetin 4,684) |
| Expiration | **May 1, 2027** |

~20,000 SU remain and there are eight months left. Do **not** make
scope-reduction decisions (driving a subset to 60/60 rather than all nine) on
the basis of the old expiry date — the premise is gone.

### ⚠️ The idle-time failure mode recurred, and worse: a 41-day gap

The last production job ended **2026-08-16**; nothing ran until **2026-09-26**.
That is a **41-day** gap with zero jobs, against the 8.30-day gap that the
"real bottleneck is IDLE TIME" section below already identified as the single
largest loss in the campaign. 15 sub-sims ≈ 2 days of GPU work were outstanding
that whole time.

This is the one thing that actually governs the finish date. The chains work
(the 08-08 → 08-10 round hit a 97 % duty cycle); what fails is that nobody
resubmits after the last chain link drains. **Check `squeue -u fcetin` whenever
you touch this campaign, and resubmit anything < 60/60 with no active job.**

### 🐛 Convergence analysis: the two real blockers, both now fixed

Publication item 1 below ("Convergence was NEVER assessed") reported the code
as fixed and the analysis as running. It was neither finished nor correct: the
2026-08-14 jobs (53444277/53444279) both exited 0 having produced **nothing
usable** — lisuride `0 / 44 processed`, lisuride_L362F_mutseq `31 / 60`, and
**not one STOP verdict anywhere**. Two independent bugs, both now fixed.

**(1) `skip=10` discarded 91–100 % of every sub-simulation.** `skip` is a
per-sub-simulation *frame* discard, and its default was calibrated for the
legacy 1 ps exchange cadence (~150 frames/segment). Measured frame counts at
the production cadence:

| System | frames/segment |
|---|---:|
| LSD, LSD_L362F_mutseq, methysergide | **9** |
| methylergonovine, methylergonovine_T140A, methysergide_A225G, lisuride 16–59 | **10** |
| LY266097, lisuride_L362F_mutseq | **11** |
| lisuride 0–15 (legacy cadence) | 147–152 |

This is by design: `sim_length=25` ns is AGGREGATE over all replicas
(`randolph.py:134`, `sim_time_per_rep = sim_time / n_replicates`), so each
replica runs only 8–10 iterations of 17.5 ps per sub-sim, and a frame is saved
every iteration (+1 initial frame). But `skip=10` therefore **empties** every
segment of seven systems and leaves **one frame per segment** for the two with
11. That is the whole story of the two failure modes seen on 08-14:

- Seven systems / lisuride 16–59 → `shape[0] == 0` → the run died instantly.
- The two 11-frame systems → MBAR got 1 frame per state and returned **exactly
  uniform weights** (1/1050, 1/1650 to the last digit, std 3.9e-14). The
  reweighting to 300 K was a silent no-op, so the 62 matrices it cached were
  meaningless — and the cache-hit test would have reused them forever.

Verified by rerun: with `skip=0` and 30 segments of data MBAR returns properly
non-uniform weights (max 1.9e-3 against a uniform 2.0e-5, many underflowing to
0 as high-T configurations should). Fixed in ChiMPSS **bb2d147**, which also
validates `skip` up-front and names the cadence as the cause.

⚠️ **Use `--skip 0`** (now the `RUN_RETRO_ANALYSIS.py` default). Let
`determine_equilibration()`/`t0` do the discard statistically; a constant frame
count is cadence-dependent and silently wrong.

**(1a) The error message told you to delete good data.** The old assertion read
`"<dir> has no frames — delete the directory and resume."` The data on disk was
perfectly intact; following that advice would have destroyed **44 completed
lisuride segments**. Now it reports the real on-disk frame count and says
explicitly not to delete. If you see that old wording, you are on pre-bb2d147 code.

**(2) pymbar's BAR initialiser raises `NameError`.** `MBAR(..., initialize='BAR')`
is a hard failure whenever a state pair lacks overlap: `_initialize_with_bar`
wraps `bar()` in `except ConvergenceError:`, but `pymbar/mbar.py` never imports
`ConvergenceError` (it is in `pymbar/utils.py:407`). `bar()` legitimately raises
`BoundsError`/`ConvergenceError` on poor overlap, and evaluating the except
clause then dies with `NameError: name 'ConvergenceError' is not defined`. This
is what killed 29 of L362F's 60 sub-sims. Overlap is necessarily poor at low
`sim_no`, so it is not rare. Fixed in **c899510** by falling back to the default
zeros initialisation — same converged weights, more solver iterations — rather
than patching site-packages.

**Not the cause (tested, ruled out):** the reduced energies are ~5e5 in float32,
which looks like it should starve MBAR of precision. It does not — float32,
float64 and per-sample recentring give byte-identical weights. Don't spend time
there.

**Invalid cache quarantined, not deleted:**

    fultonmarket_hmr/lisuride_L362F_mutseq/retro_cache.INVALID_skip10_20260926/   (124 .npy)
    fultonmarket_hmr/lisuride/retro_cache.INVALID_skip10_20260926/                (empty)

Retro jobs rerunning all six complete systems: **54460387–54460392**
(`sbatch RUN_RETRO_ANALYSIS.job <system> --skip 0`; lisuride additionally
`--sim_nos 16..59` to stay cadence-clean). The three in-production systems still
need theirs once they reach 60/60. **Convergence remains formally unassessed
until those produce STOP verdicts** — do not describe it as done before then.

### ❓ Open scientific question for the PI: every ligand is simulated NEUTRAL

Found 2026-09-26; nothing has been changed. All 13 `prep/json/*.json` ligand
SMILES carry no formal charge, and the built systems match: the `UNK` hydrogen
count in `equil_hmr/<name>/step_5.pdb` equals the **neutral** formula for every
ligand checked (LSD 25 H, lisuride 26, methylergonovine 25, methysergide 27,
LY266097 23; the +1 species would each have one more).

Why it may matter: the ergoline N6 tertiary amine (LSD pKa ≈ 7.8) and
LY266097's secondary amine are conventionally modelled **protonated** in
aminergic GPCRs, because the protonated amine forms the conserved salt bridge
with Asp3.32 (**D135** in 5-HT2B). A neutral amine can at best H-bond there. In
the equilibrated starting structures, the nearest ligand N sits 2.8-4.2 Å from
the D135 carboxylate, >4 Å for LSD and methylergonovine. That is suggestive,
not conclusive: it is one frame per system, and atom identity was not mapped
to N6.

This may have been deliberate. If it was not, it affects all nine systems
equally and the fix starts at Bridgeport, so it is a PI decision, not an agent
decision. The earlier "ligand identity confirmed by formula" check establishes
the right compounds, not the right protonation states.

### ✅ Independently re-verified this session

- **All 525 segments intact**: every one has all 6 `.npy` files, non-empty, and
  `0..N-1` contiguous per system. No corruption from 41 days of timeouts/cancels.
- **Exchange acceptance is healthy, but temperature-space mixing is incomplete**
  (from `states.npy`, current full data): replicas span 98–99 % of the ladder
  and 75–97 % reach both extremes, but full bottom→top→bottom round trips are
  only **0.3–0.9 per replica** (lisuride 1.3, helped by its legacy 1 ps
  segments). Most replicas have not completed one round trip. See the
  per-replica sampling-time finding below; the two are the same limitation.


### 🔎 Full stage 1–3 audit (2026-09-26): what is sound and what is not

**Sound, verified directly for all nine systems (not taken from earlier notes):**
- HMR: solute H raised (2.52 / 2.02 Da), heavy atoms lowered, total mass
  conserved, water H untouched, ligand H included; constraint counts match the
  plain systems; PDB and system atom counts match.
- No leftover restraint forces in the production systems; PME, 1.0 nm cutoff.
- MotorRow: zero NaN lines in steps 1–5; final T ≈ 300 K; step-5 box-volume
  drift < 0.5 %.
- Structures: no chain breaks; both disulfides (C128–C207, C350–C353) formed at
  ~2.0 Å; mutations present and WT residues correct at 140/225/362.

**Not sound or not ideal (PI decisions; nothing has been changed):**

1. **"1500 ns" is aggregate across replicas. Each replica runs only 8–10 ns.**
   `sim_time_per_rep = sim_time / n_replicates`. The 68 → 150–188 ladder
   growth divided a fixed budget among ~2.5× more replicas. Per replica over 60
   sub-sims: LSD / LSD_L362F / methysergide ≈ 8.4 ns; methylergonovine /
   T140A / A225G ≈ 9.5 ns; LY266097 / lisuride_L362F ≈ 10.5 ns. The 300 K
   ensemble is assembled from that. Report it as aggregate, and judge whether
   ~8–10 ns per replica is enough for the claims being made.
2. **Every system carries a net charge of +7 to +9 e.** Na⁺ and Cl⁻ counts are
   exactly equal in all nine systems, so salt was added but no counterions were
   added to neutralize the protein. PME silently applies a uniform neutralizing
   background, which is a known artefact source in heterogeneous (membrane)
   systems. Neutral ligands do not explain it; ligand charge is 0 in all nine.
3. **Production pressure coupling is isotropic** (`MonteCarloBarostat` via
   `ThermodynamicState(pressure=1 bar)`, and MotorRow step 5 switches to it
   from the membrane barostat). The box x/z ratio is fixed to 1e-7 in production,
   so the bilayer cannot adjust area and thickness independently, including in
   the 367 K replicas that exchange down to 300 K.
4. **Ligands are neutral** (see the protonation section above).
5. **Constructs differ across ligands at the N-terminus**: residues start at 40
   (methysergide), 42 (LSD), 44 (methylergonovine), 48 (lisuride), 49
   (LY266097), all ending at 400. WT/mutant pairs match each other, so
   mutant-vs-WT comparisons are clean; cross-ligand comparisons carry a
   construct difference too.

## Where everything lives (Expanse lustre)

    PROJ=/expanse/lustre/projects/iit127/fcetin/5ht2b
    $PROJ/work_dir/systems_hmr/<name>_FG_HMR.xml   # Bridgeport HMR system  (stage 1) ✅ all 9
    $PROJ/equil_hmr/<name>/step_5.{pdb,xml}        # MotorRow final equil   (stage 2) ✅ all 9
    $PROJ/fultonmarket_hmr/<name>/                 # FultonMarket output    (stage 3) ← in progress
    $PROJ/logs/fultonmarket_hmr/FM_HMR.<jobid>.*.out

The 9 systems:
lisuride, methylergonovine, methysergide, LSD, LY266097,
lisuride_L362F_mutseq, methylergonovine_T140A_mutseq,
methysergide_A225G_mutseq, LSD_L362F_mutseq.

## Verifying the systems are physically correct (esp. HMR)

The production runs integrate at **dt = 3.5 fs**, which is only valid if (a) the
`_FG_HMR.xml` systems have hydrogen-mass repartitioning (HMR) applied to the
SOLUTE and (b) X–H bonds are constrained. Neither is asserted by the pipeline,
so verify it explicitly. Run with the chimpss env python
(`/home/fcetin/miniconda3/envs/chimpss/bin/python`):

    import openmm as mm, numpy as np
    from openmm import unit
    P="/expanse/lustre/projects/iit127/fcetin/5ht2b"
    def masses(x):
        s=mm.XmlSerializer.deserialize(open(x).read())
        return s, np.array([s.getParticleMass(i).value_in_unit(unit.dalton)
                            for i in range(s.getNumParticles())])
    s,h = masses(f"{P}/work_dir/systems_hmr/lisuride_FG_HMR.xml")   # HMR
    _,p = masses(f"{P}/work_dir/systems/lisuride.xml")              # plain (pre-HMR)
    d=h-p

PASS criteria (observed baseline for lisuride, 2026-06-30):
- **HMR really applied:** `(d>1e-6).sum()` solute H raised, `(d<-1e-6).sum()`
  heavy atoms lowered. Baseline: 13,188 H raised, 6,865 heavy lowered.
- **Mass conserved:** `abs(h.sum()-p.sum()) < 1e-3` → True (repartition, not
  mass added). Baseline total 592,065 Da.
- **Water untouched:** median raised-H mass reflects SOLUTE only; the ~50k water
  H stay at 1.008 (rigid water). Baseline raised-H target = **2.52 Da**
  (11,723 at 2.52, 1,465 at 2.02).
- **Constraints present:** `s.getNumConstraints()` ≫ 0 (X–H + water). Baseline
  75,416.
- **pdb ↔ system atom counts match** for every system (else FM mis-maps
  coordinates). Verified True for all 9; sizes 94,383–131,146.
- **Each system has membrane + ligand + ions:** step_5.pdb HETATM resnames
  include `POP` (POPC), `NA`/`CL`, and one `UNK` (~50-atom ligand).
- **MotorRow equilibrated cleanly:** `equil_hmr/<name>/step_5.log` has zero
  `nan` lines, final Potential Energy ≈ −1.37M…−1.9M kJ/mol (scales with size),
  Temperature ≈ 300 K, stable box volume.

### ⚠️ dt = 3.5 fs is FIXED — do not "optimize" it

**3.5 fs is a directive from the PI, not a tunable parameter.** Do not lower it
to 3.0 fs or otherwise change it to chase NaN stability. Any future agent that
sees NaNs and reasons "reduce the timestep" is undoing a deliberate decision.

Related design decision (also deliberate): the equilibration was run with the
**customizable ChiMPSS MotorRow**, specifically so the 3.5 fs / HMR setup could
be produced in ONE pass — rather than running the OLD MotorRow twice. This is
why `equil_hmr/` exists alongside the older `equil/` tree; `equil_hmr/` is the
authoritative input to FultonMarket. Do not "re-equilibrate with the old
MotorRow" to reconcile them.

KNOWN CAVEAT (observation, not an action item): the HMR hydrogen target is
**2.52 Da**, lighter than the 3–4 Da often paired with 3.5 fs. It is stable
because X–H bonds are constrained. If NaN frequency ever needs reducing, the
lever is the HMR mass in Bridgeport (`hydrogenMass=3–4*amu`) — **not** the
timestep.

### Bridgeport inputs verified (mutations + ligand identity) — 2026-06-30

**Mutations ✅ correct at BOTH levels.** Mutants are built by feeding a mutated
Modeller PIR to `RepairProtein` (`RepairProtein.fasta_path` in
`prep/json/<name>.json`; `engineered_resids: null`) — the mutation is NOT applied
as a post-hoc edit, so the sequence file is the source of truth.

- Sequence level: `prep/fasta/5HT2B_{L362F,T140A,A225G}.fasta` each differ from
  `5HT2B.fasta` by **exactly one substitution at the intended position**.
  Sequence is 481 aa = native UniProt P41595 numbering (Met1), so 362/140/225
  are real residue numbers.
- Structure level (what is actually simulated): `equil_hmr/<name>/step_5.pdb`
  contains `PHE362` / `ALA140` / `GLY225` in the mutants, and `LEU362` /
  `THR140` / `ALA225` in the matched WT systems.

NOTE when re-checking: these are Modeller **PIR** files, not plain FASTA. Skip
the `>P1;NAME` line AND the `sequence; NAME:::::::::` description line, and strip
the trailing `*`. Naively treating the description line as sequence shifts the
frame and produces hundreds of bogus diffs.

**Ligand identity ✅ confirmed.** Ligands are all named `UNK` in the built PDBs,
so identity must be traced via `prep/json/<name>.json` → `Ligand.smiles`.
Verified with rdkit (available in the **`prep1`** env, NOT `chimpss`): every
SMILES atom count equals the `UNK` atom count in `step_5.pdb`, and the formulas
are the correct ones for each compound —

| formula | systems |
|---|---|
| C20H26N4O | lisuride, lisuride_L362F_mutseq |
| C20H25N3O2 | methylergonovine, methylergonovine_T140A_mutseq |
| C21H27N3O2 | methysergide, methysergide_A225G_mutseq |
| C20H25N3O | LSD, LSD_L362F_mutseq |
| C21H23ClN2O2 | LY266097 |

5 distinct ligands across 9 systems (4 WT/mutant pairs + LY266097), exactly as
the study design implies.

## REMD sampling health (lisuride, sims 0–12 ≈ 325 ns) — verified 2026-06-30

These measurements describe the legacy 1 ps exchange cadence used for saved
segments 0–15. Starting with segment 16, production uses the corrected 17.5 ps
cadence (5,000 MD steps at 3.5 fs). The completed equilibrium samples are
retained, but analyses should record the cadence transition explicitly.

A REMD run can be numerically healthy yet sample poorly. Measured, it is mixing well:

| Metric | Value | Read |
|---|---|---|
| Neighbour-swap acceptance | mean **0.707**, median 0.723, min **0.501** | Excellent |
| Pairs below the code's 0.40 threshold | **0 / 171** | None starved |
| Replicas visiting both ladder extremes | **153/172 (89 %)** | Good traversal |
| Replicas with ≥1 full round trip (bottom→top→bottom) | **136/172** | Real temperature random-walk |
| Mean per-replica state range | 165 of 171 | Near-full ladder |

This also explains the 68 → **172** ladder expansion: FultonMarket inserts states
until acceptance clears `init_overlap_thresh`, so the large ladder is the
mechanism that produced the healthy acceptance — working as designed, not a bug.

### How to reproduce (two independent routes)

Cheap route — state trajectories, no need to touch the 28 GB ncdf:

    S=np.concatenate([np.load(f"{OUT}/saved_variables/{n}/states.npy")
                      for n in range(N_DONE)],axis=0)   # (frames, n_replicas)
    # ladder traversal / round trips from S

Authoritative route — reporter mixing statistics:

    from openmmtools.multistate import MultiStateReporter
    r=MultiStateReporter(f"{OUT}/output.ncdf", open_mode='r')
    acc,prop=r.read_mixing_statistics(); r.close()

⚠️ **GOTCHA:** `read_mixing_statistics()` returns **3-D, per-iteration** arrays of
shape `(n_iterations, n_states, n_states)` — NOT a single 2-D matrix. You must
`np.nansum(..., axis=0)` over the iteration axis BEFORE taking `rate[i,i+1]`.
Indexing the 3-D array directly silently yields nonsense (it slices iterations,
not neighbour pairs) and produced a bogus "mean acceptance 0.041" on first
attempt. Sanity check: neighbour pairs must equal `n_states - 1` (171).

## How jobs are launched

    bash /home/fcetin/FultonMarket/submit_5ht2b_fultonmarket_hmr.sh            # all 9
    bash /home/fcetin/FultonMarket/submit_5ht2b_fultonmarket_hmr.sh LSD ...    # subset
      -> sbatch /home/fcetin/FultonMarket/RUN_FULTONMARKET.job <pdb> <sysxml> <outdir> --input_state ... --timestep 3.5 --T_min 300 --T_max 367 --n_replicates 68 --iter_length 0.0175 --sim_length 25 --total_sim_time 1500

Exchange attempts use 5,000 MD steps per iteration: 10 ps at the standard
2 fs timestep, scaled to **17.5 ps (0.0175 ns)** for the 3.5 fs HMR timestep.

The job runs ChiMPSS `RUN_FULTONMARKET.py` from `/home/fcetin/ChiMPSS` with
`PYTHONPATH=/home/fcetin/ChiMPSS/src`. The resume + exact time-only-stop
behaviour and corrected iteration defaults live on branch
**`feature/fm-resume-and-time-stop`**, currently at commit **15cf587**. Keep that
branch checked out (or merge it) or the jobs lose these campaign fixes.

## Inspecting past and currently-running jobs

### 🚨 SLURM state is NOT a correctness signal — read this first

- `State=COMPLETED`, `ExitCode 0:0` **can mean zero science was done.** Job
  50926204 "COMPLETED" in 14 s because pre-patch code hit
  `Trajectory already exists — skipping` and returned immediately.
- `State=FAILED`, `ExitCode 1:0` **is the normal, expected outcome of a NaN**,
  and all sampling completed before the crash is valid and saved.
- **The job emails are worse than useless for judging progress.** The job
  scripts set `--mail-type=END,FAIL`, and SLURM sends "END" for *every*
  termination — including `scancel`. On 2026-08-08 a 16-email burst arrived that
  read like a wave of finished jobs; it was 15 cancellations (mine, reverting
  the preempt experiment) plus exactly **one** real completion. Judge progress
  only by:

      for d in $PROJ/fultonmarket_hmr/*/; do
        echo "$(basename $d): $(ls $d/saved_variables 2>/dev/null | wc -l)/60"; done

**Therefore: judge progress ONLY by the `saved_variables/` count (→ 60) and the
log tail — never by `sacct` state.**

### Commands

    # currently queued/running (all jobs share the name FM_HMR)
    squeue -u fcetin -o "%.12i %.10P %.10j %.3t %.11M %R"

    # which SYSTEM each job owns — use sacct SubmitLine (works for PENDING too)
    sacct -u fcetin --starttime=today --name=FM_HMR --format=JobID%14,SubmitLine%400 -X --noheader \
    | while read -r jid rest; do
        sys=$(echo "$rest" | tr ' ' '\n' | grep -E '/fultonmarket_hmr/[A-Za-z0-9_]+$' | head -1 | awk -F/ '{print $NF}')
        echo "$jid -> $sys"
      done

    # ⚠️ do NOT use `scontrol show job <jobid> -dd | grep -o '/fultonmarket_hmr/[^ ]*'`
    # (as this doc previously advised). scontrol does not surface the script args
    # here — Command= is just the .job path — and the grep instead matches the
    # StdOut LOG path, which contains /fultonmarket_hmr/ too. It silently returns
    # the wrong thing. `sacct --format=SubmitLine` is the reliable route.

    # job history
    sacct -u fcetin --starttime=now-60days --name=FM_HMR \
      --format=JobID%14,JobName%10,State%20,Start%16,Elapsed%11,ExitCode%8,NodeList%12

    # map every past log -> system + failure signal
    cd $PROJ/logs/fultonmarket_hmr
    for f in FM_HMR.*.out; do
      jid=$(echo "$f" | cut -d. -f2)
      sys=$(grep -m1 -o 'fultonmarket_hmr/[A-Za-z0-9_]*' "$f" | head -1 | cut -d/ -f2)
      sig=$(grep -m1 -oE "Trajectory already exists[^.]*|SimulationNaNError|ModuleNotFoundError|getcontacts_script must be provided|Traceback" "$f" | head -1)
      echo "job=$jid system=${sys:-?} signal=${sig:-none}"
    done

### Forensics of all past FM_HMR jobs (all were lisuride) — every one explained

| Job | When | Elapsed | SLURM state | Root cause | Status |
|---|---|---|---|---|---|
| 50839679 | 06-15 | 6 s | FAILED | `ModuleNotFoundError: No module named 'jax'` — `conda activate` didn't take in non-interactive SLURM | **Fixed** — job script now calls the env's python by absolute path |
| 50839682 | 06-15 | 15 h | FAILED | `RuntimeError: getcontacts_script must be provided` at the convergence check | **Fixed** by the *time-only stop* half of commit 1b60417 |
| 50926204 | 06-16 | 14 s | **COMPLETED (0:0)** | `Trajectory already exists — skipping`; did **zero** work | **Fixed** by the *resume* half of commit 1b60417 |
| 50926212 | 06-16 | 24.7 h | FAILED | NaN, replica 114 / state 164 → produced sub-sims 0–5 | Expected; resumable |
| 51682492 | 06-29 | 27.1 h | FAILED | NaN, replica 49 / state 144; resumed 6→13 first | Expected; resumable |
| 52404087 | 07-23 | 11.8 h | CANCELLED | Submitted with obsolete `--iter_length 0.001`; produced completed segments 13–15 before audit caught it | Intentionally stopped; no partial segment 16 saved |
| 52425624 | 07-24 | 32.4 h | FAILED | Corrected 17.5 ps canary; resumed at 16, verified 5,000 steps/iteration, saved segments 16–26, then NaN at replica 142 / state 146 | Protocol validated; resumable from 27 |

Note this history *retroactively validates* commit 1b60417: its two halves each
fix a failure that actually occurred (50839682 and 50926204), and the canary
(51682492) then demonstrated both working.

The stale 2026-06-30 batch is no longer active. Job 52404087 was deliberately
canceled before relaunching with the corrected iteration length. Corrected
lisuride canary 52425624 validated `--timestep 3.5 --iter_length 0.0175` and
completed eleven new segments before the expected stochastic NaN.

### Batch of 2026-07-26 (jobs 52527226–52527234) — all 9 hit the 48 h TIMEOUT

All nine ran the full 48 h and were killed by the wall clock, not by a crash.
Net yield 87/540 sub-sims. Yield was very uneven, and the reason matters:

| System | Job | Sub-sims after | Ladder rebuilds |
|---|---:|---:|---:|
| lisuride (resumed at 27) | 52527226 | **43**/60 | 17 |
| methylergonovine | 52527227 | 6/60 | 22 |
| methysergide | 52527228 | 1/60 | 25 |
| LSD | 52527229 | 3/60 | 19 |
| LY266097 | 52527230 | 9/60 | 24 |
| lisuride_L362F_mutseq | 52527231 | 11/60 | 25 |
| methylergonovine_T140A_mutseq | 52527232 | 7/60 | 21 |
| methysergide_A225G_mutseq | 52527233 | 6/60 | 21 |
| LSD_L362F_mutseq | 52527234 | 1/60 | 25 |

**Why the eight fresh systems were so slow — and why it will NOT recur.**
`Randolph._run_cycle` restarts the whole sub-simulation whenever any adjacent
pair falls under the acceptance threshold: it interpolates new states, resets
`current_cycle = 0`, deletes `output.ncdf`, and starts the segment over. On
`sim_no == 0` the threshold is the strict `init_overlap_thresh = 0.5`, so each
fresh system burned most of its 48 h churning the 68-replica ladder up to
170–183 replicas, throwing away the partial segment every time.

lisuride, resuming at sim 27 with an established 172-replica ladder, shows the
steady state instead: 17 rebuilds for 16 completed sub-sims ≈ one rebuild per
segment (the mandatory `_build_simulation` call), i.e. essentially no churn.

This cost is **one-time and now paid for all 9 systems.** `_load_initial_args`
reloads `temperatures.npy` from the last saved sub-sim, so every resume inherits
the expanded ladder and uses the looser `term_overlap_thresh = 0.35`. Do not
"fix" the ladder churn — it is the mechanism that produced the healthy 0.707
acceptance documented above.

**Measured steady-state cadence** (from `saved_variables/<N>/` mtimes): 2 h 42 m
– 3 h 45 m per sub-sim, ≈ 3 h average → **~16 sub-sims per 48 h job**.

### Batch of 2026-08-05 (jobs 53009125–53009133) — the retry loop paid off

Eight hit the 48 h TIMEOUT; 53009133 (LSD_L362F_mutseq) was still running into
2026-08-07. **+129 sub-sims, 87 → 216/540** — versus +87 for the whole campaign
before it, and this time evenly spread (+12 to +17 per system) because the sim-0
ladder cost was already paid.

| System | Job | Before → after |
|---|---:|---:|
| lisuride | 53009125 | 43 → **59** |
| methylergonovine | 53009126 | 6 → 20 |
| methysergide | 53009127 | 1 → 16 |
| LSD | 53009128 | 3 → 15 |
| LY266097 | 53009129 | 9 → 25 |
| lisuride_L362F_mutseq | 53009130 | 11 → 28 |
| methylergonovine_T140A_mutseq | 53009131 | 7 → 21 |
| methysergide_A225G_mutseq | 53009132 | 6 → 19 |
| LSD_L362F_mutseq | 53009133 | 1 → 13 (still running) |

The retry loop fired exactly once and exactly as designed: lisuride NaN'd after
13,740 s (3.8 h) at sub-sim 44, the loop caught it, resumed, and carried the
system to 59. Under the old single-call script that job would have died at hour
3.8 and thrown away ~44 h of its allocation. The other eight never NaN'd at all
(`attempt 0` only) and ran clean until the wall clock.

Note the logs have no closing `=== JOB END ===` line — SLURM kills the script
mid-attempt when the wall clock expires, so that line only appears when a job
finishes on its own terms. Its absence is normal, not a crash.

All 216 `saved_variables/<N>/` dirs verified: ≥5 files each, and contiguous
`0..N-1` for every system (no lost segments from the mid-write kills).

### Current batch submitted 2026-08-07

| System | Jobs | From |
|---|---|---:|
| lisuride | 53095261 | 59/60 — needs 1 segment, then done |
| methylergonovine | 53095262 → 53095263 | 20/60 |
| methysergide | 53095264 → 53095265 | 16/60 |
| LSD | 53095266 → 53095267 | 15/60 |
| LY266097 | 53095268 → 53095269 | 25/60 |
| lisuride_L362F_mutseq | 53095270 → 53095271 | 28/60 |
| methylergonovine_T140A_mutseq | 53095272 → 53095273 | 21/60 |
| methysergide_A225G_mutseq | 53095274 → 53095275 | 19/60 |
| LSD_L362F_mutseq | 53009133 (running) → 53095276 | 13/60 |

`→` denotes an `afterany` chain link. Verified: every chained job depends on its
own system's predecessor, so no system can ever have two concurrent jobs.

## 📄 Publication readiness — audited 2026-08-14

Three findings, in order of severity. Items 1 and 2 are the ones a reviewer will
ask about.

### 1. Convergence was NEVER assessed — code fixed 2026-08-14, analysis running

Every production run took the time-only stopping path. Evidence:

- **444** log lines `no getContacts → skipping convergence check`
- both completed systems ended `Max simulation time reached — skipping convergence checks`
- **zero** `resampled_*_matrix.npy` anywhere on disk

`--total_sim_time 1500` with no `getContacts_Info` and no `skip_contacts` hits an
early return in `_evaluate_stopping_criterion` before any Frobenius/JSD
comparison runs. **1500 ns is a spent budget, not a converged endpoint.**

**Root cause of why it could not simply be turned on:** getContacts is not
installed anywhere reachable. `RUN_RETRO_ANALYSIS.py` hardcoded
`/expanse/lustre/projects/uil133/josephdb/getcontacts/...` (another project —
permission denied) and a `pyinteraph2` conda env that does not exist here. The
live path honoured `skip_contacts=True`, but the **retro path did not** —
`compute_distance_matrices` always called `getContactDistanceMatrix`, raising
`getcontacts_script must be provided`. That is the same error that killed job
50839682 back in June.

**Fixes applied (ChiMPSS):**
- `retro_convergence.compute_distance_matrices(..., skip_contacts=False)` —
  returns torsion + alpha-carbon only when set.
- `retro_convergence.build_checks(..., skip_contacts=False)` — omits the two
  contact rows. **This one matters:** `_all_pass({})` returns `False`, so leaving
  the contact rows in would have pinned `STOP` to `False` forever and silently
  reported a converged run as unconverged. Verified both ways with a unit check.
- `print_sim_report` reports only matrices actually evaluated, so a skipped
  contact matrix is not printed as "no valid previous checkpoints".
- `analysis.retro_analyze_all(..., skip_contacts=False)` threads it through, and
  its cache-hit test now expects 2 matrices instead of a hardcoded 3.
- `RUN_RETRO_ANALYSIS.py` rewritten: fixed `os.path.isdir(None)` crash (the
  documented `<input_dir>/retro_cache/` default was never applied), removed the
  foreign hardcoded paths, added `--skip_contacts`, `--sim_nos`, `--sele_str`.

**Running now:** `RUN_RETRO_ANALYSIS.job` (jobs 53444277, 53444279). It runs on
the **`shared` CPU partition**, so it charges the CPU allocation and does *not*
compete with production REMD for GPU reservation budget.

    sbatch RUN_RETRO_ANALYSIS.job <system> [--sim_nos 16,17,...]

⚠️ **Do not add `skip_contacts=True` to the live production runs.** With it, the
convergence check becomes active and can return `True` — stopping a system early
(`minimum_fraction` allows it from ~15 sub-sims). That would give systems
different endpoints and wreck the comparison. Retro analysis is the safe route
because it cannot perturb a run.

### 2. lisuride's data is structurally inconsistent — decision needed

`states.npy` size per segment:

| lisuride | bytes |
|---|---|
| segs 0–12 | 104,704 |
| segs 13–15 | 101,264 |
| **segs 16–59** | **7,008** |
| lisuride_L362F_mutseq (all 60) | 6,728 — uniform |

A ~15× discontinuity at segment 16, matching the cadence change (53 logged
iterations at **285 steps/iter** = legacy 1 ps; 637 at **5000** = 17.5 ps). Only
lisuride is affected — every other system is uniform.

Segments 0–15 hold roughly **85 % of lisuride's frames** while being 27 % of its
segments, and they are the earliest, least-equilibrated ones. Naive
concatenation would weight the analysis overwhelmingly toward early sampling and
make lisuride incomparable to its own L362F partner.

Note the physics is not wrong — REMD with a shorter exchange interval still
samples the correct equilibrium distribution. The problems are frame-count
imbalance and differing statistical inefficiency, both of which are analysis
issues. Three options:

| Option | Cost | Result |
|---|---|---|
| **(a) drop segs 0–15** | free | lisuride = 1100 ns vs others' 1500 ns — unequal |
| **(b) extend to 76 segs** (`--total_sim_time 1900`) | ~16 sub-sims ≈ one 48 h GPU job | segs 16–75 = exactly 60 uniform segments — **full parity** |
| **(c) subsample segs 0–15** to the density of 16–59 | free | keeps 1500 ns, uniform frame density |

**(c) is recommended** — it costs no GPU and preserves the designed sampling.
**(b) is the cleanest** but competes for scarce GPU with the 7 unfinished
systems, so it should wait until those are done. This is a PI call; nothing has
been done to the data either way. The retro convergence run above uses
`--sim_nos 16..59` for lisuride so the convergence verdict is cadence-clean
regardless of which option is chosen for production analysis.

### 3. What is already sound

- **Exchange mixing is healthy in every system:** mean neighbour acceptance
  0.711–0.752, minimum 0.386, essentially nothing below the 0.40 threshold
  (measured over the last 400 reported pairs per system, 2026-08-14).
- dt = 3.5 fs and HMR applied uniformly; no system deviates.
- All 447 segments intact, ≥5 files, contiguous `0..N-1` — no corruption
  despite many timeouts and cancellations.
- Mutations verified at sequence *and* structure level; ligand identities
  confirmed by formula (2026-06-30).

## 📉 The real bottleneck is IDLE TIME, not compute — measured 2026-08-07

The per-sub-simulation cost is at hardware speed and has no fat in it. Measured
on LY266097 (157 replicas, 100,928 atoms), nine consecutive sub-sims:

| Phase | Time |
|---|---|
| setup / build | 0.5 min |
| cycle 0 (5 iterations) | 89.1 min |
| cycle 1 (5 iterations) + save + convergence | 85.5 min |
| **total per sub-sim** | **175.1 min ± 0.4** |

That is 748 MD steps/s aggregate = **1.34 ms/step for a 101k-atom system on one
V100** — normal for this size — and **>98 % of the wall clock is MD**. Saving the
~2.1 GB `positions.npy` and the MBAR/convergence step together cost under 1 %.
**There is nothing to optimise here. Do not go looking.**

The campaign is slow for a purely operational reason:

| Since production started 2026-07-23 | |
|---|---|
| calendar span | 15.02 days |
| calendar with ≥1 job running | 6.06 days (40 %) |
| **nothing running at all** | **8.95 days (60 %)** |
| aggregate GPU-days delivered | 37.5 |
| duty cycle vs. 9 GPUs busy | **28 %** |

One gap dominates: **2026-07-28 → 2026-08-05, 8.30 days with zero jobs running**
— the batch timed out and nothing was resubmitted for over a week. That single
gap exceeds half the campaign's elapsed time.

**Keeping the GPUs fed is worth ~3.5× more than any code optimisation.** With
continuous occupancy the remaining 324 sub-sims are ~7 days of compute.

    # duty-cycle audit (re-run any time)
    sacct -u fcetin -S <start> --name=FM_HMR --format=JobID,Start,End,State -X -P
    # union the [Start,End) intervals; compare against calendar span

## ❌ gpu-preempt was tried and REVERTED — it livelocks. Do not retry it.

Attempted 2026-08-07, abandoned 2026-08-08 after ~28 h. **All 8 preempt jobs
saved ZERO sub-simulations.** Per-job log forensics:

    job        script_starts  cycles_advanced  subsims_saved
    53095900         1               1               0
    53095901         1               2               0
    53095902         1               2               0
    53095903         1               2               0
    53095904         1               1               0
    53095905         2               3               0
    53095906         1               1               0
    53095907         1               2               0

**Why it can never work here:** progress is only checkpointed at sub-simulation
boundaries, and one sub-sim needs 2 cycles ≈ **175 min**. Preemption on this
partition arrives sooner than that, so every preemption discards all in-flight
work and the job restarts from the same `saved_variables` count. When the
preemption interval is shorter than the checkpoint interval, throughput is
**identically zero, forever** — the jobs would have churned indefinitely.

`--requeue` + the resume patch bound the loss to "one partial sub-sim", which
sounded cheap when the trade was made. The flaw in that reasoning: if *every*
attempt is cut short before its first checkpoint, "at most one partial sub-sim"
is also *all* the work, every time.

The only progress between 2026-08-07 and 08-08 (+3 sub-sims) came from the two
surviving `gpu-shared` jobs — lisuride reaching 60/60 and LSD_L362F's original
job finishing its window.

`RUN_FULTONMARKET_PREEMPT.job` and the `PREEMPT=1` switch are left in place, but
**do not use them** unless `sim_length` is first reduced enough that a sub-sim
completes well inside the observed preemption interval. That is a science-facing
change (it alters saved segment granularity and the analysis cadence) and needs
the PI's sign-off, not an agent's.

<details>
<summary>Original migration notes (superseded — kept for context)</summary>

## 🔀 Migrated to gpu-preempt (7-day windows) — 2026-08-07

The 48 h `gpu-shared` limit is what creates the resubmission treadmill, and the
treadmill is what creates the idle gaps. `gpu-preempt` allows **7-day** jobs at
the **same SU rate** (UsageFactor 1.0), which covers nearly every remaining
system in a single window.

`RUN_FULTONMARKET_PREEMPT.job` is `RUN_FULTONMARKET.job` with exactly three
lines changed — partition, `-t 7-00:00:00`, and `--open-mode=append` (so a
preemption-requeue appends to the log instead of truncating it). Same retry
loop, same REMD parameters. Select it with `PREEMPT=1`:

    PREEMPT=1 bash submit_5ht2b_fultonmarket_hmr.sh <name> ...

**Preemption is cheap here and that is the whole point.** `--requeue` is set,
SLURM re-queues a preempted job, and the resume patch restarts it from the last
completed sub-sim — so a preemption costs at most one partial sub-sim (~3 h) and
never a completed one. The trade is unpredictable restarts for the elimination
of the 48 h treadmill.

Current assignment (`gpu-preempt-normal`: MaxJobsPU 12, MaxSubmitPU 16):

| System | Job | Partition |
|---|---|---|
| lisuride | 53095261 | gpu-shared (needs 1 segment; guaranteed, not preemptible) |
| methylergonovine | 53095900 | gpu-preempt 7 d |
| methysergide | 53095901 | gpu-preempt 7 d |
| LSD | 53095902 | gpu-preempt 7 d |
| LY266097 | 53095903 | gpu-preempt 7 d |
| lisuride_L362F_mutseq | 53095904 | gpu-preempt 7 d |
| methylergonovine_T140A_mutseq | 53095905 | gpu-preempt 7 d |
| methysergide_A225G_mutseq | 53095906 | gpu-preempt 7 d |
| LSD_L362F_mutseq | 53009133 (running) → 53095907 | gpu-shared → gpu-preempt 7 d |

The 15 superseded `gpu-shared` jobs (53095262–53095276) were cancelled **before**
the preempt jobs were submitted — all were still PENDING, so no work was lost.
Cancel-before-submit is mandatory: if both sets were eligible at once, two jobs
could land on the same system and corrupt `output.ncdf`.

</details>

## ⚠️ "Project balance is not enough" = GPU RESERVATION pressure, not a real shortfall

This error is the single most confusing thing about running this campaign, and
it has now been misdiagnosed twice. Read this before reacting to it.

    sbatch: error: Project balance is not enough to run the job.
    sbatch: error: Batch job submission failed: Job violates accounting/QOS policy
                   (job submit limit, user's size and/or time limits)

**What it actually means:** SDSC's submit filter reserves each *queued* job's
worst-case GPU cost (walltime × GPUs) against the project balance. The sum of
those reservations — not SUs actually spent — is what trips the check. It is
GPU-specific, so CPU submissions keep working while every GPU submission is
refused, which makes it look like a vanished GPU allocation.

**The budget is NOT a fixed number — do not memorise one.** Observed ceilings:

| Date | Queued 48 h jobs accepted | GPU-hours reserved |
|---|---|---|
| 2026-08-07 | 17 (18th refused) | 816 |
| 2026-08-08 | 16 | 768 |
| 2026-08-10 | **8 (9th refused)** | **384** |

The ceiling halved between 08-08 and 08-10 with no allocation change (spend went
5,374 → 5,894 of 100,000). The formula lives in SDSC's Lua submit filter and is
not readable from Slurm — `sshare` exposes no `GrpTRESRunMins`, and the error
text is theirs, not Slurm's. It plausibly tracks fairshare or recent burn, but
**that is a guess and should not be relied on.**

**Therefore: probe, never predict.** Before submitting, bisect the headroom:

    for t in 48:00:00 08:00:00 02:00:00 01:00:00; do
      printf "%s: " "$t"
      sbatch --test-only -p gpu-shared --gpus=1 -N1 --ntasks-per-node=10 \
             --mem=90G -A iit127 -t $t --wrap="hostname" 2>&1 | head -1
    done

On 2026-08-10 this showed 1 h accepted and 2 h refused — i.e. with 8 × 48 h jobs
queued the headroom was ~1 GPU-hour, and **no third chain link could be added at
any depth.** Chain depth is therefore capped by whatever the filter allows *at
that moment*; deep chains are not always possible and the campaign may need a
manual round after each batch drains.

**Reservations take ~15–30 min to be released after `scancel`.** On 2026-08-08
the 8 preempt jobs were cancelled at 19:57; a GPU submission at 19:58 was still
refused, and the identical submission at 20:23 succeeded. Nothing about the
allocation changed in between.

### ✏️ Two corrections, so nobody re-derives this a third time

1. The 2026-08-07 note called this a "**~17 concurrent job ceiling**." Roughly
   right in effect, wrong in units — the budget is in **GPU-days of reservation
   (~34–36)**, not job count. 17 is just what that buys at 48 h each. Long jobs
   consume it far faster.
2. A 2026-08-08 note then claimed the "**GPU allocation is gone**", citing
   `expanse-client project -r expanse-gpu iit127` → *"Project not found"* and a
   passing CPU test. **That was wrong.** `expanse-gpu` is simply not how this
   project's GPU time is tracked, and the CPU/GPU asymmetry is explained by the
   reservations being GPU-specific. The allocation was healthy throughout
   (100,000 total, ~5,374 spent). No PI action, renewal, or cloud fallback was
   ever required. That diagnosis was made one minute after a mass `scancel`,
   before the reservations had cleared.

**Rule of thumb:** if GPU submission fails, check `squeue` first. If you just
cancelled jobs, wait ~30 min and retry *before* concluding anything about the
allocation. Verify with a real `sbatch`, not only `--test-only`.

    # is this reservation pressure? (sum queued GPU-days)
    squeue -u fcetin -h -p gpu-shared -o "%l" | awk -F- '{d=($1~/^[0-9]+$/&&NF>1)?$1:0; print d+1}' | paste -sd+ | bc

The QOS caps are real but have never been the binding constraint:
`gpu-shared-normal` MaxSubmitPU = 24, `gpu-preempt-normal` MaxSubmitPU = 16.

### Allocation clock — ❌ OBSOLETE, see the 2026-09-26 section at the top

> The expiry below is wrong. The project moved to **TG-BIO250326**: 27,091 SU,
> 7,091 spent, **expires 2027-05-01**. Nothing in this subsection should be
> used to justify cutting scope.

The `expanse` allocation expires **2026-08-28**. The remaining 321 sub-sims are
≥7 days of continuous 8-GPU compute at a perfect duty cycle, and the historical
duty cycle is 28 %. Finishing all nine systems before expiry is unlikely; if
time runs short, ask the PI whether to drive a subset to 60/60 rather than
advance all eight partially — a partial matrix is much less analysable.
CLAUDE.md names cloud A100s as the alternative execution target.

Remaining after this batch: 324 sub-sims. At ~14/window the long poles are LSD
and LSD_L362F_mutseq (45+ each) ≈ **3 more rounds**. The 2-deep chains cover two
of those unattended; expect to resubmit once more around 2026-08-11.

Disk: 603 GB at 87 sub-sims, ~2.1 GB per sub-sim steady state → expect **~1.5 TB**
at 60/60 across all nine. (lisuride is an outlier at 493 GB because its segments
0–15 were saved at the legacy 1 ps exchange cadence and hold ~17.5× more frames
than a modern 17.5 ps segment.)

## Target & progress

- Target: `total_sim_time = 1500 ns` per system = **60 sub-sims** of 25 ns each.
- Progress = number of `fultonmarket_hmr/<name>/saved_variables/<N>/` dirs.
- Check all systems:

      for d in $PROJ/fultonmarket_hmr/*/; do
        echo "$(basename $d): $(ls $d/saved_variables 2>/dev/null | wc -l)/60 sub-sims"; done

- As of 2026-08-10: **334/540 total** — **lisuride 60/60 ✅ COMPLETE**,
  lisuride_L362F_mutseq 45, LY266097 41, methylergonovine_T140A_mutseq 35,
  methylergonovine 34, methysergide_A225G_mutseq 32, methysergide 31,
  LSD_L362F_mutseq 29, LSD 27. **1 of 9 systems done.**

### The 2026-08-08 → 08-10 round: what a healthy round looks like

**+115 sub-sims in 49 h.** All 8 first-links started within ~1 h of submission,
ran the full 48 h to TIMEOUT, and saved 12–17 sub-sims each with **zero NaN
recoveries and zero fast-failures**. Duty cycle ≈ **97 %**, against the 28 %
campaign average — chaining is what closed the gap, and this is the benchmark to
compare future rounds against.

The 8 second-links (53106088/91/93/95/97/99/101/102) had their dependencies
satisfied automatically and are queued as of 08-10 21:42 — **the chain performed
the resubmission with no human involved, which was the entire point.**

A third link could NOT be added: the reservation ceiling was saturated by the 8
queued jobs (see the probe section below). Expect to submit round 3 manually
once these drain, around 2026-08-12/13.

  lisuride validated the full stopping path end-to-end: its last job exited with
  `python exited 0 after 10151s -- target reached`, i.e. the time-only-stop patch
  and the retry loop's clean-exit branch both work as intended.

## ⚠️ The one thing you must know: NaNs recur

REMD replicas blow up stochastically (~every 150–200 ns of aggregate sampling;
seen at replica/state 114/164 and 49/144 — different each time, so it is NOT a
fixed bad site or a setup error). When it happens, openmmtools retries 4× then
raises `SimulationNaNError`; **the Python process exits and the SLURM job ends.**
`--requeue` does NOT cover this (it only covers preemption/node failure).

Resuming is safe and clean: relaunching the SAME system picks up from the last
completed sub-sim (the resume patch). The pre-crash frame is saved under
`fultonmarket_hmr/<name>/nan-error-logs/` if you ever want to inspect it.

**So the remaining work is a resubmit loop until every system reaches 60/60.**

### Resubmit safely (never run two jobs on one system — it corrupts output.ncdf)

A system is safe to (re)submit only if it has NO active job. All jobs share the
name `FM_HMR`, so map job→system via its args before resubmitting:

    squeue -u fcetin -h -o "%i %t" | while read jid st; do
      scontrol show job "$jid" -dd | grep -o '/fultonmarket_hmr/[^ ]*' | head -1
    done      # -> shows which system each running/pending job owns

Then, for each system with `<60` sub-sims and no active job:

    bash /home/fcetin/FultonMarket/submit_5ht2b_fultonmarket_hmr.sh <name>

Better: chain several 48 h windows so the campaign advances without a resubmit
every two days. `afterany` guarantees the successor starts only after the
predecessor has ended, which enforces the one-job-per-system rule for free:

    CHAIN_DEPTH=2 bash .../submit_5ht2b_fultonmarket_hmr.sh <name> ...

To extend a system whose job is still running, hang the chain off it:

    AFTER_JOB=<running_jobid> CHAIN_DEPTH=2 bash .../submit_...sh <name>

Mind the ~17-job ceiling described below when choosing the depth.

## Robustness: self-healing retry loop — ✅ APPLIED 2026-08-05

`RUN_FULTONMARKET.job` no longer makes a single python call. It now wraps it in
a bounded retry loop so a NaN auto-recovers **inside** the 48 h allocation
instead of ending the job. Each retry resumes from `saved_variables/`; python
exits 0 only once `total_sim_time` is reached (the time-stop patch), so exit 0
is the loop's only clean termination.

Tunables at the top of the job script:

    MAX_RETRIES=30
    MAX_FAST_FAILS=3
    MIN_REAL_RUN=300     # seconds

The `MIN_REAL_RUN` guard is the part worth knowing about. A genuine NaN costs at
least several minutes of GPU work before openmmtools gives up and raises, so any
failure faster than 300 s is treated as a *setup* error (bad import, corrupt save
dir, missing input) rather than a NaN, and only `MAX_FAST_FAILS` of those are
tolerated. Without it a `ModuleNotFoundError` would spin all 30 retries.

Verified against three stubbed scenarios before submitting: NaN→resume→success,
immediate setup error (stops after 3), and already-complete (exits 0 on the
first call, no retries).

Each job now logs `=== attempt N | sub-sims saved so far: M ===` per attempt and
a closing `=== JOB END | sub-sims: X -> Y | retries used: N ===`, which makes
per-job yield readable straight from the log tail.

This should turn ~10 manual resubmits per system into ~4. A resubmit is still
required each time the 48 h wall clock is hit with sub-sims < 60.

## Definition of done

All 9 systems at 60/60 sub-sims (`output.ncdf` ≈ 1500 ns), then downstream
analysis (FultonMarketAnalysis / retro convergence) — see ChiMPSS
`RUN_RETRO_ANALYSIS.py`.
