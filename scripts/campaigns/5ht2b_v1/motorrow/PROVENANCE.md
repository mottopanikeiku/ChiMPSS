# v1 equilibration provenance (5-HT2B)

- `generate_hmr_xmls.py` turned `work_dir/systems/<name>.xml` into
  `work_dir/systems_hmr/<name>_FG_HMR.xml` with
  `chimpss.bridgeport._utils.apply_force_groups` + `apply_hmr`. v2 reuses
  this recipe; it reproduces the v1 file bit-for-bit (see
  `../../5ht2b_v2/check_hmr_repro.py`).
- `RUN_CHIMPSS_MOTORROW.py` + `EQUIL_HMR.job` + `submit_5ht2b_equil_hmr.sh`
  ran ChiMPSS MotorRow at dt = 3.5 fs into `equil_hmr/` (May 2026), the
  authoritative v1 production input. `EQUIL_MEMBRANE_fcetin.job` and
  `submit_5ht2b_equil.sh` are the earlier non-HMR route into `equil/`.
- These files live untracked in `/home/fcetin/MotorRow` (origin
  `CCBatIIT/MotorRow`, commit `0977b38`); copied here unchanged.
