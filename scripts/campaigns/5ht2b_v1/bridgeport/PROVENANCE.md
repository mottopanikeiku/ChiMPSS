# v1 Bridgeport build provenance (5-HT2B, built 2026-04-13)

- Code: standalone Bridgeport clone at `/home/fcetin/Bridgeport`, origin
  `https://github.com/CCBatIIT/Bridgeport.git`, commit
  `aa4f441` ("Remove print statements").
- That clone carries uncommitted local edits (Apr 2026), saved here as
  `uncommitted_local_changes.patch`: `RepairProtein/RepairProtein.py` and
  `Bridgeport/RUN_BRIDGEPORT.job`. They were present for the v1 build.
- Env: `/home/fcetin/miniconda3/envs/prep1` (python 3.12, openmm 8.3.1).
- Inputs: `/expanse/lustre/projects/iit127/fcetin/5ht2b/prep/json/<name>.json`
  (copied to `../json/`). Logs: `5ht2b/logs/bridgeport/`.
- The job and submit scripts here are copies of the unversioned files in that
  clone, as they were when the systems were built.
