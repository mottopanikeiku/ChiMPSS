# v2 provenance (5-HT2B, 2026-10-06)

| stage | code | env |
|---|---|---|
| build (`build_v2.py`) | standalone Bridgeport `/home/fcetin/Bridgeport` @ `aa4f441` (same clone and uncommitted edits as v1, see `../5ht2b_v1/bridgeport/`; the edited RepairProtein is not exercised, since repair is skipped and the v1 environment reused) + ChiMPSS `chimpss.shared.system_charge` | prep1 |
| neutralization | ChiMPSS `0a3c324` for 8 systems; methysergide_A225G_mutseq rebuilt at `9a59cce` after the residue-renumbering fix (the first attempt lost 3 atoms on PDB re-read and stopped) | prep1 |
| HMR + force groups | `chimpss.bridgeport._utils`, unchanged since v1, verified bit-for-bit | prep1 |
| equilibration | ChiMPSS MotorRow @ `0a3c324`+ (step-5 membrane barostat option) | prep1 |
| production | **frozen snapshot** `5ht2b/v2/code` = ChiMPSS `0a3c324`, tag `5ht2b-v2-production` | chimpss |
| convergence | ChiMPSS `chimpss-convergence` (`3f36d8a`+), commit logged in each CONV log | chimpss |

Build and verification logs: `5ht2b/v2/logs/bridgeport/`. All nine systems
passed `verify_v2.py`.
