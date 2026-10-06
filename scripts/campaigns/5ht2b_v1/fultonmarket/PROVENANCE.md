# v1 production provenance (5-HT2B, 2026-06 to 2026-09)

- Ran ChiMPSS `RUN_FULTONMARKET.py` from the live `/home/fcetin/ChiMPSS`
  working tree (branch `feature/fm-resume-and-time-stop`), not a snapshot.
- **All 540 v1 sub-simulations carry the replica-reset bug** (fixed in
  `24e6d44`), so `fultonmarket_hmr/` is not valid REMD. See
  `FULTONMARKET_HANDOFF.md` on Expanse.
- `RUN_FULTONMARKET_PREEMPT.job` was tried and abandoned: it livelocks.
- These files live untracked in `/home/fcetin/FultonMarket` (origin
  `CCBatIIT/FultonMarket`); copied here unchanged.
