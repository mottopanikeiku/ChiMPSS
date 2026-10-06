#!/bin/bash
# =============================================================================
# Submit 9 Bridgeport prep jobs for:
# "Structural determinants of 5-HT2B receptor activation and biased agonism"
# Fig. 4 (lisuride, methylergonovine, methysergide, LSD — WT + mutant)
# Fig. 5 (LY266097 WT)
# Date: 2026-04-13
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/prep/json"
JOB_SCRIPT="${SCRIPT_DIR}/RUN_BRIDGEPORT.job"

echo "============================================================"
echo "Submitting 9 5-HT2B biased agonism Bridgeport prep jobs"
echo "============================================================"
echo ""

# ---- Fig. 4 & 5: WT systems ------------------------------------------------
echo "--- WT systems ---"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/lisuride.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methylergonovine.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methysergide.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/LSD.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/LY266097.json"

# ---- Fig. 4: Mutant systems -------------------------------------------------
echo ""
echo "--- Mutant systems ---"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/lisuride_L362F.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methylergonovine_T140A.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methysergide_A225G.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/LSD_L362F.json"

echo ""
echo "============================================================"
echo "All 9 jobs submitted. Check status with: squeue -u \$USER"
echo "============================================================"
