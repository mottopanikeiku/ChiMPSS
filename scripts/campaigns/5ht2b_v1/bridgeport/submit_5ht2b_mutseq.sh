#!/bin/bash
# =============================================================================
# Submit 4 Bridgeport prep jobs for mutant systems using mutant FASTA sequences
# (MODELLER introduces the point mutation via the sequence file)
# L362F: lisuride + LSD  |  T140A: methylergonovine  |  A225G: methysergide
# Date: 2026-04-13
# Separate from the 9-job batch (submit_5ht2b_biased_agonism.sh)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/prep/json"
JOB_SCRIPT="${SCRIPT_DIR}/RUN_BRIDGEPORT.job"

echo "============================================================"
echo "Submitting 4 5-HT2B mutant-sequence Bridgeport prep jobs"
echo "(mutant FASTA — sequence-introduced point mutations)"
echo "============================================================"
echo ""

sbatch "${JOB_SCRIPT}" "${JSON_DIR}/lisuride_L362F_mutseq.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methylergonovine_T140A_mutseq.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/methysergide_A225G_mutseq.json"
sbatch "${JOB_SCRIPT}" "${JSON_DIR}/LSD_L362F_mutseq.json"

echo ""
echo "============================================================"
echo "All 4 mutseq jobs submitted. Check status with: squeue -u \$USER"
echo "============================================================"
