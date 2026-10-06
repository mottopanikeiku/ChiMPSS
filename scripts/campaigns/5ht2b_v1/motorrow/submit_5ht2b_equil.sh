#!/bin/bash
# =============================================================================
# Submit 9 MotorRow equilibration jobs for:
# "Structural determinants of 5-HT2B receptor activation and biased agonism"
# Fig. 4 (lisuride, methylergonovine, methysergide, LSD — WT + mutseq mutant)
# Fig. 5 (LY266097 WT)
# Date: 2026-04-21
# =============================================================================

set -e

JOB_SCRIPT="/home/fcetin/MotorRow/EQUIL_MEMBRANE_fcetin.job"
INPUT_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems"
OUTPUT_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/equil"

echo "============================================================"
echo "Submitting 9 5-HT2B biased agonism MotorRow equil jobs"
echo "============================================================"
echo ""

# ---- WT systems ------------------------------------------------------------
echo "--- WT systems ---"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" lisuride           "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methylergonovine   "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methysergide       "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LSD                "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LY266097           "${OUTPUT_DIR}" UNK

# ---- Mutant systems (sequence-introduced mutations) ------------------------
echo ""
echo "--- Mutant systems ---"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" lisuride_L362F_mutseq         "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methylergonovine_T140A_mutseq "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methysergide_A225G_mutseq     "${OUTPUT_DIR}" UNK
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LSD_L362F_mutseq              "${OUTPUT_DIR}" UNK

echo ""
echo "============================================================"
echo "All 9 equil jobs submitted. Check status with: squeue -u \$USER"
echo "============================================================"
