#!/bin/bash
# =============================================================================
# Submit 9 ChiMPSS MotorRow equilibration jobs WITH HMR @ dt = 3.5 fs for:
# "Structural determinants of 5-HT2B receptor activation and biased agonism"
# Fig. 4 (lisuride, methylergonovine, methysergide, LSD — WT + mutseq mutant)
# Fig. 5 (LY266097 WT)
# Date: 2026-05-22
# =============================================================================

set -e

JOB_SCRIPT="/home/fcetin/MotorRow/EQUIL_HMR.job"
INPUT_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems"
HMR_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/work_dir/systems_hmr"
OUTPUT_DIR="/expanse/lustre/projects/iit127/fcetin/5ht2b/equil_hmr"
LIG_RESNAME="UNK"
DT_FS="3.5"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /expanse/lustre/projects/iit127/fcetin/5ht2b/logs/equil_hmr

echo "============================================================"
echo "Submitting 9 5-HT2B biased agonism HMR MotorRow equil jobs"
echo "  dt = ${DT_FS} fs"
echo "  PDBs from:  ${INPUT_DIR}"
echo "  HMR XMLs:   ${HMR_DIR}"
echo "  Output to:  ${OUTPUT_DIR}/<name>/"
echo "============================================================"
echo ""

# ---- WT systems ------------------------------------------------------------
echo "--- WT systems ---"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" lisuride           "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methylergonovine   "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methysergide       "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LSD                "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LY266097           "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"

# ---- Mutant systems (sequence-introduced mutations) ------------------------
echo ""
echo "--- Mutant systems ---"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" lisuride_L362F_mutseq         "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methylergonovine_T140A_mutseq "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" methysergide_A225G_mutseq     "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"
sbatch "${JOB_SCRIPT}" "${INPUT_DIR}" LSD_L362F_mutseq              "${HMR_DIR}" "${OUTPUT_DIR}" "${LIG_RESNAME}" "${DT_FS}"

echo ""
echo "============================================================"
echo "All 9 HMR equil jobs submitted. Check status with: squeue -u \$USER"
echo "============================================================"
