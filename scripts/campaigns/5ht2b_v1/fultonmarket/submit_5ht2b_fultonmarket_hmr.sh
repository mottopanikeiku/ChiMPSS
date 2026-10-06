#!/bin/bash
# =============================================================================
# Submit ChiMPSS FultonMarket REMD jobs (HMR, dt = 3.5 fs, 17.5 ps swaps)
# for the 9 5-HT2B
# biased-agonism systems.
#
# Usage:
#   bash submit_5ht2b_fultonmarket_hmr.sh                 # submit all 9
#   bash submit_5ht2b_fultonmarket_hmr.sh lisuride        # submit just lisuride
#   bash submit_5ht2b_fultonmarket_hmr.sh lisuride LSD    # submit a subset
#
# Chaining (optional, via env vars):
#   CHAIN_DEPTH=3 bash submit_5ht2b_fultonmarket_hmr.sh LSD
#       -> submits 3 jobs for LSD, each --dependency=afterany on the previous.
#          A 48 h TIMEOUT is the NORMAL end state of these jobs, so `afterany`
#          (not afterok) is the correct dependency: the successor must start
#          after the predecessor ENDS, whatever its exit state.
#          This is also what keeps the "never two jobs on one system" rule --
#          a chained job cannot start until its predecessor has terminated.
#
#   AFTER_JOB=<jobid> bash submit_5ht2b_fultonmarket_hmr.sh LSD
#       -> hangs the new chain off an ALREADY-RUNNING job for that system.
#          Use this to extend a system whose job is still running.
#
# ⚠️ SLURM caps gpu-shared at 24 jobs per user (running + pending, QOS
#    gpu-shared-normal MaxSubmitPU). Chain depth must respect that budget --
#    check `squeue -u $USER -h -p gpu-shared | wc -l` before submitting.
#
# Per-system inputs:
#   input_pdb     = equil_hmr/<name>/step_5.pdb
#   input_system  = work_dir/systems_hmr/<name>_FG_HMR.xml
#   --input_state = equil_hmr/<name>/step_5.xml
#   --timestep    = 3.5
#   --iter_length = 0.0175 ns (17.5 ps; 5,000 MD steps)
# Output per system: fultonmarket_hmr/<name>/
# =============================================================================

set -e

# PREEMPT=1 selects the 7-day gpu-preempt job script instead of the 48 h
# gpu-shared one. Same REMD parameters, same retry loop -- only the partition,
# walltime and --open-mode differ. Preempted jobs are requeued by SLURM and
# resume from saved_variables/, so a preemption costs at most one partial
# sub-simulation (~3 h), never a completed one.
if [ "${PREEMPT:-0}" = "1" ]; then
    JOB_SCRIPT="/home/fcetin/FultonMarket/RUN_FULTONMARKET_PREEMPT.job"
else
    JOB_SCRIPT="/home/fcetin/FultonMarket/RUN_FULTONMARKET.job"
fi

PROJ="/expanse/lustre/projects/iit127/fcetin/5ht2b"
SYS_HMR_DIR="${PROJ}/work_dir/systems_hmr"
EQUIL_DIR="${PROJ}/equil_hmr"
OUTPUT_DIR="${PROJ}/fultonmarket_hmr"
LOG_DIR="${PROJ}/logs/fultonmarket_hmr"

# REMD parameters
TIMESTEP=3.5
T_MIN=300
T_MAX=367
N_REPLICATES=68
SIM_LENGTH=25            # ns per sub-simulation / checkpoint interval
TOTAL_SIM_TIME=1500      # ns aggregate target
ITER_LENGTH=0.0175       # ns between swaps: 17.5 ps = 5,000 steps at 3.5 fs

ALL_SYSTEMS=(
    lisuride
    methylergonovine
    methysergide
    LSD
    LY266097
    lisuride_L362F_mutseq
    methylergonovine_T140A_mutseq
    methysergide_A225G_mutseq
    LSD_L362F_mutseq
)

if [ $# -gt 0 ]; then
    SYSTEMS=("$@")
else
    SYSTEMS=("${ALL_SYSTEMS[@]}")
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "Submitting FultonMarket HMR REMD jobs"
echo "  systems       : ${SYSTEMS[*]}"
echo "  timestep      : ${TIMESTEP} fs"
echo "  iter_length   : ${ITER_LENGTH} ns (17.5 ps; 5,000 MD steps)"
echo "  T ladder      : ${T_MIN}-${T_MAX} K, ${N_REPLICATES} replicates"
echo "  sim_length    : ${SIM_LENGTH} ns / checkpoint"
echo "  total_sim_time: ${TOTAL_SIM_TIME} ns"
echo "  equil dir     : ${EQUIL_DIR}/<name>/"
echo "  HMR system dir: ${SYS_HMR_DIR}/"
echo "  output        : ${OUTPUT_DIR}/<name>/"
echo "============================================================"

for name in "${SYSTEMS[@]}"; do
    PDB="${EQUIL_DIR}/${name}/step_5.pdb"
    STATE="${EQUIL_DIR}/${name}/step_5.xml"
    SYSXML="${SYS_HMR_DIR}/${name}_FG_HMR.xml"
    OUT="${OUTPUT_DIR}/${name}"

    # Preflight: every required input must exist
    for f in "${PDB}" "${STATE}" "${SYSXML}"; do
        if [ ! -f "${f}" ]; then
            echo "ERROR [${name}]: missing input ${f}" >&2
            exit 1
        fi
    done
    mkdir -p "${OUT}"

    echo ""
    echo "--- ${name} ---"

    # Chain CHAIN_DEPTH jobs, each waiting on the previous one to END.
    prev="${AFTER_JOB:-}"
    for ((link = 0; link < ${CHAIN_DEPTH:-1}; link++)); do
        dep=()
        [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")

        out=$(sbatch "${dep[@]}" "${JOB_SCRIPT}" \
            "${PDB}" \
            "${SYSXML}" \
            "${OUT}" \
            --input_state "${STATE}" \
            --timestep "${TIMESTEP}" \
            --T_min "${T_MIN}" \
            --T_max "${T_MAX}" \
            --n_replicates "${N_REPLICATES}" \
            --iter_length "${ITER_LENGTH}" \
            --sim_length "${SIM_LENGTH}" \
            --total_sim_time "${TOTAL_SIM_TIME}")

        jid=$(echo "${out}" | awk '{print $NF}')
        if [ -n "${prev}" ]; then
            echo "  link ${link}: ${jid}  (waits on ${prev})"
        else
            echo "  link ${link}: ${jid}"
        fi
        prev="${jid}"
    done
done

echo ""
echo "============================================================"
echo "All ${#SYSTEMS[@]} job(s) submitted. squeue -u \$USER to check."
echo "============================================================"
