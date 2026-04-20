#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/home/svu/e1553870/DSA5204-Group-Project}"
PBS_SCRIPT="${PBS_SCRIPT:-${REPO_ROOT}/pbs/train_f30k_shared_backbone.pbs}"

QUEUE="${QUEUE:-batch_gpu}"
SELECT_SPEC="${SELECT_SPEC:-1:ncpus=8:ngpus=1:mem=64gb:gpu-type=nvidia}"
WALLTIME="${WALLTIME:-24:00:00}"

RUNS_ROOT="${RUNS_ROOT:-/scratch/e1553870/DSA5204-Group-Project/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/e1553870/DSA5204-Group-Project/output/laps_f30k_shared}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
F30K_IMG_PATH="${F30K_IMG_PATH:-/scratch/e1553870/datasets/flickr30k-images}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
VIT_TYPE="${VIT_TYPE:-vit}"
EMBED_SIZE="${EMBED_SIZE:-512}"
SEED="${SEED:-0}"
VAL_SPLIT="${VAL_SPLIT:-dev}"
TEST_SPLIT="${TEST_SPLIT:-test}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
AMP="${AMP:-0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-1}"
RESUME_PATH="${RESUME_PATH:-auto}"
DEPEND_MODE="${DEPEND_MODE:-chain}"

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "Missing PBS script: ${PBS_SCRIPT}" >&2
  exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available in PATH" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${RUNS_ROOT}"

submit_one() {
  local model_variant="$1"
  local run_name="$2"
  local extra_args="${3:-}"
  local job_name="${4:-laps-${model_variant}}"
  local -a qsub_args
  local vars
  local job_id

  qsub_args=(
    -N "${job_name}"
    -q "${QUEUE}"
    -l "select=${SELECT_SPEC}"
    -l "walltime=${WALLTIME}"
  )

  if [[ "${DEPEND_MODE}" == "chain" && -n "${PREV_JOB_ID:-}" ]]; then
    qsub_args+=(-W "depend=afterok:${PREV_JOB_ID}")
  fi

  vars="REPO_ROOT=${REPO_ROOT},VENV_PATH=${VENV_PATH},RUNS_ROOT=${RUNS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_VARIANT=${model_variant},RUN_NAME=${run_name}${RUN_SUFFIX},RUN_SUFFIX=${RUN_SUFFIX},F30K_IMG_PATH=${F30K_IMG_PATH},BATCH_SIZE=${BATCH_SIZE},NUM_EPOCHS=${NUM_EPOCHS},WORKERS=${WORKERS},VIT_TYPE=${VIT_TYPE},EMBED_SIZE=${EMBED_SIZE},SEED=${SEED},VAL_SPLIT=${VAL_SPLIT},TEST_SPLIT=${TEST_SPLIT},AMP=${AMP},AMP_DTYPE=${AMP_DTYPE},SAVE_LAST_CHECKPOINT=${SAVE_LAST_CHECKPOINT},RESUME_PATH=${RESUME_PATH},EXTRA_ARGS=${extra_args}"

  job_id="$(qsub "${qsub_args[@]}" -v "${vars}" "${PBS_SCRIPT}")"
  PREV_JOB_ID="${job_id}"
  echo "${run_name}: ${job_id}"
}

echo "[submit_f30k_shared_backbone_chain] queue=${QUEUE} select=${SELECT_SPEC} walltime=${WALLTIME}"
echo "[submit_f30k_shared_backbone_chain] batch_size=${BATCH_SIZE} epochs=${NUM_EPOCHS} vit_type=${VIT_TYPE}"
echo "[submit_f30k_shared_backbone_chain] depend_mode=${DEPEND_MODE}"

PREV_JOB_ID=""

submit_one "vsepp_shared" "f30k_vsepp_shared" ""
submit_one "scan_shared" "f30k_scan_shared" ""
submit_one "basealign" "f30k_basealign" ""
submit_one "sparse" "f30k_sparse" ""
submit_one "laps" "f30k_laps" ""
