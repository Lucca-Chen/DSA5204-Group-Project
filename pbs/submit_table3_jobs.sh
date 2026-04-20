#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/home/svu/e1553870/DSA5204-Group-Project}"
TRAIN_PBS="${TRAIN_PBS:-${REPO_ROOT}/pbs/train_f30k_clip_table3.pbs}"
EVAL_PBS="${EVAL_PBS:-${REPO_ROOT}/pbs/eval_table3_grounding.pbs}"

TRAIN_QUEUE="${TRAIN_QUEUE:-batch_gpu}"
EVAL_QUEUE="${EVAL_QUEUE:-batch_gpu}"
FREE_EVAL_QUEUE="${FREE_EVAL_QUEUE:-gpu}"

submit_train() {
  local variant="$1"
  qsub -q "${TRAIN_QUEUE}" -v MODEL_VARIANT="${variant}" "${TRAIN_PBS}"
}

submit_eval_checkpoint() {
  local variant="$1"
  local queue="$2"
  qsub -q "${queue}" -v MODEL_TYPE=checkpoint,MODEL_VARIANT="${variant}" "${EVAL_PBS}"
}

submit_eval_vanilla() {
  local queue="$1"
  qsub -q "${queue}" -v MODEL_TYPE=vanilla_clip "${EVAL_PBS}"
}

echo "[submit_table3_jobs] train_pbs=${TRAIN_PBS}"
echo "[submit_table3_jobs] eval_pbs=${EVAL_PBS}"

case "${1:-}" in
  train_all)
    submit_train vsepp_shared
    submit_train scan_shared
    submit_train sgr_shared
    submit_train laps
    ;;
  eval_all)
    submit_eval_vanilla "${FREE_EVAL_QUEUE}"
    submit_eval_checkpoint vsepp_shared "${EVAL_QUEUE}"
    submit_eval_checkpoint scan_shared "${EVAL_QUEUE}"
    submit_eval_checkpoint sgr_shared "${EVAL_QUEUE}"
    submit_eval_checkpoint laps "${EVAL_QUEUE}"
    ;;
  *)
    echo "Usage: $0 {train_all|eval_all}" >&2
    exit 1
    ;;
esac
