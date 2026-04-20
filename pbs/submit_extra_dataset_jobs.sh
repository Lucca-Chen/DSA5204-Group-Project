#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-}"
DATASET="${DATASET:-${2:-}}"
VIT_TYPE="${VIT_TYPE:-vit}"

if [[ -z "${ACTION}" ]]; then
  echo "Usage: ./pbs/submit_extra_dataset_jobs.sh <train_all|eval_all> [dataset]" >&2
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "Please provide DATASET as an environment variable or the second argument." >&2
  exit 1
fi

VARIANTS=(vsepp_shared scan_shared sgr_shared chan_shared laps)

case "${ACTION}" in
  train_all)
    for variant in "${VARIANTS[@]}"; do
      qsub -v DATASET="${DATASET}",MODEL_VARIANT="${variant}",VIT_TYPE="${VIT_TYPE}" ./pbs/train_extra_dataset_shared_backbone.pbs
    done
    ;;
  eval_all)
    for variant in "${VARIANTS[@]}"; do
      qsub -v DATASET="${DATASET}",MODEL_VARIANT="${variant}",VIT_TYPE="${VIT_TYPE}" ./pbs/eval_extra_dataset_shared_backbone.pbs
    done
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    exit 1
    ;;
esac
