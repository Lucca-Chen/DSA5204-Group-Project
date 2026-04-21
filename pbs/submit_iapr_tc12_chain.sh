#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/home/svu/e1553870/DSA5204-Group-Project}"
RAW_ROOT="${RAW_ROOT:-/scratch/e1553870/datasets/iapr_tc12_raw}"
OUTPUT_DATA_ROOT="${OUTPUT_DATA_ROOT:-${REPO_ROOT}/data/iapr_tc12}"
CPU_QUEUE="${CPU_QUEUE:-auto_free}"
GPU_QUEUE="${GPU_QUEUE:-auto}"
VIT_TYPE="${VIT_TYPE:-vit}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

download_job=$(qsub -N iapr-dl -q "${CPU_QUEUE}" \
  -v RAW_ROOT="${RAW_ROOT}" \
  "${REPO_ROOT}/pbs/download_iapr_tc12_cpu.pbs")
echo "download_job=${download_job}"

prep_job=$(qsub -N iapr-prep -q "${CPU_QUEUE}" -W "depend=afterok:${download_job}" \
  -v RAW_ROOT="${RAW_ROOT}",OUTPUT_ROOT="${OUTPUT_DATA_ROOT}" \
  "${REPO_ROOT}/pbs/prepare_iapr_tc12_cpu.pbs")
echo "prep_job=${prep_job}"

dispatch_job=$(qsub -N iapr-dispatch -q "${CPU_QUEUE}" -W "depend=afterok:${prep_job}" \
  -v REPO_ROOT="${REPO_ROOT}",GPU_QUEUE="${GPU_QUEUE}",VIT_TYPE="${VIT_TYPE}",RUN_SUFFIX="${RUN_SUFFIX}" \
  "${REPO_ROOT}/pbs/dispatch_iapr_tc12_gpu_jobs_cpu.pbs")
echo "dispatch_job=${dispatch_job}"
