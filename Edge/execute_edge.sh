#!/usr/bin/env bash
set -euo pipefail

# Minimal entrypoint for the Edge pipeline.
# All three datasets (os, bgl, hdfs) share the same
# stages; os is the reference example.
#
# Usage:
#   bash execute_edge.sh [DATASET] [STAGE]
#
# Defaults:
#   DATASET: os
#   STAGE:   all
#
# DATASET: os | bgl | hdfs
# STAGE:   scores | thresh | predict | routing | hybrid | all | cloud

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_DIR="${ROOT_DIR}/Edge"

usage() {
  echo "Usage: bash execute_edge.sh [DATASET] [STAGE]" >&2
  echo "  DATASET: os | bgl | hdfs (default: os)" >&2
  echo "  STAGE: scores | thresh | predict | routing | hybrid | all | cloud (default: all)" >&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

DATASET=${1:-os}
STAGE=${2:-all}

case "${DATASET}" in
  os)
    PAR_SCRIPT="edge_scripts/parallel_execute_os.sh" ;;
  bgl)
    PAR_SCRIPT="edge_scripts/parallel_execute_bgl.sh" ;;
  hdfs)
    PAR_SCRIPT="edge_scripts/parallel_execute_hdfs.sh" ;;
  *)
    echo "[execute_edge] Unknown DATASET: ${DATASET} (expected: os | bgl | hdfs)" >&2
    usage
    exit 1 ;;
esac

cd "${EDGE_DIR}"

case "${STAGE}" in
  scores)
    echo "[execute_edge] ${DATASET}: ExecuTorch scoring (${PAR_SCRIPT})"
    bash "${PAR_SCRIPT}"
    ;;
  thresh)
    echo "[execute_edge] ${DATASET}: thresholds (thresh.sh; os example)"
    bash edge_scripts/thresh.sh
    ;;
  predict)
    echo "[execute_edge] ${DATASET}: edge predictions (predict.sh; os example)"
    bash edge_scripts/predict.sh
    ;;
  routing)
    echo "[execute_edge] ${DATASET}: routing (routing.sh; os example)"
    bash edge_scripts/routing.sh
    ;;
  hybrid)
    echo "[execute_edge] ${DATASET}: hybrid evaluation (hybrid.sh; os example)"
    bash edge_scripts/hybrid.sh
    ;;
  cloud)
    if [[ "${DATASET}" != "os" ]]; then
      echo "[execute_edge] STAGE 'cloud' is only supported for DATASET=os" >&2
      usage
      exit 1
    fi
    echo "[execute_edge] os: full Edge↔Cloud pipeline via ./collaborative_execution.sh"
    bash "${ROOT_DIR}/scripts/collaborative_execution.sh"
    ;;
  all)
    echo "[execute_edge] ${DATASET}: scores -> thresh -> predict -> routing -> hybrid"
    bash "${PAR_SCRIPT}"
    bash edge_scripts/thresh.sh
    bash edge_scripts/predict.sh
    bash edge_scripts/routing.sh
    bash edge_scripts/hybrid.sh
    ;;
  *)
    echo "[execute_edge] Invalid STAGE: ${STAGE}" >&2
    usage
    exit 1
    ;;
esac

