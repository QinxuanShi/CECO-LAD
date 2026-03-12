#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the full OS Edge→Cloud→Edge pipeline:
#  1. Run Edge pipeline for OS (scores, thresh, predict, routing).
#  2. Copy selected samples from Edge to Cloud.
#  3. On Cloud, run execute_selected_cloud.sh to get cloud preds.
#  4. Copy cloud predictions back to Edge.
#  5. Run Edge hybrid evaluation.
#
# This script is intended to be run on the EDGE device.
# You must configure the Cloud SSH information below.

# --- CONFIGURE THESE FOR YOUR ENVIRONMENT ---
CLOUD_USER="your_cloud_username"
CLOUD_HOST="your.cloud.hostname.or.ip"
# Absolute path to the CECO-LAD/Cloud directory on the Cloud device
CLOUD_CLOUD_DIR="/path/to/CECO-LAD/Cloud"
# -------------------------------------------

if [[ "${CLOUD_USER}" == "your_cloud_username" || "${CLOUD_HOST}" == "your.cloud.hostname.or.ip" || "${CLOUD_CLOUD_DIR}" == "/path/to/CECO-LAD/Cloud" ]]; then
  echo "[cloud-edge collaboration] Please edit this script and set CLOUD_USER, CLOUD_HOST and CLOUD_CLOUD_DIR first." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_DIR="${ROOT_DIR}/Edge"

EDGE_SELECTED_REL="prediction_results/os_selected.txt"
EDGE_PRED_REL="prediction_results/cloud_selected_ensemble_preds.txt"

CLOUD_SELECTED_REMOTE="${CLOUD_CLOUD_DIR}/selected_samples/os_selected.txt"
CLOUD_PRED_REMOTE="${CLOUD_CLOUD_DIR}/selected_samples/cloud_selected_ensemble_preds.txt"

echo "[cloud-edge collaboration] Step 1: Edge OS pipeline (prediction (anomaly score) -> threshold calculation -> anomaly prediction (binary) -> routing) via execute_edge.sh"
bash "${EDGE_DIR}/execute_edge.sh" os scores
bash "${EDGE_DIR}/execute_edge.sh" os thresh
bash "${EDGE_DIR}/execute_edge.sh" os predict
bash "${EDGE_DIR}/execute_edge.sh" os routing

if [[ ! -f "${EDGE_SELECTED_REL}" ]]; then
  echo "[cloud-edge collaboration] ERROR: Edge selected file not found: ${EDGE_SELECTED_REL}" >&2
  exit 1
fi

echo "[cloud-edge collaboration] Step 2: Copy selected samples Edge -> Cloud"
scp "${EDGE_SELECTED_REL}" "${CLOUD_USER}@${CLOUD_HOST}:${CLOUD_SELECTED_REMOTE}"

echo "[cloud-edge collaboration] Step 3: Run cloud selected inference on Cloud"
ssh "${CLOUD_USER}@${CLOUD_HOST}" "cd ${CLOUD_CLOUD_DIR} && ./execute_selected_cloud.sh"

echo "[cloud-edge collaboration] Step 4: Copy cloud predictions Cloud -> Edge"
scp "${CLOUD_USER}@${CLOUD_HOST}:${CLOUD_PRED_REMOTE}" "${EDGE_PRED_REL}"

if [[ ! -f "${EDGE_PRED_REL}" ]]; then
  echo "[cloud-edge collaboration] ERROR: Cloud prediction file not found after copy: ${EDGE_PRED_REL}" >&2
  exit 1
fi

echo "[cloud-edge collaboration] Step 5: Edge hybrid evaluation (using cloud-selected preds)"
cd "${EDGE_DIR}"
bash ./execute_edge.sh os hybrid

echo "[cloud-edge collaboration] Done: full OS Edge↔Cloud pipeline complete."
