#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
JOBS="${JOBS:-5}"
"$PY" step1_download.py
"$PY" step2_train_boosting.py --jobs "$JOBS"
"$PY" step3_train_network.py --jobs "$JOBS"
"$PY" step4_predict.py
