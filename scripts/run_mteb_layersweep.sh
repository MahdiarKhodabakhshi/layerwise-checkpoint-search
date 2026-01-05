#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/run_mteb_layersweep.sh

export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

python -m layer_time.cli mteb-layersweep --config configs/mteb_layersweep.yaml
