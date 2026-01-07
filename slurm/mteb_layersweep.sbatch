#!/bin/bash
#SBATCH --job-name=pythia-mteb
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.log
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.log

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Make sure log dir exists (Slurm won't create parent dirs)
mkdir -p /scratch/$USER/slurm_logs

# Clean environment (do NOT rely on dotfiles)
module --force purge
module load StdEnv/2023
module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0   # IMPORTANT: before venv activation

# Repo + env
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

# Make src importable even if editable install isn't present
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# HF caches on scratch
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# Diagnostics / fail-fast preflight
pwd
which python
python -V
python -c "import sys; print('executable=', sys.executable)"
python -c "import layer_time; print('layer_time=', layer_time.__file__)"
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import numpy, psutil, pandas, scipy; import pyarrow; import datasets, mteb; print('deps_ok')"

# Stable run id for resume (reuse RUN_ID to resume)
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
echo "RUN_ID=$RUN_ID"

# Run
python -m layer_time.cli mteb-layersweep \
  --config configs/mteb_layersweep.yaml \
  --run-id "$RUN_ID"