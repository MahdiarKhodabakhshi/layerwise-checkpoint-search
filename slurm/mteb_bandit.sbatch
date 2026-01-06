#!/bin/bash
#SBATCH --job-name=pythia-bandit
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.log
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.log

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Make sure log dir exists
mkdir -p /scratch/$USER/slurm_logs

# Clean environment
module --force purge
module load StdEnv/2023
module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0

# Repo + env
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

# Make src importable
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# HF caches on scratch
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# Diagnostics
pwd
which python
python -V
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import layer_time.mteb_bandit_runner; print('bandit_runner_ok')"

# Run ID (reuse to resume)
RUN_ID="${RUN_ID:-bandit_$(date +%Y%m%d_%H%M%S)}"
echo "RUN_ID=$RUN_ID"

# Run bandit workflow
python -m layer_time.cli mteb-layersweep \
  --config configs/mteb_layersweep.yaml \
  --run-id "$RUN_ID"

