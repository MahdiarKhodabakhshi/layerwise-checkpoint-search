#!/bin/bash
#SBATCH --job-name=exp1-fixed
#SBATCH --account=aip-btaati
#SBATCH --array=0-3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/slurm_logs/%x_%A_%a.log
#SBATCH --error=/scratch/%u/slurm_logs/%x_%A_%a.log

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

mkdir -p /scratch/$USER/slurm_logs

module --force purge
module load StdEnv/2023
module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0

cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# shard controls for 410m model
export LAYER_TIME_ONLY_SIZE="410m"
export LAYER_TIME_NUM_SHARDS="4"
export LAYER_TIME_SHARD_ID="${SLURM_ARRAY_TASK_ID}"

# run-id is specified in the config file
python -m layer_time.cli mteb-layersweep \
  --config configs/exp1_main_final9layers_fixed.yaml
