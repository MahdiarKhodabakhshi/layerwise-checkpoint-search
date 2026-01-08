#!/bin/bash
#SBATCH --job-name=layer-by-layer-exp2
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.log
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.log

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

# Ensure output directory exists
OUTPUT_ROOT="/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer"
mkdir -p "$OUTPUT_ROOT"
echo "Output directory: $OUTPUT_ROOT"

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
python -c "import layer_time.mteb_runner; print('mteb_runner_ok')"

echo "=========================================="
echo "Experiment 2: Final 50 Checkpoints, Final 4 Layers"
echo "=========================================="
echo "Model: Pythia 410m"
echo "Checkpoints: step94000 through step143000 (50 checkpoints)"
echo "Layers: 20-23 (final 4 layers)"
echo "Tasks: All 32 MTEB tasks"
echo "=========================================="

# Run Experiment 2
python -m layer_time.cli mteb-layersweep \
  --config pipeline_layer_by_layer/configs/exp2_final50checkpoints_final4layers.yaml \
  --run-id exp2_final50checkpoints_final4layers

echo "Experiment 2 completed successfully"
