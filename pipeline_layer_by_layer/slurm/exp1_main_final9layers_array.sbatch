#!/bin/bash
#SBATCH --job-name=layer-by-layer-exp1-array
#SBATCH --account=aip-btaati
#SBATCH --array=0-8%9              # 9 array tasks (one per layer), 9 concurrent (uses 9 GPUs)
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
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

OUTPUT_ROOT="/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer"
mkdir -p "$OUTPUT_ROOT"

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

# Map array task ID to layer (15-23)
LAYERS=(15 16 17 18 19 20 21 22 23)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Experiment 1: Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Layer: ${LAYER}"
echo "Checkpoint: main"
echo "Tasks: All 32 MTEB tasks"
echo "Expected evaluations: 1 checkpoint × 1 layer × 32 tasks = 32"
echo "=========================================="

# Create temporary config for this layer
TEMP_CONFIG="/tmp/config_exp1_layer_${LAYER}_$$.yaml"
export LAYER TEMP_CONFIG
python3 << EOF
import yaml
import os
from pathlib import Path

layer = os.environ['LAYER']
temp_config = os.environ['TEMP_CONFIG']

base_config_path = Path("pipeline_layer_by_layer/configs/exp1_main_final9layers.yaml")
with base_config_path.open() as f:
    config = yaml.safe_load(f)

# Filter to only this layer
config['embedding']['layers'] = layer

with open(temp_config, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config: {temp_config}")
print(f"Layer: {layer}")
EOF

export LAYER_TIME_NUM_SHARDS=1
export LAYER_TIME_SHARD_ID=0

python -m layer_time.cli mteb-layersweep \
  --config "${TEMP_CONFIG}" \
  --run-id exp1_main_final9layers_v1

rm -f "${TEMP_CONFIG}"
echo "Array task ${SLURM_ARRAY_TASK_ID} (layer ${LAYER}) completed successfully"
