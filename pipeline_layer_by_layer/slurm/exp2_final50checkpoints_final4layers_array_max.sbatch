#!/bin/bash
#SBATCH --job-name=layer-by-layer-exp2-max
#SBATCH --account=aip-btaati
#SBATCH --array=0-199%60            # 200 array tasks (one per checkpoint-layer pair), max 60 concurrent
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

# Map array task ID to (checkpoint, layer) pair
# 50 checkpoints × 4 layers = 200 pairs
CHECKPOINTS=(
    "step94000" "step95000" "step96000" "step97000" "step98000" "step99000"
    "step100000" "step101000" "step102000" "step103000" "step104000" "step105000"
    "step106000" "step107000" "step108000" "step109000" "step110000" "step111000"
    "step112000" "step113000" "step114000" "step115000" "step116000" "step117000"
    "step118000" "step119000" "step120000" "step121000" "step122000" "step123000"
    "step124000" "step125000" "step126000" "step127000" "step128000" "step129000"
    "step130000" "step131000" "step132000" "step133000" "step134000" "step135000"
    "step136000" "step137000" "step138000" "step139000" "step140000" "step141000"
    "step142000" "step143000"
)
LAYERS=(20 21 22 23)

CHECKPOINT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % 4))
CHECKPOINT=${CHECKPOINTS[$CHECKPOINT_IDX]}
LAYER=${LAYERS[$LAYER_IDX]}

echo "=========================================="
echo "Experiment 2: Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Layer: ${LAYER}"
echo "Tasks: All 32 MTEB tasks"
echo "Expected evaluations: 1 checkpoint × 1 layer × 32 tasks = 32"
echo "=========================================="

# Create temporary config for this checkpoint-layer pair
TEMP_CONFIG="/tmp/config_exp2_${CHECKPOINT}_layer${LAYER}_$$.yaml"
export CHECKPOINT LAYER TEMP_CONFIG
python3 << EOF
import yaml
import os
from pathlib import Path

checkpoint = os.environ['CHECKPOINT']
layer = os.environ['LAYER']
temp_config = os.environ['TEMP_CONFIG']

base_config_path = Path("pipeline_layer_by_layer/configs/exp2_final50checkpoints_final4layers.yaml")
with base_config_path.open() as f:
    config = yaml.safe_load(f)

# Filter to only this checkpoint and layer
config['hf']['revisions'] = [checkpoint]
config['embedding']['layers'] = layer

with open(temp_config, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config: {temp_config}")
print(f"Checkpoint: {checkpoint}, Layer: {layer}")
EOF

export LAYER_TIME_NUM_SHARDS=1
export LAYER_TIME_SHARD_ID=0

python -m layer_time.cli mteb-layersweep \
  --config "${TEMP_CONFIG}" \
  --run-id exp2_final50checkpoints_final4layers

rm -f "${TEMP_CONFIG}"
echo "Array task ${SLURM_ARRAY_TASK_ID} (checkpoint ${CHECKPOINT}, layer ${LAYER}) completed successfully"
