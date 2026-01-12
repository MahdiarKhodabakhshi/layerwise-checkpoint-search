#!/bin/bash
#SBATCH --job-name=pythia-validation-main-last4-array
#SBATCH --account=aip-btaati
#SBATCH --array=0-3%4              # 4 array tasks (one per layer), max 4 concurrent
#SBATCH --gres=gpu:l40s:1           # 1 GPU per array task
#SBATCH --cpus-per-task=8           # 8 CPUs per task
#SBATCH --mem=120G                  # Per-task memory
#SBATCH --time=24:00:00             # 24 hour limit per task
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

# Ensure output directory exists on scratch
OUTPUT_ROOT="/scratch/mahdiar/pythia-layer-time-runs"
mkdir -p "$OUTPUT_ROOT"
echo "Output directory: $OUTPUT_ROOT"
df -h "$OUTPUT_ROOT" | head -2

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

# Run ID shared across all array tasks
RUN_ID="${RUN_ID:-validation_main_last4_array_$(date +%Y%m%d_%H%M%S)}"
echo "RUN_ID=$RUN_ID"
echo "Array task: ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"

# Map array task ID to layer: 0→20, 1→21, 2→22, 3→23
LAYERS=(20 21 22 23)
LAYER=${LAYERS[${SLURM_ARRAY_TASK_ID}]}

echo "=========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID}: Processing layer ${LAYER}"
echo "Checkpoint: main"
echo "Layer: ${LAYER}"
echo "Tasks: All 32 MTEB tasks"
echo "Expected evaluations: 1 checkpoint × 1 layer × 32 tasks = 32"
echo "=========================================="

# Create a temporary config file with single layer
TEMP_CONFIG="/tmp/config_validation_array_${SLURM_ARRAY_TASK_ID}.yaml"
python3 << EOF
import yaml
from pathlib import Path

# Load base config
base_config_path = Path("configs/mteb_validation_main_last4.yaml")
with base_config_path.open() as f:
    config = yaml.safe_load(f)

# Set single layer for this array task
config['embedding']['layers'] = "${LAYER}"

# Write filtered config
with open("${TEMP_CONFIG}", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created filtered config: ${TEMP_CONFIG}")
print(f"Layer in config: ${LAYER}")
EOF

# Run the evaluation with filtered config
# No sharding needed since each array task handles its own layer
export LAYER_TIME_NUM_SHARDS=1
export LAYER_TIME_SHARD_ID=0

python -m layer_time.cli mteb-layersweep \
  --config "${TEMP_CONFIG}" \
  --run-id "$RUN_ID"

# Cleanup temp config
rm -f "${TEMP_CONFIG}"

echo "Array task ${SLURM_ARRAY_TASK_ID} (layer ${LAYER}) completed successfully"
