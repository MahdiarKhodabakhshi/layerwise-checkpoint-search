#!/bin/bash
# Submit Experiment 2 with parallelization (50 separate jobs)
# Each checkpoint gets its own job for faster completion
# Usage: ./scripts/submit_exp2_parallel.sh [RUN_ID]

set -euo pipefail

RUN_ID="${1:-exp2_final50checkpoints_final4layers}"
echo "Using RUN_ID: $RUN_ID"
echo "Submitting 50 separate jobs for Experiment 2..."
echo ""

cd "$(dirname "$0")/../.."

# Final 50 checkpoints
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

JOB_IDS=()
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    
    JOB_ID=$(sbatch << EOF | grep -oP '\d+'
#!/bin/bash
#SBATCH --job-name=exp2-checkpoint-${i}
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

mkdir -p /scratch/\$USER/slurm_logs

module --force purge
module load StdEnv/2023
module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0

cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

export PYTHONPATH="\$PWD/src:\${PYTHONPATH:-}"

OUTPUT_ROOT="/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer"
mkdir -p "\$OUTPUT_ROOT"

export HF_HOME=/scratch/\$USER/hf
export HF_DATASETS_CACHE=\$HF_HOME/datasets
export XDG_CACHE_HOME=\$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$XDG_CACHE_HOME"

# Diagnostics
pwd
which python
python -V
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import layer_time.mteb_runner; print('mteb_runner_ok')"

echo "=========================================="
echo "Experiment 2: Checkpoint ${checkpoint}"
echo "Layers: 20-23 (final 4 layers)"
echo "Tasks: All 32 MTEB tasks"
echo "=========================================="

# Create temporary config for this checkpoint
TEMP_CONFIG="/tmp/config_exp2_${checkpoint}_\$\$.yaml"
python3 << PYEOF
import yaml
from pathlib import Path

base_config_path = Path("pipeline_layer_by_layer/configs/exp2_final50checkpoints_final4layers.yaml")
with base_config_path.open() as f:
    config = yaml.safe_load(f)

# Filter to only this checkpoint
config['hf']['revisions'] = ["${checkpoint}"]

with open("\${TEMP_CONFIG}", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config: \${TEMP_CONFIG}")
PYEOF

export LAYER_TIME_NUM_SHARDS=1
export LAYER_TIME_SHARD_ID=0

python -m layer_time.cli mteb-layersweep \\
  --config "\${TEMP_CONFIG}" \\
  --run-id "${RUN_ID}"

rm -f "\${TEMP_CONFIG}"
echo "Checkpoint ${checkpoint} completed successfully"
EOF
)
    
    JOB_IDS+=($JOB_ID)
    if [ $((i % 10)) -eq 0 ] || [ $i -eq $((${#CHECKPOINTS[@]} - 1)) ]; then
        echo "Submitted checkpoint $i/$((${#CHECKPOINTS[@]} - 1)): $checkpoint (Job ID $JOB_ID)"
    fi
done

echo ""
echo "=========================================="
echo "All 50 jobs submitted for Experiment 2!"
echo "=========================================="
echo "RUN_ID: $RUN_ID"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Jobs will start incrementally as GPUs become available"
echo "Monitor with: squeue -u mahdiar | grep exp2-checkpoint"
