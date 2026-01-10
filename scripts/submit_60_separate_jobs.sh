#!/bin/bash
# Submit 60 separate SLURM jobs for faster start time
# Each job can start independently as GPUs become available
# Usage: ./scripts/submit_60_separate_jobs.sh [RUN_ID]

set -euo pipefail

# Get RUN_ID from argument or generate new one
RUN_ID="${1:-all_checkpoints_last4_separate_60_$(date +%Y%m%d_%H%M%S)}"
echo "Using RUN_ID: $RUN_ID"
echo "Submitting 60 separate jobs..."
echo ""

# Change to project directory
cd "$(dirname "$0")/.."

# Submit 60 jobs
JOB_IDS=()
for i in $(seq 0 59); do
    # Get checkpoints for this job
    CHECKPOINTS_FOR_JOB=$(python3 scripts/filter_checkpoints_for_array.py 60 $i)
    CHECKPOINT_COUNT=$(echo $CHECKPOINTS_FOR_JOB | wc -w)
    
    # Submit job with inline script
    JOB_ID=$(sbatch << EOF | grep -oP '\d+'
#!/bin/bash
#SBATCH --job-name=pythia-checkpoints-sep-${i}
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

OUTPUT_ROOT="/scratch/mahdiar/pythia-layer-time-runs"
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

# Checkpoints for this job
CHECKPOINTS="${CHECKPOINTS_FOR_JOB}"
CHECKPOINT_COUNT=${CHECKPOINT_COUNT}

echo "=========================================="
echo "Job ${i}: Processing \${CHECKPOINT_COUNT} checkpoints"
echo "Checkpoints: \${CHECKPOINTS}"
echo "Layers: 20, 21, 22, 23 (last 4)"
echo "Tasks: All 32 MTEB tasks"
echo "Expected evaluations: \${CHECKPOINT_COUNT} × 4 layers × 32 tasks = \$((CHECKPOINT_COUNT * 4 * 32))"
echo "=========================================="

# Create temporary config file
TEMP_CONFIG="/tmp/config_job_${i}_\$\$.yaml"
python3 << PYEOF
import yaml
from pathlib import Path

base_config_path = Path("configs/mteb_all_checkpoints_last4.yaml")
with base_config_path.open() as f:
    config = yaml.safe_load(f)

checkpoints_list = "${CHECKPOINTS_FOR_JOB}".split()
config['hf']['revisions'] = checkpoints_list

with open("\${TEMP_CONFIG}", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config: \${TEMP_CONFIG}")
print(f"Checkpoints: {len(checkpoints_list)}")
PYEOF

export LAYER_TIME_NUM_SHARDS=1
export LAYER_TIME_SHARD_ID=0

python -m layer_time.cli mteb-layersweep \\
  --config "\${TEMP_CONFIG}" \\
  --run-id "${RUN_ID}"

rm -f "\${TEMP_CONFIG}"
echo "Job ${i} completed successfully"
EOF
)
    
    JOB_IDS+=($JOB_ID)
    echo "Submitted job $i: Job ID $JOB_ID (${CHECKPOINT_COUNT} checkpoints)"
done

echo ""
echo "=========================================="
echo "All 60 jobs submitted!"
echo "=========================================="
echo "RUN_ID: $RUN_ID"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u mahdiar | grep pythia-checkpoints-sep"
echo ""
echo "Cancel all with:"
echo "  scancel ${JOB_IDS[*]}"
echo ""
echo "Jobs will start incrementally as GPUs become available!"
