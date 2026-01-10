#!/bin/bash
# Submit separate SLURM jobs for faster start time with backfilling
# Each job can start independently as GPUs become available
# Usage: ./scripts/submit_separate_jobs.sh [NUM_JOBS] [RUN_ID]
#   NUM_JOBS: Number of jobs to submit (default: 100, recommended: 100-120)
#   RUN_ID: Optional run ID to resume

set -euo pipefail

# Get parameters
NUM_JOBS="${1:-100}"
RUN_ID="${2:-all_checkpoints_last4_separate_${NUM_JOBS}_$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "Submitting $NUM_JOBS separate jobs"
echo "RUN_ID: $RUN_ID"
echo "=========================================="
echo ""

# Validate
if [ "$NUM_JOBS" -lt 60 ]; then
    echo "Warning: Less than 60 jobs means you won't fully utilize your 60 GPU limit"
    echo "Consider using at least 100 jobs for better backfilling"
fi

# Change to project directory
cd "$(dirname "$0")/.."

TOTAL_CHECKPOINTS=154
CHECKPOINTS_PER_JOB=$((TOTAL_CHECKPOINTS / NUM_JOBS))
REMAINDER=$((TOTAL_CHECKPOINTS % NUM_JOBS))

echo "Configuration:"
echo "  • Total checkpoints: $TOTAL_CHECKPOINTS"
echo "  • Number of jobs: $NUM_JOBS"
echo "  • Checkpoints per job: $CHECKPOINTS_PER_JOB"$([ $REMAINDER -gt 0 ] && echo "-$((CHECKPOINTS_PER_JOB + 1))" || echo "")
echo "  • First $((NUM_JOBS - REMAINDER)) jobs: $CHECKPOINTS_PER_JOB checkpoints"
[ $REMAINDER -gt 0 ] && echo "  • Last $REMAINDER jobs: $((CHECKPOINTS_PER_JOB + 1)) checkpoints"
echo "  • Max concurrent GPUs: 60"
echo "  • Jobs in queue: $NUM_JOBS (first 60 start immediately)"
echo "  • Backfill capacity: $((NUM_JOBS - 60)) jobs waiting"
echo ""

# Submit jobs
JOB_IDS=()
for i in $(seq 0 $((NUM_JOBS - 1))); do
    # Get checkpoints for this job
    CHECKPOINTS_FOR_JOB=$(python3 scripts/filter_checkpoints_for_array.py $NUM_JOBS $i)
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
    if [ $((i % 10)) -eq 0 ] || [ $i -eq $((NUM_JOBS - 1)) ]; then
        echo "Submitted job $i/$((NUM_JOBS - 1)): Job ID $JOB_ID (${CHECKPOINT_COUNT} checkpoints)"
    fi
done

echo ""
echo "=========================================="
echo "All $NUM_JOBS jobs submitted!"
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
echo "Benefits of $NUM_JOBS jobs:"
echo "  • First 60 jobs start immediately (use all GPUs)"
echo "  • Remaining $((NUM_JOBS - 60)) jobs wait in queue"
echo "  • As jobs finish, waiting jobs start automatically"
echo "  • GPUs stay busy, no idle time!"
echo "  • Better load balancing than 60 jobs"
