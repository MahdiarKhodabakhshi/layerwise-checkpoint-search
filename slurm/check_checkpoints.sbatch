#!/bin/bash
#SBATCH --job-name=check-pythia-checkpoints
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/slurm_logs/check_checkpoints_%j.log
#SBATCH --error=/scratch/%u/slurm_logs/check_checkpoints_%j.log

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# Make sure log dir exists
mkdir -p /scratch/$USER/slurm_logs

# Clean environment
module --force purge
module load StdEnv/2023
module load python/3.11

# Repo + env
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

# Make src importable
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# HF caches on scratch
export HF_HOME=/scratch/$USER/hf
mkdir -p "$HF_HOME"

# Run checkpoint checker for all model sizes
echo "=========================================="
echo "Checking Pythia checkpoints on HuggingFace"
echo "=========================================="
echo ""

MODEL_SIZES="${MODEL_SIZES:-14m 70m 410m}"

for MODEL_SIZE in $MODEL_SIZES; do
    echo ""
    echo "Processing pythia-${MODEL_SIZE}..."
    python scripts/list_available_checkpoints.py \
        --model-size "$MODEL_SIZE" \
        --format json \
        --output "checkpoints_${MODEL_SIZE}.json"
done

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
for MODEL_SIZE in $MODEL_SIZES; do
    if [ -f "checkpoints_${MODEL_SIZE}.json" ]; then
        echo ""
        echo "Checkpoints for pythia-${MODEL_SIZE}:"
        python -c "import json; data=json.load(open('checkpoints_${MODEL_SIZE}.json')); cp=data.get('${MODEL_SIZE}', []); print(f'  Total: {len(cp)} checkpoints'); print(f'  First: {cp[0] if cp else \"N/A\"}'); print(f'  Last: {cp[-1] if cp else \"N/A\"}')"
    fi
done

echo ""
echo "Checkpoint files saved in:"
ls -lh checkpoints_*.json 2>/dev/null || echo "  No checkpoint files found"

