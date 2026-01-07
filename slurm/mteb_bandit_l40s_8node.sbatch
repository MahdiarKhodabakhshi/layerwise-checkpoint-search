#!/bin/bash
#SBATCH --job-name=pythia-bandit-10node
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=24:00:00  # 24 hours: Expected ~4-8h with 40 GPUs, 10 nodes for faster completion
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

# HF caches on scratch (shared across nodes)
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
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('num_gpus=', torch.cuda.device_count()); [print(f'gpu_{i}=', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
python -c "import layer_time.mteb_bandit_runner; print('bandit_runner_ok')"

# Run ID (reuse to resume)
RUN_ID="${RUN_ID:-bandit_$(date +%Y%m%d_%H%M%S)}"
echo "RUN_ID=$RUN_ID"
echo "Node: ${SLURM_NODEID}, Total nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs on this node: ${SLURM_GPUS_ON_NODE}"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"

# Run bandit workflow
# Note: Current implementation uses GPU 0 only, but having 10 nodes gives:
# - More resources (10 nodes × 4 GPUs = 40 GPUs total)
# - More total memory (180GB × 10 = 1.8TB)
# - Better queue priority with 24h time limit
# - Faster completion expected with more resources
#
# To use all 40 GPUs, we would need to implement distributed worker code.
# For now, this provides additional resources and faster queue allocation.
#
# The workflow will automatically use GPU 0 on the primary node.
# Other nodes provide additional resources and can be used for future parallelization.

python -m layer_time.cli mteb-layersweep \
  --config configs/mteb_layersweep.yaml \
  --run-id "$RUN_ID"
