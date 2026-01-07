#!/bin/bash
#SBATCH --job-name=pythia-bandit-10node
#SBATCH --account=aip-btaati
#SBATCH --nodes=10
#SBATCH --ntasks=40  # One task per GPU (10 nodes × 4 GPUs = 40 tasks)
#SBATCH --ntasks-per-node=4  # 4 tasks per node (one per GPU)
#SBATCH --gres=gpu:l40s:4  # 4 GPUs per node (1 GPU per task)
#SBATCH --cpus-per-task=8  # 8 CPUs per task (4 tasks × 8 CPUs = 32 CPUs per node)
#SBATCH --mem=180G
#SBATCH --time=24:00:00  # 24 hours: Expected ~4-8h with 40 GPUs in parallel
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.log
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.log

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

mkdir -p /scratch/$USER/slurm_logs

module --force purge
module load StdEnv/2023

# Load Slurm explicitly so srun exists inside the job environment
module load slurm/killarney/24.05.7

module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0

cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

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
echo "Total nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total tasks: ${SLURM_NTASKS}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"

# Distributed mode: Use srun to launch workers across all GPUs
# Each task gets: SLURM_PROCID (0-39), SLURM_LOCALID (0-3), SLURM_NTASKS (40)
# Worker 0 is the master, others are workers
echo "Launching ${SLURM_NTASKS} distributed workers (one per GPU)"
echo "Worker ID: ${SLURM_PROCID:-0}, Local ID: ${SLURM_LOCALID:-0}"

# Use srun to launch workers (SLURM will distribute across nodes automatically)
srun python -m layer_time.cli mteb-layersweep \
  --config configs/mteb_layersweep.yaml \
  --run-id "$RUN_ID"
