#!/bin/bash
#SBATCH --job-name=pythia-metrics-l40s-max
#SBATCH --account=aip-btaati
#SBATCH --nodes=8
#SBATCH --ntasks=32              # 32 tasks total (one per GPU)
#SBATCH --ntasks-per-node=4      # 4 tasks per node
#SBATCH --gres=gpu:l40s:4        # 4 GPUs per node
#SBATCH --cpus-per-task=8        # 8 CPUs per task (4 tasks × 8 CPUs = 32 CPUs per node)
#SBATCH --mem=180G
#SBATCH --time=18:00:00          # Increased walltime to comfortably finish remaining work
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

# Ensure output directory exists on scratch (has more space than project)
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
echo "SLURM_PROCID=${SLURM_PROCID:-N/A}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-N/A}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-N/A}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('num_gpus=', torch.cuda.device_count()); [print(f'gpu_{i}=', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
python -c "import layer_time.mteb_bandit_runner; print('bandit_runner_ok')"

# Run ID (reuse to resume)
RUN_ID="${RUN_ID:-metrics_410m_last8_$(date +%Y%m%d_%H%M%S)}"
echo "RUN_ID=$RUN_ID"

# This job computes representation metrics for all (checkpoint, layer) pairs
# Using bandit workflow with distributed Phase 1 (metrics computation)
# Workers will compute metrics in parallel, master will collect results and exit
echo "Running metrics computation with distributed Phase 1"
echo "Total workers: ${SLURM_NTASKS:-12}"

# Set environment variable to exit after Phase 1 (metrics-only mode)
export LAYER_TIME_METRICS_ONLY=true

# Use srun to launch 12 processes, each processing different work items
# Each task gets SLURM_PROCID (0-11) and SLURM_LOCALID (0-3 per node)
# We set CUDA_VISIBLE_DEVICES per-task using SLURM_LOCALID so each process uses a different GPU
srun bash -c '
  export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
  export LAYER_TIME_METRICS_ONLY=true
  echo "Task ${SLURM_PROCID}: SLURM_LOCALID=${SLURM_LOCALID}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, Node=${SLURMD_NODENAME:-N/A}"
  python -m layer_time.cli mteb-layersweep \
    --config configs/mteb_metrics_last4_layers6plus.yaml \
    --run-id "$RUN_ID"
'
