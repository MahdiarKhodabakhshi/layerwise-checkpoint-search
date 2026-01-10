# SLURM Run Guide - Complete Step-by-Step Instructions

## Overview

This guide provides exact steps to run the project on SLURM, with recommendations for maximum speed using available GPUs.

## Prerequisites

1. **Virtual Environment**: The project uses `lbl` virtual environment located at `/project/6101803/mahdiar/pythia-layer-time/lbl/`
2. **Config File**: Edit `configs/mteb_layersweep.yaml` before running
3. **GPU Resources**: Available GPUs: L40s (good), H100 (fastest - recommended)

## Quick Start: Two Running Modes

### Option 1: Bandit Workflow (Recommended for Budgeted Evaluation)
- **Use Case**: When you want to evaluate a limited number of (checkpoint, layer) pairs intelligently
- **Best For**: Exploring many checkpoints with limited compute budget
- **Speed**: Faster overall (evaluates fewer combinations)
- **GPU**: 1x H100 recommended (faster GPUs = faster metrics computation)

### Option 2: Brute-Force Sweep (Recommended for Complete Evaluation)
- **Use Case**: When you want to evaluate ALL (checkpoint, layer, task) combinations
- **Best For**: Final evaluation or when you need complete results
- **Speed**: Slower but complete (evaluates all combinations)
- **GPU**: Array jobs with multiple L40s or H100s for parallelization

---

## Step-by-Step: Bandit Workflow

### Step 1: Configure the Bandit Workflow

Edit `configs/mteb_layersweep.yaml`:

```yaml
bandit:
  enabled: true  # Enable bandit mode
  alpha: 1.0     # Exploration parameter (1.0 is good default)
  budget: 100    # Number of evaluations (adjust based on your needs)
  baseline_checkpoint: "main"

metrics:
  corpus_max_examples_per_task: 1000  # Limit for faster metrics (optional)

# Configure model sizes and checkpoints you want to explore
hf:
  model_sizes: ["410m"]  # Start with one size for testing
  revisions: ["main"]    # Add more checkpoints here if available

# Reduce batch size if you get OOM errors
embedding:
  batch_size: 32  # Reduce from 64 if needed
  device: "cuda"
  dtype: "float16"  # Use float16 for faster computation
```

### Step 2: Submit Bandit Job

**Option A: Use H100 (Fastest - Recommended)**
```bash
cd /project/6101803/mahdiar/pythia-layer-time

# Submit the job
sbatch slurm/mteb_bandit.sbatch

# Or set custom RUN_ID to resume:
RUN_ID=bandit_20250101_120000 sbatch slurm/mteb_bandit.sbatch
```

**Option B: Use L40 (Cheaper, Slower)**
Edit `slurm/mteb_bandit.sbatch` and change:
```bash
#SBATCH --gres=gpu:h100:1  →  #SBATCH --gres=gpu:l40s:1
#SBATCH --mem=120G  →  #SBATCH --mem=64G
```

Then submit:
```bash
sbatch slurm/mteb_bandit.sbatch
```

### Step 3: Monitor the Job

```bash
# Check job status
squeue -u mahdiar

# Watch logs (replace JOBID with your job ID)
tail -f /scratch/mahdiar/slurm_logs/pythia-bandit_<JOBID>.log

# Or check run directory
tail -f /project/6101803/mahdiar/pythia-layer-time/runs/bandit_*/logs/bandit_runner.log
```

### Step 4: Check Results

After completion, results are in:
```
runs/bandit_<RUN_ID>/
├── bandit_results.json      # Best arm and summary
├── bandit_progress.csv      # Progress tracking
├── bandit_state.json        # Bandit state (for resume)
└── cache/                   # Cached embeddings and metrics
```

View best arm:
```bash
cat runs/bandit_<RUN_ID>/bandit_results.json
```

### Step 5: Resume if Needed

If job was interrupted, resume with same RUN_ID:
```bash
RUN_ID=bandit_20250101_120000 sbatch slurm/mteb_bandit.sbatch
```

---

## Step-by-Step: Brute-Force Sweep (Complete Evaluation)

### Step 1: Configure Brute-Force Mode

Edit `configs/mteb_layersweep.yaml`:

```yaml
bandit:
  enabled: false  # Disable bandit (use brute-force)

hf:
  model_sizes: ["14m", "70m", "410m"]  # All sizes
  revisions: ["main"]

embedding:
  batch_size: 64
  device: "cuda"
  dtype: "float16"
```

### Step 2: Submit Array Jobs (Parallel Execution)

**For 70m model (2 parallel jobs):**
```bash
cd /project/6101803/mahdiar/pythia-layer-time

# Submit array job (2 shards)
sbatch slurm/mteb_70m_array.sbatch

# With custom RUN_ID:
RUN_ID=70m_20250101_120000 sbatch slurm/mteb_70m_array.sbatch
```

**For 410m model (4 parallel jobs - recommended for speed):**
```bash
# Submit array job (4 shards)
sbatch slurm/mteb_410m_array.sbatch

# With custom RUN_ID:
RUN_ID=410m_20250101_120000 sbatch slurm/mteb_410m_array.sbatch
```

**For single job (smaller models or testing):**
```bash
sbatch slurm/mteb_layersweep.sbatch
```

### Step 3: Monitor Array Jobs

```bash
# Check all jobs in array
squeue -u mahdiar

# View logs for specific array task
tail -f /scratch/mahdiar/slurm_logs/pythia-mteb-410m_<ARRAY_JOB_ID>_<TASK_ID>.log

# Check progress
tail -f /project/6101803/mahdiar/pythia-layer-time/runs/<RUN_ID>/logs/runner_shard*.log
```

### Step 4: Check Results

Results are in:
```
runs/<RUN_ID>/
├── progress.csv (or progress_shard*.csv for array jobs)
├── outputs/mteb/... (MTEB results)
└── logs/runner*.log
```

Collect all results:
```bash
python -m layer_time.analysis.collect_results \
  --run-dir runs/<RUN_ID> \
  --out runs/<RUN_ID>/summary.csv
```

---

## Recommended GPU Configurations

### For Bandit Workflow

| Model Size | GPU Type | Memory | CPUs | Time Limit | Recommendation |
|------------|----------|--------|------|------------|----------------|
| 14m        | H100     | 120GB  | 16   | 12h        | Fastest        |
| 70m        | H100     | 120GB  | 16   | 24h        | Fastest        |
| 410m       | H100     | 120GB  | 16   | 24h        | Fastest        |
| 410m       | L40s     | 96GB   | 8    | 36h        | Cheaper option |

**Edit `slurm/mteb_bandit.sbatch`** to change GPU type.

### For Brute-Force Sweep

| Model Size | Array Jobs | GPU per Job | Memory | CPUs | Recommendation |
|------------|------------|-------------|--------|------|----------------|
| 14m        | 1          | H100        | 120GB  | 16   | Single job OK  |
| 70m        | 2-4        | L40s/H100   | 64GB   | 8    | 2 shards good  |
| 410m       | 4-8        | L40s/H100   | 96GB   | 8    | 4 shards good  |

**Edit array job scripts** to adjust shard count:
- `slurm/mteb_70m_array.sbatch`: Change `#SBATCH --array=0-1` to `#SBATCH --array=0-3` for 4 shards
- `slurm/mteb_410m_array.sbatch`: Change `#SBATCH --array=0-3` to `#SBATCH --array=0-7` for 8 shards

---

## Detailed Steps for First-Time Setup

### 1. Navigate to Project Directory
```bash
cd /project/6101803/mahdiar/pythia-layer-time
```

### 2. Verify Virtual Environment
```bash
# Check if lbl environment exists
ls -la lbl/bin/activate

# If missing, you may need to recreate it (see README.md)
```

### 3. Test Configuration
```bash
# Activate environment
source lbl/bin/activate

# Test import
python -c "import layer_time; print('OK')"

# Check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 4. Edit Configuration File
```bash
# Edit config
nano configs/mteb_layersweep.yaml

# Or use your preferred editor
vim configs/mteb_layersweep.yaml
```

### 5. Submit Job
Choose based on your mode:

**Bandit mode:**
```bash
sbatch slurm/mteb_bandit.sbatch
```

**Brute-force mode:**
```bash
# For 410m (recommended for speed with 4 parallel jobs)
sbatch slurm/mteb_410m_array.sbatch
```

---

## Monitoring and Debugging

### Check Job Status
```bash
# Your jobs
squeue -u mahdiar

# Pending jobs
pending  # (if you have the alias)

# Running jobs  
running  # (if you have the alias)
```

### View Logs
```bash
# SLURM logs
tail -f /scratch/mahdiar/slurm_logs/pythia-*.log

# Application logs
tail -f runs/<RUN_ID>/logs/*.log
```

### Check GPU Usage (if on compute node)
```bash
# If you have interactive access to compute node
nvidia-smi
```

### Check Progress
```bash
# For bandit workflow
tail -f runs/<RUN_ID>/bandit_progress.csv

# For brute-force
tail -f runs/<RUN_ID>/progress.csv
```

### Common Issues

**Issue: Out of Memory (OOM)**
- Solution: Reduce `batch_size` in config (try 32 or 16)
- Solution: Use `dtype: "float16"` instead of "auto"

**Issue: Job Timeout**
- Solution: Increase `--time` in SLURM script
- Solution: Use array jobs to split work

**Issue: Module Not Found**
- Solution: Ensure virtual environment is activated in script
- Solution: Check `PYTHONPATH` is set correctly

---

## Performance Recommendations

### Maximum Speed Setup

**For Bandit Workflow:**
1. Use **H100 GPU** (fastest)
2. Set `dtype: "float16"` in config
3. Set `batch_size: 64` (or maximum that fits)
4. Use `corpus_max_examples_per_task: 1000` for faster metrics
5. Start with smaller `budget` (e.g., 50) to test, then increase

**For Brute-Force:**
1. Use **array jobs** with 4-8 shards (depending on model size)
2. Use **H100 GPUs** if available, else L40s
3. Set `dtype: "float16"`
4. Set `batch_size: 64`
5. Run multiple model sizes in parallel (separate array jobs)

### Resource Optimization

**Memory:**
- 14m/70m: 64GB is enough
- 410m: 96-120GB recommended

**GPU:**
- H100: ~2-3x faster than L40s
- L40s: Good for parallel array jobs (more available)

**CPU:**
- 8 CPUs: Sufficient for most cases
- 16 CPUs: Better for larger batch sizes

---

## Example: Complete Bandit Workflow Run

```bash
# 1. Navigate to project
cd /project/6101803/mahdiar/pythia-layer-time

# 2. Edit config to enable bandit mode
nano configs/mteb_layersweep.yaml
# Set: bandit.enabled: true, budget: 100, model_sizes: ["410m"]

# 3. Submit job
sbatch slurm/mteb_bandit.sbatch

# 4. Monitor (replace JOBID)
watch -n 30 'squeue -u mahdiar'
tail -f /scratch/mahdiar/slurm_logs/pythia-bandit_<JOBID>.log

# 5. After completion, check results
cat runs/bandit_*/bandit_results.json
```

---

## Example: Complete Brute-Force Run (410m, 4 shards)

```bash
# 1. Navigate to project
cd /project/6101803/mahdiar/pythia-layer-time

# 2. Edit config (bandit.enabled: false)
nano configs/mteb_layersweep.yaml

# 3. Submit array job (4 parallel jobs)
RUN_ID=410m_complete_$(date +%Y%m%d_%H%M%S) sbatch slurm/mteb_410m_array.sbatch

# 4. Monitor
squeue -u mahdiar

# 5. After completion, collect results
python -m layer_time.analysis.collect_results \
  --run-dir runs/410m_complete_* \
  --out runs/410m_complete_*/summary.csv
```

---

## Quick Reference

### Key Files
- Config: `configs/mteb_layersweep.yaml`
- Bandit script: `slurm/mteb_bandit.sbatch`
- Brute-force scripts: `slurm/mteb_*_array.sbatch`
- Results: `runs/<RUN_ID>/`

### Key Commands
```bash
# Submit bandit job
sbatch slurm/mteb_bandit.sbatch

# Submit brute-force (410m, 4 shards)
sbatch slurm/mteb_410m_array.sbatch

# Monitor jobs
squeue -u mahdiar

# View logs
tail -f /scratch/mahdiar/slurm_logs/pythia-*.log

# Collect results
python -m layer_time.analysis.collect_results --run-dir runs/<RUN_ID> --out runs/<RUN_ID>/summary.csv
```

### Environment Variables
- `RUN_ID`: Set to resume existing run
- `LAYER_TIME_ONLY_SIZE`: For array jobs (set in script)
- `LAYER_TIME_NUM_SHARDS`: For array jobs (set in script)
- `LAYER_TIME_SHARD_ID`: For array jobs (set automatically by SLURM)

---

## Summary

**For fastest execution:**
1. Use **H100 GPUs** when available
2. Use **bandit workflow** for budgeted evaluation
3. Use **array jobs** for brute-force (4-8 shards)
4. Set `dtype: "float16"` and optimize `batch_size`
5. Start with smaller test runs, then scale up

**For maximum parallelism:**
1. Use **array jobs** with multiple shards
2. Run different model sizes in separate array jobs
3. Use **L40s** if H100s are limited (more available)

