# Quick Start Guide - Exact Steps to Run

## 🚀 Fastest Setup (Bandit Workflow with H100)

### Step 1: Edit Config (2 minutes)
```bash
cd /project/6101803/mahdiar/pythia-layer-time
nano configs/mteb_layersweep.yaml
```

Change these lines:
```yaml
bandit:
  enabled: true    # ← Change from false to true
  budget: 100      # ← Number of evaluations (adjust as needed)

hf:
  model_sizes: ["410m"]  # ← Start with one size for testing

embedding:
  dtype: "float16"  # ← Change from "auto" for faster computation
```

### Step 2: Submit Job (10 seconds)
```bash
sbatch slurm/mteb_bandit.sbatch
```

### Step 3: Monitor (optional)
```bash
# Check status
squeue -u mahdiar

# View logs (replace JOBID)
tail -f /scratch/mahdiar/slurm_logs/pythia-bandit_<JOBID>.log
```

### Step 4: Check Results (after completion)
```bash
# View best arm
cat runs/bandit_*/bandit_results.json

# View progress
cat runs/bandit_*/bandit_progress.csv
```

**That's it!** The job will run and save results to `runs/bandit_<timestamp>/`

---

## 🎯 Complete Evaluation (Brute-Force with Array Jobs)

### Step 1: Edit Config (2 minutes)
```bash
cd /project/6101803/mahdiar/pythia-layer-time
nano configs/mteb_layersweep.yaml
```

Ensure:
```yaml
bandit:
  enabled: false  # ← Keep false for brute-force

hf:
  model_sizes: ["410m"]  # ← Your target size

embedding:
  dtype: "float16"
```

### Step 2: Submit Array Job (10 seconds)
```bash
# For 410m model (4 parallel jobs - recommended)
sbatch slurm/mteb_410m_array.sbatch

# For 70m model (2 parallel jobs)
sbatch slurm/mteb_70m_array.sbatch
```

### Step 3: Monitor
```bash
squeue -u mahdiar
```

### Step 4: Collect Results (after completion)
```bash
# Replace RUN_ID with your actual run ID
python -m layer_time.analysis.collect_results \
  --run-dir runs/410m_* \
  --out runs/410m_*/summary.csv
```

---

## 📋 Command Cheat Sheet

```bash
# Navigate to project
cd /project/6101803/mahdiar/pythia-layer-time

# Edit config
nano configs/mteb_layersweep.yaml

# Submit bandit job (H100, fastest)
sbatch slurm/mteb_bandit.sbatch

# Submit brute-force array job (410m, 4 shards)
sbatch slurm/mteb_410m_array.sbatch

# Check job status
squeue -u mahdiar

# View logs
tail -f /scratch/mahdiar/slurm_logs/pythia-*.log

# Resume with same RUN_ID (if interrupted)
RUN_ID=bandit_20250101_120000 sbatch slurm/mteb_bandit.sbatch
```

---

## ⚡ GPU Recommendations

**For Maximum Speed:**
- **Bandit workflow**: Use `slurm/mteb_bandit.sbatch` (H100, 1 GPU)
- **Brute-force**: Use array jobs with H100s if available

**For Maximum Parallelism:**
- Use array jobs (`mteb_410m_array.sbatch`) - 4 parallel jobs
- Can run multiple model sizes simultaneously (separate submissions)

**GPU Options:**
- **H100**: Fastest (~2-3x faster than L40s) - Recommended
- **L40s**: Good for parallel jobs (more available)

To change GPU type, edit the `.sbatch` file:
```bash
# Change this line:
#SBATCH --gres=gpu:h100:1  →  #SBATCH --gres=gpu:l40s:1
```

---

## 🔧 Common Configurations

### Test Run (Fast)
```yaml
bandit:
  enabled: true
  budget: 20  # Small budget for testing

hf:
  model_sizes: ["14m"]  # Smallest model
  revisions: ["main"]
```

### Production Run (Bandit)
```yaml
bandit:
  enabled: true
  budget: 200  # Larger budget

hf:
  model_sizes: ["410m"]
  revisions: ["main"]  # Add more checkpoints if available
```

### Complete Evaluation (Brute-Force)
```yaml
bandit:
  enabled: false

hf:
  model_sizes: ["410m"]
  revisions: ["main"]
```

---

## 📖 Full Documentation

See `SLURM_RUN_GUIDE.md` for:
- Detailed explanations
- Troubleshooting
- Performance optimization
- Advanced configurations

