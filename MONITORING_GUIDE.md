# Monitoring Guide - Check Workflow Progress

This guide provides simple commands to check if each step of the bandit workflow is working properly while it's running.

## Quick Status Check

```bash
# Replace RUN_ID with your actual run ID (e.g., bandit_20260104_131923)
RUN_ID="bandit_20260104_131923"

# Quick overview
echo "=== Quick Status ==="
echo "Log file exists: $(test -f runs/${RUN_ID}/logs/bandit_runner.log && echo 'YES' || echo 'NO')"
echo "Corpus cached: $(test -f runs/${RUN_ID}/cache/representation_corpus.json && echo 'YES' || echo 'NO')"
echo "Bandit state exists: $(test -f runs/${RUN_ID}/bandit_state.json && echo 'YES' || echo 'NO')"
echo "Results file exists: $(test -f runs/${RUN_ID}/bandit_results.json && echo 'YES' || echo 'NO')"
```

---

## Step-by-Step Monitoring

### Step 1: Check if Job is Running

```bash
# Check SLURM job status
squeue -u mahdiar

# Check latest log file
ls -lht /scratch/mahdiar/slurm_logs/*bandit*.log | head -1
```

### Step 2: Monitor Log File (Real-Time)

```bash
# Get the latest log file
LOG_FILE=$(ls -t /scratch/mahdiar/slurm_logs/*bandit*.log | head -1)

# Watch log in real-time (press Ctrl+C to stop)
tail -f "$LOG_FILE"

# Or check last 50 lines
tail -50 "$LOG_FILE"
```

### Step 3: Check Step 1 - Corpus Building

```bash
RUN_ID="bandit_20260104_131923"  # Replace with your run ID

# Check if corpus file exists and its size
ls -lh runs/${RUN_ID}/cache/representation_corpus.json 2>/dev/null || echo "Corpus not created yet"

# Check corpus size (number of examples)
python3 -c "
import json
from pathlib import Path
corpus_path = Path('runs/${RUN_ID}/cache/representation_corpus.json')
if corpus_path.exists():
    corpus = json.load(corpus_path.open())
    print(f'Corpus size: {len(corpus)} examples')
    if len(corpus) > 0:
        print(f'First example (first 100 chars): {corpus[0][:100]}...')
else:
    print('Corpus file does not exist yet')
"

# Check log for corpus building messages
grep -i "corpus" runs/${RUN_ID}/logs/bandit_runner.log 2>/dev/null | tail -5
```

**Expected output:**
- Corpus file should exist after ~10-15 minutes
- Corpus size should be > 0 (typically thousands of examples)
- Log should show: "Corpus size: X examples"

---

### Step 4: Check Step 2 - Metrics Pre-computation

```bash
RUN_ID="bandit_20260104_131923"  # Replace with your run ID

# Check if metrics cache directory exists
ls -ld runs/${RUN_ID}/cache/metrics/*/ 2>/dev/null || echo "Metrics cache not created yet"

# Count how many metric files have been computed
find runs/${RUN_ID}/cache/metrics -name "*.json" 2>/dev/null | wc -l

# List metric files for a specific model/checkpoint
ls -lh runs/${RUN_ID}/cache/metrics/410m/main/ 2>/dev/null | head -10

# Check a sample metric file
python3 -c "
import json
from pathlib import Path
import glob

metric_files = list(Path('runs/${RUN_ID}/cache/metrics').glob('**/layer_*.json'))
if metric_files:
    sample = json.load(metric_files[0].open())
    print(f'Found {len(metric_files)} metric files')
    print(f'Sample metrics (first file): {list(sample.keys())}')
    print(f'Values: {sample}')
else:
    print('No metric files found yet')
"

# Check log for metrics computation
grep -i "metric\|layer\|Pre-computing" runs/${RUN_ID}/logs/bandit_runner.log 2>/dev/null | tail -10
```

**Expected output:**
- Metrics directory should appear after corpus is built
- Metric files should be created: `layer_0.json`, `layer_1.json`, etc.
- Each file contains metrics like: `prompt_entropy`, `dataset_entropy`, `curvature`, `effective_rank`

---

### Step 5: Check Step 3 - Bandit Loop (Evaluations)

```bash
RUN_ID="bandit_20260104_131923"  # Replace with your run ID

# Check if bandit state file exists (created after first evaluation)
ls -lh runs/${RUN_ID}/bandit_state.json 2>/dev/null || echo "Bandit state not created yet (evaluations not started)"

# Check bandit state (number of evaluations so far)
python3 -c "
import json
from pathlib import Path

state_path = Path('runs/${RUN_ID}/bandit_state.json')
if state_path.exists():
    state = json.load(state_path.open())
    trajectory = state.get('trajectory', [])
    arm_counts = state.get('arm_counts', {})
    print(f'Evaluations completed: {len(trajectory)}')
    print(f'Arms evaluated: {len(arm_counts)}')
    if trajectory:
        print(f'Last evaluation: {trajectory[-1]}')
else:
    print('Bandit state file does not exist yet')
"

# Check MTEB output directories (actual evaluations)
find runs/${RUN_ID}/outputs -type d -name "*layer_*" 2>/dev/null | wc -l

# List recent evaluation outputs
ls -lht runs/${RUN_ID}/outputs/mteb/*/ 2>/dev/null | head -10

# Check log for evaluation messages
grep -i "evaluat\|selected\|reward\|budget" runs/${RUN_ID}/logs/bandit_runner.log 2>/dev/null | tail -15
```

**Expected output:**
- Bandit state file should appear after first evaluation
- Trajectory length should increase with each evaluation
- MTEB output directories should be created for each evaluation

---

### Step 6: Check Final Results

```bash
RUN_ID="bandit_20260104_131923"  # Replace with your run ID

# Check if results file exists
ls -lh runs/${RUN_ID}/bandit_results.json 2>/dev/null || echo "Results file not created yet (workflow not complete)"

# View results
cat runs/${RUN_ID}/bandit_results.json 2>/dev/null | python3 -m json.tool

# Check completion message in log
tail -20 runs/${RUN_ID}/logs/bandit_runner.log | grep -i "complete\|best arm\|finished"
```

**Expected output:**
- Results file should exist when workflow completes
- Contains: `best_arm`, `trajectory_length`, `arms_evaluated`, `budget_used`
- Log should show: "Bandit workflow complete"

---

## All-in-One Monitoring Script

Save this as `check_progress.sh`:

```bash
#!/bin/bash
# Usage: ./check_progress.sh <RUN_ID>

RUN_ID="${1:-bandit_20260104_131923}"

echo "=== Bandit Workflow Progress Check ==="
echo "Run ID: $RUN_ID"
echo ""

# Step 1: Corpus
echo "Step 1 - Corpus Building:"
if [ -f "runs/${RUN_ID}/cache/representation_corpus.json" ]; then
    SIZE=$(python3 -c "import json; print(len(json.load(open('runs/${RUN_ID}/cache/representation_corpus.json'))))" 2>/dev/null)
    echo "  ✅ Corpus built: $SIZE examples"
else
    echo "  ⏳ Corpus not built yet"
fi

# Step 2: Metrics
echo ""
echo "Step 2 - Metrics Pre-computation:"
METRIC_COUNT=$(find runs/${RUN_ID}/cache/metrics -name "*.json" 2>/dev/null | wc -l)
if [ "$METRIC_COUNT" -gt 0 ]; then
    echo "  ✅ Metrics computed: $METRIC_COUNT files"
else
    echo "  ⏳ Metrics not computed yet"
fi

# Step 3: Bandit State
echo ""
echo "Step 3 - Bandit Evaluations:"
if [ -f "runs/${RUN_ID}/bandit_state.json" ]; then
    EVALS=$(python3 -c "import json; print(len(json.load(open('runs/${RUN_ID}/bandit_state.json')).get('trajectory', [])))" 2>/dev/null)
    echo "  ✅ Evaluations completed: $EVALS"
else
    echo "  ⏳ Evaluations not started yet"
fi

# Step 4: Results
echo ""
echo "Step 4 - Final Results:"
if [ -f "runs/${RUN_ID}/bandit_results.json" ]; then
    echo "  ✅ Workflow complete!"
    python3 -c "import json; r=json.load(open('runs/${RUN_ID}/bandit_results.json')); print(f\"  Best arm: {r.get('best_arm')}\"); print(f\"  Budget used: {r.get('budget_used')}\")" 2>/dev/null
else
    echo "  ⏳ Workflow still running"
fi

# Latest log entries
echo ""
echo "Latest log entries:"
tail -5 "runs/${RUN_ID}/logs/bandit_runner.log" 2>/dev/null || echo "  Log file not found"
```

Make it executable and run:
```bash
chmod +x check_progress.sh
./check_progress.sh bandit_20260104_131923
```

---

## Common Issues and Checks

### Issue: Corpus is empty (0 examples)
```bash
# Check log for errors
grep -i "error\|warning\|failed" runs/${RUN_ID}/logs/bandit_runner.log | tail -10
```

### Issue: Metrics not being computed
```bash
# Check if embeddings cache exists
ls -lh runs/${RUN_ID}/cache/embeddings/*/main/ 2>/dev/null | head -10

# Check log for metric computation errors
grep -i "metric\|embedding\|error" runs/${RUN_ID}/logs/bandit_runner.log | tail -10
```

### Issue: Evaluations not starting
```bash
# Check bandit initialization
grep -i "bandit\|initialize\|arm" runs/${RUN_ID}/logs/bandit_runner.log | tail -10

# Check if metrics were computed for all arms
python3 -c "
from pathlib import Path
import json
metric_files = list(Path('runs/${RUN_ID}/cache/metrics').glob('**/layer_*.json'))
print(f'Total metric files: {len(metric_files)}')
"
```

### Issue: Job appears stuck
```bash
# Check if process is actually running
squeue -u mahdiar

# Check latest log activity (should update every few minutes)
tail -1 runs/${RUN_ID}/logs/bandit_runner.log

# Check system resources (if on compute node)
# ssh to compute node and check GPU usage
nvidia-smi
```

---

## Quick Reference

| Step | File to Check | Command |
|------|---------------|---------|
| **Step 1: Corpus** | `cache/representation_corpus.json` | `wc -l runs/${RUN_ID}/cache/representation_corpus.json` |
| **Step 2: Metrics** | `cache/metrics/*/layer_*.json` | `find runs/${RUN_ID}/cache/metrics -name "*.json" \| wc -l` |
| **Step 3: Evaluations** | `bandit_state.json` | `cat runs/${RUN_ID}/bandit_state.json \| grep trajectory` |
| **Step 4: Results** | `bandit_results.json` | `cat runs/${RUN_ID}/bandit_results.json` |
| **Live Log** | `logs/bandit_runner.log` | `tail -f runs/${RUN_ID}/logs/bandit_runner.log` |

---

## Time Estimates

- **Step 1 (Corpus)**: 10-15 minutes
- **Step 2 (Metrics)**: Depends on model size and number of layers
  - 14m: ~5-10 minutes
  - 70m: ~20-30 minutes  
  - 410m: ~1-2 hours
- **Step 3 (Bandit Loop)**: Depends on budget
  - Each evaluation: ~5-15 minutes (32 tasks)
  - Budget 100: ~8-25 hours total
- **Step 4 (Results)**: < 1 minute

**Total estimated time for budget=100, 410m model: ~10-30 hours**

