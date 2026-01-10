# Resume Mechanism Guide

## How Resume Works

The resume mechanism works at the **individual evaluation level**:
- Each `(checkpoint, layer, task)` combination creates a `done.json` file when complete
- Location: `/scratch/mahdiar/pythia-layer-time-runs/<RUN_ID>/outputs/mteb/Pythia/410m/<checkpoint>/layer_<N>/<task>/done.json`
- When `resume=true` (default), the code **skips** any evaluation that already has a `done.json` file

## Key Points

1. **RUN_ID determines the output directory**
   - Same RUN_ID = same output directory = resume capability
   - Different RUN_ID = new output directory = fresh run

2. **Array task splitting doesn't affect resume**
   - Each checkpoint is processed independently
   - Multiple array tasks can work on different checkpoints simultaneously
   - Resume checks `done.json` files, not which array task created them

3. **Different array configurations are compatible**
   - 16-way split vs 32-way split vs 64-way split all work together
   - They just assign different checkpoints to different array task IDs
   - All write to the same output directory (if same RUN_ID)

## Your Current Situation

**Job 1856916**: `RUN_ID=all_checkpoints_last4_array_20260115_045214`
- 2 tasks running (0, 1)
- 16-way split (tasks 0-15, max 10 concurrent)

**Job 1858946**: `RUN_ID=all_checkpoints_last4_array_20260115_120312`
- 10 tasks running (0-9), 6 pending (10-15)
- 16-way split (tasks 0-15, max 10 concurrent)

## Your Options

### Option 1: Let Current Jobs Finish (Recommended if they're progressing well)
- Current jobs continue running
- Use new scripts for future runs with different RUN_ID
- No resume needed - separate runs

### Option 2: Cancel and Restart with New Scripts (Faster completion)
```bash
# Cancel current jobs
scancel 1856916 1858946

# Wait a moment for cancellation
sleep 5

# Submit new optimized script with SAME RUN_ID to resume
RUN_ID=all_checkpoints_last4_array_20260115_045214 sbatch slurm/mteb_all_checkpoints_last4_array_32tasks.sbatch
```
- New scripts will automatically skip already-completed work (via done.json)
- Faster completion due to better parallelization
- All progress is preserved

### Option 3: Run New Scripts in Parallel (Continue current + start new)
```bash
# Keep current jobs running
# Submit new script with SAME RUN_ID - it will skip completed work
RUN_ID=all_checkpoints_last4_array_20260115_045214 sbatch slurm/mteb_all_checkpoints_last4_array_32tasks.sbatch
```
- Both old and new jobs can run simultaneously
- They'll skip each other's completed work
- May cause some duplicate work but resume will handle it
- **Not recommended** - wastes resources

## Recommended Approach

**Option 2 (Cancel and Restart)** is recommended because:
1. ✅ Current jobs may not complete in 24 hours
2. ✅ New scripts will complete much faster (better parallelization)
3. ✅ All progress is preserved (done.json files)
4. ✅ Resume automatically skips completed work
5. ✅ No wasted resources

## How to Resume with New Scripts

```bash
# Use the RUN_ID from your current job
RUN_ID=all_checkpoints_last4_array_20260115_045214 sbatch slurm/mteb_all_checkpoints_last4_array_32tasks.sbatch

# Or for the other job
RUN_ID=all_checkpoints_last4_array_20260115_120312 sbatch slurm/mteb_all_checkpoints_last4_array_32tasks.sbatch
```

The new scripts will:
1. Check for existing `done.json` files
2. Skip any `(checkpoint, layer, task)` that's already done
3. Continue with remaining work
4. Process checkpoints in their assigned chunks (32-way split instead of 16-way)

## Verification

To check what's already completed:
```bash
# Count done.json files
find /scratch/mahdiar/pythia-layer-time-runs/all_checkpoints_last4_array_20260115_045214/outputs -name "done.json" | wc -l

# Check progress CSV
wc -l /scratch/mahdiar/pythia-layer-time-runs/all_checkpoints_last4_array_20260115_045214/progress.csv
```
