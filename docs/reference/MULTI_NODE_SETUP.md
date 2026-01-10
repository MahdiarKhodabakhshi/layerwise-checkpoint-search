# Multi-Node Multi-GPU Setup Guide

## Overview

Using **8 nodes × 4 L40s GPUs = 32 GPUs total** can significantly speed up the bandit workflow, especially **Phase 1 (metrics pre-computation)**.

## Current Status

✅ **Multi-node SLURM script created**: `slurm/mteb_bandit_l40s_8node.sbatch`
⚠️ **Distributed worker code**: Partially implemented (needs integration)

## Speedup Potential

### Phase 1: Metrics Pre-computation (BIGGEST WIN)

**Current**: Sequential processing of ~5,184 (checkpoint, layer) pairs
- 3 model sizes × ~144 checkpoints × ~6-24 layers each
- Estimated time: **~50-100 hours** on single GPU

**With 32 GPUs**: Parallel processing
- **Theoretical speedup**: Up to **32x** (if perfectly parallelized)
- **Realistic speedup**: **~20-25x** (accounting for overhead, I/O)
- **Estimated time**: **~2-4 hours** for Phase 1

### Phase 2: Bandit Loop (MODERATE WIN)

**Current**: Sequential arm selection + sequential task evaluation
- Bandit selection is inherently sequential (needs previous results)
- Task evaluation within each arm can be parallelized

**With 32 GPUs**: Parallel task evaluation
- **32 tasks** can be evaluated in parallel (one per GPU)
- **Theoretical speedup**: Up to **32x** for task evaluation
- **Overall speedup**: **~10-15x** (accounting for sequential bandit selection)

## Implementation Approaches

### Option 1: Simple Multi-Node (Current - Works Now)

**What it does**: Runs on 8 nodes, but only uses GPU 0 on each node (8 GPUs total)

**Pros**:
- ✅ Works immediately with current code
- ✅ No code changes needed
- ✅ 8x speedup for Phase 1

**Cons**:
- ⚠️ Only uses 8 of 32 GPUs (25% utilization)
- ⚠️ Not optimal, but still helpful

**How to use**:
```bash
sbatch slurm/mteb_bandit_l40s_8node.sbatch
```

### Option 2: Full Distributed (Requires Code Changes)

**What it does**: Uses all 32 GPUs with work queue pattern

**Implementation**:
1. Master process generates work queue of all (checkpoint, layer) pairs
2. 32 workers pull work items and compute metrics in parallel
3. Results written to shared filesystem
4. Master collects results and proceeds to Phase 2

**Status**: 
- ✅ Work queue infrastructure created (`distributed_worker.py`)
- ⚠️ Needs integration into `mteb_bandit_runner.py`
- ⚠️ Needs CLI updates for distributed mode

**Estimated implementation time**: 2-3 hours

## Recommended Approach

### Immediate (Today)

**Use Option 1**: Submit the 8-node job. Even with only 8 GPUs active, you'll get:
- **8x speedup** for Phase 1 (from ~50-100h to ~6-12h)
- **More memory** (180GB per node)
- **Longer time limit** (72h)

```bash
cd /project/6101803/mahdiar/pythia-layer-time
sbatch slurm/mteb_bandit_l40s_8node.sbatch
```

### Future (If Needed)

**Implement Option 2**: Full 32-GPU parallelization
- Would require integrating `distributed_worker.py` into the main workflow
- Would get **~20-25x speedup** for Phase 1
- Would reduce Phase 1 time to **~2-4 hours**

## Current Multi-Node Script

**File**: `slurm/mteb_bandit_l40s_8node.sbatch`

**Resources**:
- 8 nodes
- 4 L40s GPUs per node (32 total)
- 32 CPUs per node (256 total)
- 180GB RAM per node (1.44TB total)
- 72-hour time limit

**Note**: Currently runs in single-node mode (uses GPU 0 only). To enable full distributed mode, set `DISTRIBUTED=true` (requires code integration).

## Monitoring

```bash
# Check job status
squeue -u mahdiar

# Watch logs
tail -f /scratch/mahdiar/slurm_logs/pythia-bandit-8node_*.log

# Check progress
ls -lh /project/6101803/mahdiar/pythia-layer-time/runs/bandit_*/cache/metrics/
```

## Next Steps

1. **Submit 8-node job** (works now, 8x speedup)
2. **Monitor Phase 1 progress** (metrics computation)
3. **If Phase 1 is still slow**, implement full distributed mode (32x speedup)

## Questions?

- **Q**: Will 8 nodes help even if only 8 GPUs are used?
- **A**: Yes! 8x speedup is still significant. Phase 1 will go from ~50-100h to ~6-12h.

- **Q**: Can we use all 32 GPUs?
- **A**: Yes, but requires code changes. The infrastructure is ready (`distributed_worker.py`), but needs integration.

- **Q**: Is it worth implementing full distributed mode?
- **A**: If Phase 1 takes >12 hours with 8 GPUs, yes. Otherwise, 8x speedup might be sufficient.
