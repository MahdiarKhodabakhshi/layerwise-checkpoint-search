# Time Limit Explanation: Multi-GPU Jobs

## How SLURM `--time` Works

**Important**: The `--time` parameter in SLURM is the **wall-clock time limit** for the **entire job**, NOT per GPU.

- With **1 GPU**: Job might take 50 hours → need `--time=72:00:00` (safety buffer)
- With **32 GPUs in parallel**: Job finishes faster (~2-4 hours) → still need reasonable time limit

## Time Limit Logic

### Why You Don't Reduce Time Limit Much

Even with 32 GPUs:
- ✅ **Actual compute time**: ~2-4 hours (with perfect parallelization)
- ✅ **Time limit needed**: Still keep `--time=24:00:00` to `72:00:00` for safety

**Reasons to keep reasonable time limit**:
1. **Variability**: Some checkpoints/models load slower
2. **I/O overhead**: Network, disk I/O can slow things down
3. **Phase 2**: Bandit loop still takes time (sequential selection)
4. **Safety buffer**: Prevents job from being killed mid-computation

### Can You Reduce Time Limit?

**Yes, but carefully**:

| Setup | Expected Time | Recommended Time Limit | Risky Limit |
|-------|--------------|----------------------|-------------|
| **1 GPU** | 50-100 hours | 72-96 hours | 48 hours |
| **8 GPUs** | 6-12 hours | 24-48 hours | 12 hours |
| **32 GPUs** | 2-4 hours | 12-24 hours | 6 hours ⚠️ |

## Recommended Time Limits

### For 8 Nodes × 4 GPUs (32 GPUs total)

**Conservative (Recommended)**:
```bash
#SBATCH --time=48:00:00  # 48 hours - safe buffer
```

**Balanced**:
```bash
#SBATCH --time=24:00:00  # 24 hours - reasonable buffer
```

**Aggressive (Risky)**:
```bash
#SBATCH --time=12:00:00  # 12 hours - might be too tight
```

## Time Breakdown (32 GPUs, parallelized)

### Phase 1: Metrics Pre-computation
- **Sequential (1 GPU)**: ~50-100 hours
- **Parallel (32 GPUs)**: ~2-4 hours
- **Speedup**: ~20-25x

### Phase 2: Bandit Loop
- **Sequential arm selection**: Inherently sequential
- **Task evaluation**: Can be parallelized
- **Total Phase 2**: ~4-8 hours (budget=100)

### Total Job Time
- **Expected**: ~6-12 hours (with 32 GPUs)
- **Recommended limit**: 24-48 hours (2-4x buffer)
- **Aggressive limit**: 12-24 hours (risky but might work)

## My Recommendation

For your 8-node × 4-GPU setup:

**Keep `--time=48:00:00` (48 hours)** because:
1. ✅ Safe buffer (4-8x expected time)
2. ✅ Accounts for Phase 2 sequential parts
3. ✅ Handles variability in checkpoint loading
4. ✅ Prevents job from being killed
5. ✅ Better queue priority (longer time limits often get priority)

**Don't reduce too much** because:
- ⚠️ Phase 2 (bandit loop) is still sequential
- ⚠️ Network I/O between nodes can add overhead
- ⚠️ Some checkpoints load slower than others
- ⚠️ Job might get killed if it goes slightly over

## Summary

**Question**: "Can I use less time with more GPUs?"

**Answer**: 
- **Actual compute time**: YES, much less (~2-4h vs 50-100h)
- **SLURM time limit**: NO, keep it reasonable (24-48h) for safety

The time limit is a **safety cap**, not a target. With more GPUs, your job finishes faster, but you still need a reasonable buffer to handle variability.
