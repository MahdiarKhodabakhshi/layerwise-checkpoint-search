# GPU Recommendations for Bandit Workflow

## Quick Comparison

| Script | GPUs | Speed | Memory | Time Limit | Best For |
|--------|------|-------|--------|------------|----------|
| `mteb_bandit.sbatch` | 1x H100 | ⭐⭐⭐⭐⭐ | 120GB | 24h | Quick tests, small runs |
| `mteb_bandit_l40s.sbatch` | 1x L40s | ⭐⭐⭐ | 120GB | 24h | Budget-friendly |
| `mteb_bandit_h100_multi.sbatch` | 3x H100 | ⭐⭐⭐⭐⭐ | 240GB | 48h | **Production runs (RECOMMENDED)** |
| `mteb_bandit_l40s_multi.sbatch` | 4x L40s | ⭐⭐⭐ | 180GB | 72h | Long runs, more availability |

## Recommendation for Your Production Run

**Use `mteb_bandit_h100_multi.sbatch` (3x H100)** because:

1. ✅ **Fastest GPUs** - H100s are ~2-3x faster than L40s
2. ✅ **More memory** - 240GB vs 120GB (can handle larger batch sizes)
3. ✅ **Longer time limit** - 48h vs 24h (important for full checkpoint sweep)
4. ✅ **More CPUs** - 48 vs 16 (faster data loading)
5. ✅ **Better for production** - Can handle all 155 checkpoints comfortably

Even though only 1 GPU is used at a time, having 3 H100s gives you:
- Headroom for memory-intensive operations
- Ability to increase batch_size in config if needed
- Faster overall processing due to GPU speed

## How to Use Multi-GPU Script

```bash
cd /project/6101803/mahdiar/pythia-layer-time

# Submit with multi-H100 (RECOMMENDED)
sbatch slurm/mteb_bandit_h100_multi.sbatch

# Or multi-L40s (if H100s not available)
sbatch slurm/mteb_bandit_l40s_multi.sbatch
```

## Notes on Current Implementation

The current bandit workflow processes checkpoints **sequentially** (one at a time):
- Phase 1: Metrics pre-computation (sequential)
- Phase 2: Bandit loop (sequential - selects arm → evaluates → selects next)

**Only GPU 0 is currently used**, even with multiple GPUs available. However, having multiple GPUs still helps:
- More total memory across GPUs (though current code uses GPU 0 only)
- Can increase batch_size in config if you hit OOM
- Better resource allocation (longer time limits)
- Room for future parallelization improvements

## Future Improvements (Optional)

To fully utilize multiple GPUs, we would need to parallelize:
- **Phase 1**: Parallel metric computation across GPUs (different checkpoints)
- **Phase 2**: This is harder because bandit selection depends on previous results

This would require code changes and is not necessary for current runs.

## For Now: Use Multi-GPU Scripts

**Recommendation**: Use `mteb_bandit_h100_multi.sbatch` for your production run. The extra resources help even if only one GPU is actively used at a time.
