# Pythia 154 Checkpoints: Configuration Guide

## ✅ Confirmed Information

According to Pythia HuggingFace card:
- **154 intermediate checkpoints** per model
- **All model sizes** (14m, 70m, 410m, etc.) trained on **same data, same order**
- **Same checkpoints available** for all model sizes
- Checkpoints hosted on HuggingFace as **branches** (step-*)

## 📊 Search Space Impact

### With Sparse Sampling (~8 checkpoints):
- 410m: 8 checkpoints × 24 layers = **192 arms**
- 70m: 8 checkpoints × 6 layers = **48 arms**
- 14m: 8 checkpoints × 6 layers = **48 arms**
- **Total: 288 arms**

### With Uniform Sampling (~30 checkpoints):
- 410m: 30 checkpoints × 24 layers = **720 arms**
- 70m: 30 checkpoints × 6 layers = **180 arms**
- 14m: 30 checkpoints × 6 layers = **180 arms**
- **Total: 1,080 arms**

### With All 154 Checkpoints:
- 410m: 154 checkpoints × 24 layers = **3,696 arms**
- 70m: 154 checkpoints × 6 layers = **924 arms**
- 14m: 154 checkpoints × 6 layers = **924 arms**
- **Total: 5,544 arms** (too many for most experiments!)

## 🎯 Recommended Strategies

### 1. Sparse Sampling (Recommended to Start)
**Use case**: Initial exploration, limited compute budget
```bash
python generate_checkpoint_list.py --strategy sparse --format yaml
```
**Checkpoints**: step-1000, step-2000, step-5000, step-10000, step-20000, step-40000, step-80000, main
**Total**: ~8 checkpoints

### 2. Uniform Sampling (Balanced)
**Use case**: Thorough exploration with reasonable compute
```bash
python generate_checkpoint_list.py --strategy uniform --interval 5000 --format yaml
```
**Checkpoints**: Every 5000 steps (step-5000, step-10000, ..., step-143000, main)
**Total**: ~30 checkpoints

### 3. Logarithmic Sampling (Capture Early Changes)
**Use case**: Want more checkpoints early in training
```bash
python generate_checkpoint_list.py --strategy log --format yaml
```
**Checkpoints**: Dense early (every 1000), sparse late (every 10000)
**Total**: ~20 checkpoints

### 4. Dense Sampling (Very Thorough)
**Use case**: Complete coverage, large compute budget
```bash
python generate_checkpoint_list.py --strategy dense --interval 1000 --format yaml
```
**Checkpoints**: Every 1000 steps
**Total**: ~143 checkpoints

## 📝 Config Examples

### Sparse (Current Default):
```yaml
revisions: ["step-1000", "step-2000", "step-5000", "step-10000", "step-20000", "step-40000", "step-80000", "main"]
```

### Uniform (Every 5000 steps):
```yaml
revisions: ["step-5000", "step-10000", "step-15000", "step-20000", "step-25000", 
            "step-30000", "step-35000", "step-40000", "step-45000", "step-50000",
            "step-55000", "step-60000", "step-65000", "step-70000", "step-75000",
            "step-80000", "step-85000", "step-90000", "step-95000", "step-100000",
            "step-105000", "step-110000", "step-115000", "step-120000", "step-125000",
            "step-130000", "step-135000", "step-140000", "main"]
```

## ⚠️ Important Notes

1. **Computation Time**: Scales linearly with number of checkpoints
   - Sparse (8): ~8x longer than single checkpoint
   - Uniform (30): ~30x longer
   - Dense (143): ~143x longer

2. **Storage**: Each checkpoint-layer pair needs:
   - Embeddings cache: ~few MB per layer
   - Metrics cache: ~few KB per layer
   - Total for 410m (24 layers) × 8 checkpoints: ~few GB

3. **Bandit Budget**: With more checkpoints, you may want to increase budget
   ```yaml
   bandit:
     budget: 200  # Increase from 100 for larger search space
   ```

4. **Verification**: The code will skip unavailable checkpoints with warnings
   - If a checkpoint doesn't exist, it's skipped
   - Check logs for warnings

## 🚀 Quick Start

1. **Start with sparse sampling** (already in config):
   ```yaml
   revisions: ["step-1000", "step-2000", "step-5000", "step-10000", "step-20000", "step-40000", "step-80000", "main"]
   ```

2. **Run experiment** and verify checkpoints work

3. **Expand if needed**:
   ```bash
   python generate_checkpoint_list.py --strategy uniform --format yaml >> configs/mteb_layersweep.yaml
   ```

## 📚 References

- Pythia HuggingFace: https://huggingface.co/EleutherAI/pythia-410m
- Pythia Paper: Check HuggingFace model card for details
- Checkpoint documentation: See model card for checkpoint schedule

