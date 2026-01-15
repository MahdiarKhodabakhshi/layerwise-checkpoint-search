# Discrepancy Analysis: Our Results vs Paper

## Executive Summary

Our average main scores for layers 15-23 are systematically lower than the paper's reported results. After investigation, we found **NO final layer norm issue** (outputs.last_hidden_state == hidden_states[-1]), but several other potential causes remain.

## Our Results

| Layer | Average Main Score | Change from Layer 15 |
|-------|-------------------|---------------------|
| 15    | 0.304192          | baseline            |
| 16    | 0.297618          | -2.16%              |
| 17    | 0.292147          | -3.96%              |
| 18    | 0.284230          | -6.56%              |
| 19    | 0.284478          | -6.48%              |
| 20    | 0.282944          | -6.99%              |
| 21    | 0.290864          | -4.38%              |
| 22    | 0.347597          | +14.28%             |
| 23    | 0.422393          | +38.86%             |

**Mean across layers: 0.311829**
**Pattern: Layers 15-20 are relatively flat, then sharp increase at layers 22-23**

## Sample Task Results

### STS12
- Layer 15: 0.2508 → Layer 23: 0.4741 (+89.1%)
- Values show improvement, but absolute scores may be lower than paper

### STSBenchmark  
- Layer 15: 0.2147 → Layer 23: 0.3957 (+84.3%)
- Similar pattern to STS12

### AmazonCounterfactualClassification
- Layer 15: 0.6071 → Layer 23: 0.6011 (-1.0%)
- Classification shows minimal change (expected pattern)

### ArxivClusteringS2S
- Layer 15: 0.1931 → Layer 23: 0.2857 (+48.0%)
- Clustering shows improvement but may be lower than paper

## What We Verified

### ✓ Layer Indexing is Correct
- `hidden_states[0]` = input embeddings
- `hidden_states[1..24]` = after transformer blocks 0..23
- Our code: `idx = layer_index + 1`, extracting `hidden_states[idx]`
- **Status: CORRECT**

### ✓ No Final Layer Norm Issue
- Tested: `outputs.last_hidden_state == hidden_states[-1]`
- **Result: They are identical (no final layer norm)**
- **Status: NOT THE ISSUE**

### ✓ Embedding Extraction Parameters
- Pooling: mean pooling over non-padding tokens ✓
- Normalization: L2 normalization after pooling ✓
- Max length: 256 tokens ✓
- **Status: APPEARS CORRECT**

### ✓ MTEB Version and Tasks
- Using MTEB v1.14.19 (matches paper) ✓
- Using all 32 tasks from layer_by_layer_32 preset ✓
- **Status: APPEARS CORRECT**

## Potential Issues (Unverified)

### 1. ⚠️ Extraction Point Within Transformer Block
**Issue:** We extract `hidden_states[idx]` which is after the complete transformer block.
However, the paper might extract from a different point:
- After attention but before MLP?
- After input layer norm but before attention?
- After post-attention layer norm but before MLP?

**Impact:** Could cause systematic score differences across all layers.

**Evidence:** Our scores show the right pattern (improvement towards final layers) but are systematically lower, suggesting a consistent extraction difference.

### 2. ⚠️ Preprocessing/Tokenization Differences
**Issue:** Differences in:
- Text normalization (lowercasing, punctuation)
- Tokenizer configuration (padding, truncation)
- Special token handling

**Impact:** Could affect embedding quality, especially for similarity tasks.

**Evidence:** STS tasks show lower absolute scores, which could indicate tokenization/preprocessing issues.

### 3. ⚠️ Model Checkpoint/Version Mismatch
**Issue:** 
- Are we using the exact same checkpoint as the paper?
- Are there multiple "main" checkpoints with different training stages?
- Did the model receive additional fine-tuning?

**Impact:** Different weights would produce different representations.

**Evidence:** The fact that our pattern is correct (improvement towards final layers) but scores are systematically lower suggests the model might be from a different training stage.

### 4. ⚠️ Evaluation Settings
**Issue:**
- Random seeds (affecting clustering tasks)
- Number of runs per task
- Data splits (train/test/validation)
- Evaluation metrics computation

**Impact:** Could cause differences in task scores, especially for clustering tasks that use multiple runs.

**Evidence:** Our clustering tasks show variability that might be affected by random seeds.

### 5. ⚠️ Numerical Precision/Implementation Details
**Issue:**
- Float precision (float16 vs float32)
- Batch size effects on numerical stability
- Different implementations of similarity metrics

**Impact:** Small numerical differences that accumulate.

**Evidence:** Less likely to cause large systematic differences.

## Recommendations

### Priority 1: Check Paper's Repository/Code
1. **Find the paper's official repository** (OFSkean/information_flow)
2. **Compare extraction code** - How exactly do they extract embeddings?
3. **Compare model checkpoint** - Which exact checkpoint do they use?
4. **Compare preprocessing** - What tokenizer settings do they use?

### Priority 2: Diagnostic Tests
1. **Compare single-task scores** with paper's reported values
   - Pick 2-3 tasks with paper-reported values
   - Compare layer-by-layer scores
   - Identify which tasks are most different

2. **Test different extraction points**
   - Extract before vs after layer norm (within block)
   - Compare results to see which matches paper better

3. **Verify model checkpoint**
   - Check model card for training stage
   - Compare model hash/commit with paper's

### Priority 3: Code Modifications (if needed)
1. If extraction point is wrong, modify `embedder.py` to extract from correct location
2. If preprocessing differs, update tokenization/preprocessing
3. If checkpoint differs, switch to correct checkpoint

## Conclusion

Our implementation appears to follow correct methodology (correct layer indexing, no final layer norm issue, correct pooling/normalization). However, **absolute scores are systematically lower** than expected.

**Most likely causes:**
1. **Different extraction point** within transformer blocks (before/after specific layer norms)
2. **Model checkpoint mismatch** (different training stage or fine-tuning)
3. **Preprocessing/tokenization differences** (subtle but impactful)

**Next steps:**
- Locate paper's repository and compare extraction code
- Compare single-task scores with paper's reported values
- Test alternative extraction points to see which matches paper's results
