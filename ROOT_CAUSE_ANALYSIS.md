# Root Cause Analysis: Why Our Results Differ from Paper

## Executive Summary

**ROOT CAUSE IDENTIFIED:** Three critical implementation differences explain why our results are 30-45% lower than the paper's:

1. **Pooling Method** (MOST CRITICAL): Paper pools over LAST tokens, we pool over ALL tokens
2. **Tokenizer Padding Side**: Paper uses "left", we use "right"  
3. **Max Sequence Length**: Paper uses 2048, we use 256

## Comparison Results

### Average Main Score by Layer

| Layer | Paper Avg | Our Avg | Difference | % Lower |
|-------|-----------|---------|------------|---------|
| 15    | **0.492121** | 0.301648 | -0.190473 | **-38.70%** |
| 16    | **0.495890** | 0.295798 | -0.200092 | **-40.35%** |
| 17    | **0.519564** | 0.290674 | -0.228891 | **-44.05%** |
| 18    | **0.516978** | 0.282937 | -0.234042 | **-45.27%** |
| 19    | **0.516785** | 0.282874 | -0.233911 | **-45.26%** |
| 20    | **0.495707** | 0.281113 | -0.214594 | **-43.29%** |
| 21    | **0.485260** | 0.289337 | -0.195923 | **-40.37%** |
| 22    | **0.483525** | 0.344136 | -0.139389 | **-28.83%** |
| 23    | **0.478863** | 0.418214 | -0.060649 | **-12.67%** |

**Summary:**
- Mean absolute difference: **0.188663**
- Mean % difference: **37.64%**
- All layers: Ours < Paper (systematic underestimation)

### Sample Task Comparisons

#### STS12
- Layer 15: Paper **0.444506** vs Ours 0.250770 (**-43.58%**)
- Layer 23: Paper **0.474375** vs Ours 0.474131 (**-0.05%** - almost matches!)

#### STSBenchmark  
- Layer 15: Paper **0.520231** vs Ours 0.214742 (**-58.72%**)
- Layer 23: Paper **0.454221** vs Ours 0.395710 (**-12.88%**)

#### AmazonCounterfactualClassification
- Layer 15: Paper **0.831259** vs Ours 0.607057 (**-26.97%**)
- Layer 23: Paper **0.751124** vs Ours 0.601051 (**-19.98%**)

#### ArxivClusteringS2S
- Layer 15: Paper **0.277539** vs Ours 0.193071 (**-30.43%**)
- Layer 23: Paper **0.304133** vs Ours 0.285719 (**-6.05%** - closest match!)

**Observation:** Layer 23 shows the smallest differences, suggesting the pooling method has less impact on final layers.

## Critical Differences Found

### 1. ⚠️ POOLING METHOD (MOST CRITICAL)

**Paper's Implementation:**
```python
def _get_pooled_hidden_states(self, hidden_states, attention_mask=None, method="mean"):
    if method == "mean":
        seq_lengths = attention_mask.sum(dim=-1)
        return torch.stack(
            [
                hidden_states[i, -length:, :].mean(dim=0)  # <-- LAST tokens!
                for i, length in enumerate(seq_lengths)
            ],
            dim=0,
        )
```
**Pools over LAST 'length' tokens (from the END of sequence)**

**Our Implementation:**
```python
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)  # Sums ALL non-padding tokens
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
```
**Pools over ALL non-padding tokens (from START to END)**

**Why This Matters:**
- For **causal models** (like Pythia), tokens are processed left-to-right
- **Last tokens** have seen all previous context → more informative
- **Early tokens** have less context → less informative
- Pooling over last tokens gives **better representations** for downstream tasks
- This explains why paper's scores are **systematically higher**

### 2. ⚠️ TOKENIZER PADDING SIDE

**Paper:** `tokenizer.padding_side = "left"` (line 127)
**Ours:** `tokenizer.padding_side = "right"` (line 432)

**Impact:**
- With left padding, real tokens are at the END
- With right padding, real tokens are at the START
- Combined with "last tokens" pooling, this affects which tokens are used
- Paper's approach ensures last tokens are always the real content (not padding)

### 3. ⚠️ MAX SEQUENCE LENGTH

**Paper:** `max_sample_length = 2048` (for causal models)
**Ours:** `max_length = 256`

**Impact:**
- Longer sequences provide more context
- More tokens to pool from (especially last tokens)
- Could contribute to higher scores, though less critical than pooling method

## Code Locations

### Paper's Code
- Repository: `OFSkean/information_flow`
- File: `experiments/utils/model_definitions/text_automodel_wrapper.py`
- Lines: 277-289 (pooling), 127 (padding side), 173 (max length)

### Our Code
- File: `src/layer_time/embedder.py`
- Lines: 257-261 (pooling), 432 (padding side), 282 (max length)

## Fixes Required

### Priority 1: Fix Pooling Method
**File:** `src/layer_time/embedder.py`

**Change:**
```python
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # OLD: mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    #      summed = (last_hidden * mask).sum(dim=1)
    #      counts = mask.sum(dim=1).clamp(min=1e-6)
    #      return summed / counts
    
    # NEW: Match paper's implementation
    seq_lengths = attention_mask.sum(dim=-1)
    return torch.stack(
        [
            last_hidden[i, -length:, :].mean(dim=0)
            for i, length in enumerate(seq_lengths)
        ],
        dim=0,
    )
```

### Priority 2: Fix Padding Side
**File:** `src/layer_time/embedder.py` (line 432)

**Change:**
```python
# OLD: self._tokenizer.padding_side = "right"
# NEW:
self._tokenizer.padding_side = "left"  # Match paper
```

### Priority 3: Increase Max Length (Optional)
**File:** `src/layer_time/embedder.py` (line 282) and configs

**Change:**
```python
# OLD: max_length: int = 256
# NEW:
max_length: int = 2048  # Match paper (if memory allows)
```

## Expected Impact

After applying these fixes:
- **Layer 15-20**: Should see ~40-45% improvement (matching paper)
- **Layer 21-23**: Should see ~12-30% improvement (matching paper)
- **Overall**: Average main scores should match paper's values

## Next Steps

1. ✅ **Identified root causes** (pooling, padding, max_length)
2. ⏳ **Update embedder.py** with paper's pooling method
3. ⏳ **Change padding_side to "left"**
4. ⏳ **Re-run Experiment 1** with corrected implementation
5. ⏳ **Verify results match paper**

## Files Created

- `PAPER_COMPARISON_ANALYSIS.md` - Detailed comparison
- `ROOT_CAUSE_ANALYSIS.md` - This file
- `/scratch/.../paper_vs_our_comparison.json` - Full numerical comparison
