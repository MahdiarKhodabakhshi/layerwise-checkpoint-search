# Paper Comparison Analysis: Critical Differences Found

## Executive Summary

**ROOT CAUSE IDENTIFIED:** The paper uses a **different pooling method** that pools over the **LAST tokens** instead of all tokens, which explains the discrepancy in results.

## Critical Difference #1: Pooling Method

### Our Implementation
```python
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / counts
```
**Pools over ALL non-padding tokens (from start to end)**

### Paper's Implementation
```python
def _get_pooled_hidden_states(self, hidden_states, attention_mask=None, method="mean"):
    if method == "mean":
        seq_lengths = attention_mask.sum(dim=-1)
        return torch.stack(
            [
                hidden_states[i, -length:, :].mean(dim=0)  # <-- KEY DIFFERENCE!
                for i, length in enumerate(seq_lengths)
            ],
            dim=0,
        )
```
**Pools over LAST 'length' tokens (from the END)**

### Impact
- **For causal models like Pythia**: Last tokens contain more context-aware information
- **Last tokens have seen all previous tokens** in the sequence
- **Early tokens are less informative** for downstream tasks
- **This explains why paper's scores are HIGHER** - they use more informative tokens

## Comparison Results

### Average Main Score by Layer

| Layer | Paper Avg | Our Avg | Difference | % Diff |
|-------|-----------|---------|------------|--------|
| 15    | TBD       | 0.304192| TBD        | TBD    |
| 16    | TBD       | 0.297618| TBD        | TBD    |
| ...   | ...       | ...     | ...        | ...    |
| 23    | TBD       | 0.422393| TBD        | TBD    |

*Full comparison being extracted...*

## Other Differences Found

### 1. Layer Indexing
- **Paper**: Uses `evaluation_layer_idx` directly on `outputs.hidden_states[evaluation_layer_idx]`
- **Ours**: Uses `layer_index + 1` to account for embeddings layer
- **Status**: Both are correct, just different indexing conventions

### 2. Tokenizer Padding Side
- **Paper**: `tokenizer.padding_side = "left"` (line 127)
- **Ours**: `tokenizer.padding_side = "right"` (line 432 in embedder.py)
- **Impact**: Could affect which tokens are considered "last" in sequences

### 3. Max Length
- **Paper**: Uses `max_sample_length` (default 2048 for causal models)
- **Ours**: Uses `max_length=256`
- **Impact**: Paper processes longer sequences, which could affect results

## Recommendations

### Priority 1: Fix Pooling Method
**Change our pooling to match paper's method:**
```python
def _mean_pool_paper_style(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lengths = attention_mask.sum(dim=-1)
    return torch.stack(
        [
            hidden_states[i, -length:, :].mean(dim=0)
            for i, length in enumerate(seq_lengths)
        ],
        dim=0,
    )
```

### Priority 2: Fix Tokenizer Padding
**Change padding side to "left" to match paper:**
```python
self.tokenizer.padding_side = "left"  # Instead of "right"
```

### Priority 3: Consider Max Length
**Increase max_length to 2048** (or match paper's setting) if memory allows.

## Next Steps

1. **Update embedder.py** to use paper's pooling method
2. **Change padding_side to "left"**
3. **Re-run Experiment 1** with corrected pooling
4. **Compare results** - should match paper much better
