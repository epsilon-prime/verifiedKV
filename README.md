# Scoring-Based KV Cache Offloading for vLLM


## Overview

This system intelligently offloads less-important KV blocks from GPU to CPU during inference:
- **Scores** blocks using V-norm / K-norm ratio after prefill and every W tokens
- **Evicts** low-scoring blocks to CPU when GPU budget exceeded
- **Restores** blocks on-demand from CPU to GPU staging area during attention
- **Supports** per-request W-token intervals for adaptive scheduling


### Offload time
1. When prefill complete: score all tokens, offload required fraction
2. Lowest-scoring tokens copied to CPU
3. blocks compacted to free blocks

### Decode Phase
1. Each decode token sampled
2. Every W tokens: check if should offload
3. If yes: score and offload more low-scoring blocks
4. Attention needs blocks? Fetch from CPU to staging on-demand
5. Sparse attention kernel uses patched indices to staging
