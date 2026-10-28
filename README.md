# Tree Training

Reference implementation for **Tree Training: Efficient LLM Training on Tree-Structured Trajectories** (NeurIPS 2026). Paper: https://openreview.net/forum?id=MMS2MVw0et

This repository contains the core implementation of Tree Training, including DFS
serialization, tree-aware SSM state routing, and Redundancy-Free Tree Partitioning
with differentiable partition boundaries.

## File Overview

| File | Description |
|------|-------------|
| `dfs_tree_forward.py` | Core DFS tree forward: `TreeNode`, `build_dfs_tree_info`, tree attention mask, per-chunk projection, position IDs |
| `tree_dfs_ssm.py` | Tree-aware GDN SSM forward: `torch_chunk_gated_delta_rule_tree_varlen` (parent-state routing), `_tree_correct_conv_varlen` |
| `tree_grad_ckpt_opath.py` | `GatewayCache`: HuggingFace KV-cache interface for differentiable partition boundaries |
| `tree_grad_hybrid.py` | Main algorithm: `tree_grad_hybrid()` — DFS packing within partitions + gateway gradient chaining between partitions |
| `tree_grad_hybrid_ssm.py` | SSM gateway extension: captures recurrent state / conv context at cut nodes, injects into child partitions |
| `verify_hybrid.py` | Gradient verification for pure-attention models (Qwen3-4B) |
| `verify_hybrid_ssm.py` | Gradient verification for SSM hybrid models (Qwen3.5-35B-A3B) |

## Requirements

```
torch >= 2.0
transformers >= 4.45
```

Models used in verification:
- **Pure-attention**: `Qwen/Qwen3-4B` (or any HF causal LM)
- **SSM hybrid**: `Qwen/Qwen3.5-35B-A3B` (GDN hybrid architecture)

## Reproducing Gradient Verification

### Setup

Deterministic cuBLAS is required for exact float32 comparison:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Pure-attention model (Qwen3-4B)

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  python verify_hybrid.py \
    --model_path /path/to/Qwen3-4B \
    --dtype float32
```

Expected output:

```
── Partition sanity checks ──
  Partition [1000 tok]: 1 subtrees, ...
  Partition [  20 tok]: N subtrees, ...
  Partition [  10 tok]: N subtrees, ...

── tree_grad_hybrid  max_tokens=1000 ──
  ✓  [max_tokens=1000]  max_rel=<1e-6  (threshold rel<1e-04)

── tree_grad_hybrid  max_tokens=20 ──
  ✓  [max_tokens=20]   max_rel=<1e-4  (threshold rel<1e-04)

── tree_grad_hybrid  max_tokens=10 ──
  ✓  [max_tokens=10]   max_rel=<1e-4  (threshold rel<1e-04)

✓  ALL PASS
```

### SSM hybrid model (Qwen3.5-35B-A3B, 2-layer float32)

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python verify_hybrid_ssm.py \
    --model_path /path/to/Qwen3.5-35B-A3B \
    --n_layers 2 \
    --dtype float32
```

Expected output: `max_rel < 2e-5` for all partition configurations (no-cut,
moderate-cut, aggressive-cut), confirming the SSM state gateway introduces no
additional numerical error.

## Algorithm Summary

### `tree_grad_hybrid(model, roots, leaf_loss_fn, max_tokens)`

1. **Partition** the tree into subtrees of at most `max_tokens` tokens each
   (`partition_tree`). Each partition is a connected subtree; the algorithm is
   greedy DFS.

2. **Top-down forward**: for each partition (in topological order), run a single
   DFS forward pass using the tree attention mask. At partition boundaries (cut
   nodes), create differentiable gateway tensors (detached KV and SSM states
   with gradient tracking) for child partitions.

3. **Bottom-up backward**: for each partition in reverse order, compute leaf
   losses and call backward; then chain the accumulated gateway gradients back
   through the parent partition's retained computation graph. When a cut node
   has multiple child partitions, all children share one gateway and gradients
   are chained exactly once.

### Key correctness properties

- **Ancestor-aware attention bias**: the gateway KV may contain DFS-sibling
  tokens from the parent partition; these are masked with `-inf` in the child's
  attention so only true ancestors are visible.
- **Depth-based position offset**: computed from the cut node's depth in the DFS
  sequence, not from `past_len`, so sibling tokens do not inflate the child's
  position IDs.
- **Single gateway per cut node**: when the same cut node leads to multiple child
  partitions, they share one gateway object; `_chain_gateway_grads` is called
  exactly once to avoid gradient double-counting.
- **Float32 gradient accumulation**: a hook on each gateway tensor captures
  gradients in float32 accumulators and suppresses the native bfloat16 `.grad`
  write, avoiding multi-child accumulation rounding errors.

### Test tree (verify_hybrid.py)

```
root_a (3 tok)
  └─ b_left (4 tok)
       └─ c_left (3 tok)
            ├─ d_left (4 tok)
            │    ├─ e  (5 tok)  [leaf P0]
            │    └─ q  (3 tok)  [leaf P2]
            └─ f  (3 tok)
                 └─ g  (4 tok)  [leaf P1]

root_b2 (4 tok)
  ├─ c2 (3 tok)
  │    └─ d2 (4 tok)            [leaf P3]
  └─ x  (3 tok)
       └─ y  (4 tok)            [leaf P4]
```

Total unique tokens: 47 (across 5 leaf paths, 2 root trees).

## License

MIT — see [LICENSE](LICENSE).
