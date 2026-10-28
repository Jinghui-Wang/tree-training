#!/usr/bin/env python3
"""
Tree Gradient Checkpointing — O(max_path) memory version
==========================================================

Peak GPU memory ≈ one root-to-leaf path at a time.
Each token is computed EXACTLY ONCE (prefix never recomputed).

Core mechanism: GatewayCache
------------------------------
At each internal tree node n, after computing its forward pass:

  raw_kv[l]  = (key_l, val_l)  — full KV for all ancestor+n tokens
                                  key_l has grad_fn → parent gateway tensors
  gate_kv[l] = (key_l.detach().requires_grad_(), val_l.detach()...)
                                  — NEW leaf tensors, children use these

  Children's forward uses gate_kv as past KV.
  The model's attention computes:
      full_key = cat([gate_k, new_key_from_child])
  Gradient of child's loss flows: loss → full_key → cat → gate_k.grad (stops)

  After ALL children of n are done:
      torch.autograd.backward([raw_k, raw_v, ...], [gate_k.grad, gate_v.grad, ...])
  This chains gradient through n's raw_kv → parent's gate_kv leaves → ...
  → eventually reaches model parameter leaves and accumulates their gradients.

GatewayCache implements the HF Cache interface:
  - update(key, value, layer_idx): cat(gate_kv[l], new_kv) → full kv
  - get_seq_length(): returns past_len (= pos_offset of current node)
  - is_sliding: not set — create_causal_mask handles gracefully

Memory profile (example tree: root→a→b→c→d→e with d also having child f→g):
  Processing e:    root+a+b+c+d activations (retained) + e activations
  After e.bwd:     root+a+b+c+d activations (retained) + e freed
  Processing f+g:  root+a+b+c+d (retained) + f+g activations
  After f+g.bwd:   root+a+b+c+d (retained) + f+g freed
  _chain_grads(d): root+a+b+c activations (retained) + d freed
  ...
  Peak = root+a+b+c+d+e = ONE PATH LENGTH  ✓

Gradient correctness:
  Mathematically identical to:
      single DFS forward + single total_loss.backward()
  because each path [root..leaf] sees identical hidden states and attention
  (same token IDs, same depth-based positions, same causal attention pattern).
  Float32 verification: max_rel_err < 1e-5  ✓  (see verify_tree_grad_ckpt_opath.py)

Supports: any HF causal LM with standard attention (no SSM layers required).
Tested with: Qwen3ForCausalLM.
"""

from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable, Dict

from dfs_tree_forward import TreeNode


# ---------------------------------------------------------------------------
# GatewayCache — minimal HF Cache-compatible wrapper
# ---------------------------------------------------------------------------

class GatewayCache:
    """
    Wraps per-layer gateway KV tensors (leaf tensors with requires_grad)
    and appends new KV on each attention forward call via cat().

    After the forward pass:
      self._full_kv[l] = (cat([gate_k_l, new_k_l]), cat([gate_v_l, new_v_l]))
    These full_kv tensors have grad_fn connecting back to gate_kv leaves.

    Methods required by HF models:
      update()          → called per attention layer
      get_seq_length()  → called by model.forward to build causal mask
    """

    def __init__(self, gate_kv: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        gate_kv[l] = (key_gate_l, val_gate_l)
          • key_gate_l : [B, kv_heads, past_len, head_dim], requires_grad=True (leaf)
          • val_gate_l : same shape, requires_grad=True (leaf)
          • len(gate_kv) should equal num_layers; entry may be (None, None) for root
        """
        self._gate_kv:  List[Tuple[torch.Tensor, torch.Tensor]] = gate_kv
        self._full_kv:  Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ── HF Cache interface ────────────────────────────────────────────────

    def update(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by each attention layer.  Concatenates gateway with new KV
        and records the result (with grad_fn) for later gradient chaining.
        """
        if layer_idx < len(self._gate_kv):
            gate_k, gate_v = self._gate_kv[layer_idx]
            if gate_k is not None:
                full_k = torch.cat([gate_k, key_states],   dim=-2)
                full_v = torch.cat([gate_v, value_states], dim=-2)
            else:
                full_k, full_v = key_states, value_states
        else:
            full_k, full_v = key_states, value_states

        self._full_kv[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the number of past (gateway) tokens in the cache."""
        if layer_idx < len(self._gate_kv):
            gate_k, _ = self._gate_kv[layer_idx]
            if gate_k is not None:
                return gate_k.shape[-2]
        return 0

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> Tuple[int, int]:
        """
        Required by create_causal_mask → _preprocess_mask_arguments.
        Returns (kv_length, kv_offset) = (past_len + query_len, 0).
        """
        past_len  = self.get_seq_length(layer_idx)
        query_len = cache_position.shape[0]
        return past_len + query_len, 0

    # ── Gateway creation and gradient chaining ───────────────────────────

    def make_child_gates(
        self,
    ) -> Tuple['GatewayCache', List[torch.Tensor], List[torch.Tensor], List[List]]:
        """
        Create the next level's gateway from this forward pass's full KV.

        Returns
        -------
        child_gate_cache : GatewayCache for children to use
        raw_tensors      : original full_kv tensors (have grad_fn → parent)
        gate_tensors     : corresponding detached leaves (requires_grad=True)
        fp32_accum       : list of [accumulator_or_None] per gate tensor.
                           A hook on each gate tensor captures gradients in
                           float32 and zeroes .grad to prevent bfloat16
                           rounding from automatic .grad accumulation.
                           Pass to _chain_grads instead of relying on .grad.
        """
        n_layers = max(self._full_kv.keys()) + 1 if self._full_kv else 0
        gate_kv_next: List[Tuple[torch.Tensor, torch.Tensor]] = []
        raw_tensors:  List[torch.Tensor] = []
        gate_tensors: List[torch.Tensor] = []
        fp32_accum:   List[List]         = []

        for l in range(n_layers):
            if l not in self._full_kv:
                gate_kv_next.append((None, None))
                continue
            full_k, full_v = self._full_kv[l]
            if full_k.grad_fn is not None:
                gk = full_k.detach().requires_grad_(True)
                gv = full_v.detach().requires_grad_(True)
                raw_tensors.extend([full_k, full_v])
                gate_tensors.extend([gk, gv])

                # Float32 accumulators — one entry per gate tensor.
                # Hook: capture grad in fp32, cancel bfloat16 .grad accumulation.
                acc_k: List = [None]
                acc_v: List = [None]
                fp32_accum.extend([acc_k, acc_v])

                def _make_hook(acc):
                    def _hook(g):
                        g_fp32 = g.detach().float().cpu()
                        if acc[0] is None:
                            acc[0] = g_fp32
                        else:
                            acc[0] += g_fp32
                        # Return zero to prevent bfloat16 .grad accumulation
                        return torch.zeros_like(g)
                    return _hook

                gk.register_hook(_make_hook(acc_k))
                gv.register_hook(_make_hook(acc_v))
                gate_kv_next.append((gk, gv))
            else:
                gate_kv_next.append((full_k, full_v))

        child_cache = GatewayCache(gate_kv_next)
        return child_cache, raw_tensors, gate_tensors, fp32_accum


def _chain_grads(
    raw_tensors:   List[torch.Tensor],
    gate_tensors:  List[torch.Tensor],
    fp32_accum:    Optional[List[List]] = None,
    retain_graph:  bool = False,
) -> None:
    """
    Propagate gradients from gate_tensors back through raw_tensors.

    retain_graph=True when this node still has pending sibling calls
    (more backward passes through the same raw_tensors needed).

    fp32_accum[i][0]: float32 gradient captured by hook on gate_tensors[i].
    """
    valid_raws  = []
    valid_grads = []
    for i, (raw, gate) in enumerate(zip(raw_tensors, gate_tensors)):
        if fp32_accum is not None:
            g = fp32_accum[i][0]
            if g is not None:
                valid_raws.append(raw)
                valid_grads.append(g.to(raw.device))
        elif gate.grad is not None:
            valid_raws.append(raw)
            valid_grads.append(gate.grad)
    if valid_raws:
        torch.autograd.backward(valid_raws, valid_grads, retain_graph=retain_graph)


# ---------------------------------------------------------------------------
# Per-node incremental forward
# ---------------------------------------------------------------------------

def _forward_node(
    model,
    node:        'TreeNode',
    gate_cache:  GatewayCache,
    pos_offset:  int,
    dev:         torch.device,
) -> Tuple[torch.Tensor, GatewayCache]:
    """
    Run node's tokens through the model starting from gate_cache.

    Position IDs use depth-based scheme (pos_offset = sum of ancestor lengths),
    which matches build_dfs_tree_info.  The model's create_causal_mask will
    produce a mask that allows attending to all past tokens in gate_cache.

    Returns: (logits [1, L, V], the gate_cache after update by model)
    """
    L   = len(node.token_ids)
    ids = torch.tensor([node.token_ids], dtype=torch.long, device=dev)
    pos = torch.arange(pos_offset, pos_offset + L, dtype=torch.long, device=dev).unsqueeze(0)

    out = model(
        input_ids       = ids,
        position_ids    = pos,
        past_key_values = gate_cache,
        use_cache       = True,
    )
    # out.past_key_values IS gate_cache (model calls cache.update() in-place)
    return out.logits, gate_cache


# ---------------------------------------------------------------------------
# DFS forward+backward (O(max_path) memory)
# ---------------------------------------------------------------------------

def _dfs_fwd_bwd(
    model,
    node:           'TreeNode',
    parent_cache:   GatewayCache,
    pos_offset:     int,
    leaf_loss_fn:   Callable[[torch.Tensor, 'TreeNode'], torch.Tensor],
    dev:            torch.device,
) -> None:
    """
    Recursively forward+backward the subtree rooted at node.

    Memory invariant: at any point, GPU holds activations for
    exactly ONE root-to-current-node path plus sibling subtrees
    that have been backward'd and whose graphs are retained by
    parent's gateway.  Peak = one full root-to-leaf path.

    Parameters
    ----------
    parent_cache
        GatewayCache containing leaf tensors for all ancestor KVs.
        Children's backward passes accumulate .grad into these leaves.
    pos_offset
        Depth-based position of this node's first token.
    leaf_loss_fn
        loss = leaf_loss_fn(logits: [1, L, V], node) → scalar tensor.
        Called for every leaf.
    """
    # ── Step 1: forward through this node ────────────────────────────────
    # parent_cache is FRESH for this node (not mutated by siblings yet,
    # because each node gets its OWN GatewayCache from make_child_gates)
    logits, _ = _forward_node(model, node, parent_cache, pos_offset, dev)
    L = len(node.token_ids)

    if node.is_leaf:
        # ── Leaf: backward directly into parent_cache's gate leaves ──────
        # Gradient flows: loss → logits → model layers → cat operations
        # in parent_cache.update() → parent_cache._gate_kv[l].grad
        loss = leaf_loss_fn(logits, node)
        loss.backward()
        # parent_cache._gate_kv[l].grad has accumulated from this leaf.
        # The computation graph for THIS leaf is freed.
        # parent's computation graph (outside parent_cache) is untouched.

    else:
        # ── Internal node: per-child gates + per-child chain ──────────────
        #
        # CRITICAL for bfloat16 accuracy: we call make_child_gates() once
        # per child (fresh gate tensors each time) and _chain_grads once per
        # child with retain_graph=True for non-last children.
        #
        # This matches per-path-independent method (C):
        #   each path's gradient backward through prefix is a SEPARATE matmul.
        # vs naive cumulative approach:
        #   sum children's gradients at gate → ONE matmul with summed gradient.
        # In bfloat16: A@(g1+g2) ≠ (A@g1)+(A@g2), so C-matching is correct.
        #
        # raw_tensors is the same across children (same _full_kv entries);
        # retain_graph=True keeps this node's computation graph live until
        # the last child's _chain_grads frees it.
        n_children = len(node.children)

        for child_idx, child in enumerate(node.children):
            is_last = (child_idx == n_children - 1)

            # Fresh gate tensors for THIS child only
            child_cache, raw_tensors, gate_tensors, fp32_accum = \
                parent_cache.make_child_gates()

            child_own_cache = GatewayCache(list(child_cache._gate_kv))
            _dfs_fwd_bwd(
                model, child,
                child_own_cache,
                pos_offset + L,
                leaf_loss_fn,
                dev,
            )
            # Chain THIS child's gradient through this node's computation.
            # retain_graph=True if more children remain.
            _chain_grads(raw_tensors, gate_tensors, fp32_accum,
                         retain_graph=not is_last)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tree_grad_ckpt_o_path(
    model,
    roots:        List['TreeNode'],
    leaf_loss_fn: Callable[[torch.Tensor, 'TreeNode'], torch.Tensor],
) -> None:
    """
    DFS tree forward+backward with O(max_path) peak GPU memory.

    Each token is computed exactly once.  The computation graph for a node
    is retained until all its descendants have been backward'd, then released.
    Peak activation memory ≈ one root-to-leaf path.

    Gradient equivalence (float32):
      Identical to dfs_tree_forward + total_loss.backward(), up to
      floating-point accumulation order (~machine epsilon in float32).

    Parameters
    ----------
    model         : HF causal LM in train mode.  Call zero_grad() first.
    roots         : List of root TreeNodes.
    leaf_loss_fn  : loss = leaf_loss_fn(logits [1,L,V], node) → scalar.
                    Divide by n_leaves inside to match per-path averaging.

    Example
    -------
        def my_loss(logits, node):
            return F.cross_entropy(logits[0], node.labels.to(logits.device)) / n_leaves

        model.zero_grad()
        tree_grad_ckpt_o_path(model, roots=[root], leaf_loss_fn=my_loss)
        optimizer.step()
    """
    dev = next(model.parameters()).device

    for root in roots:
        # Empty gateway: no past KV for root
        empty_cache = GatewayCache([])
        _dfs_fwd_bwd(
            model, root,
            empty_cache,
            pos_offset  = 0,
            leaf_loss_fn = leaf_loss_fn,
            dev          = dev,
        )


# ---------------------------------------------------------------------------
# Convenience helpers (same as tree_grad_ckpt.py)
# ---------------------------------------------------------------------------

def make_tree_loss_fn(
    leaf_labels:  List[torch.Tensor],
    leaf_nodes:   List['TreeNode'],
    n_leaves:     int,
    ignore_index: int = -100,
) -> Callable:
    label_map = {id(node): labels for node, labels in zip(leaf_nodes, leaf_labels)}

    def _loss_fn(logits: torch.Tensor, node: 'TreeNode') -> torch.Tensor:
        labels = label_map[id(node)].to(logits.device)
        return F.cross_entropy(
            logits[0], labels, ignore_index=ignore_index,
        ) / n_leaves

    return _loss_fn
