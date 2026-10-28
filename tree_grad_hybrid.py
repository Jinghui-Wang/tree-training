#!/usr/bin/env python3
"""
tree_grad_hybrid.py — DFS-packed intra-subtree + gateway inter-subtree

Two-level approach:
  Within each subtree : single DFS forward (tree attn mask), dense GPU compute
  Between subtrees    : GatewayKV leaf-tensor chaining (tree_grad_ckpt_opath style)

Key correctness property
------------------------
When a parent partition P has multiple cut nodes (e.g. c2 and x both in P,
each leading to a child subtree), the child subtree of x must:
  1. Only attend to *ancestors* of x in P's gateway (root_b2 + x),
     NOT to siblings like c2 that happen to precede x in P's DFS sequence.
  2. Use depth-based pos_offset = P.pos_offset + x's depth position in P,
     NOT gateway.past_len (which counts all gateway tokens incl. siblings).

Both are computed in _make_child_gateways() from the partition DFS info.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable

from dfs_tree_forward import TreeNode, build_dfs_tree_info, DFSTreeInfo
from tree_grad_ckpt_opath import GatewayCache

# SSM gateway extension (optional — only used when model has SSM layers)
try:
    from tree_grad_hybrid_ssm import (
        SSMGateway, _make_ssm_gateway, _chain_ssm_gateway_grads,
        _make_ssm_forward_parent, _make_ssm_forward_child,
        _patch_ssm_layers, _unpatch_ssm_layers, _has_ssm,
    )
    _SSM_AVAILABLE = True
except ImportError:
    _SSM_AVAILABLE = False
    SSMGateway = None
    _has_ssm = lambda m: False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Partition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubtreePartition:
    entry_nodes:     List[TreeNode]
    dfs_nodes:       List[TreeNode] = field(default_factory=list)
    cut_node_set:    set            = field(default_factory=set)
    leaf_node_set:   set            = field(default_factory=set)
    parent:          Optional['SubtreePartition'] = None
    parent_cut_node: Optional[TreeNode]           = None
    pos_offset:      int = 0
    token_count:     int = 0


def partition_tree(roots: List[TreeNode], max_tokens: int) -> List[SubtreePartition]:
    """
    Greedy DFS.  Each forest root starts its own independent top-level partition.
    A "cut node" stays in the parent partition; its over-budget children each
    become the entry of a new child partition.
    """
    partitions: List[SubtreePartition] = []

    def _build(entry: List[TreeNode], parent, parent_cut_node):
        p = SubtreePartition(entry_nodes=entry, parent=parent,
                              parent_cut_node=parent_cut_node)
        partitions.append(p)
        deferred = []

        def _dfs(node, is_entry):
            L = len(node.token_ids)
            if not is_entry and p.token_count > 0 and p.token_count + L > max_tokens:
                return False
            p.dfs_nodes.append(node)
            p.token_count += L
            for child in node.children:
                if not _dfs(child, False):
                    p.cut_node_set.add(id(node))
                    deferred.append((node, child))
            if not node.children and id(node) not in p.cut_node_set:
                p.leaf_node_set.add(id(node))
            return True

        for r in entry:
            _dfs(r, True)
        for cut_node, cut_child in deferred:
            _build([cut_child], p, cut_node)

    for root in roots:          # each forest root → independent top-level partition
        _build([root], None, None)
    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# 2. DFS info for one partition
# ─────────────────────────────────────────────────────────────────────────────

def _dfs_info_for_partition(
    partition: SubtreePartition,
) -> Tuple[DFSTreeInfo,
           Dict[int, int],   # orig_id → start pos in DFS token seq
           Dict[int, int],   # orig_id → end   pos in DFS token seq
           Dict[int, int],   # orig_id → id(parent_orig) in this partition
           Dict[int, int],   # orig_id → chunk_idx in DFS (position in chunk_boundaries list)
           Dict[int, TreeNode]]:  # id(copy) → orig
    dfs_node_ids = {id(n) for n in partition.dfs_nodes}

    copies:       Dict[int, TreeNode] = {}
    copy_to_orig: Dict[int, TreeNode] = {}
    copy_parent:  Dict[int, TreeNode] = {}

    def _copy(orig: TreeNode, par_copy=None) -> TreeNode:
        if id(orig) in copies:
            return copies[id(orig)]
        c = TreeNode(list(orig.token_ids))
        copies[id(orig)]    = c
        copy_to_orig[id(c)] = orig
        if par_copy is not None:
            copy_parent[id(c)] = par_copy
        for child in orig.children:
            if id(child) in dfs_node_ids:
                c.children.append(_copy(child, c))
        return c

    entry_copies = [_copy(n) for n in partition.entry_nodes if id(n) in dfs_node_ids]
    dfs_info = build_dfs_tree_info(entry_copies)

    orig_id_to_start:     Dict[int, int] = {}
    orig_id_to_end:       Dict[int, int] = {}
    orig_id_to_par:       Dict[int, int] = {}
    orig_id_to_chunk_idx: Dict[int, int] = {}   # NEW: DFS chunk index for SSM state extraction
    ci = [0]

    def _walk(c_node: TreeNode, par_orig_id: int):
        idx         = ci[0]; ci[0] += 1
        s, e        = dfs_info.chunk_boundaries[idx]
        orig        = copy_to_orig[id(c_node)]
        orig_id_to_start[id(orig)]     = s
        orig_id_to_end[id(orig)]       = e
        orig_id_to_par[id(orig)]       = par_orig_id
        orig_id_to_chunk_idx[id(orig)] = idx
        for child in c_node.children:
            _walk(child, id(orig))

    for ec in entry_copies:
        _walk(ec, -1)

    return (dfs_info, orig_id_to_start, orig_id_to_end,
            orig_id_to_par, orig_id_to_chunk_idx, copy_to_orig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gateway
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubtreeGateway:
    key_gates:        List[torch.Tensor]   # detached leaves [1,kv_heads,past_len,d]
    val_gates:        List[torch.Tensor]
    raw_keys:         List[torch.Tensor]   # with grad_fn
    raw_vals:         List[torch.Tensor]
    fp32_k:           List[List]
    fp32_v:           List[List]
    past_len:         int                  # total tokens in gateway KV
    attn_bias:        torch.Tensor         # [past_len] additive bias: 0 or -inf
    pos_offset:       int                  # depth-based position offset for child
    ssm:              Optional[object] = None   # SSMGateway or None (pure-attention)


def _make_child_gateways(
    full_kv:             List[Tuple[torch.Tensor, torch.Tensor]],
    dfs_info:            DFSTreeInfo,
    orig_id_to_start:    Dict[int, int],
    orig_id_to_end:      Dict[int, int],
    orig_id_to_par:      Dict[int, int],
    orig_id_to_chunk_idx: Dict[int, int],
    cut_nodes:           List[TreeNode],
    parent_pos_offset:   int,
    parent_past_len:     int,
    # SSM states captured during parent partition forward (may be empty for pure-attn)
    ssm_rec_captured:    Dict[int, Dict[int, torch.Tensor]],  # chunk_idx→{layer→state}
    ssm_conv_captured:   Dict[int, Dict[int, torch.Tensor]],
) -> Dict[int, SubtreeGateway]:
    gateways: Dict[int, SubtreeGateway] = {}

    def _ancestors(cut_node: TreeNode):
        ranges = []
        nid = id(cut_node)
        while nid != -1 and nid in orig_id_to_start:
            ranges.append((orig_id_to_start[nid], orig_id_to_end[nid]))
            nid = orig_id_to_par.get(nid, -1)
        return ranges

    for cn in cut_nodes:
        cut_end  = orig_id_to_end[id(cn)]
        total    = parent_past_len + cut_end

        cn_start     = orig_id_to_start[id(cn)]
        cn_depth_pos = dfs_info.dfs_pos_ids[cn_start]
        pos_off      = parent_pos_offset + cn_depth_pos + len(cn.token_ids)

        bias = torch.full((total,), float('-inf'))
        bias[:parent_past_len] = 0.0
        for (s, e) in _ancestors(cn):
            bias[parent_past_len + s : parent_past_len + e] = 0.0

        # ── KV gateway ────────────────────────────────────────────────────────
        key_gates, val_gates, raw_keys, raw_vals, fp32_k, fp32_v = [], [], [], [], [], []
        for fk, fv in full_kv:
            rk = fk[:, :, :total, :]
            rv = fv[:, :, :total, :]
            gk = rk.detach().requires_grad_(True)
            gv = rv.detach().requires_grad_(True)
            ak: List = [None];  av: List = [None]

            def _h(acc):
                def hook(g):
                    g32 = g.detach().float().cpu()
                    acc[0] = g32 if acc[0] is None else acc[0] + g32
                    return torch.zeros_like(g)
                return hook

            gk.register_hook(_h(ak));  gv.register_hook(_h(av))
            key_gates.append(gk);  val_gates.append(gv)
            raw_keys.append(rk);   raw_vals.append(rv)
            fp32_k.append(ak);     fp32_v.append(av)

        # ── SSM gateway (if states were captured) ─────────────────────────────
        ssm_gw = None
        c_idx = orig_id_to_chunk_idx.get(id(cn))
        if (_SSM_AVAILABLE and c_idx is not None
                and c_idx in ssm_rec_captured and c_idx in ssm_conv_captured):
            ssm_gw = _make_ssm_gateway(
                rec_states_by_layer  = ssm_rec_captured[c_idx],
                conv_states_by_layer = ssm_conv_captured[c_idx],
            )

        gateways[id(cn)] = SubtreeGateway(
            key_gates=key_gates, val_gates=val_gates,
            raw_keys=raw_keys,   raw_vals=raw_vals,
            fp32_k=fp32_k,       fp32_v=fp32_v,
            past_len=total,      attn_bias=bias,
            pos_offset=pos_off,  ssm=ssm_gw,
        )

    return gateways


def _chain_gateway_grads(gw: SubtreeGateway, retain_graph: bool = False) -> None:
    """Chain KV and SSM gateway gradients back through raw tensors."""
    raws, grads = [], []
    for rk, rv, ak, av in zip(gw.raw_keys, gw.raw_vals, gw.fp32_k, gw.fp32_v):
        if ak[0] is not None:
            raws.append(rk);  grads.append(ak[0].to(rk.device))
        if av[0] is not None:
            raws.append(rv);  grads.append(av[0].to(rv.device))

    # SSM gateway grads — collect but pass retain_graph only at the end
    if _SSM_AVAILABLE and gw.ssm is not None:
        for raw, acc in zip(gw.ssm.rec_raw, gw.ssm.rec_fp32):
            if raw is not None and acc[0] is not None:
                raws.append(raw); grads.append(acc[0].to(raw.device))
        for raw, acc in zip(gw.ssm.conv_raw, gw.ssm.conv_fp32):
            if raw is not None and acc[0] is not None:
                raws.append(raw); grads.append(acc[0].to(raw.device))

    if raws:
        torch.autograd.backward(raws, grads, retain_graph=retain_graph)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Subtree forward
# ─────────────────────────────────────────────────────────────────────────────

def _subtree_forward(
    model,
    partition: SubtreePartition,
    gateway:   Optional[SubtreeGateway],
    dev:       torch.device,
    dtype:     torch.dtype,
) -> Tuple[torch.Tensor,
           DFSTreeInfo,
           List[Tuple[torch.Tensor, torch.Tensor]],
           Dict[int, int],   # orig_id_to_start
           Dict[int, int],   # orig_id_to_end
           Dict[int, int],   # orig_id_to_par
           Dict[int, int],   # orig_id_to_chunk_idx  (NEW — for SSM state extraction)
           Dict[int, TreeNode]]:
    """
    DFS forward for one partition with optional past-KV+SSM gateway.

    Returns
    -------
    logits, dfs_info, full_kv,
    orig_id_to_start, orig_id_to_end, orig_id_to_par,
    orig_id_to_chunk_idx, copy_id_to_orig
    """
    (dfs_info, orig_id_to_start, orig_id_to_end,
     orig_id_to_par, orig_id_to_chunk_idx, copy_id_to_orig) = _dfs_info_for_partition(partition)
    N        = dfs_info.N
    past_len = gateway.past_len if gateway is not None else 0

    ids_t = torch.tensor([dfs_info.dfs_token_ids], device=dev)
    pos_t = torch.tensor(
        [[p + partition.pos_offset for p in dfs_info.dfs_pos_ids]], device=dev
    )

    # Combined attention mask
    attn_4d = torch.full((1, 1, N, past_len + N), float('-inf'), dtype=dtype, device=dev)
    if past_len > 0:
        attn_4d[0, 0, :, :past_len] = gateway.attn_bias.to(dev)
    rows, cols = dfs_info.attn_mask_bool.nonzero(as_tuple=True)
    attn_4d[0, 0, rows, cols + past_len] = 0.0

    # Past KV as GatewayCache
    if gateway is not None:
        past_cache = GatewayCache(list(zip(gateway.key_gates, gateway.val_gates)))
    else:
        past_cache = GatewayCache([])

    # ── SSM patching ──────────────────────────────────────────────────────────
    # Determine cut_chunk_indices: which DFS chunk indices correspond to cut nodes
    cut_chunk_indices = {
        orig_id_to_chunk_idx[nid]
        for nid in partition.cut_node_set
        if nid in orig_id_to_chunk_idx
    }

    ssm_rec_captured:  Dict[int, Dict[int, torch.Tensor]] = {}  # chunk_idx → {layer_idx: state}
    ssm_conv_captured: Dict[int, Dict[int, torch.Tensor]] = {}

    use_ssm = _SSM_AVAILABLE and _has_ssm(model)
    ssm_saved = {}

    if use_ssm:
        has_gateway_ssm = (gateway is not None and gateway.ssm is not None)

        if cut_chunk_indices:
            # PARENT partition: capture SSM states at cut node positions
            # Build per-layer rec_store / conv_store dicts keyed by layer_idx
            # We'll use a single factory that wraps all SSM layers
            per_layer_rec_store  = {}   # {layer_idx: {chunk_idx: tensor}}
            per_layer_conv_store = {}

            def make_parent_patch(ssm_layer):
                l = ssm_layer.layer_idx
                per_layer_rec_store[l]  = {}
                per_layer_conv_store[l] = {}
                return _make_ssm_forward_parent(
                    ssm_layer,
                    dfs_info.chunk_boundaries,
                    dfs_info.chunk_parent_map,
                    cut_chunk_indices,
                    per_layer_rec_store[l],
                    per_layer_conv_store[l],
                )

            ssm_saved = _patch_ssm_layers(model, make_parent_patch)
            # After forward, per_layer_rec/conv_store will be populated
        elif has_gateway_ssm:
            # CHILD partition: inject SSM states from gateway
            gw_ssm = gateway.ssm

            def make_child_patch(ssm_layer):
                l = ssm_layer.layer_idx
                g_rec  = gw_ssm.rec_gate[l]  if l < len(gw_ssm.rec_gate)  else None
                g_conv = gw_ssm.conv_gate[l] if l < len(gw_ssm.conv_gate) else None
                if g_rec is None or g_conv is None:
                    # Fallback: standard DFS patch (zero initial state)
                    from dfs_tree_forward import _make_dfs_ssm_forward
                    return _make_dfs_ssm_forward(
                        ssm_layer,
                        dfs_info.chunk_boundaries,
                        dfs_info.chunk_parent_map,
                    )
                return _make_ssm_forward_child(
                    ssm_layer,
                    dfs_info.chunk_boundaries,
                    dfs_info.chunk_parent_map,
                    g_rec, g_conv,
                )

            ssm_saved = _patch_ssm_layers(model, make_child_patch)
        else:
            # Root partition with no cuts, or no SSM gateway: standard DFS SSM patch
            from dfs_tree_forward import _patch_ssm, _unpatch_ssm as _dfs_unpatch
            ssm_saved = _patch_ssm(model,
                                    dfs_info.chunk_boundaries,
                                    dfs_info.chunk_parent_map)

    try:
        out = model(
            input_ids       = ids_t,
            attention_mask  = attn_4d,
            position_ids    = pos_t,
            past_key_values = past_cache,
            use_cache       = True,
        )
    finally:
        if use_ssm:
            _unpatch_ssm_layers(ssm_saved)
        else:
            # Try standard DFS unpatch (pure-attention fallback was already set up above)
            pass

    logits   = out.logits
    n_layers = len(past_cache._full_kv)
    full_kv  = [(past_cache._full_kv[l][0], past_cache._full_kv[l][1])
                for l in range(n_layers)]

    # Reorganise captured SSM states: {chunk_idx: {layer_idx: state}}
    if use_ssm and cut_chunk_indices:
        for l, rec_by_chunk in per_layer_rec_store.items():
            for c_idx, state in rec_by_chunk.items():
                ssm_rec_captured.setdefault(c_idx, {})[l] = state
        for l, conv_by_chunk in per_layer_conv_store.items():
            for c_idx, state in conv_by_chunk.items():
                ssm_conv_captured.setdefault(c_idx, {})[l] = state

    return (logits, dfs_info, full_kv,
            orig_id_to_start, orig_id_to_end, orig_id_to_par,
            orig_id_to_chunk_idx, copy_id_to_orig,
            ssm_rec_captured, ssm_conv_captured)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main algorithm
# ─────────────────────────────────────────────────────────────────────────────

def tree_grad_hybrid(
    model,
    roots:        List[TreeNode],
    leaf_loss_fn: Callable[[torch.Tensor, TreeNode], torch.Tensor],
    max_tokens:   int,
) -> None:
    """
    Hybrid tree training: DFS-packed intra-subtree + gateway inter-subtree.

    Parameters
    ----------
    model        : HF causal LM, train mode, zero_grad() already called
    roots        : forest roots
    leaf_loss_fn : loss = leaf_loss_fn(logits [1,L,V], node) → scalar
    max_tokens   : max DFS token count per subtree
    """
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    partitions = partition_tree(roots, max_tokens)

    # id(p) → [(cut_node, child_p)]
    children_of: Dict[int, List[Tuple[TreeNode, SubtreePartition]]] = {
        id(p): [] for p in partitions
    }
    for p in partitions:
        if p.parent is not None:
            children_of[id(p.parent)].append((p.parent_cut_node, p))

    # ── Top-down forward ──────────────────────────────────────────────────────
    # gateways[id(p)] → SubtreeGateway or None
    gateways:     Dict[int, Optional[SubtreeGateway]] = {}
    fwd_cache:    Dict[int, tuple] = {}

    for p in partitions:
        gw = gateways.get(id(p), None)
        p.pos_offset = gw.pos_offset if gw is not None else 0

        (logits, dfs_info, full_kv,
         s_map, e_map, par_map, ci_map, cto,
         ssm_rec, ssm_conv) = _subtree_forward(model, p, gw, dev, dtype)
        fwd_cache[id(p)] = (logits, dfs_info, full_kv, s_map, e_map, par_map, ci_map, cto)

        # Deduplicate cut nodes and create gateways
        cut_nodes_seen = set()
        cut_nodes_for_p = []
        for cn, _ in children_of[id(p)]:
            if id(cn) not in cut_nodes_seen:
                cut_nodes_seen.add(id(cn))
                cut_nodes_for_p.append(cn)
        if cut_nodes_for_p:
            child_gws = _make_child_gateways(
                full_kv              = full_kv,
                dfs_info             = dfs_info,
                orig_id_to_start     = s_map,
                orig_id_to_end       = e_map,
                orig_id_to_par       = par_map,
                orig_id_to_chunk_idx = ci_map,
                cut_nodes            = cut_nodes_for_p,
                parent_pos_offset    = p.pos_offset,
                parent_past_len      = gw.past_len if gw is not None else 0,
                ssm_rec_captured     = ssm_rec,
                ssm_conv_captured    = ssm_conv,
            )
            for cn, child_p in children_of[id(p)]:
                gateways[id(child_p)] = child_gws[id(cn)]

    # ── Bottom-up backward ────────────────────────────────────────────────────
    for p in reversed(partitions):
        logits, dfs_info, full_kv, _, _, _, _, copy_to_orig = fwd_cache[id(p)]

        leaf_chunks = []
        for i, lc in enumerate(dfs_info.leaf_chunk_ids):
            s, e      = dfs_info.chunk_boundaries[lc]
            copy_leaf = dfs_info.leaf_nodes[i]
            orig_leaf = copy_to_orig[id(copy_leaf)]
            if id(orig_leaf) in p.leaf_node_set:
                leaf_chunks.append((s, e, orig_leaf))

        cut_chains = children_of[id(p)]

        # Deduplicate: when the same cut node leads to multiple child partitions
        # (e.g. d_left → P_q and d_left → P_f), they share one gateway object.
        # _chain_gateway_grads must be called EXACTLY ONCE per unique gateway.
        seen_gw_ids = set()
        unique_cut_chains = []
        for cut_node, child_p in cut_chains:
            gw = gateways.get(id(child_p))
            if gw is not None and id(gw) not in seen_gw_ids:
                seen_gw_ids.add(id(gw))
                unique_cut_chains.append((cut_node, child_p))

        total_ops = len(leaf_chunks) + len(unique_cut_chains)
        remaining  = [total_ops]

        def _retain():
            remaining[0] -= 1
            return remaining[0] > 0

        for s, e, orig_leaf in leaf_chunks:
            loss = leaf_loss_fn(logits[:, s:e, :], orig_leaf)
            loss.backward(retain_graph=_retain())

        for cut_node, child_p in unique_cut_chains:
            child_gw = gateways[id(child_p)]
            _chain_gateway_grads(child_gw, retain_graph=_retain())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_tree_loss_fn(
    leaf_labels:  List[torch.Tensor],
    leaf_nodes:   List[TreeNode],
    n_leaves:     int,
    ignore_index: int = -100,
) -> Callable:
    label_map = {id(n): lab for n, lab in zip(leaf_nodes, leaf_labels)}

    def _fn(logits: torch.Tensor, node: TreeNode) -> torch.Tensor:
        labels = label_map[id(node)].to(logits.device)
        return F.cross_entropy(logits[0], labels,
                               ignore_index=ignore_index) / n_leaves
    return _fn
