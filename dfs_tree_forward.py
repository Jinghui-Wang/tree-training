#!/usr/bin/env python3
"""
DFS Tree Forward/Backward — Clean Production Implementation

One forward pass processes an entire tree (or forest) of generation paths
that share common prefixes. Verified EXACT-0 forward alignment and
EXACT-0 backward gradient equivalence (see README_dfs_alignment.md).

Supported model architecture: Qwen3.5-35B-A3B (hybrid SSM + attention MoE).
The SSM patching mechanism is generic for any model with
`Qwen3_5MoeGatedDeltaNet` layers accessed as `layer.linear_attn`.

Usage
-----
    from dfs_tree_forward import TreeNode, dfs_tree_forward

    # Build tree
    root = TreeNode([tok_a, tok_b, tok_c])   # shared prefix
    leaf0 = root.add_child([tok_d, tok_e])
    leaf1 = root.add_child([tok_f, tok_g])

    # Single forward pass
    result = dfs_tree_forward(model, roots=[root])

    # Compute loss (predicts next token = EOS at each leaf's last position)
    total_loss = result.total_loss(eos_id)
    total_loss.backward()
    optimizer.step()
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet, l2norm,
)


# ============================================================
# Part 1: Tree data structure
# ============================================================

@dataclass
class TreeNode:
    """
    One node in the generation tree.

    token_ids : token IDs for this node (the "choice" at this depth)
    children  : child nodes; empty = leaf node
    """
    token_ids: List[int]
    children: List[TreeNode] = field(default_factory=list)

    def add_child(self, token_ids: List[int]) -> TreeNode:
        """Append a child node and return it for chaining."""
        child = TreeNode(token_ids)
        self.children.append(child)
        return child

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ============================================================
# Part 2: DFS sequence builder (arbitrary forest)
# ============================================================

@dataclass
class DFSTreeInfo:
    """All tensors and metadata needed to run one DFS tree forward pass."""
    # Input sequences
    dfs_token_ids:    List[int]           # flat DFS token sequence
    dfs_pos_ids:      List[int]           # position IDs (depth-based)
    # Tree topology
    chunk_boundaries: List[Tuple[int,int]]# (start, end) token index per chunk
    chunk_parent_map: List[int]           # parent chunk index per chunk (-1 = root)
    # Attention mask
    attn_mask_bool:   torch.BoolTensor    # [N, N], True = allowed to attend
    # Leaf info
    leaf_chunk_ids:   List[int]           # which chunks are leaves
    leaf_nodes:       List[TreeNode]      # leaf TreeNode objects (same order)
    # Sizes
    N:                int                 # total tokens in DFS sequence
    n_chunks:         int


def build_dfs_tree_info(roots: List[TreeNode]) -> DFSTreeInfo:
    """
    Convert a forest (list of root TreeNodes) into a DFSTreeInfo.

    DFS traversal order: pre-order (parent before children, children
    visited left-to-right). chunk_parent_map[i] < i for all i > 0,
    guaranteeing parent SSM states are always computed before children.

    Position IDs: depth-based. Token at offset j in chunk c gets
    position = (total tokens in ancestor chunks) + j.
    This ensures sibling branches reuse the same position indices,
    matching what a separate per-path forward would use.
    """
    dfs_token_ids:    List[int]           = []
    dfs_pos_ids:      List[int]           = []
    chunk_boundaries: List[Tuple[int,int]]= []
    chunk_parent_map: List[int]           = []
    leaf_chunk_ids:   List[int]           = []
    leaf_nodes:       List[TreeNode]      = []

    chunk_idx = 0

    def _dfs(node: TreeNode, parent_chunk: int, depth_offset: int):
        """
        depth_offset: cumulative token count of all ancestor chunks
                      (= position ID of the first token in this chunk)
        """
        nonlocal chunk_idx
        my_chunk = chunk_idx
        chunk_idx += 1

        c_start = len(dfs_token_ids)
        dfs_token_ids.extend(node.token_ids)
        L = len(node.token_ids)
        for j in range(L):
            dfs_pos_ids.append(depth_offset + j)

        chunk_parent_map.append(parent_chunk)
        chunk_boundaries.append((c_start, c_start + L))

        if node.is_leaf:
            leaf_chunk_ids.append(my_chunk)
            leaf_nodes.append(node)
        else:
            for child in node.children:
                _dfs(child, my_chunk, depth_offset + L)

    for root in roots:
        _dfs(root, parent_chunk=-1, depth_offset=0)

    N        = len(dfs_token_ids)
    n_chunks = len(chunk_parent_map)

    # ---- Ancestor sets (for attention mask) ----
    ancestors: List[Optional[List[int]]] = [None] * n_chunks

    def _get_ancestors(c: int) -> List[int]:
        if ancestors[c] is not None:
            return ancestors[c]
        p = chunk_parent_map[c]
        ancestors[c] = [c] if p == -1 else _get_ancestors(p) + [c]
        return ancestors[c]

    for c in range(n_chunks):
        _get_ancestors(c)

    # ---- N×N boolean attention mask ----
    # Token i can attend to token j iff j is in an ancestor chunk of i
    # (full attention to entire ancestor chunk) or j is in the same chunk
    # as i and j <= i (causal within chunk).
    attn_mask_bool = torch.zeros(N, N, dtype=torch.bool)
    for c in range(n_chunks):
        c_start, c_end = chunk_boundaries[c]
        for anc in ancestors[c]:
            a_start, a_end = chunk_boundaries[anc]
            for i in range(c_start, c_end):
                if anc == c:                      # causal within own chunk
                    for j in range(c_start, i + 1):
                        attn_mask_bool[i, j] = True
                else:                             # full access to ancestor
                    for j in range(a_start, a_end):
                        attn_mask_bool[i, j] = True

    return DFSTreeInfo(
        dfs_token_ids    = dfs_token_ids,
        dfs_pos_ids      = dfs_pos_ids,
        chunk_boundaries = chunk_boundaries,
        chunk_parent_map = chunk_parent_map,
        attn_mask_bool   = attn_mask_bool,
        leaf_chunk_ids   = leaf_chunk_ids,
        leaf_nodes       = leaf_nodes,
        N                = N,
        n_chunks         = n_chunks,
    )


# ============================================================
# Part 3: Tree-correct conv (variable-length chunks)
# ============================================================

def _tree_correct_conv_varlen(
    ssm,
    mixed_qkv_raw: torch.Tensor,
    chunk_boundaries: List[Tuple[int, int]],
    chunk_parent_map: List[int],
) -> torch.Tensor:
    """
    Compute tree-correct conv1d output for a DFS sequence.

    At a DFS backtrack, the standard causal conv1d window would cross
    into sibling tokens. Instead, each chunk uses its PARENT node's
    last (K-1) effective tokens as conv context.

    conv_states[c] stores the last K-1 tokens of the effective sequence
    [ancestor_context, chunk_c], so grandchildren see the full K-1
    history — matching what a separate per-path forward would compute.
    """
    K = ssm.conv_kernel_size
    chunk_outputs: List[torch.Tensor] = []
    conv_states:   Dict[int, torch.Tensor] = {}   # chunk_idx → [B, conv_dim, K-1]

    for c, (c_start, c_end) in enumerate(chunk_boundaries):
        L     = c_end - c_start
        chunk = mixed_qkv_raw[:, :, c_start:c_end]   # [B, conv_dim, L]

        parent_c     = chunk_parent_map[c]
        parent_state = conv_states.get(parent_c, None)   # None if root

        if parent_state is not None:
            # Prepend parent's effective K-1 history
            ext = torch.cat([parent_state.to(chunk.device), chunk], dim=-1)  # [B, conv_dim, K-1+L]
            if ssm.causal_conv1d_fn is not None:
                out = ssm.causal_conv1d_fn(
                    x=ext, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                out = F.silu(ssm.conv1d(ext)[:, :, :ext.shape[-1]])
            chunk_out = out[:, :, (K - 1):]              # drop prepended context → [B, conv_dim, L]
            # Save last K-1 of effective sequence for this chunk's children
            conv_states[c] = ext[:, :, -(K - 1):].clone()
        else:
            # Root chunk: no ancestor context
            if ssm.causal_conv1d_fn is not None:
                chunk_out = ssm.causal_conv1d_fn(
                    x=chunk, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                chunk_out = F.silu(ssm.conv1d(chunk)[:, :, :L])
            # Effective history = [zeros..., chunk]; keep last K-1
            conv_states[c] = F.pad(chunk, (max(0, K - 1 - L), 0))[:, :, -(K - 1):].clone()

        chunk_outputs.append(chunk_out)

    return torch.cat(chunk_outputs, dim=2)   # [B, conv_dim, N]


# ============================================================
# Part 4: Tree-aware SSM (GatedDeltaNet with tree state routing)
# ============================================================

def _tree_gated_delta_net_varlen(
    query:  torch.Tensor,      # [B, S, H, Dk]
    key:    torch.Tensor,
    value:  torch.Tensor,      # [B, S, H, Dv]
    g:      torch.Tensor,      # [B, S, H]
    beta:   torch.Tensor,      # [B, S, H]
    chunk_boundaries: List[Tuple[int, int]],
    chunk_parent_map: List[int],
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """
    Tree-aware GatedDeltaNet forward.

    Identical to standard chunk GatedDeltaNet EXCEPT:
      - Each chunk reads its initial SSM state from all_states[parent+1]
        instead of the sequentially preceding chunk's state.
      - DFS order guarantees parent state is always computed before child.

    Gradient: sibling chunks share the same parent state tensor in
    all_states. PyTorch autograd accumulates their gradients at that
    tensor automatically — no special handling required.

    Returns core_out [B, S, H, Dv] (same dtype as inputs).
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key   = l2norm(key,   dim=-1, eps=1e-6)

    # Work in float32 for numerical stability
    q  = query.transpose(1, 2).contiguous().float()   # [B, H, S, Dk]
    k  = key.transpose(1, 2).contiguous().float()
    v  = value.transpose(1, 2).contiguous().float()   # [B, H, S, Dv]
    bt = beta.transpose(1, 2).contiguous().float()    # [B, H, S]
    g  = g.transpose(1, 2).contiguous().float()       # [B, H, S]

    B, H, S, Dk = k.shape
    Dv  = v.shape[-1]
    dev = q.device

    q = q * (Dk ** -0.5)

    init = (initial_state.float().to(dev) if initial_state is not None
            else torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev))

    # all_states[0] = initial state; all_states[i+1] = state after chunk i
    all_states:  List[torch.Tensor] = [init]
    all_outputs: List[torch.Tensor] = []

    for i, (c_start, c_end) in enumerate(chunk_boundaries):
        L = c_end - c_start

        q_i  = q[:, :, c_start:c_end]     # [B, H, L, Dk]
        k_i  = k[:, :, c_start:c_end]
        v_i  = v[:, :, c_start:c_end]
        bt_i = bt[:, :, c_start:c_end]    # [B, H, L]
        g_i  = g[:, :, c_start:c_end]     # [B, H, L]

        v_beta_i = v_i * bt_i.unsqueeze(-1)   # [B, H, L, Dv]
        k_beta_i = k_i * bt_i.unsqueeze(-1)   # [B, H, L, Dk]

        g_cum = g_i.cumsum(dim=-1)             # [B, H, L]

        # Decay mask: lower-triangular, [j,k] = exp(g_cum[j] - g_cum[k]) for j>=k
        decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()

        mask_upper_incl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev))
        mask_upper_excl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

        # Within-chunk correction (state-aware attention)
        attn_raw = -(k_beta_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_incl, 0)
        attn_rows = [attn_raw[..., 0:1, :]]
        for j in range(1, L):
            corrected_j = attn_raw[..., j:j+1, :] + attn_raw[..., j:j+1, :j] @ attn_raw[..., :j, :]
            attn_rows.append(corrected_j)
        attn       = torch.cat(attn_rows, dim=-2) + torch.eye(L, dtype=attn_raw.dtype, device=dev)
        value_corr = attn @ v_beta_i                                       # [B, H, L, Dv]
        k_cumdecay = attn @ (k_beta_i * g_cum.exp().unsqueeze(-1))        # [B, H, L, Dk]

        # Cross-chunk: read from PARENT state (tree routing)
        parent_idx    = int(chunk_parent_map[i]) + 1   # -1 → 0, j → j+1
        current_state = all_states[parent_idx]          # [B, H, Dk, Dv]

        v_prime = k_cumdecay @ current_state            # [B, H, L, Dv]
        v_new   = value_corr - v_prime                  # [B, H, L, Dv]

        attn_within = (q_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_excl, 0)
        attn_inter  = (q_i * g_cum.exp().unsqueeze(-1)) @ current_state   # [B, H, L, Dv]
        out_i = attn_inter + attn_within @ v_new                           # [B, H, L, Dv]
        all_outputs.append(out_i)

        # Update state for this chunk's children
        new_state = (
            current_state * g_cum[:, :, -1, None, None].exp()
            + (k_i * (g_cum[:, :, -1:] - g_cum).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )
        all_states.append(new_state)

    core_out = torch.cat(all_outputs, dim=2)                           # [B, H, S, Dv]
    return core_out.transpose(1, 2).contiguous().to(initial_dtype)    # [B, S, H, Dv]


# ============================================================
# Part 5: SSM layer patched forward
# ============================================================

def _make_dfs_ssm_forward(ssm_layer, chunk_boundaries, chunk_parent_map):
    """
    Return a patched forward function for one SSM (GatedDeltaNet) layer.

    Key properties:
    1. Per-chunk projections: each chunk uses [B, Lc, d] GEMM, matching
       the shape of a separate per-path forward exactly. Ensures cuBLAS
       picks the same kernel → EXACT 0 forward alignment.
    2. Tree-correct conv: uses parent's K-1 history at backtrack points.
    3. Tree-aware SSM: each chunk starts from its parent's recurrent state.
    4. Per-chunk norm+out_proj: same GEMM shape as separate paths.
    """
    def patched_forward(hidden_states, cache_params=None,
                        cache_position=None, attention_mask=None):
        self = ssm_layer
        B, S, _ = hidden_states.shape

        # ── Per-chunk input projections ─────────────────────────────
        qkv_chunks, z_chunks, b_chunks, a_chunks = [], [], [], []
        for c_start, c_end in chunk_boundaries:
            h_c = hidden_states[:, c_start:c_end, :]            # [B, Lc, d]
            qkv_chunks.append(self.in_proj_qkv(h_c).transpose(1, 2))  # [B, conv_dim, Lc]
            z_chunks.append(self.in_proj_z(h_c))
            b_chunks.append(self.in_proj_b(h_c))
            a_chunks.append(self.in_proj_a(h_c))

        mixed_qkv = torch.cat(qkv_chunks, dim=2)                      # [B, conv_dim, S]
        z = torch.cat(z_chunks, dim=1).reshape(B, S, -1, self.head_v_dim)
        b = torch.cat(b_chunks, dim=1)
        a = torch.cat(a_chunks, dim=1)

        # ── Tree-correct conv ────────────────────────────────────────
        mixed_qkv = _tree_correct_conv_varlen(
            self, mixed_qkv, chunk_boundaries, chunk_parent_map
        )

        # ── Split Q, K, V ────────────────────────────────────────────
        mixed_qkv = mixed_qkv.transpose(1, 2)                         # [B, S, conv_dim]
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, S, -1, self.head_k_dim)
        k = k.reshape(B, S, -1, self.head_k_dim)
        v = v.reshape(B, S, -1, self.head_v_dim)

        beta = b.sigmoid()
        g    = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA expansion
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)

        # ── Tree-aware SSM ───────────────────────────────────────────
        core_attn_out = _tree_gated_delta_net_varlen(
            q, k, v, g, beta,
            chunk_boundaries = chunk_boundaries,
            chunk_parent_map = chunk_parent_map,
            use_qk_l2norm    = True,
        )

        # ── Per-chunk norm + out_proj ────────────────────────────────
        out_chunks = []
        for c_start, c_end in chunk_boundaries:
            Lc   = c_end - c_start
            ca_c = core_attn_out[:, c_start:c_end, :, :]    # [B, Lc, H, Dv]
            z_c  = z[:, c_start:c_end, :, :]                # [B, Lc, H, Dv]
            ca_c = self.norm(
                ca_c.reshape(-1, self.head_v_dim),
                z_c.reshape(-1, self.head_v_dim),
            )                                                # [B*Lc*H, Dv]
            ca_c = ca_c.reshape(B, Lc, -1)                  # [B, Lc, H*Dv]
            out_chunks.append(self.out_proj(ca_c))           # [B, Lc, d]

        return torch.cat(out_chunks, dim=1)                  # [B, S, d]

    return patched_forward


# ============================================================
# Part 6: SSM patching (works with or without accelerate device_map)
# ============================================================

def _patch_ssm(model, chunk_boundaries, chunk_parent_map) -> dict:
    """
    Patch all GatedDeltaNet layers with tree-aware forward.
    Compatible with both accelerate device_map='auto' and single-device.
    Returns saved state for restoring via _unpatch_ssm().
    """
    saved = {}
    for layer in model.model.layers:
        if not hasattr(layer, 'linear_attn'):
            continue
        ssm = layer.linear_attn
        patched = _make_dfs_ssm_forward(ssm, chunk_boundaries, chunk_parent_map)
        if hasattr(ssm, '_old_forward'):
            # accelerate wraps forward → patch _old_forward
            saved[ssm] = ('_old_forward', ssm._old_forward)
            ssm._old_forward = patched
        else:
            # single-device: patch forward directly
            saved[ssm] = ('forward', ssm.forward)
            ssm.forward = patched
    return saved


def _unpatch_ssm(saved: dict):
    for ssm, (attr, orig) in saved.items():
        setattr(ssm, attr, orig)


# ============================================================
# Part 7: Main forward result
# ============================================================

@dataclass
class DFSTreeResult:
    """
    Output of dfs_tree_forward().

    leaf_logits: list of [1, Lc, V] tensors, one per leaf, in DFS order.
                 All tensors are views into the single forward pass output.
                 .backward() works correctly on any computation derived
                 from these tensors.
    info:        DFSTreeInfo with sequence metadata.
    """
    leaf_logits: List[torch.Tensor]
    leaf_nodes:  List[TreeNode]
    info:        DFSTreeInfo

    def total_loss(self, labels: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute sum of cross-entropy losses over all leaves.

        labels: list of 1-D LongTensors, one per leaf.
                For "predict EOS after last token": labels[i] = tensor([eos_id]).
                For multi-token leaf chunks: labels[i] = next_token_ids (length Lc).

        Returns scalar loss; calling .backward() gives correct gradients
        for all model parameters (equivalent to per-path separate backward).
        """
        assert len(labels) == len(self.leaf_logits), \
            f"Expected {len(self.leaf_logits)} label tensors, got {len(labels)}"
        total = sum(
            F.cross_entropy(logits.squeeze(0), lab)   # [Lc, V], [Lc]
            for logits, lab in zip(self.leaf_logits, labels)
        )
        return total


# ============================================================
# Part 8: Main API
# ============================================================

def dfs_tree_forward(
    model,
    roots:         List[TreeNode],
    dtype:         Optional[torch.dtype] = None,
) -> DFSTreeResult:
    """
    Run one DFS-packed forward pass over a forest of generation paths.

    Parameters
    ----------
    model : AutoModelForCausalLM (Qwen3.5-35B-A3B or compatible)
    roots : list of TreeNode, each a root of a subtree (or the single root)
    dtype : override attention mask dtype (default: model's parameter dtype)

    Returns
    -------
    DFSTreeResult with per-leaf logit tensors (with gradient support).

    Notes
    -----
    - model.eval() recommended; model.train() works but batch norm etc. differ.
    - Gradients flow correctly through the returned logits.
    - For EXACT-0 forward alignment (bfloat16): requires deterministic mode.
      Set CUBLAS_WORKSPACE_CONFIG=:4096:8 and use_deterministic_algorithms(True).
    - For EXACT-0 backward verification: use float32 model.
    """
    dev   = next(model.parameters()).device
    mdtype = dtype or next(model.parameters()).dtype

    # ---- Build DFS sequences ----
    info = build_dfs_tree_info(roots)

    ids_t = torch.tensor([info.dfs_token_ids], device=dev)
    pos_t = torch.tensor([info.dfs_pos_ids],   device=dev)

    # Additive attention mask: 0 where allowed, -inf where blocked
    attn_4d = torch.full((1, 1, info.N, info.N), float('-inf'), dtype=mdtype, device=dev)
    attn_4d[0, 0][info.attn_mask_bool.to(dev)] = 0.0

    # ---- Single forward pass ----
    saved = _patch_ssm(model, info.chunk_boundaries, info.chunk_parent_map)
    try:
        out = model(
            input_ids      = ids_t,
            attention_mask = attn_4d,
            position_ids   = pos_t,
            use_cache      = False,
        )
    finally:
        _unpatch_ssm(saved)   # always restore, even if forward raises

    logits = out.logits   # [1, N, V]

    # ---- Extract leaf logits ----
    leaf_logits = []
    for lc in info.leaf_chunk_ids:
        s, e = info.chunk_boundaries[lc]
        leaf_logits.append(logits[:, s:e, :])   # [1, Lc, V], gradient preserved

    return DFSTreeResult(
        leaf_logits = leaf_logits,
        leaf_nodes  = info.leaf_nodes,
        info        = info,
    )
