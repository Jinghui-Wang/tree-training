#!/usr/bin/env python3
"""
Approach B: DFS-packed tree with tree-aware SSM (one forward pass, memory-efficient).

Variable-length chunks — NO intra-node padding.
Each tree node contributes exactly its actual token count to the DFS sequence.

Core innovation: torch_chunk_gated_delta_rule_tree_varlen
  - chunk_boundaries: list of (start, end) token positions per tree node
  - chunk_parent_map[i] = parent chunk index (-1 = use initial_state)
  - Chunk i uses all_states[chunk_parent_map[i]+1] as initial SSM state
    instead of the sequentially preceding state
  - Gradient flows correctly: sibling chunks share the same parent state tensor
    and their gradients accumulate at the parent via autograd

Memory: O(N × d) where N = sum of actual node token counts (no padding inflation)

DFS sequence (binary tree depth=2, prefix=P, L1=A, L2=B tokens):
  [p_0..p_{P-1} | L1a_toks | L2aa_toks | L2ab_toks | L1b_toks | L2ba_toks | L2bb_toks]
   chunk 0        chunk 1    chunk 2     chunk 3      chunk 4    chunk 5     chunk 6

chunk_parent_map:  [-1, 0, 1, 1, 0, 4, 4]
chunk_boundaries:  [(0,P),(P,P+A),(P+A,P+A+B),(P+A+B,P+A+2B),(P+2A,P+2A+B),...]
"""

import copy, itertools
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet, l2norm,
)


# ============================================================================
# Part 1: Variable-length tree-aware GatedDeltaNet chunk algorithm
# ============================================================================

def torch_chunk_gated_delta_rule_tree_varlen(
    query, key, value, g, beta,
    chunk_boundaries,       # list[(start,end)] token index range per tree node
    chunk_parent_map,       # list[int] parent chunk idx per node; -1 = initial_state
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """
    Tree-aware GatedDeltaNet with variable-length chunks (no intra-node padding).

    Mathematically identical to torch_chunk_gated_delta_rule EXCEPT:
      - Each chunk can have a different length L_i
      - all_states is a list; chunk i reads: current_state = all_states[parent_map[i]+1]
        (i.e. -1 -> index 0 = initial_state, j -> index j+1 = state after chunk j)

    DFS order guarantees chunk_parent_map[i] < i for all i > 0, so each parent
    state is computed before it is needed.

    Args:
        query, key, value: [B, S, H, d]  (H heads; q/k may have been repeat_interleaved)
        g, beta:           [B, S, H]     (scalar per head per token)
        chunk_boundaries:  [(s0,e0), ...]  token index ranges summing to S
        chunk_parent_map:  tree topology
    Returns:
        core_out:   [B, S, H, Dv]
        last_state: [B, H, Dk, Dv] or None
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key   = l2norm(key,   dim=-1, eps=1e-6)

    # Transpose to [B, H, S, d] and use float32 for precision
    q  = query.transpose(1, 2).contiguous().float()   # [B, H, S, Dk]
    k  = key.transpose(1, 2).contiguous().float()
    v  = value.transpose(1, 2).contiguous().float()   # [B, H, S, Dv]
    bt = beta.transpose(1, 2).contiguous().float()    # [B, H, S]
    g  = g.transpose(1, 2).contiguous().float()       # [B, H, S]

    B, H, S, Dk = k.shape
    Dv  = v.shape[-1]
    dev = q.device

    q = q * (1.0 / (Dk ** 0.5))

    init = (initial_state.float().to(dev) if initial_state is not None
            else torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev))

    # all_states[0] = init, all_states[i+1] = state after processing chunk i
    # DFS order ensures parent always appears earlier in the list.
    all_states  = [init]
    all_outputs = []   # one [B, H, L_i, Dv] tensor per chunk

    for i, (c_start, c_end) in enumerate(chunk_boundaries):
        L = c_end - c_start

        # Extract chunk slices [B, H, L, d]
        q_i  = q[:, :, c_start:c_end]
        k_i  = k[:, :, c_start:c_end]
        v_i  = v[:, :, c_start:c_end]
        bt_i = bt[:, :, c_start:c_end]   # [B, H, L]
        g_i  = g[:, :, c_start:c_end]    # [B, H, L]

        v_beta_i = v_i * bt_i.unsqueeze(-1)    # [B, H, L, Dv]
        k_beta_i = k_i * bt_i.unsqueeze(-1)    # [B, H, L, Dk]

        # Cumulative decay within this chunk
        g_cum = g_i.cumsum(dim=-1)              # [B, H, L]

        # Decay mask: lower-triangular, entry[j,k] = exp(g_cum[j]-g_cum[k]) for j>=k
        decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()  # [B,H,L,L]

        mask_upper_incl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev))
        mask_upper_excl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

        # ---- Within-chunk SSM correction ----
        # Computes the state-corrected values and key-cumdecay for cross-chunk use
        attn_raw = -(k_beta_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_incl, 0)
        # Build corrected attn row-by-row (out-of-place to preserve autograd graph).
        # For row j: corrected[j] = raw[j] + raw[j, :j] @ raw[:j, :]
        attn_rows = [attn_raw[..., 0:1, :]]   # first row unchanged (no predecessors)
        for j in range(1, L):
            row_j = attn_raw[..., j:j+1, :j]    # [B, H, 1, j]  (first j entries of row j)
            sub_j = attn_raw[..., :j, :]         # [B, H, j, L]
            # row_j @ sub_j → [B, H, 1, L]; combined with raw row j → [B, H, 1, L]
            corrected_j = attn_raw[..., j:j+1, :] + row_j @ sub_j
            attn_rows.append(corrected_j)
        attn       = torch.cat(attn_rows, dim=-2) + torch.eye(L, dtype=attn_raw.dtype, device=dev)
        value_corr = attn @ v_beta_i                                        # [B, H, L, Dv]
        k_cumdecay = attn @ (k_beta_i * g_cum.exp().unsqueeze(-1))         # [B, H, L, Dk]

        # ---- Cross-chunk contribution: read from PARENT state (tree topology) ----
        parent_list_idx = int(chunk_parent_map[i]) + 1   # -1 -> 0, j -> j+1
        current_state   = all_states[parent_list_idx]     # [B, H, Dk, Dv]

        v_prime     = k_cumdecay @ current_state           # [B, H, L, Dv]
        v_new       = value_corr - v_prime                 # [B, H, L, Dv]

        attn_within = (q_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_excl, 0)
        attn_inter  = (q_i * g_cum.exp().unsqueeze(-1)) @ current_state     # [B, H, L, Dv]

        out_i = attn_inter + attn_within @ v_new           # [B, H, L, Dv]
        all_outputs.append(out_i)

        # ---- Store new state for children ----
        new_state = (
            current_state * g_cum[:, :, -1, None, None].exp()
            + (k_i * (g_cum[:, :, -1:] - g_cum).exp().unsqueeze(-1)).transpose(-1, -2)
            @ v_new
        )    # [B, H, Dk, Dv]
        all_states.append(new_state)

    # Reassemble: cat over L_i dims, then transpose back to [B, S, H, Dv]
    core_out = torch.cat(all_outputs, dim=2)                          # [B, H, S, Dv]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype) # [B, S, H, Dv]

    last_state = all_states[-1] if output_final_state else None
    return core_out, last_state


# ============================================================================
# Part 2: DFS tree sequence builder — NO padding
# ============================================================================

def build_dfs_info(prefix_ids, level_choices_ids, prefix_pos_start=0):
    """
    Build all data structures for DFS-packed tree processing.

    NO intra-node padding: DFS sequence = concatenation of raw node token lists.
    Sibling nodes naturally share the same position ID range (DFS backtracking).

    Args:
        prefix_ids:        list[int]  root prefix token IDs
        level_choices_ids: list of D lists, each = list of B token lists
                           All choices at the same depth must have the same length.
        prefix_pos_start:  starting position ID

    Returns dict:
        dfs_token_ids:    list[int]       full DFS token sequence (no padding)
        dfs_pos_ids:      list[int]       position IDs
        chunk_parent_map: list[int]       parent chunk per node (-1 = initial)
        chunk_boundaries: list[(int,int)] token index range per chunk
        node_token_ranges: same as chunk_boundaries (alias)
        attn_mask_bool:   BoolTensor[N,N] True = allowed to attend
        leaf_chunk_ids:   list[int]       which chunks are leaf nodes
        n_chunks:         int
        N:                int             total DFS sequence length
        prefix_len:       int
        level_lens:       list[int]       tokens per level choice
    """
    DEPTH     = len(level_choices_ids)
    BRANCHING = len(level_choices_ids[0])
    level_lens = [len(level_choices_ids[d][0]) for d in range(DEPTH)]
    prefix_len = len(prefix_ids)

    def pos_for_depth(d):
        start = prefix_pos_start + prefix_len + sum(level_lens[:d])
        return list(range(start, start + level_lens[d]))

    dfs_token_ids    = []
    dfs_pos_ids      = []
    chunk_parent_map = []
    chunk_boundaries = []
    leaf_chunk_ids   = []
    chunk_idx        = 0

    def dfs(depth, parent_chunk, tokens, pos_ids):
        nonlocal chunk_idx
        my_chunk  = chunk_idx
        chunk_idx += 1
        c_start   = len(dfs_token_ids)
        dfs_token_ids.extend(tokens)
        dfs_pos_ids.extend(pos_ids)
        chunk_parent_map.append(parent_chunk)
        chunk_boundaries.append((c_start, len(dfs_token_ids)))
        if depth == DEPTH:
            leaf_chunk_ids.append(my_chunk)
            return
        for b in range(BRANCHING):
            dfs(depth + 1, my_chunk,
                level_choices_ids[depth][b],
                pos_for_depth(depth))

    # Root: prefix (no padding)
    dfs_token_ids.extend(prefix_ids)
    dfs_pos_ids.extend(range(prefix_pos_start, prefix_pos_start + prefix_len))
    chunk_parent_map.append(-1)
    chunk_boundaries.append((0, prefix_len))
    root_chunk = 0
    chunk_idx  = 1

    for b in range(BRANCHING):
        dfs(1, root_chunk,
            level_choices_ids[0][b],
            pos_for_depth(0))

    N        = len(dfs_token_ids)
    n_chunks = len(chunk_parent_map)

    # ---- Ancestor sets ----
    ancestors = [None] * n_chunks
    def get_ancestors(c):
        if ancestors[c] is not None:
            return ancestors[c]
        p = chunk_parent_map[c]
        ancestors[c] = [c] if p == -1 else get_ancestors(p) + [c]
        return ancestors[c]
    for c in range(n_chunks):
        get_ancestors(c)

    # ---- N×N bool attention mask ----
    attn_mask_bool = torch.zeros(N, N, dtype=torch.bool)
    for c in range(n_chunks):
        c_start, c_end = chunk_boundaries[c]
        for anc_c in ancestors[c]:
            a_start, a_end = chunk_boundaries[anc_c]
            if anc_c == c:
                # Causal within own chunk
                for i in range(c_start, c_end):
                    for j in range(c_start, i + 1):
                        attn_mask_bool[i, j] = True
            else:
                # Full access to all ancestor chunk tokens
                for i in range(c_start, c_end):
                    for j in range(a_start, a_end):
                        attn_mask_bool[i, j] = True

    return {
        'dfs_token_ids':    dfs_token_ids,
        'dfs_pos_ids':      dfs_pos_ids,
        'chunk_parent_map': chunk_parent_map,
        'chunk_boundaries': chunk_boundaries,
        'node_token_ranges': chunk_boundaries,   # alias for compatibility
        'attn_mask_bool':   attn_mask_bool,
        'leaf_chunk_ids':   leaf_chunk_ids,
        'n_chunks':         n_chunks,
        'N':                N,
        'prefix_len':       prefix_len,
        'level_lens':       level_lens,
    }


# ============================================================================
# Part 3: Tree-correct conv (variable-length chunks)
# ============================================================================

def _tree_correct_conv_varlen(ssm, mixed_qkv_raw, chunk_boundaries, chunk_parent_map):
    """
    Compute tree-correct conv output for a DFS sequence with variable-length chunks.

    At a DFS backtrack position, the conv1d window would cross into sibling tokens
    (incorrect). Instead, each chunk uses its PARENT node's last (K-1) raw QKV tokens
    as conv context, giving mathematically the same result as running the sub-path
    [root → parent → this_node] separately.

    DFS order guarantees chunk_parent_map[c] < c for c > 0, so each parent conv
    state is saved before any child processes it.

    Args:
        ssm:              Qwen3_5MoeGatedDeltaNet instance
        mixed_qkv_raw:    [B, conv_dim, N]  raw QKV projections (before conv)
        chunk_boundaries: list of (start, end) per chunk
        chunk_parent_map: list of parent chunk indices
    Returns:
        output: [B, conv_dim, N]  tree-correct conv output
    """
    K      = ssm.conv_kernel_size
    chunk_outputs = []   # collect per-chunk outputs, cat at the end (no in-place)
    conv_states   = {}   # chunk_idx → [B, conv_dim, K] detached conv context for children

    for c, (c_start, c_end) in enumerate(chunk_boundaries):
        L     = c_end - c_start
        chunk = mixed_qkv_raw[:, :, c_start:c_end]   # [B, conv_dim, L]

        parent_c     = chunk_parent_map[c]
        parent_state = conv_states.get(parent_c, None)

        if parent_state is not None:
            ctx = parent_state   # already exactly K-1 tokens
            ext = torch.cat([ctx.to(chunk.device), chunk], dim=-1)   # [B, conv_dim, K-1+L]
            if ssm.causal_conv1d_fn is not None:
                out = ssm.causal_conv1d_fn(
                    x=ext, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                out = F.silu(ssm.conv1d(ext)[:, :, :ext.shape[-1]])
            chunk_out = out[:, :, (K - 1):]   # drop prepended context → [B, conv_dim, L]
        else:
            if ssm.causal_conv1d_fn is not None:
                chunk_out = ssm.causal_conv1d_fn(
                    x=chunk, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                chunk_out = F.silu(ssm.conv1d(chunk)[:, :, :L])

        chunk_outputs.append(chunk_out)

        # Save last K-1 EFFECTIVE tokens (including ancestor context) for children.
        # Must include ancestor tokens so grandchildren see the full K-1 history,
        # matching what a separate per-path forward would compute.
        if parent_state is not None:
            # ext = [parent_ctx (K-1 tokens), chunk (L tokens)]; take last K-1
            conv_states[c] = ext[:, :, -(K - 1):].clone()
        else:
            # No ancestor: effective sequence is [zeros..., chunk]; take last K-1
            conv_states[c] = F.pad(chunk, (max(0, K - 1 - L), 0))[:, :, -(K - 1):].clone()

    return torch.cat(chunk_outputs, dim=2)


# ============================================================================
# Part 4: SSM layer patch for DFS tree forward
# ============================================================================

def _make_dfs_ssm_forward(ssm_layer, chunk_boundaries, chunk_parent_map):
    """
    Create a patched _old_forward for an SSM (GatedDeltaNet) layer that uses:
      1. tree-correct conv  (_tree_correct_conv_varlen)
      2. tree-aware SSM     (torch_chunk_gated_delta_rule_tree_varlen)

    Patching _old_forward (not forward) is required for compatibility with
    accelerate's device-map hooks, which store the original in _old_forward.
    """
    def patched_forward(hidden_states, cache_params=None, cache_position=None,
                        attention_mask=None):
        self    = ssm_layer
        B, S, _ = hidden_states.shape

        # Per-chunk projections: each chunk uses [B, Lc, d] GEMM, matching the
        # separate per-path forward's GEMM shape exactly → exact 0 alignment in bfloat16.
        # Projecting the full S-token sequence would use a different GEMM shape than
        # separate paths (S=5 or S=3), causing cuBLAS to pick a different accumulation
        # order and producing small but non-zero precision differences.
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

        # Tree-correct conv: replaces the standard causal_conv1d which would
        # use sibling tokens as context at DFS backtrack positions
        mixed_qkv = _tree_correct_conv_varlen(
            self, mixed_qkv, chunk_boundaries, chunk_parent_map
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)   # [B, S, conv_dim]
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, S, -1, self.head_k_dim)
        k = k.reshape(B, S, -1, self.head_k_dim)
        v = v.reshape(B, S, -1, self.head_v_dim)

        beta = b.sigmoid()
        g    = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)
            # beta and g are already [B, S, num_v_heads]; no repeat needed

        # Tree-aware SSM: each chunk starts from its parent's recurrent state
        core_attn_out, _ = torch_chunk_gated_delta_rule_tree_varlen(
            q, k, v, g, beta,
            chunk_boundaries=chunk_boundaries,
            chunk_parent_map=chunk_parent_map,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Per-chunk norm + out_proj to ensure GEMM shapes match separate paths.
        out_chunks = []
        for c_start, c_end in chunk_boundaries:
            Lc   = c_end - c_start
            ca_c = core_attn_out[:, c_start:c_end, :, :]       # [B, Lc, H, Dv]
            z_c  = z[:, c_start:c_end, :, :]                   # [B, Lc, H, Dv]
            ca_c = self.norm(ca_c.reshape(-1, self.head_v_dim),
                             z_c.reshape(-1, self.head_v_dim))  # [B*Lc*H, Dv]
            ca_c = ca_c.reshape(B, Lc, -1)                     # [B, Lc, H*Dv]
            out_chunks.append(self.out_proj(ca_c))              # [B, Lc, d]
        return torch.cat(out_chunks, dim=1)                     # [B, S, d]

    return patched_forward


def patch_ssm_for_dfs(model, chunk_boundaries, chunk_parent_map):
    """Patch all SSM layers via _old_forward (accelerate-compatible)."""
    saved = {}
    for layer in model.model.layers:
        if hasattr(layer, 'linear_attn'):
            ssm = layer.linear_attn
            saved[ssm] = ssm._old_forward
            ssm._old_forward = _make_dfs_ssm_forward(ssm, chunk_boundaries, chunk_parent_map)
    return saved


def unpatch_ssm(saved):
    for ssm, orig in saved.items():
        ssm._old_forward = orig


# ============================================================================
# Part 5: Main DFS tree forward
# ============================================================================

def tree_forward_dfs(model, prefix_ids, level_choices_ids, use_grad=False):
    """
    Full tree forward using DFS packing with variable-length chunks.

    One model forward call processes the entire tree:
      - SSM layers: tree-aware chunk algorithm with chunk_parent_map
      - Attention layers: tree attention mask (ancestors visible, siblings blocked)
      - Memory: O(N × d), no O(B^D) cache copies

    Args:
        model:             Qwen3.5 or compatible model
        prefix_ids:        list[int]
        level_choices_ids: list of D lists; level_choices_ids[d][b] = token list
        use_grad:          True to enable gradient tracking

    Returns:
        prefix_logits: [1, prefix_len, V]
        leaf_logits:   list of [1, L_last, V], one per leaf in DFS order
                       leaf_logits[0] = leftmost path (choice 0 at every level)
    """
    dev   = next(model.parameters()).device
    dtype = model.dtype

    info             = build_dfs_info(prefix_ids, level_choices_ids)
    chunk_parent_map = info['chunk_parent_map']
    chunk_boundaries = info['chunk_boundaries']
    N                = info['N']
    leaf_chunk_ids   = info['leaf_chunk_ids']
    prefix_len       = info['prefix_len']

    ids_t = torch.tensor([info['dfs_token_ids']], device=dev)
    pos_t = torch.tensor([info['dfs_pos_ids']],   device=dev)

    # Tree attention mask: [1, 1, N, N] additive (0 or -inf)
    attn_bool = info['attn_mask_bool'].to(dev)
    attn_4d   = torch.zeros(1, 1, N, N, dtype=dtype, device=dev)
    attn_4d[0, 0][~attn_bool] = float('-inf')

    saved = patch_ssm_for_dfs(model, chunk_boundaries, chunk_parent_map)
    try:
        ctx = torch.no_grad() if not use_grad else torch.enable_grad()
        with ctx:
            out = model(
                input_ids=ids_t,
                attention_mask=attn_4d,
                position_ids=pos_t,
                use_cache=False,
            )
    finally:
        unpatch_ssm(saved)

    logits = out.logits   # [1, N, V]

    prefix_logits = logits[:, :prefix_len]

    leaf_logits = []
    for c in leaf_chunk_ids:
        c_start, c_end = chunk_boundaries[c]
        leaf_logits.append(logits[:, c_start:c_end])

    return prefix_logits, leaf_logits
