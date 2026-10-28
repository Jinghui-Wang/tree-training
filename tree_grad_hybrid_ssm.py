#!/usr/bin/env python3
"""
tree_grad_hybrid_ssm.py
=======================

SSM-aware extension of tree_grad_hybrid.  Adds recurrent-state and
conv-state gateways so that Qwen3.5-style hybrid (SSM+attention) models
can use KV-cache gradient checkpointing across partition boundaries.

Design
------
KV gateway (attention layers) — already handled by tree_grad_hybrid.py.
SSM gateway (per SSM layer, per cut node):
  rec_gate[l]  = all_states[cut_chunk_idx + 1]  # [B, H, Dk, Dv]  with grad_fn
  conv_gate[l] = conv_states[cut_chunk_idx]      # [B, conv_dim, K-1]  with grad_fn

Parent partition forward:
  Run DFS SSM forward; capture rec_state and conv_state at each cut_node position.

Child partition forward:
  - SSM rec:  pass gateway rec_state as initial_state → root chunks start from parent state
  - SSM conv: pre-populate conv_states[-1] = gateway conv_state → root chunks get correct context

Gradient flow:
  Same as KV: detach → leaf tensor with requires_grad → fp32 hook → _chain_grads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Callable

from tree_dfs_ssm import _tree_correct_conv_varlen

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm
    HAS_QWEN35 = True
except ImportError:
    HAS_QWEN35 = False
    l2norm = None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SSM recurrent function  —  captures states at requested chunk positions
# ─────────────────────────────────────────────────────────────────────────────

def _ssm_varlen_capture(
    query, key, value, g, beta,
    chunk_boundaries:    List[Tuple[int, int]],
    chunk_parent_map:    List[int],
    initial_state:       Optional[torch.Tensor] = None,
    capture_states_at:   Optional[Set[int]]     = None,   # chunk indices
    use_qk_l2norm:       bool = True,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Like torch_chunk_gated_delta_rule_tree_varlen but also returns
    all_states[c+1] for each c in capture_states_at.

    Returns
    -------
    core_out      : [B, S, H, Dv]
    captured      : {chunk_idx: state_after_chunk}  states have grad_fn
    """
    if use_qk_l2norm and l2norm is not None:
        query = l2norm(query, dim=-1, eps=1e-6)
        key   = l2norm(key,   dim=-1, eps=1e-6)

    initial_dtype = query.dtype
    q  = query.transpose(1, 2).contiguous().float()
    k  = key.transpose(1, 2).contiguous().float()
    v  = value.transpose(1, 2).contiguous().float()
    bt = beta.transpose(1, 2).contiguous().float()
    g  = g.transpose(1, 2).contiguous().float()

    B, H, S, Dk = k.shape
    Dv  = v.shape[-1]
    dev = q.device
    q   = q * (Dk ** -0.5)

    init = (initial_state.float().to(dev) if initial_state is not None
            else torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev))

    all_states:  List[torch.Tensor] = [init]
    all_outputs: List[torch.Tensor] = []
    captured:    Dict[int, torch.Tensor] = {}

    for i, (c_start, c_end) in enumerate(chunk_boundaries):
        L = c_end - c_start

        q_i  = q[:, :, c_start:c_end]
        k_i  = k[:, :, c_start:c_end]
        v_i  = v[:, :, c_start:c_end]
        bt_i = bt[:, :, c_start:c_end]
        g_i  = g[:, :, c_start:c_end]

        v_beta_i = v_i * bt_i.unsqueeze(-1)
        k_beta_i = k_i * bt_i.unsqueeze(-1)
        g_cum    = g_i.cumsum(dim=-1)
        decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()

        mask_upper_incl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev))
        mask_upper_excl = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

        attn_raw  = -(k_beta_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_incl, 0)
        attn_rows = [attn_raw[..., 0:1, :]]
        for j in range(1, L):
            corrected_j = attn_raw[..., j:j+1, :] + attn_raw[..., j:j+1, :j] @ attn_raw[..., :j, :]
            attn_rows.append(corrected_j)
        attn       = torch.cat(attn_rows, dim=-2) + torch.eye(L, dtype=attn_raw.dtype, device=dev)
        value_corr = attn @ v_beta_i
        k_cumdecay = attn @ (k_beta_i * g_cum.exp().unsqueeze(-1))

        parent_idx    = int(chunk_parent_map[i]) + 1
        current_state = all_states[parent_idx]

        v_prime = k_cumdecay @ current_state
        v_new   = value_corr - v_prime

        attn_within = (q_i @ k_i.transpose(-1, -2) * decay_mask).masked_fill(mask_upper_excl, 0)
        attn_inter  = (q_i * g_cum.exp().unsqueeze(-1)) @ current_state
        out_i = attn_inter + attn_within @ v_new
        all_outputs.append(out_i)

        new_state = (
            current_state * g_cum[:, :, -1, None, None].exp()
            + (k_i * (g_cum[:, :, -1:] - g_cum).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )
        all_states.append(new_state)

        if capture_states_at and i in capture_states_at:
            captured[i] = new_state   # has grad_fn → parent computation graph

    core_out = torch.cat(all_outputs, dim=2).transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, captured


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Conv function  —  accepts pre-seeded parent conv states and captures
# ─────────────────────────────────────────────────────────────────────────────

def _conv_varlen_capture(
    ssm,
    mixed_qkv_raw:          torch.Tensor,        # [B, conv_dim, S]
    chunk_boundaries:        List[Tuple[int, int]],
    chunk_parent_map:        List[int],
    init_conv_states:        Optional[Dict[int, torch.Tensor]] = None,
    # pre-seeded parent states; key = parent chunk index (-1 for root chunks)
    capture_states_at:       Optional[Set[int]] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Like _tree_correct_conv_varlen but:
      - init_conv_states[parent_idx] seeds the conv context for chunks whose
        DFS parent is parent_idx (particularly -1 for root chunks)
      - captures conv_states[c] for c in capture_states_at

    Returns (conv_output [B, conv_dim, S], captured {chunk_idx: conv_state}).
    """
    K = ssm.conv_kernel_size
    chunk_outputs: List[torch.Tensor]        = []
    conv_states:   Dict[int, torch.Tensor]   = {}
    captured:      Dict[int, torch.Tensor]   = {}

    # Pre-seed from gateway (or external source)
    if init_conv_states:
        conv_states.update(init_conv_states)

    for c, (c_start, c_end) in enumerate(chunk_boundaries):
        L     = c_end - c_start
        chunk = mixed_qkv_raw[:, :, c_start:c_end]   # [B, conv_dim, L]

        parent_c     = chunk_parent_map[c]
        parent_state = conv_states.get(parent_c, None)

        if parent_state is not None:
            ctx = parent_state
            ext = torch.cat([ctx.to(chunk.device), chunk], dim=-1)
            if ssm.causal_conv1d_fn is not None:
                out = ssm.causal_conv1d_fn(
                    x=ext, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                out = F.silu(ssm.conv1d(ext)[:, :, :ext.shape[-1]])
            chunk_out = out[:, :, (K - 1):]
            conv_states[c] = ext[:, :, -(K - 1):].clone()
        else:
            if ssm.causal_conv1d_fn is not None:
                chunk_out = ssm.causal_conv1d_fn(
                    x=chunk, weight=ssm.conv1d.weight.squeeze(1),
                    bias=ssm.conv1d.bias, activation=ssm.activation, seq_idx=None,
                )
            else:
                chunk_out = F.silu(ssm.conv1d(chunk)[:, :, :L])
            conv_states[c] = F.pad(chunk, (max(0, K - 1 - L), 0))[:, :, -(K - 1):].clone()

        chunk_outputs.append(chunk_out)

        if capture_states_at and c in capture_states_at:
            captured[c] = conv_states[c]   # has grad_fn if chunk had gradient

    return torch.cat(chunk_outputs, dim=2), captured


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SSM forward factories for parent (capture) and child (inject) partitions
# ─────────────────────────────────────────────────────────────────────────────

def _make_ssm_forward_parent(
    ssm_layer,
    chunk_boundaries:  List[Tuple[int, int]],
    chunk_parent_map:  List[int],
    cut_chunk_indices: Set[int],           # chunk indices of cut nodes
    rec_store:         Dict,               # filled in-place: {chunk_idx: state}
    conv_store:        Dict,               # filled in-place: {chunk_idx: conv_state}
):
    """
    SSM forward for the PARENT partition.
    Captures recurrent and conv states at each cut_node's chunk position.
    These states (with grad_fn) become the SSM gateway for child partitions.
    """
    def patched(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        self = ssm_layer
        B, S, _ = hidden_states.shape

        # Per-chunk projections
        qkv_chunks, z_chunks, b_chunks, a_chunks = [], [], [], []
        for c_start, c_end in chunk_boundaries:
            h_c = hidden_states[:, c_start:c_end, :]
            qkv_chunks.append(self.in_proj_qkv(h_c).transpose(1, 2))
            z_chunks.append(self.in_proj_z(h_c))
            b_chunks.append(self.in_proj_b(h_c))
            a_chunks.append(self.in_proj_a(h_c))
        mixed_qkv = torch.cat(qkv_chunks, dim=2)
        z = torch.cat(z_chunks, dim=1).reshape(B, S, -1, self.head_v_dim)
        b = torch.cat(b_chunks, dim=1)
        a = torch.cat(a_chunks, dim=1)

        # Tree-correct conv — capture conv states at cut positions
        mixed_qkv, captured_conv = _conv_varlen_capture(
            self, mixed_qkv, chunk_boundaries, chunk_parent_map,
            capture_states_at=cut_chunk_indices,
        )
        conv_store.update(captured_conv)

        mixed_qkv = mixed_qkv.transpose(1, 2)
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

        # Tree-aware SSM — capture rec states at cut positions
        core_out, captured_rec = _ssm_varlen_capture(
            q, k, v, g, beta,
            chunk_boundaries  = chunk_boundaries,
            chunk_parent_map  = chunk_parent_map,
            capture_states_at = cut_chunk_indices,
            use_qk_l2norm     = True,
        )
        rec_store.update(captured_rec)

        # Per-chunk norm + out_proj
        out_chunks = []
        for c_start, c_end in chunk_boundaries:
            Lc   = c_end - c_start
            ca_c = core_out[:, c_start:c_end, :, :]
            z_c  = z[:, c_start:c_end, :, :]
            ca_c = self.norm(ca_c.reshape(-1, self.head_v_dim),
                             z_c.reshape(-1, self.head_v_dim))
            ca_c = ca_c.reshape(B, Lc, -1)
            out_chunks.append(self.out_proj(ca_c))
        return torch.cat(out_chunks, dim=1)

    return patched


def _make_ssm_forward_child(
    ssm_layer,
    chunk_boundaries:  List[Tuple[int, int]],
    chunk_parent_map:  List[int],
    gateway_rec:       torch.Tensor,   # leaf tensor [B, H, Dk, Dv], requires_grad
    gateway_conv:      torch.Tensor,   # leaf tensor [B, conv_dim, K-1], requires_grad
):
    """
    SSM forward for the CHILD partition.
    Injects gateway recurrent and conv states for root chunks (parent_map = -1).
    """
    def patched(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        self = ssm_layer
        B, S, _ = hidden_states.shape

        qkv_chunks, z_chunks, b_chunks, a_chunks = [], [], [], []
        for c_start, c_end in chunk_boundaries:
            h_c = hidden_states[:, c_start:c_end, :]
            qkv_chunks.append(self.in_proj_qkv(h_c).transpose(1, 2))
            z_chunks.append(self.in_proj_z(h_c))
            b_chunks.append(self.in_proj_b(h_c))
            a_chunks.append(self.in_proj_a(h_c))
        mixed_qkv = torch.cat(qkv_chunks, dim=2)
        z = torch.cat(z_chunks, dim=1).reshape(B, S, -1, self.head_v_dim)
        b = torch.cat(b_chunks, dim=1)
        a = torch.cat(a_chunks, dim=1)

        # Inject gateway conv state for root chunks (parent_map == -1)
        # conv_states[-1] = gateway_conv → root chunks will use it as parent context
        mixed_qkv, _ = _conv_varlen_capture(
            self, mixed_qkv, chunk_boundaries, chunk_parent_map,
            init_conv_states={-1: gateway_conv},
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
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

        # Inject gateway rec state as initial_state for root chunks
        core_out, _ = _ssm_varlen_capture(
            q, k, v, g, beta,
            chunk_boundaries = chunk_boundaries,
            chunk_parent_map = chunk_parent_map,
            initial_state    = gateway_rec.float(),   # root chunks read from this
            use_qk_l2norm    = True,
        )

        out_chunks = []
        for c_start, c_end in chunk_boundaries:
            Lc   = c_end - c_start
            ca_c = core_out[:, c_start:c_end, :, :]
            z_c  = z[:, c_start:c_end, :, :]
            ca_c = self.norm(ca_c.reshape(-1, self.head_v_dim),
                             z_c.reshape(-1, self.head_v_dim))
            ca_c = ca_c.reshape(B, Lc, -1)
            out_chunks.append(self.out_proj(ca_c))
        return torch.cat(out_chunks, dim=1)

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Patch / unpatch helpers (accelerate-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_ssm_layers(model, patch_fn_factory) -> dict:
    """
    Apply a per-layer SSM patch.  patch_fn_factory(ssm_layer) returns the
    patched forward function.  Compatible with accelerate device_map.
    """
    saved = {}
    for layer in model.model.layers:
        if not hasattr(layer, 'linear_attn'):
            continue
        ssm = layer.linear_attn
        patched = patch_fn_factory(ssm)
        if hasattr(ssm, '_old_forward'):
            saved[ssm] = ('_old_forward', ssm._old_forward)
            ssm._old_forward = patched
        else:
            saved[ssm] = ('forward', ssm.forward)
            ssm.forward = patched
    return saved


def _unpatch_ssm_layers(saved: dict):
    for ssm, (attr, orig) in saved.items():
        setattr(ssm, attr, orig)


def _has_ssm(model) -> bool:
    return any(hasattr(l, 'linear_attn') for l in model.model.layers)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SSM gateway dataclass  (attached to SubtreeGateway in tree_grad_hybrid)
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class SSMGateway:
    """
    Per-layer SSM state gateway for one cut node.
    Each field is a list indexed by SSM layer_idx.

    rec_gate[l]  : leaf tensor [B, H, Dk, Dv], requires_grad=True
    rec_raw[l]   : original tensor with grad_fn → parent computation
    rec_fp32[l]  : [acc_or_None]  float32 gradient accumulator
    conv_gate[l] : leaf tensor [B, conv_dim, K-1], requires_grad=True
    conv_raw[l]  : original tensor with grad_fn
    conv_fp32[l] : [acc_or_None]
    """
    rec_gate:  List[torch.Tensor]
    rec_raw:   List[torch.Tensor]
    rec_fp32:  List[List]
    conv_gate: List[torch.Tensor]
    conv_raw:  List[torch.Tensor]
    conv_fp32: List[List]


def _make_ssm_gateway(
    rec_states_by_layer:   Dict[int, torch.Tensor],   # layer_idx -> state tensor
    conv_states_by_layer:  Dict[int, torch.Tensor],   # layer_idx -> conv tensor
) -> SSMGateway:
    """
    Create an SSMGateway from raw tensors (which have grad_fn).
    Registers float32 hooks and cancels bf16 .grad accumulation.
    """
    n_layers = max(max(rec_states_by_layer.keys(), default=-1),
                   max(conv_states_by_layer.keys(), default=-1)) + 1

    rec_gate, rec_raw, rec_fp32    = [], [], []
    conv_gate, conv_raw, conv_fp32 = [], [], []

    def _gate_tensor(raw_t):
        g = raw_t.detach().requires_grad_(True)
        acc: List = [None]

        def _h(a):
            def hook(grad):
                g32 = grad.detach().float().cpu()
                a[0] = g32 if a[0] is None else a[0] + g32
                return torch.zeros_like(grad)
            return hook

        g.register_hook(_h(acc))
        return g, acc

    for l in range(n_layers):
        if l in rec_states_by_layer:
            raw = rec_states_by_layer[l]
            g, acc = _gate_tensor(raw)
            rec_raw.append(raw); rec_gate.append(g); rec_fp32.append(acc)
        else:
            rec_raw.append(None); rec_gate.append(None); rec_fp32.append([None])

        if l in conv_states_by_layer:
            raw = conv_states_by_layer[l]
            g, acc = _gate_tensor(raw)
            conv_raw.append(raw); conv_gate.append(g); conv_fp32.append(acc)
        else:
            conv_raw.append(None); conv_gate.append(None); conv_fp32.append([None])

    return SSMGateway(rec_gate, rec_raw, rec_fp32, conv_gate, conv_raw, conv_fp32)


def _chain_ssm_gateway_grads(ssm_gw: SSMGateway, retain_graph: bool = False):
    """Chain accumulated SSM gateway gradients back through raw tensors."""
    raws, grads = [], []
    for raw, acc in zip(ssm_gw.rec_raw, ssm_gw.rec_fp32):
        if raw is not None and acc[0] is not None:
            raws.append(raw); grads.append(acc[0].to(raw.device))
    for raw, acc in zip(ssm_gw.conv_raw, ssm_gw.conv_fp32):
        if raw is not None and acc[0] is not None:
            raws.append(raw); grads.append(acc[0].to(raw.device))
    if raws:
        torch.autograd.backward(raws, grads, retain_graph=retain_graph)
