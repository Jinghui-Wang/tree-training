#!/usr/bin/env python3
"""
Verify tree_grad_hybrid vs tree_grad_ckpt_sequential (reference).

Tests three scenarios:
  1. max_tokens=1000  → no cuts, entire tree in one subtree (= sequential)
  2. max_tokens=20    → 2–3 subtrees (cuts at intermediate nodes)
  3. max_tokens=10    → aggressive cuts (many small subtrees)

Uses the case-2 tree (5 leaves, forest of 2 roots) for thorough coverage.

Run: CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
     python verify_hybrid.py [--dtype float32|bfloat16]
"""

import os, sys, argparse, time
import torch
import torch.nn.functional as F

assert 'CUBLAS_WORKSPACE_CONFIG' in os.environ, \
    "Must run: CUBLAS_WORKSPACE_CONFIG=:4096:8 python verify_hybrid.py"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.dirname(__file__))
from dfs_tree_forward import TreeNode
from tree_grad_ckpt    import tree_grad_ckpt_sequential, make_tree_loss_fn as mf_seq
from tree_grad_hybrid  import (tree_grad_hybrid, make_tree_loss_fn as mf_hyb,
                                partition_tree)


# ─────────────────────────────────────────────────────────────────────────────
# Tree (same as verify_opath_case2.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_tree(vocab_size, seed=42):
    torch.manual_seed(seed)
    V = min(vocab_size - 1, 1000)
    def rnd(n): return torch.randint(1, V, (n,)).tolist()

    root_a  = TreeNode(rnd(3))
    b_left  = root_a.add_child(rnd(4))
    c_left  = b_left.add_child(rnd(3))
    d_left  = c_left.add_child(rnd(4))
    e_node  = d_left.add_child(rnd(5))   # leaf P0
    q_node  = d_left.add_child(rnd(3))   # leaf P2
    f_node  = c_left.add_child(rnd(3))
    g_node  = f_node.add_child(rnd(4))   # leaf P1

    root_b2 = TreeNode(rnd(4))
    c2_node = root_b2.add_child(rnd(3))
    d2_node = c2_node.add_child(rnd(4))  # leaf P3
    x_node  = root_b2.add_child(rnd(3))
    y_node  = x_node.add_child(rnd(4))   # leaf P4

    roots  = [root_a, root_b2]
    leaves = [e_node, g_node, q_node, d2_node, y_node]
    return roots, leaves


def make_labels(leaves, vocab_size, device, seed=99):
    torch.manual_seed(seed)
    V = min(vocab_size - 1, 1000)
    return [torch.randint(1, V, (len(n.token_ids),), device=device)
            for n in leaves]


# ─────────────────────────────────────────────────────────────────────────────
# Gradient hook
# ─────────────────────────────────────────────────────────────────────────────

def register_fp32_hooks(model):
    accum = {}; handles = []
    for pname, p in model.named_parameters():
        def make_hook(name):
            def hook(grad):
                g = grad.detach().cpu().float()
                accum[name] = accum[name] + g if name in accum else g.clone()
                return grad
            return hook
        handles.append(p.register_hook(make_hook(pname)))
    return accum, handles

def remove_hooks(handles):
    for h in handles: h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Run reference (tree_grad_ckpt_sequential)
# ─────────────────────────────────────────────────────────────────────────────

def run_reference(model, roots, leaves, leaf_labels):
    model.zero_grad()
    loss_fn = mf_seq(leaf_labels, leaves, len(leaves))
    accum, handles = register_fp32_hooks(model)
    tree_grad_ckpt_sequential(model, roots=roots, leaf_loss_fn=loss_fn)
    remove_hooks(handles)
    return accum


# ─────────────────────────────────────────────────────────────────────────────
# Run hybrid
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid(model, roots, leaves, leaf_labels, max_tokens):
    model.zero_grad()
    loss_fn = mf_hyb(leaf_labels, leaves, len(leaves))
    accum, handles = register_fp32_hooks(model)
    tree_grad_hybrid(model, roots=roots, leaf_loss_fn=loss_fn, max_tokens=max_tokens)
    remove_hooks(handles)
    return accum


# ─────────────────────────────────────────────────────────────────────────────
# Compare
# ─────────────────────────────────────────────────────────────────────────────

def compare(ref, new, label, dtype_str):
    max_abs = max_rel = 0.0
    n_params = 0
    worst = []
    for name in ref:
        gr = ref.get(name); gn = new.get(name)
        if gr is None and gn is None: continue
        if gr is None or gn is None:
            worst.append(f'  {name}: one side None'); continue
        ae = (gr - gn).abs().max().item()
        re = ae / (gr.abs().max().item() + 1e-12)
        max_abs = max(max_abs, ae)
        max_rel = max(max_rel, re)
        n_params += 1
        if ae > 0:
            worst.append((ae, re, name))
    if worst and isinstance(worst[0], tuple):
        worst.sort(reverse=True)
        worst_str = [f'  abs={ae:.3e} rel={re:.3e}  {nm}'
                     for ae, re, nm in worst[:5]]
    else:
        worst_str = worst[:5]

    thr = 1e-3 if dtype_str == 'float32' else 5e-2
    ok  = max_rel < thr
    tag = '✓' if ok else '✗'
    print(f'  {tag}  [{label}]  params={n_params}  '
          f'max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  '
          f'(threshold rel<{thr:.0e})')
    if not ok:
        for s in worst_str: print(s)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Partition sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_partition(roots, max_tokens):
    parts = partition_tree(roots, max_tokens)
    total_tokens = sum(p.token_count for p in parts)
    all_node_ids = set()
    for p in parts:
        for n in p.dfs_nodes:
            assert id(n) not in all_node_ids, f'Node appears in 2 partitions!'
            all_node_ids.add(id(n))
    for p in parts:
        assert p.token_count <= max(max_tokens,
                                    max(len(n.token_ids) for n in p.dfs_nodes)), \
            f'Partition exceeds budget: {p.token_count} > {max_tokens}'
    print(f'  Partition [{max_tokens:4d} tok]: '
          f'{len(parts)} subtrees, '
          f'sizes={[p.token_count for p in parts]}, '
          f'total={total_tokens}')
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='/home/intel/dest_path/model/Qwen3-4B')
    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    p.add_argument('--seed',  type=int, default=42)
    args = p.parse_args()

    torch_dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    print(f'dtype={args.dtype}  seed={args.seed}')

    from transformers import AutoModelForCausalLM, AutoConfig
    print(f'\nLoading {args.model_path} ...')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch_dtype, device_map='cuda:0')
    model.train()
    print(f'Loaded in {time.time()-t0:.1f}s')

    dev = next(model.parameters()).device
    cfg = AutoConfig.from_pretrained(args.model_path)
    roots, leaves = build_tree(cfg.vocab_size, args.seed)
    labels = make_labels(leaves, cfg.vocab_size, dev, seed=99)

    total_toks = sum(len(n.token_ids) for root in roots
                     for n in _all_nodes(root))
    print(f'Tree: {len(leaves)} leaves, total unique tokens = {total_toks}')

    # ── Partition sanity checks ──────────────────────────────────────────────
    print('\n── Partition sanity checks ──')
    check_partition(roots, 1000)   # no cut
    check_partition(roots, 20)     # moderate cuts
    check_partition(roots, 10)     # aggressive cuts

    # ── Reference ────────────────────────────────────────────────────────────
    print('\n── Reference (tree_grad_ckpt_sequential) ──')
    t0 = time.time()
    ref_grads = run_reference(model, roots, leaves, labels)
    print(f'Done in {time.time()-t0:.1f}s')

    # ── Hybrid with different budgets ─────────────────────────────────────────
    all_ok = True
    for max_tok in [1000, 20, 10]:
        print(f'\n── tree_grad_hybrid  max_tokens={max_tok} ──')
        t0 = time.time()
        try:
            hyb_grads = run_hybrid(model, roots, leaves, labels, max_tok)
            elapsed = time.time() - t0
            print(f'Done in {elapsed:.1f}s')
            ok = compare(ref_grads, hyb_grads, f'max_tokens={max_tok}', args.dtype)
            all_ok = all_ok and ok
        except Exception as e:
            print(f'  ✗  ERROR: {e}')
            import traceback; traceback.print_exc()
            all_ok = False

    print('\n' + ('✓  ALL PASS' if all_ok else '✗  SOME FAILED'))
    sys.exit(0 if all_ok else 1)


def _all_nodes(root):
    yield root
    for c in root.children:
        yield from _all_nodes(c)


if __name__ == '__main__':
    main()
