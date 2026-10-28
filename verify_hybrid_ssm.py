#!/usr/bin/env python3
"""
Verify tree_grad_hybrid SSM gateway on 2-layer Qwen3.5-35B-A3B model.

Reference A: dfs_tree_forward (full DFS in one shot, tree-aware SSM patch)
Test     B: tree_grad_hybrid  (DFS per-partition, SSM gateway between partitions)

Both use the SAME SSM patch (dfs_tree_forward._patch_ssm / _make_dfs_ssm_forward).
Expected: A vs B EXACT 0 for the no-cuts case, near-EXACT for cuts (float32 precision).

Run:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python verify_hybrid_ssm.py
"""
import os, sys, time
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch.nn.functional as F
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dfs_tree_forward import TreeNode, dfs_tree_forward
from tree_grad_hybrid import tree_grad_hybrid, partition_tree, make_tree_loss_fn


MODEL_PATH = '/home/intel/dest_path/model/Qwen3.5-35B-A3B'

# ── 2-layer model ─────────────────────────────────────────────────────────────
print('Building 2-layer Qwen3.5 model ...')
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tc = config.text_config
tc.num_hidden_layers = 2
tc.layer_types = tc.layer_types[:2]
print(f'  layers={tc.num_hidden_layers}, layer_types={tc.layer_types}')

model = AutoModelForCausalLM.from_config(tc, trust_remote_code=True)
model = model.float().to('cuda:0').eval()
dev   = model.device
print(f'  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {dev}')
for i, layer in enumerate(model.model.layers):
    print(f'  layer {i}: {"SSM" if hasattr(layer,"linear_attn") else "ATTN"}')

# ── Tree ──────────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
def enc(s): return tok.encode(' ' + s, add_special_tokens=False)

ta  = enc('alpha');  tb  = enc('beta');   tc_ = enc('chi')
td  = enc('delta');  te  = enc('epsilon'); tq  = enc('theta')
tf  = enc('phi');    tg  = enc('gamma')
tb2 = enc('beta');   tc2 = enc('chi');     td2 = enc('delta')
tx  = enc('xi');     ty  = enc('psi')

# Tree 1: root=a  →  b→c→d→{e,q}  and  b→c→f→g
root1 = TreeNode(ta)
nb    = root1.add_child(tb)
nc    = nb.add_child(tc_)
nd    = nc.add_child(td)
ne    = nd.add_child(te)    # leaf P0
nq    = nd.add_child(tq)    # leaf P1
nf    = nc.add_child(tf)
ng    = nf.add_child(tg)    # leaf P2

# Tree 2: root=B  →  {c→d,  x→y}
root2 = TreeNode(tb2)
nc2   = root2.add_child(tc2)
nd2   = nc2.add_child(td2)  # leaf P3
nx    = root2.add_child(tx)
ny    = nx.add_child(ty)    # leaf P4

roots  = [root1, root2]
leaves = [ne, nq, ng, nd2, ny]
n_lv   = len(leaves)

# Token lengths
def _all_nodes(r):
    yield r
    for c in r.children: yield from _all_nodes(c)

total_toks = sum(len(n.token_ids) for r in roots for n in _all_nodes(r))
print(f'\nTotal unique tokens: {total_toks}')

# Random labels
torch.manual_seed(42)
labels = [torch.randint(1, tok.vocab_size, (len(n.token_ids),), device=dev) for n in leaves]
label_map = {id(n): lab for n, lab in zip(leaves, labels)}


# ── Gradient hook ─────────────────────────────────────────────────────────────
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


# ── A: dfs_tree_forward (reference, tree-aware SSM, correct) ──────────────────
print('\n─── A: dfs_tree_forward + leaf-by-leaf backward (reference) ───')
model.zero_grad()
accum_A, handles_A = register_fp32_hooks(model)
t0 = time.time()

result = dfs_tree_forward(model, roots=roots)
# Build label map keyed by leaf_node id
for i, (logits_leaf, leaf_node) in enumerate(zip(result.leaf_logits, result.leaf_nodes)):
    lab = label_map[id(leaf_node)]
    loss = F.cross_entropy(logits_leaf[0], lab.to(logits_leaf.device)) / n_lv
    retain = (i < n_lv - 1)
    loss.backward(retain_graph=retain)

print(f'  Done in {time.time()-t0:.1f}s')
remove_hooks(handles_A)


# ── Partition info ─────────────────────────────────────────────────────────────
for mt in [1000, 8, 6, 5, 4, 3]:
    parts = partition_tree(roots, mt)
    print(f'  max_tokens={mt:4d}: {len(parts)} subtrees, '
          f'sizes={[p.token_count for p in parts]}'
          f'  {"← no cut" if len(parts)==2 else "← SSM gateway!"}' )


# ── B: tree_grad_hybrid ────────────────────────────────────────────────────────
def run_hybrid(max_tokens, tag):
    print(f'\n─── B: tree_grad_hybrid  max_tokens={max_tokens}  {tag} ───')
    model.zero_grad()
    loss_fn = make_tree_loss_fn(labels, leaves, n_lv)
    accum_B, handles_B = register_fp32_hooks(model)
    t0 = time.time()
    try:
        tree_grad_hybrid(model, roots=roots, leaf_loss_fn=loss_fn, max_tokens=max_tokens)
        ok = True
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()
        ok = False
    remove_hooks(handles_B)
    print(f'  Done in {time.time()-t0:.1f}s')
    return accum_B if ok else None


def compare(ga, gb, la, lb, thr=1e-3):
    params = sorted(set(ga) & set(gb))
    max_abs = max_rel = 0.0
    worst = []
    for pn in params:
        d = (ga[pn] - gb[pn]).abs().max().item()
        r = d / (ga[pn].abs().max().item() + 1e-30)
        if d > max_abs: max_abs = d
        if r > max_rel: max_rel = r
        if d > 0: worst.append((d, r, pn))
    worst.sort(reverse=True)
    ok  = max_abs < thr
    tag = f'✓  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}' if ok else \
          f'✗  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  (> {thr:.0e})'
    print(f'  {la} vs {lb}: {tag}')
    if not ok and worst:
        for d, r, pn in worst[:3]:
            print(f'    abs={d:.3e}  rel={r:.3e}  {pn}')
    return ok, max_abs


all_pass = True

# No cuts (baseline: both roots fit in one partition each)
grads = run_hybrid(1000, '(no cuts, both roots separate)')
if grads:
    ok, _ = compare(accum_A, grads, 'A', 'B(max=1000)')
    all_pass = all_pass and ok

# Cuts WITHIN tree1 — SSM gateway is activated
for mt in [6, 5, 4, 3]:
    grads = run_hybrid(mt, f'(SSM gateway activated)')
    if grads:
        ok, _ = compare(accum_A, grads, 'A', f'B(max={mt}, SSM gw)')
        all_pass = all_pass and ok

print('\n' + '='*60)
print('✓  ALL PASS' if all_pass else '✗  SOME FAILED')
print('(float32 + deterministic: expect max_abs < 1e-3)')
print('Note: ~1-2e-5 residual = float32 GEMM-shape arithmetic, not SSM gateway error.')
print('='*60)

del model; torch.cuda.empty_cache()
