"""Score gate files against true KVzip scores on held-out tokens (the trainer's split: every 100th token).

    python eval_gate.py <model dir name> <gate.pt> [<gate.pt> ...]      (on a GPU node; see job_eval.sbatch)

For the Qwen3-4B validation: our retrained gate (gates_validation/) next to the authors' released gate,
both on OUR extracted Qwen3-4B data. Reports per gate the authors' own metrics (BCE loss "cnt" and the
0.05-threshold agreement "acc", vendor/utils.py), plus the Spearman-like rank agreement of predicted vs true
scores per (layer, head). The held-out tokens of our data are unseen by our gate in training; the released
gate never saw this data at all (it was trained on the authors' own extraction), so a small edge for ours
is expected and a large gap either way is the signal.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import optim  # noqa: E402
from utils import acc_fn, loss_fn  # noqa: E402

name, gate_files = sys.argv[1], sys.argv[2:]
root = os.path.join(HERE, "data", name)
xs, ys = [], []
offset = 0  # global token index of this file's first token in the trainer's concatenation
for folder, n in (("fineweb_10k", 29), ("fineweb_10k_cat", 5)):
    for i in range(n):
        fs = os.path.join(root, folder, f"score_{i}.pt")
        if not os.path.isfile(fs):
            continue
        s = torch.load(fs)
        h = torch.load(fs.replace("score_", "hidden_"), mmap=True)
        idx = torch.arange(s.shape[-1])
        # vendor/load_data.split_fn's rule on the concatenation's global token index
        test = ((idx + offset + 1) % 100) == 0
        offset += s.shape[-1]
        xs.append(h[:, test].clone())
        ys.append(s[..., test].clone())
x = torch.cat(xs, dim=1).cuda()  # [n_layer, n, hidden]
y = torch.cat(ys, dim=-1).cuda()  # [n_layer, n_kv, n]
print(f"{name}: {y.shape[-1]} held-out tokens, {y.shape[0]} layers", flush=True)


def rank_corr(a, b):
    ra = a.argsort(-1).argsort(-1).float()
    rb = b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return ((ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))).mean()


for gf in gate_files:
    state = torch.load(gf, weights_only=False, map_location="cuda")["module"]
    sd0 = state[0]
    head_group, input_dim = sd0["q_proj.weight"].shape
    nhead_x_dim, _ = sd0["k_proj.weight"].shape
    out_dim = sd0["q_norm.weight"].shape[-1]
    nhead, ngroup = nhead_x_dim // out_dim, head_group // nhead_x_dim
    losses, accs, corrs = [], [], []
    with torch.no_grad():
        for l, sd in enumerate(state):
            w = optim.Weight(l, input_dim, out_dim, nhead, ngroup, sd["q_proj.weight"].dtype,
                             sink=sd["k_base"].shape[2]).cuda()
            w.load_state_dict(sd)
            pred = w(x[l].to(sd["q_proj.weight"].dtype)).float()  # [n_kv, n]
            losses.append(loss_fn(pred, y[l].float(), type="cnt").item())
            accs.append(acc_fn(pred, y[l].float()).item())
            corrs.append(rank_corr(pred, y[l].float()).item())
    print(f"{os.path.relpath(gf, HERE)}: loss {sum(losses) / len(losses):.4f}  acc {100 * sum(accs) / len(accs):.2f}%  "
          f"rank corr {sum(corrs) / len(corrs):.3f}  (worst layer acc {100 * min(accs):.1f}%)", flush=True)
