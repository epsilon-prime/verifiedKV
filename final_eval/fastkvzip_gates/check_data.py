"""Scan extracted gate-training data for values the trainer cannot take: non-finite hidden states or scores,
and scores outside [0, 1] (the BCE target range). One line per file with any problem, then a total."""
import glob
import os
import sys

import torch

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", sys.argv[1])
bad_tok = 0
n_tok = 0
for fs in sorted(glob.glob(os.path.join(root, "*", "score_*.pt"))):
    fh = fs.replace("score_", "hidden_")
    s = torch.load(fs)  # [n_layer, n_kv, n]
    h = torch.load(fh, mmap=True)  # [n_layer, n, hidden]
    s_bad = ~torch.isfinite(s) | (s < 0) | (s > 1)
    h_bad = torch.zeros(h.shape[1], dtype=torch.bool)
    for l in range(h.shape[0]):
        h_bad |= ~torch.isfinite(h[l].float()).all(-1)
    tok_bad = h_bad | s_bad.any(0).any(0)
    n_tok += h.shape[1]
    bad_tok += int(tok_bad.sum())
    hmax = max(float(h[l].float().abs().max()) for l in range(h.shape[0]))
    if tok_bad.any() or hmax > 1e4:
        print(f"{os.path.relpath(fs, root)}: {int(tok_bad.sum())} bad tokens (hidden non-finite {int(h_bad.sum())}, "
              f"score bad {int(s_bad.sum())}, score range {float(s[torch.isfinite(s)].min()):.3g}..{float(s[torch.isfinite(s)].max()):.3g}), |hidden| max {hmax:.3g}", flush=True)
print(f"total: {bad_tok} bad of {n_tok} tokens")
