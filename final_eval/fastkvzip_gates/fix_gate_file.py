"""Rewrite a gate file trained before the save fix: curves numpy -> torch tensors, weights -> CPU, so that
FastKVzipPress's weights_only load accepts it. Keeps a .orig copy. Our own files only (trusted pickle)."""
import shutil
import sys

import numpy as np
import torch

for f in sys.argv[1:]:
    d = torch.load(f, weights_only=False, map_location="cpu")
    if all(not isinstance(v, np.ndarray) for v in d.values()):
        print(f"{f}: already fine"); continue
    shutil.copy(f, f + ".orig")
    out = {"module": [{k: v.cpu() for k, v in sd.items()} for sd in d["module"]]}
    out.update({k: torch.as_tensor(v) for k, v in d.items() if k != "module"})
    torch.save(out, f)
    print(f"{f}: rewritten")
