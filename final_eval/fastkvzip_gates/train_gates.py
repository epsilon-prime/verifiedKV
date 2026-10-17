"""Train Fast KVzip gates with the authors' own optimiser on features from ``extract_features.py``.

    MODEL=... NGROUP=4 sbatch -p dgx --qos=dgx --mem=700G job_train.sbatch

Calls ``vendor/optim.py::optimize`` unchanged (SGD, lr 0.2, momentum 0.9, 5000 steps, batch 1000,
BCE on KVzip scores, gate dim 16, 16 sink keys, every 100th token held out). Only the loader differs
from ``vendor/load_data.py``: it reads the same files in the same order but fills one preallocated
tensor, because the official ``torch.cat`` of per-context tensors needs ~3x the feature size in RAM
(Ministral 3: ~300 GB of hidden states). The saved file has the official layout
(``{"module": [state_dict per layer], "loss", "acc", "loss_train", "acc_train"}``), named
``gates/<model>/q<ngroup>_dim16_sink16.pt`` as ``FastKVzipPress`` expects.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import optim  # noqa: E402  the authors' trainer, unmodified

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ngroup", type=int, required=True, help="query heads per KV head (VLM configs nest it)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bf16_params", action="store_true",
                help="train the gate weights in the features' dtype as the authors' code does (bf16); the default "
                     "trains them in fp32 because bf16 gates diverged to NaN on Qwen3.5 (see below)")
ap.add_argument("--out", default=os.path.join(HERE, "gates"),
                help="FastKVzipPress reads $KVPRESS_FASTKVZIP_GATES=<this>/<model>/...; a model with published "
                     "gates (Qwen3-4B) must be retrained into another folder, or it would shadow them")
a = ap.parse_args()

name = a.model.split("/")[-1].lower()
root = os.path.join(HERE, "data", name)
files = [(os.path.join(root, folder, f"hidden_{i}.pt"), os.path.join(root, folder, f"score_{i}.pt"))
         for folder, n in (("fineweb_10k", 29), ("fineweb_10k_cat", 5)) for i in range(n)]
files = [f for f in files if os.path.isfile(f[1])]  # load_folder also skips missing indices
print(f"{len(files)} contexts", flush=True)

# sizes from the score files (small), then one allocation for the hidden states
lens = [torch.load(s, mmap=True).shape[-1] for _, s in files]
first_h = torch.load(files[0][0], mmap=True)
n_layer, _, dim = first_h.shape
first_s = torch.load(files[0][1])
inputs = torch.empty(n_layer, sum(lens), dim, dtype=first_h.dtype)
scores = torch.empty(n_layer, first_s.shape[1], sum(lens), dtype=first_s.dtype)
pos = 0
for (fh, fs), n in zip(files, lens):
    inputs[:, pos:pos + n] = torch.load(fh, mmap=True)
    scores[..., pos:pos + n] = torch.load(fs)
    pos += n
print(f"Total x: {tuple(inputs.shape)}, y: {tuple(scores.shape)}, {inputs.dtype}", flush=True)

# fp32 gate weights. optim.optimize builds each gate as Weight(..., inputs.dtype) and trains it with SGD at
# lr 0.2; with bf16 weights a saturated prediction (exactly 0 or 1 in bf16) gives a ~1e12 BCE gradient, a weight
# becomes inf and the next forward NaN. Qwen3.5 hit this at step ~1800 (data checked clean: check_data.py).
# Same optimiser, loss, schedule and data; only the arithmetic is fp32 (features stay bf16 in RAM, cast per
# batch). The saved gates are cast back to the released gates' dtypes (bf16 projections and bias, fp32 norms
# and sink keys), which is what FastKVzipPress builds its modules from.
REF_DTYPES = {k: v.dtype for k, v in optim.Weight(0, 8, 16, 1, 1, inputs.dtype, sink=16).state_dict().items()}
if not a.bf16_params:
    class Weight32(optim.Weight):
        def __init__(self, index, input_dim, output_dim, nhead, ngroup, dtype, sink=16):
            super().__init__(index, input_dim, output_dim, nhead, ngroup, torch.float32, sink=sink)

        def forward(self, hidden_states):
            return super().forward(hidden_states.float())

    optim.Weight = Weight32  # optimize() looks the class up in optim's globals

args = argparse.Namespace(model=a.model, dim=16, ngroup=a.ngroup, sink=16, loss="cnt", lr=2e-1, seed=a.seed, tag="")
torch.manual_seed(args.seed)
modules, losses, acc, losses_test, acc_test = optim.optimize(args, inputs, scores)

out = os.path.join(a.out, name)
os.makedirs(out, exist_ok=True)
tag = f"q{args.ngroup}_dim{args.dim}_sink{args.sink}"
state = [{k: v.to(REF_DTYPES[k]) for k, v in m.state_dict().items()} for m in modules]
assert all(torch.isfinite(v.float()).all() for sd in state for v in sd.values()), "non-finite gate weights"
# curves as tensors, weights on CPU: FastKVzipPress loads with weights_only=True, whose allowlist does not
# cover numpy 2's array pickles (numpy._core.multiarray._reconstruct); the authors' files predate numpy 2
curves = {k: torch.as_tensor(v) for k, v in
          {"loss": losses_test, "acc": acc_test, "loss_train": losses, "acc_train": acc}.items()}
torch.save({"module": [{k: v.cpu() for k, v in sd.items()} for sd in state], **curves}, os.path.join(out, f"{tag}.pt"))
optim.plot(losses, losses_test, name=f"loss_{tag}", log=True, path=out)
optim.plot(acc, acc_test, name=f"acc_{tag}", path=out)
print(f"saved {out}/{tag}.pt; final held-out acc {acc_test[-1].mean() * 100:.1f}%, loss {losses_test[-1].mean():.3f}")
