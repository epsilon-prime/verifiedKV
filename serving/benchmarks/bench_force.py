"""Forced decoding for the benches: every config decodes the same reference tokens.

With BENCH_FORCE_TOKENS=<json list of token ids>, the bench adds this logits
processor. At decode step t it records, from the config's own logits, whether
the argmax equals ref[t] and the NLL of ref[t], then masks the logits so ref[t]
is sampled. Queries, and therefore sparse selections and fetches, then follow
the same text in every config, and the recorded agreement / NLL measure how far
each config's attention drifts from the reference.

Stats stay on the GPU (no per-step sync); `ForcedTokens.instance.summary()`
reads them after generation (in-process engine). With a multiprocess engine
the processor lives in the EngineCore process: it saves its vectors to
BENCH_FORCE_DUMP itself when the last reference step is scored, and the
bench reads the summary back from that file (`summary_from_file`).
"""
import json
import os

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate, LogitsProcessor, MoveDirectionality)


class ForcedTokens(LogitsProcessor):
    instance = None

    def __init__(self, vllm_config, device, is_pin_memory):
        with open(os.environ['BENCH_FORCE_TOKENS']) as f:
            self.ref = [int(t) for t in json.load(f)]
        n = len(self.ref)
        self.nll = torch.zeros(n, dtype=torch.float32, device=device)
        self.agree = torch.zeros(n, dtype=torch.float32, device=device)
        self.seen = torch.zeros(n, dtype=torch.float32, device=device)
        self.rows: dict[int, list[int]] = {}   # batch index -> running output ids
        ForcedTokens.instance = self

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for i in batch_update.removed:
            self.rows.pop(i, None)
        for i, _params, _prompt, out in batch_update.added:
            self.rows[i] = out
        for a, b, d in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if ra is not None:
                self.rows[b] = ra
            if d == MoveDirectionality.SWAP and rb is not None:
                self.rows[a] = rb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for i, out in self.rows.items():
            t = len(out)
            if t >= len(self.ref):
                continue
            tok = self.ref[t]
            row = logits[i]
            self.nll[t] = -torch.log_softmax(row.float(), -1)[tok]
            self.agree[t] = (row.argmax() == tok).float()
            self.seen[t] = 1.0
            if t == len(self.ref) - 1 and os.environ.get("BENCH_FORCE_DUMP"):
                self.dump(os.environ["BENCH_FORCE_DUMP"])
            row.fill_(float('-inf'))
            row[tok] = 0.0
        return logits

    def reset(self) -> None:
        for b in (self.nll, self.agree, self.seen):
            b.zero_()

    def dump(self, path: str) -> None:
        """Per-step vectors (for exact run-vs-run comparison): nll, agree."""
        n = int(self.seen.sum().item())
        torch.save({"nll": self.nll[:n].cpu(), "agree": self.agree[:n].cpu()}, path)

    @staticmethod
    def summary_from_file(path: str) -> dict:
        d = torch.load(path)
        n = int(d["nll"].numel())
        miss = d["agree"] == 0
        return {"steps": n, "agree": float(d["agree"].sum()) / max(n, 1),
                "nll": float(d["nll"].sum()) / max(n, 1),
                "first_disagree": int(miss.nonzero()[0]) if miss.any() else -1}

    def summary(self) -> dict:
        n = int(self.seen.sum().item())
        agree = self.agree.cpu()
        miss = (agree == 0) & (self.seen.cpu() > 0)
        return {
            'steps': n,
            'agree': float(agree.sum()) / max(n, 1),
            'nll': float(self.nll.sum()) / max(n, 1),
            'first_disagree': int(miss.nonzero()[0]) if miss.any() else -1,
        }
