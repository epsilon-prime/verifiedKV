# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.adapters import get_adapter
from kvpress.presses.kvzip_press import KVzipPress


@dataclass
class CachedKVzipPress(KVzipPress):
    """
    KVzipPress whose scores are cached on disk per (model, context), so the repeat-context scoring pass
    runs once per context and every arm built on KVzip reuses it (research harness only).

    The scores are a deterministic function of the model and the context tokens, so a cached run
    gives exactly the uncached result. The cache key is a hash of the context token ids, the model
    name, ``n_sink`` and the context length. Writes are atomic, so parallel jobs can share the
    directory. Set ``KVPRESS_NO_SCORE_CACHE=1`` to bypass it.

    The directory is ``$KVPRESS_SCORE_CACHE`` if set (read when kvpress is imported), else
    ``~/.cache/kvpress/kvzip_scores``. The key does not include the code version or the GPU, and the first
    writer wins: clear the directory after changing the scoring code, and do not share one directory
    between GPU types if runs must be comparable bit for bit.
    """

    cache_dir: str = os.environ.get(
        "KVPRESS_SCORE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "kvpress", "kvzip_scores")
    )

    def _score_cache_path(self, model: PreTrainedModel) -> str:
        ids = self._context_ids.detach().cpu().numpy().tobytes()
        key = hashlib.sha1(ids).hexdigest()[:24]
        name = str(getattr(model.config, "name_or_path", "model")).replace("/", "--")
        return os.path.join(self.cache_dir, f"{name}_{key}_sink{self.n_sink}_n{self._context_ids.shape[1]}.pt")

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        if os.environ.get("KVPRESS_NO_SCORE_CACHE"):
            return super()._perform_kvzip_compression(model, tokenizer)
        self.context_length = self._context_ids.shape[1]
        path = self._score_cache_path(model)
        if os.path.exists(path):
            # what prepare() would have set; compress_post reads it
            modules = get_adapter(model).compressible_modules(model)
            self._score_row_by_layer = {int(module.layer_idx): row for row, module in enumerate(modules)}
            score_val = torch.load(path, map_location=model.device)
            n_kv = getattr(model.config, "text_config", model.config).num_key_value_heads  # VLM wrappers
            expected = (len(self._score_row_by_layer), 1, n_kv, self.context_length)
            assert tuple(score_val.shape) == expected, f"cached score shape {tuple(score_val.shape)} != {expected}"
            self.score_val = score_val.to(model.dtype)
            self.compress_post(model)  # the parent's last step; the repeat passes are skipped
            return
        super()._perform_kvzip_compression(model, tokenizer)
        if self.score_val is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            torch.save(self.score_val.detach().cpu(), tmp)
            os.replace(tmp, path)
