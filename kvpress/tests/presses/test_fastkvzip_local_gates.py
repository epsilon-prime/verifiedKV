# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress.presses import fastkvzip_press
from kvpress.presses.fastkvzip_press import FastKVzipPress


def test_local_gates_are_preferred_over_the_hub(tmp_path, monkeypatch):
    """Gates trained for a model the authors did not release load from $KVPRESS_FASTKVZIP_GATES."""
    gate_id = "some-model/q4_dim16_sink16.pt"
    (tmp_path / "some-model").mkdir()
    torch.save({"module": [{"w": torch.ones(1)}]}, tmp_path / gate_id)
    monkeypatch.setattr(fastkvzip_press, "get_gate_id", lambda name: gate_id)
    monkeypatch.setenv("KVPRESS_FASTKVZIP_GATES", str(tmp_path))

    def no_hub(*a, **k):
        raise AssertionError("hub must not be used when a local gate exists")

    monkeypatch.setattr(fastkvzip_press, "hf_hub_download", no_hub)
    weights, got = fastkvzip_press.get_gate_weight("org/some-model")
    assert got == gate_id and torch.equal(weights[0]["w"], torch.ones(1))


def test_hub_is_used_without_a_local_gate(tmp_path, monkeypatch):
    gate_id = "other-model/q4_dim16_sink16.pt"
    monkeypatch.setattr(fastkvzip_press, "get_gate_id", lambda name: gate_id)
    monkeypatch.setenv("KVPRESS_FASTKVZIP_GATES", str(tmp_path))
    path = tmp_path / "hub.pt"
    torch.save({"module": [{"w": torch.zeros(1)}]}, path)
    calls = []
    monkeypatch.setattr(fastkvzip_press, "hf_hub_download", lambda **k: calls.append(k) or str(path))
    weights, _ = fastkvzip_press.get_gate_weight("org/other-model")
    assert calls and calls[0]["filename"] == gate_id and torch.equal(weights[0]["w"], torch.zeros(1))


@pytest.mark.parametrize("window_size, expected", [(4096, 4096), (None, 800)])
def test_window_at_long_context(window_size, expected):
    """At >= 32k tokens the stock press protects 4096 recent tokens, which is the whole kept budget at
    32k from 8x on; window_size=None keeps the 2% window (800 of 40000 tokens) at every length."""
    import types

    press = FastKVzipPress(window_size=window_size)
    press._row = {0: 0}
    class ZeroGate(torch.nn.Module):  # every token scored 0: only the sinks and the window are protected
        def forward(self, h):
            return torch.zeros(1, 2, h.shape[1])

    press.gates = [ZeroGate()]
    press.score_val = [None]
    press._score_fast(types.SimpleNamespace(layer_idx=0), torch.zeros(1, 40000, 8))
    protected = (press.score_val[0][0, 0, press.n_sink:] == 1.0).sum().item()
    assert protected == expected
