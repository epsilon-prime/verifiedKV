# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from kvpress.metric_logging.head_stats import (
    compute_head_stats,
    keep_mask_from_indices,
    valid_key_mask,
)

B, HKV, G, D = 1, 2, 2, 4
HQ = HKV * G
KLEN, CTX = 8, 6


def _tensors(seed=0, q_len=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, HQ, q_len, D, generator=g)
    k = torch.randn(B, HKV, KLEN, D, generator=g)
    v = torch.randn(B, HKV, KLEN, D, generator=g)
    return q, k, v


def _dot(q, k, *, head, kv, j):
    """Pure-Python float64 dot product. Summing 0-dim torch tensors would accumulate in their own
    dtype, which for float32 inputs leaves a ~1e-7 error and makes this no longer a reference."""
    return sum(float(q[0, head, -1, d]) * float(k[0, kv, j, d]) for d in range(D))


def _brute(q, k, v, scaling, *, head, valid, weight_of):
    """float64 Python reference for one query head: returns (B_ctx, B_suf, A_full, Abar_full)."""
    kv = head // G
    b_ctx = b_suf = 0.0
    a = [0.0] * D
    abar = 0.0
    mx = max(_dot(q, k, head=head, kv=kv, j=j) * scaling for j in range(KLEN) if valid(j))
    for j in range(KLEN):
        if not valid(j):
            continue
        logit = _dot(q, k, head=head, kv=kv, j=j) * scaling
        w = math.exp(logit - mx)
        wj = w * weight_of(j)
        if j < CTX:
            b_ctx += wj
        else:
            b_suf += wj
        for d in range(D):
            a[d] += wj * float(v[0, kv, j, d])
        abar += wj * math.sqrt(sum(float(v[0, kv, j, d]) ** 2 for d in range(D)))
    return b_ctx, b_suf, a, abar, mx


def test_no_op_bias_reads_zero_error():
    """A bias of all zeros keeps everything and reweights nothing: every error must be ~0."""
    q, k, v = _tensors()
    bias = torch.zeros(B, HKV, CTX)
    st = compute_head_stats(query=q, key=k, value=v, attention_mask=None, scaling=D**-0.5, ctx_len=CTX, bias=bias)
    torch.testing.assert_close(st.log_b_sparse, st.log_b_dense, atol=1e-9, rtol=0)
    assert st.num_err.max() < 1e-12
    assert st.attn_err.max() < 1e-12
    assert (st.n_kept_ctx == CTX).all()


def test_hand_computed_denominator_and_numerator():
    """Against a float64 Python loop, with a non-trivial bias (one dropped key, one reweighted)."""
    q, k, v = _tensors(seed=3)
    # built in float64: log(2) rounded to float32 first would leave a 6e-8 relative error on that
    # weight and this is meant to be an exact check of the algebra
    bias = torch.zeros(B, HKV, CTX, dtype=torch.float64)
    bias[:, :, 0] = -float("inf")       # dropped
    bias[:, :, 1] = math.log(2.0)       # weight doubled
    st = compute_head_stats(
        query=q.double(), key=k.double(), value=v.double(), attention_mask=None,
        scaling=D**-0.5, ctx_len=CTX, bias=bias, compute_dtype=torch.float64,
    )

    for head in range(HQ):
        wf = lambda j: 0.0 if j == 0 else (2.0 if j == 1 else 1.0)  # noqa: E731
        b_ctx_s, b_suf, a_s, _, mx = _brute(q, k, v, D**-0.5, head=head, valid=lambda j: True, weight_of=wf)
        b_ctx_d, _, a_d, abar_d, _ = _brute(q, k, v, D**-0.5, head=head, valid=lambda j: True, weight_of=lambda j: 1.0)

        assert abs(float(st.log_b_sparse[0, head]) - (math.log(b_ctx_s) + mx)) < 1e-12
        assert abs(float(st.log_b_dense[0, head]) - (math.log(b_ctx_d) + mx)) < 1e-12
        assert abs(float(st.log_b_suffix[0, head]) - (math.log(b_suf) + mx)) < 1e-12

        dn = math.sqrt(sum((a_s[d] - a_d[d]) ** 2 for d in range(D)))
        na = math.sqrt(sum(a_d[d] ** 2 for d in range(D)))
        assert abs(float(st.num_err[0, head]) - dn / na) < 1e-12
        assert abs(float(st.num_err_abar[0, head]) - dn / abar_d) < 1e-12
        assert st.n_kept_ctx[0, head] == CTX - 1


def test_gqa_mapping_matches_add_attention_bias():
    """The (n_kv, groups) -> n_q reshape must equal repeat_interleave, which add_attention_bias assumes."""
    bias = torch.randn(B, HKV, CTX)
    expanded = bias.repeat_interleave(G, dim=1)
    for h in range(HQ):
        torch.testing.assert_close(expanded[0, h], bias[0, h // G])

    # and end to end: giving one KV head a huge bias must move exactly its own query heads
    q, k, v = _tensors(seed=5)
    b2 = torch.zeros(B, HKV, CTX)
    b2[:, 1, :] = -float("inf")
    st = compute_head_stats(query=q, key=k, value=v, attention_mask=None, scaling=D**-0.5, ctx_len=CTX, bias=b2)
    assert (st.n_kept_ctx[0, :G] == CTX).all(), "kv head 0 must be untouched"
    assert (st.n_kept_ctx[0, G:] == 0).all(), "kv head 1 must be fully dropped"


def test_logaddexp_identity_is_the_softmax_denominator():
    """B_ctx + B_suffix must equal logsumexp over every valid logit."""
    q, k, v = _tensors(seed=7)
    bias = torch.zeros(B, HKV, CTX)
    st = compute_head_stats(
        query=q.double(), key=k.double(), value=v.double(), attention_mask=None,
        scaling=D**-0.5, ctx_len=CTX, bias=bias.double(), compute_dtype=torch.float64,
    )
    qq = q[:, :, -1, :].view(B, HKV, G, D)
    logits = (torch.einsum("bhgd,bhnd->bhgn", qq.double(), k.double()) * D**-0.5).reshape(B, HQ, KLEN)
    torch.testing.assert_close(
        torch.logaddexp(st.log_b_dense, st.log_b_suffix), torch.logsumexp(logits, dim=-1), atol=1e-12, rtol=0
    )


def test_boolean_mask_is_not_mutated_and_stays_bool():
    """The stride-0 expand view must come back untouched; masked keys contribute exactly zero."""
    q, k, v = _tensors(seed=9, q_len=3)
    m = torch.ones(B, 1, 3, KLEN, dtype=torch.bool).expand(B, 1, 3, KLEN)
    m = m.clone().expand(B, 1, 3, KLEN)
    before = m.clone()
    bias = torch.zeros(B, HKV, CTX)
    compute_head_stats(query=q, key=k, value=v, attention_mask=m, scaling=D**-0.5, ctx_len=CTX, bias=bias)
    assert m.dtype is torch.bool
    assert torch.equal(m, before)


def test_mask_forms_agree():
    """None, boolean and additive-float masks encoding the same pattern must agree exactly."""
    q, k, v = _tensors(seed=11)
    bias = torch.zeros(B, HKV, CTX)
    kw = dict(query=q, key=k, value=v, scaling=D**-0.5, ctx_len=CTX, bias=bias)

    a = compute_head_stats(attention_mask=None, **kw)
    bool_m = torch.ones(B, 1, 1, KLEN, dtype=torch.bool)
    b = compute_head_stats(attention_mask=bool_m, **kw)
    f = torch.zeros(B, 1, 1, KLEN)
    c = compute_head_stats(attention_mask=f, **kw)
    torch.testing.assert_close(a.log_b_dense, b.log_b_dense, atol=0, rtol=0)
    torch.testing.assert_close(a.log_b_dense, c.log_b_dense, atol=0, rtol=0)

    # a float mask with finfo.min must drop those keys
    f2 = torch.zeros(B, 1, 1, KLEN)
    f2[..., 0] = torch.finfo(torch.float32).min
    d = compute_head_stats(attention_mask=f2, **kw)
    assert (d.log_b_dense < a.log_b_dense).all()


def test_causal_alignment_is_bottom_right():
    """q_len=3, k_len=8: row 0 must see keys 0..5, not just key 0 (top-left would give the latter)."""
    q_len = 3
    valid = valid_key_mask(None, q_index=0, q_len=q_len, k_len=KLEN, bsz=1, device=torch.device("cpu"))
    assert valid[0, 0].tolist() == [True] * (KLEN - q_len + 1) + [False] * (q_len - 1)
    last = valid_key_mask(None, q_index=q_len - 1, q_len=q_len, k_len=KLEN, bsz=1, device=torch.device("cpu"))
    assert last[0, 0].all()


def test_q_len_one_sees_every_key():
    valid = valid_key_mask(None, q_index=0, q_len=1, k_len=KLEN, bsz=1, device=torch.device("cpu"))
    assert valid[0, 0].all()


def test_keep_mask_path_uses_real_keys():
    """Top-k has no reweighting: B_sparse is the plain sum over kept keys, and never exceeds B_dense."""
    q, k, v = _tensors(seed=13)
    keep = torch.ones(B, HKV, CTX, dtype=torch.bool)
    keep[:, :, 2] = False
    st = compute_head_stats(
        query=q.double(), key=k.double(), value=v.double(), attention_mask=None,
        scaling=D**-0.5, ctx_len=CTX, keep_mask=keep, compute_dtype=torch.float64,
    )
    assert (st.log_b_sparse < st.log_b_dense).all(), "dropping mass must lower the denominator"
    assert (st.n_kept_ctx == CTX - 1).all()

    for head in range(HQ):
        b_s, _, _, _, mx = _brute(
            q, k, v, D**-0.5, head=head, valid=lambda j: True, weight_of=lambda j: 0.0 if j == 2 else 1.0
        )
        assert abs(float(st.log_b_sparse[0, head]) - (math.log(b_s) + mx)) < 1e-12


def test_keep_mask_from_indices_matches_index_set():
    idx = (torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([1, 4]))
    keep = keep_mask_from_indices(idx, bsz=1, n_kv_heads=HKV, ctx_len=CTX, device=torch.device("cpu"))
    assert not keep[0, 0, 1] and not keep[0, 1, 4]
    assert keep.sum() == HKV * CTX - 2


def test_top_k_never_over_estimates():
    """A theorem for the unreweighted arm: a breach is a bug, not a result."""
    for seed in range(20):
        q, k, v = _tensors(seed=seed)
        g = torch.Generator().manual_seed(seed)
        keep = torch.rand(B, HKV, CTX, generator=g) > 0.5
        keep[:, :, 0] = True
        st = compute_head_stats(
            query=q, key=k, value=v, attention_mask=None, scaling=D**-0.5, ctx_len=CTX, keep_mask=keep
        )
        assert (st.log_b_sparse <= st.log_b_dense + 1e-9).all()


def test_horvitz_thompson_is_unbiased():
    """Bernoulli + log(1/r) must give mean B_hat/B == 1. A GQA mis-map or a sign flip shows up here."""
    q, k, v = _tensors(seed=17)
    r = torch.full((B, HKV, CTX), 0.4)
    g = torch.Generator().manual_seed(1234)
    ratios = []
    for _ in range(4000):
        keep = torch.rand(B, HKV, CTX, generator=g) < r
        bias = torch.where(keep, -torch.log(r), torch.full_like(r, -float("inf")))
        st = compute_head_stats(
            query=q, key=k, value=v, attention_mask=None, scaling=D**-0.5, ctx_len=CTX, bias=bias
        )
        ratios.append(torch.exp(st.log_b_sparse - st.log_b_dense))
    m = torch.stack(ratios).mean(0)
    se = torch.stack(ratios).std(0) / math.sqrt(len(ratios))
    assert ((m - 1.0).abs() < 4 * se).all(), f"HT estimator is biased: {m.tolist()}"


def test_exactly_one_of_bias_or_keep_mask():
    q, k, v = _tensors()
    with pytest.raises(ValueError):
        compute_head_stats(query=q, key=k, value=v, attention_mask=None, scaling=1.0, ctx_len=CTX)
    with pytest.raises(ValueError):
        compute_head_stats(
            query=q, key=k, value=v, attention_mask=None, scaling=1.0, ctx_len=CTX,
            bias=torch.zeros(B, HKV, CTX), keep_mask=torch.ones(B, HKV, CTX, dtype=torch.bool),
        )


def test_float32_default_is_accurate_enough():
    """
    The default float32 compute must agree with float64 to ~1e-7 relative. That is four orders of
    magnitude below the compression errors this metric exists to measure, and it is why the matmul
    is not run in float64 (which costs ~30x for nothing).
    """
    q, k, v = _tensors(seed=23)
    bias = torch.zeros(B, HKV, CTX)
    kw = dict(attention_mask=None, scaling=D**-0.5, ctx_len=CTX)
    f32 = compute_head_stats(query=q, key=k, value=v, bias=bias, **kw)
    f64 = compute_head_stats(
        query=q.double(), key=k.double(), value=v.double(), bias=bias.double(),
        compute_dtype=torch.float64, **kw
    )
    rel = ((f32.log_b_dense - f64.log_b_dense).abs() / f64.log_b_dense.abs().clamp_min(1e-30)).max()
    assert rel < 1e-6, rel


# ---------------------------------------------------------------------------------------------------------------------
# The fake-key write goes into the KV cache's own storage, so without a restore the real keys are
# gone for every later decode step and any dense reference silently becomes the compressed one.
# ---------------------------------------------------------------------------------------------------------------------


def _decode_run(model, tok, n_steps, collect):
    import tempfile

    from kvpress import KVzipPress
    from kvpress.metric_logging import HeadStatsCollector

    ctx = "The quick brown fox jumps over the lazy dog. " * 40
    d = tempfile.mkdtemp()
    HeadStatsCollector.reset_for_testing()
    if collect:
        HeadStatsCollector().configure(log_path=d, run_config={})
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    out = pipe(ctx, questions=["What jumps?"], press=KVzipPress(compression_ratio=0.5),
               max_new_tokens=n_steps)
    if collect:
        HeadStatsCollector().close()
    return out["answers"][0], d


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = "MaxJeblick/llama2-0b-unit-test"
    m = AutoModelForCausalLM.from_pretrained(ckpt, attn_implementation="sdpa", dtype=torch.float32)
    return m, AutoTokenizer.from_pretrained(ckpt)


def test_dense_reference_survives_every_decode_step(tiny):
    """
    Without the restore the metric reads a difference on the FIRST decode forward only: from the
    second onward the cache holds fake keys, so the 'dense' denominator is the compressed one and
    the error collapses to zero. This asserts a real difference at every step.
    """
    import pandas as pd

    model, tok = tiny
    _, d = _decode_run(model, tok, 4, collect=True)
    df = pd.read_parquet(f"{d}/head_stats.parquet")
    assert df.decode_step.nunique() >= 3, df.decode_step.unique()
    moved = df.groupby("decode_step").apply(
        lambda g: float((g.log_b_sparse < g.log_b_dense - 1e-6).mean()), include_groups=False
    )
    assert (moved > 0.5).all(), f"steps where the dense reference was lost: {moved.to_dict()}"


def test_restore_does_not_change_the_answer(tiny):
    """The restore must be output-neutral: same generation with and without the metric."""
    model, tok = tiny
    a, _ = _decode_run(model, tok, 6, collect=False)
    b, _ = _decode_run(model, tok, 6, collect=True)
    assert a == b, (a, b)
