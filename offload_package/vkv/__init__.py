"""Vkv on per-head token offloading (port of the research harness's vkv_final: press_final.py,
algo.py, chernoff.py on branch vkv_final_algo).

Enable with ``kv_connector_extra_config = {"offload_granularity": "token", "token_policy": "vkv",
"vkv_<field>": ...}`` (fields of ``state.VkvConfig``) and ``SKYLIGHT_SPARSE_METHOD=vkv``,
``enforce_eager=True``.

  reference.py          the harness's algo.py verbatim (+ the kvpress sketch helpers)
  reference_chernoff.py the harness's chernoff.py verbatim (the Chernoff certificate, kstar_fast)
  state.py              config, per-engine state, Algorithm 1 at the request's single round
  sketch.py             2-bit needle-fetch key sketch (storage, build, Triton scorer)
  decision.py           Algorithm 2 (a)-(d) in torch with capped sorts (== reference)
  kernels.py            the fused Triton decode step
  decode.py             one decode layer: shortlist, decision, staging of fetched entries,
                        attention (nothing persists)
  wrapper.py            the Skylight decode-wrapper adapter
"""
