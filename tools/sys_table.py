"""System-metrics table of a ruler_vllm.py --sysmon run: one row per config (rows marked profiled give
the kernel breakdown, the others the counters).

    python tools/sys_table.py out/sys32k --csv <file> [--hbm-peak 4800 --pcie-avail 55.5]

Columns, means over the measured rows (all tasks):
  answer phase (decode after the forced question) and prefill phase (prefill + the round):
    HBM bandwidth (GPM DRAM utilisation x the HBM peak), PCIe host->GPU / GPU->host throughput and its
    share of the measured link bandwidth, SM active / occupancy, tensor / FP64 / FP32 pipe use, power;
  energy per answer token (answer-phase energy / answer tokens);
  GPU kernel time per answer step by category (profiled rows);
  vkv decisions: k* mean, rows fetching, fallback; KV footprint per request on GPU and in host memory.
"""
import argparse
import csv
import glob
import json
import re
import statistics as S
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--csv", default=None)
ap.add_argument("--hbm-peak", type=float, default=4800.0, help="GB/s that GPM's 100%% DRAM utilisation is")
ap.add_argument("--pcie-avail", type=float, default=55.5, help="measured GB/s per direction (bw_bench)")
args = ap.parse_args()

KV_TOKEN = 36 * 8 * 2 * 128 * 2          # bytes of one token's K+V over all layers (Qwen3-4B, bf16)
SKETCH = 0.0176                           # vkv's 2-bit sketch, fraction of the full KV
MIB = 1048576 / 1e9                       # MiB/s -> GB/s

out = []
for f in sorted(glob.glob(f"{args.run}/*/rows.jsonl")):
    arm = Path(f).parent.name
    m = re.match(r"vkv-fkvz_e([\d.]+)_d([\d.]+)$", arm)
    method = ("vkv" if m else "dense_recall" if "recall" in arm else "dense" if arm == "dense"
              else "fastkvzip_evict")
    rs = list(map(json.loads, open(f)))
    meas = [r for r in rs if not r.get("profiled") and r.get("sys")]
    prof = [r for r in rs if r.get("profiled") and r.get("kernels")]
    if not meas:
        continue

    def ph(phase, key, scale=1.0):
        v = [r["sys"][phase][key] for r in meas if r["sys"].get(phase) and key in r["sys"][phase]]
        return round(scale * S.mean(v), 3) if v else None

    row = {"method": method, "eps": m[1] if m else "", "delta": m[2] if m else "",
           "n_rows": len(meas), "n_profiled": len(prof),
           "tpot_ms": round(1e3 * S.mean(r["tpot"] for r in meas), 2),
           "decode_tok_s": round(S.mean(r["decode_tok_s"] for r in meas), 2),
           "ttft_s": round(S.mean(r["ttft"] for r in meas), 3),
           "prefill_s": round(S.mean(r["prefill_s"] for r in meas), 3)}
    for p in ("answer", "prefill"):
        row.update({
            f"{p}_hbm_GBs": ph(p, "hbm_bw_util_pct", args.hbm_peak / 100),
            f"{p}_hbm_util_pct": ph(p, "hbm_bw_util_pct"),
            f"{p}_pcie_h2d_GBs": ph(p, "pcie_rx_MiBs", MIB),
            f"{p}_pcie_d2h_GBs": ph(p, "pcie_tx_MiBs", MIB),
            f"{p}_sm_active_pct": ph(p, "sm_active_pct"),
            f"{p}_power_W": ph(p, "power_W")})
        for d in ("h2d", "d2h"):
            v = row[f"{p}_pcie_{d}_GBs"]
            row[f"{p}_pcie_{d}_util_pct"] = round(100 * v / args.pcie_avail, 2) if v is not None else None
    row.update({"answer_sm_occupancy_pct": ph("answer", "sm_occupancy_pct"),
                "answer_tensor_pct": ph("answer", "tensor_pct"),
                "answer_fp64_pct": ph("answer", "fp64_pct"),
                "answer_fp32_pct": ph("answer", "fp32_pct"),
                "energy_J_per_answer_token": round(S.mean(
                    r["sys"]["answer"]["energy_J"] / max(r["n_out"] - 1, 1) for r in meas
                    if r["sys"].get("answer")), 3)})
    if prof:
        for k in prof[0]["kernels"]:
            row[k.replace("gpu_ms_", "step_ms_")] = round(S.mean(r["kernels"][k] for r in prof), 3)
    vs = [r["vkv"] for r in meas if r.get("vkv")]
    if vs and method == "vkv":
        dsum = sum(v["decisions"] for v in vs)
        row.update({"k_star_mean": round(sum(v["k_sum"] for v in vs) / dsum, 1),
                    "rows_fetching_pct": round(100 * sum(v["fetch_any"] for v in vs) / dsum, 2),
                    "fallback_pct": round(100 * sum(v["fallback"] for v in vs) / dsum, 3),
                    "screened_pct": round(100 * sum(v["screened"] for v in vs) / dsum, 2)})
    kept = 1.0 if method == "dense" else S.mean(r.get("kept") or 0.0 for r in meas)
    full = S.mean(r["n_prompt"] for r in meas) * KV_TOKEN / 1e9
    on_gpu = kept + (SKETCH if method == "vkv" else 0.0)
    row.update({"kv_full_GB": round(full, 3), "kv_gpu_GB": round(full * on_gpu, 3),
                "kv_host_GB": round(full * (1 - kept), 3),
                "fetch_GB_per_request": round(S.mean(r.get("fetch_bytes", 0.0) for r in meas) / 1e9, 3)})
    out.append(row)

out.sort(key=lambda r: ({"dense": 0, "dense_recall": 1, "vkv": 2, "fastkvzip_evict": 3}[r["method"]],
                       float(r["eps"] or 0)))
cols = list(dict.fromkeys(k for r in out for k in r))
for r in out:
    print({k: r.get(k) for k in ("method", "eps", "tpot_ms", "answer_hbm_GBs", "answer_pcie_h2d_GBs",
                                  "answer_sm_active_pct", "step_ms_decide", "step_ms_fetch")})
if args.csv:
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
