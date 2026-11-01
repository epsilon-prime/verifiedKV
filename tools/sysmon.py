"""System metrics for ruler_vllm.py --sysmon: NVML GPM counters sampled in a background thread, and a
per-category GPU kernel time breakdown from torch.profiler.

GpmSampler (no profiling permission needed, Hopper GPM through NVML): every ``interval`` s one window
of DRAM (HBM) bandwidth utilisation, PCIe TX / RX throughput, SM activity / occupancy, tensor and
FP64 / FP32 / FP16 pipe utilisation, plus power and device memory in use. ``window(ta, tb)`` averages
the samples overlapping [ta, tb] (weighted by overlap) and integrates power to energy.

kernel_breakdown(prof, n_steps): CUDA kernel time per decode step by category (vkv decide, fetch /
staging / H2D, attention, GEMM, other).
"""
from __future__ import annotations

import re
import threading
import time

import pynvml as N

# (name, GPM metric id, unit)
GPM = [("hbm_bw_util_pct", N.NVML_GPM_METRIC_DRAM_BW_UTIL),
       ("pcie_tx_MiBs", N.NVML_GPM_METRIC_PCIE_TX_PER_SEC),       # GPU -> host
       ("pcie_rx_MiBs", N.NVML_GPM_METRIC_PCIE_RX_PER_SEC),       # host -> GPU
       ("sm_active_pct", N.NVML_GPM_METRIC_SM_UTIL),
       ("sm_occupancy_pct", N.NVML_GPM_METRIC_SM_OCCUPANCY),
       ("tensor_pct", N.NVML_GPM_METRIC_ANY_TENSOR_UTIL),
       ("fp64_pct", N.NVML_GPM_METRIC_FP64_UTIL),
       ("fp32_pct", N.NVML_GPM_METRIC_FP32_UTIL),
       ("fp16_pct", N.NVML_GPM_METRIC_FP16_UTIL)]


def handle_for_torch_device(dev: int = 0):
    import torch
    N.nvmlInit()
    return N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(dev).uuid))


class GpmSampler(threading.Thread):
    def __init__(self, handle, interval: float = 0.1):    # DRAM bandwidth reads NaN below ~0.1 s
        super().__init__(daemon=True)
        self.h, self.interval = handle, interval
        self.samples = []                     # (t0, t1, {metric: value})
        self._stop = threading.Event()
        assert N.nvmlGpmQueryDeviceSupport(self.h).isSupportedDevice, "GPM unsupported"

    def _metrics(self, s1, s2):
        mg = N.c_nvmlGpmMetricsGet_t()
        mg.version = N.NVML_GPM_METRICS_GET_VERSION
        mg.numMetrics = len(GPM)
        mg.sample1, mg.sample2 = s1, s2
        for i, (_, mid) in enumerate(GPM):
            mg.metrics[i].metricId = mid
        N.nvmlGpmMetricsGet(mg)
        return {name: float(mg.metrics[i].value) for i, (name, _) in enumerate(GPM)}

    def run(self):
        a, b = N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()
        N.nvmlGpmSampleGet(self.h, a)
        ta = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval)
            N.nvmlGpmSampleGet(self.h, b)
            tb = time.perf_counter()
            m = self._metrics(a, b)
            m["power_W"] = N.nvmlDeviceGetPowerUsage(self.h) / 1e3
            m["mem_used_GB"] = N.nvmlDeviceGetMemoryInfo(self.h).used / 1e9
            self.samples.append((ta, tb, m))
            a, b, ta = b, a, tb

    def stop(self):
        self._stop.set()

    def window(self, ta: float, tb: float) -> dict:
        """Overlap-weighted means over [ta, tb]; energy_J = mean power x (tb - ta); peak memory."""
        acc, wts, wsum, peak = {}, {}, 0.0, 0.0
        for s0, s1, m in list(self.samples):
            w = min(s1, tb) - max(s0, ta)
            if w <= 0:
                continue
            wsum += w
            peak = max(peak, m["mem_used_GB"])
            for k, v in m.items():
                if v == v:                                   # a metric may read NaN in a window
                    acc[k] = acc.get(k, 0.0) + w * v
                    wts[k] = wts.get(k, 0.0) + w
        if wsum == 0:
            return {}
        out = {k: v / wts[k] for k, v in acc.items()}
        out["energy_J"] = out["power_W"] * (tb - ta)
        out["mem_peak_GB"] = peak
        out["covered_s"] = wsum
        return out


CATS = [("decide", r"decide_kernel|fallback_kernel|chern|kstar|shortlist|group_best|exact_scores|"
                   r"weights_kernel|tail_max|sure_kernel|prep_kernel|walk_kernel|totals_kernel|"
                   r"dtdv|gradkey|rest_kernel|curve_kernel|flags_kernel|stats_kernel|cold_kernel"),
        ("fetch", r"emit_kernel|tok_|place_kernel|restore_place|recall_copy|stage|Memcpy HtoD|"
                  r"Memcpy DtoH"),
        ("attention", r"sparse_decode|flash|fmha|fa_table|lse_merge|attention|reshape_and_cache"),
        ("gemm", r"nvjet|gemm|cutlass|splitK|sm90_xmma|cublas")]


def kernel_breakdown(prof, n_steps: int) -> dict:
    """ms per decode step of CUDA kernel time by category (``other``: everything else)."""
    out = {c: 0.0 for c, _ in CATS}
    out["other"] = 0.0
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0.0)
        if not t:
            continue
        for c, pat in CATS:
            if re.search(pat, e.key):
                out[c] += t
                break
        else:
            out["other"] += t
    return {f"gpu_ms_{c}": v / 1e3 / max(n_steps, 1) for c, v in out.items()}
