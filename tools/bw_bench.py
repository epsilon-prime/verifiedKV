"""Available bandwidth on this GPU: PCIe host <-> device with pinned memcpy (each direction, and
both at once), and HBM device-to-device copy (read + write counted). Prints JSON.

    CUDA_VISIBLE_DEVICES=2 python tools/bw_bench.py
"""
import json

import torch


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / reps


GB = 1 << 30
h = torch.empty(GB, dtype=torch.uint8, pin_memory=True)
h2 = torch.empty(GB, dtype=torch.uint8, pin_memory=True)
d = torch.empty(GB, dtype=torch.uint8, device="cuda")
d2 = torch.empty(GB, dtype=torch.uint8, device="cuda")
big = torch.empty(4 * GB, dtype=torch.uint8, device="cuda")
big2 = torch.empty(4 * GB, dtype=torch.uint8, device="cuda")
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()


def both():
    with torch.cuda.stream(s1):
        d.copy_(h, non_blocking=True)
    with torch.cuda.stream(s2):
        h2.copy_(d2, non_blocking=True)
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)


out = {"gpu": torch.cuda.get_device_name(0),
       "pcie_h2d_GBs": GB / timed(lambda: d.copy_(h, non_blocking=True)) / 1e9,
       "pcie_d2h_GBs": GB / timed(lambda: h.copy_(d, non_blocking=True)) / 1e9,
       "pcie_bidir_GBs": 2 * GB / timed(both) / 1e9,
       "hbm_d2d_GBs": 2 * 4 * GB / timed(lambda: big2.copy_(big)) / 1e9,
       "pcie_spec_GBs_per_dir": 64.0, "hbm_spec_GBs": 4800.0}
print(json.dumps(out, indent=1))
