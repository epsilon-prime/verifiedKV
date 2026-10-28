"""Needle-fetch key sketch, stored compactly per (layer, request slot, KV head).

Same quantities as the harness (``reference.top_energy_pairs`` + ``reference.quant_sketch``,
i.e. kvpress ``NeedleFetchPress._attach_sketch``): per (layer, KV head) the ``n_pairs`` RoPE
pairs of largest mean key energy over the context keys (post-RoPE, as cached), and a ``bits``-bit
uniform quantisation of those channels in groups of ``group`` context tokens (per channel zero =
group minimum, scale = range / levels, both rounded through fp16; the last group padded with the
last key). The harness keeps the dequantised keys; here the codes are stored packed (``8 / bits``
per byte) with the fp16 zero / scale, and ``score`` dequantises with the same expression
(``zero + code * scale`` in fp32), so the sketched keys are bit-identical.

Buffers (GPU):
  channels [L, S, H, nch]                int64  (nch = 2 * n_pairs)
  codes    [L, S, H, max_len, nch*bits/8] uint8
  zero     [L, S, H, n_groups, nch]       fp16
  scale    [L, S, H, n_groups, nch]       fp16
"""
from __future__ import annotations

import torch

from . import reference as ref

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:
    @triton.jit(do_not_specialize=["n", "c_base", "z_base", "c_s", "c_h", "z_s", "z_h"])
    def _score_kernel(codes, zero, scale, qs, slots, out, n, c_base, z_base,
                      c_s, c_h, z_s, z_h,
                      H: tl.constexpr, G: tl.constexpr, NCH: tl.constexpr, NB: tl.constexpr,
                      PER: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
                      BLOCK_N: tl.constexpr):
        """One (row, KV head) x BLOCK_N positions: unpack the codes, dequantise with the group's
        zero / scale (same fp32 expression as the torch path) and dot with the G queries."""
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        slot = tl.load(slots + b).to(tl.int64)
        n0 = tl.program_id(1) * BLOCK_N
        offs = n0 + tl.arange(0, BLOCK_N)
        live = offs < n
        c = tl.arange(0, NCH)
        base = c_base + slot * c_s + h * c_h
        byte = tl.load(codes + base + offs[:, None].to(tl.int64) * NB + (c // PER)[None, :],
                       mask=live[:, None], other=0)
        code = ((byte >> ((c % PER) * BITS).to(tl.uint8)[None, :]) & ((1 << BITS) - 1))
        zb = z_base + slot * z_s + h * z_h + (n0 // GROUP) * NCH
        z = tl.load(zero + zb + c).to(tl.float32)
        sc = tl.load(scale + zb + c).to(tl.float32)
        kh = z[None, :] + code.to(tl.float32) * sc[None, :]                 # [BLOCK_N, NCH]
        for g in tl.static_range(G):
            row = (b * H + h) * G + g
            qv = tl.load(qs + row * NCH + c)
            acc = tl.sum(kh * qv[None, :], axis=1)
            tl.store(out + row.to(tl.int64) * n + offs, acc, mask=live)


class SketchStore:
    def __init__(self, *, L: int, H: int, D: int, max_len: int, n_pairs: int, bits: int,
                 group: int, device) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(f"vkv sketch: bits={bits} not supported (2, 4 or 8)")
        if 2 * n_pairs > D:
            raise ValueError(f"vkv sketch: {n_pairs} pairs > head_dim / 2")
        self.L, self.H, self.D = int(L), int(H), int(D)
        self.max_len = int(max_len)
        self.n_pairs, self.bits, self.group = int(n_pairs), int(bits), int(group)
        self.nch = 2 * self.n_pairs
        self.per_byte = 8 // self.bits
        if self.nch % self.per_byte:
            raise ValueError("vkv sketch: channel count must fill whole bytes")
        self.nb = self.nch // self.per_byte
        self.levels = 2 ** self.bits - 1
        self.n_groups = -(-self.max_len // self.group)
        self.dev = device
        self.S = 0
        self.channels = self.codes = self.zero = self.scale = None
        # Bit shifts of the channels packed into one byte (channel c -> byte c // per_byte).
        self._shift = (torch.arange(self.per_byte, device=device, dtype=torch.uint8) * self.bits)

    def grow(self, S: int) -> None:
        if S <= self.S:
            return
        L, H = self.L, self.H
        with torch.inference_mode(False):
            ch = torch.zeros((L, S, H, self.nch), dtype=torch.long, device=self.dev)
            codes = torch.zeros((L, S, H, self.max_len, self.nb), dtype=torch.uint8, device=self.dev)
            zero = torch.zeros((L, S, H, self.n_groups, self.nch), dtype=torch.float16, device=self.dev)
            scale = torch.ones((L, S, H, self.n_groups, self.nch), dtype=torch.float16, device=self.dev)
        if self.S:
            ch[:, :self.S].copy_(self.channels)
            codes[:, :self.S].copy_(self.codes)
            zero[:, :self.S].copy_(self.zero)
            scale[:, :self.S].copy_(self.scale)
        self.channels, self.codes, self.zero, self.scale = ch, codes, zero, scale
        self.S = S

    def nbytes(self) -> int:
        return sum(int(t.numel()) * t.element_size()
                   for t in (self.channels, self.codes, self.zero, self.scale) if t is not None)

    # ---- build ---------------------------------------------------------------------------------

    @torch.no_grad()
    def build(self, layer: int, slot: int, keys: torch.Tensor) -> None:
        """Sketch one layer of one request from its context keys ``keys`` [H, ctx, D] (post-RoPE,
        any float dtype)."""
        H, n, D = keys.shape
        assert H == self.H and D == self.D and n <= self.max_len, (keys.shape, self.H, self.D)
        k4 = keys.unsqueeze(0)                                            # [1, H, n, D]
        ch = ref.top_energy_pairs(k4, self.n_pairs)[0]                    # [H, nch]
        kc = keys.float().gather(-1, ch[:, None, :].expand(H, n, self.nch))   # [H, n, nch]
        codes, zero, scale = self._quantise(kc)
        ng = zero.shape[1]
        self.channels[layer, slot] = ch
        self.codes[layer, slot, :, :n] = self._pack(codes)
        self.zero[layer, slot, :, :ng] = zero
        self.scale[layer, slot, :, :ng] = scale

    def _quantise(self, k: torch.Tensor):
        """``reference.quant_sketch`` split into codes and fp16 zero / scale: k [H, n, c]."""
        H, n, c = k.shape
        g = self.group
        ng = -(-n // g)
        pad = ng * g - n
        if pad:
            k = torch.cat([k, k[:, -1:].expand(H, pad, c)], dim=1)
        k = k.view(H, ng, g, c)
        zero = k.amin(2, keepdim=True).half().float()
        scale = ((k.amax(2, keepdim=True) - zero) / self.levels).clamp_min(1e-6).half().float()
        codes = ((k - zero) / scale).round().clamp(0, self.levels).to(torch.uint8)
        return (codes.view(H, ng * g, c)[:, :n], zero[:, :, 0].half(), scale[:, :, 0].half())

    def _pack(self, codes: torch.Tensor) -> torch.Tensor:
        """[..., nch] uint8 codes -> [..., nb] bytes (channel c at byte c // per_byte)."""
        c = codes.view(*codes.shape[:-1], self.nb, self.per_byte)
        return (c << self._shift).sum(-1, dtype=torch.uint8)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """[..., nb] bytes -> [..., nch] uint8 codes."""
        c = (packed.unsqueeze(-1) >> self._shift) & self.levels
        return c.view(*packed.shape[:-1], self.nch)

    # ---- decode-time scoring -------------------------------------------------------------------

    def k_hat(self, layer: int, slots: torch.Tensor, n: int) -> torch.Tensor:
        """Dequantised sketch [B, H, n, nch] fp32 of positions [0, n) (harness ``k_hat``)."""
        return self._k_hat_range(layer, slots, 0, n)

    @torch.no_grad()
    def score(self, layer: int, slots: torch.Tensor, q: torch.Tensor, n: int,
              chunk: int = 32768) -> torch.Tensor:
        """Sketch logits ``q[ch] . k_hat`` [B, Hq, n] fp32 for positions [0, n). ``q`` [B, Hq, D]
        fp32, already multiplied by the softmax scale (as the harness's ``q_all``)."""
        B, Hq, D = q.shape
        H = self.H
        G = Hq // H
        ch = self.channels[layer, slots]                                   # [B, H, nch]
        q_s = q.view(B, H, G, D).gather(-1, ch[:, :, None, :].expand(B, H, G, self.nch))
        out = torch.empty((B, H, G, n), dtype=torch.float32, device=q.device)
        blk = min(128, self.group)
        if triton is not None and q.is_cuda and self.group % blk == 0 and n > 0:
            cs, zs = self.codes.stride(), self.zero.stride()
            _score_kernel[(B * H, triton.cdiv(n, blk))](
                self.codes, self.zero, self.scale, q_s.contiguous(), slots.contiguous(), out, n,
                layer * cs[0], layer * zs[0], cs[1], cs[2], zs[1], zs[2],
                H=H, G=G, NCH=self.nch, NB=self.nb, PER=self.per_byte, BITS=self.bits,
                GROUP=self.group, BLOCK_N=blk)
            return out.view(B, Hq, n)
        return self.score_torch(layer, slots, q, n, chunk)

    def score_torch(self, layer, slots, q, n, chunk: int = 32768) -> torch.Tensor:
        """``score`` without the kernel (reference for its test)."""
        B, Hq, D = q.shape
        H, G = self.H, Hq // self.H
        ch = self.channels[layer, slots]
        q_s = q.view(B, H, G, D).gather(-1, ch[:, :, None, :].expand(B, H, G, self.nch))
        out = torch.empty((B, H, G, n), dtype=torch.float32, device=q.device)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            out[..., a:b] = torch.matmul(q_s, self._k_hat_range(layer, slots, a, b).transpose(-1, -2))
        return out.view(B, Hq, n)

    def _k_hat_range(self, layer, slots, a, b):
        codes = self._unpack(self.codes[layer, slots, :, a:b]).float()   # [B, H, b - a, nch]
        g = torch.div(torch.arange(a, b, device=self.dev), self.group, rounding_mode="floor")
        zero = self.zero[layer, slots][:, :, g].float()
        scale = self.scale[layer, slots][:, :, g].float()
        return zero + codes * scale
