"""Offload-round arithmetic from counts alone (no GPU data).

``HeadTokenOffloadEngine.run_round`` decides WHICH cells to evict from the
GPU scores, but how MANY -- and therefore how many pages survive, the new
logical-physical offset and which tail pages are freed -- depends only on
counts. ``plan_round`` computes exactly that, so the vLLM scheduler can free
blocks and size allocations without touching the engine (which lives in the
worker process under multiprocessing), and the worker can assert its round
did the same (see MULTIPROCESS_PLAN.md).

Why the count is exact, for every per-head scope (global / layer / uniform):

* ``K = max(keep_count(Lg), n_protected(Lg))``, protected = sink + window.
* Protected positions are always resident: a sink is never evictable, and a
  position in this round's window was in every earlier round's window (or
  not yet written). So a round's evictable cells are exactly the resident
  unprotected ones, ``R - L*H*n_prot`` of them, and the requested count
  ``R - L*H*K`` (``K >= n_prot``) is always available: ``E`` equals it.
* Every scope's budget is symmetric across layers / heads (global: total
  ``L*H*K``; layer: ``H*K`` per layer; uniform: ``K`` per (layer, head)),
  and every written token adds one resident cell to every unit, so the
  per-layer / per-unit residents stay equal under layer / uniform scope and
  only the total matters under global: in all three, ``E = max(0, R -
  L*H*K)`` and ``T = R - E`` resident cells survive.

Not for the shared engine (``TokenOffloadEngine``), whose arithmetic differs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PER_HEAD_SCOPES = ("global", "layer", "uniform")


@dataclass(frozen=True)
class RoundPlan:
    """One applied round of one request (``plan_round`` returns None when the
    round is skipped: nothing to evict, or it would free no page)."""
    req_id: str
    Lg: int                 # logical tokens written (== mapped) at the round
    K: int                  # kept tokens per (layer, head)
    E: int                  # cells offloaded this round (all layers / heads)
    T: int                  # resident cells after the round
    nb: int                 # whole blocks the survivors are packed into
    offset: int             # logical - physical length after the round
    keep_pages: int         # pages kept (nb + the scheduled tokens' pages)
    extra_tokens: int       # tokens scheduled this step (kept pages for them)


def keep_count(logical_len: int, fraction: float, budget: int | None) -> int:
    """Same as ``HeadTokenOffloadEngine.keep_count``."""
    k = budget if budget is not None else math.ceil((1.0 - fraction) * logical_len)
    return max(0, int(k))


def n_protected(n: int, sink: int, window: int) -> int:
    """Same as ``HeadTokenOffloadEngine._n_protected``."""
    s = min(n, sink)
    return s + min(window, max(0, n - s))


def plan_round(*, req_id: str, scope: str, L: int, H: int, P: int, Lg: int,
               resident: int, n_pages: int, extra_tokens: int, fraction: float,
               budget: int | None, sink: int, window: int) -> RoundPlan | None:
    """The round ``run_round`` would apply to a request with ``Lg`` written
    tokens, ``resident`` resident cells (``L*H*Lg`` minus all cells evicted
    so far) and ``n_pages`` pages; None if it would be skipped."""
    if scope not in PER_HEAD_SCOPES:
        raise ValueError(f"plan_round: scope {scope!r} is not a per-head scope")
    if Lg <= 0:
        return None
    K = max(keep_count(Lg, fraction, budget), n_protected(Lg, sink, window))
    E = max(0, int(resident) - L * H * K)
    if E == 0:
        return None
    T = int(resident) - E
    nb = -(-T // (L * H * P))
    if Lg - nb * P <= 0:
        return None
    keep_pages = min(int(n_pages), nb + -(-int(extra_tokens) // P))
    return RoundPlan(req_id=req_id, Lg=int(Lg), K=K, E=E, T=T, nb=nb,
                     offset=int(Lg) - nb * P, keep_pages=keep_pages,
                     extra_tokens=int(extra_tokens))
