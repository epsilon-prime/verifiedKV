# Two ways to cut Vkv1's fetch without weakening the per-decision guarantee (2026-10-24)

(1) **hx**: a tighter, still rigorous failure bound (exact on the heavy terms, Chernoff on the light remainder).
(2) **store**: a capped, budget-charged persistent store whose contents never depend on the draw.

## 0. Setup (one decision = one (layer, query head, position) row)

Context entries i = 1..n with exact weights w_i > 0 (perfect sketch), keep probabilities r_i, and the draw
Z_i ~ Bernoulli(r_i), independent across i and independent of the weights. The tail is the set of entries with
r_i < 1 that are not in the store. The plan fixes, as functions of (w, r) only:
- T, the truncated set (MSE rule);
- a fetch order;
- k, the number of entries fetched (F = the first k).

With c_i = w_i on T and c_i = w_i / r_i on the Horvitz-Thompson part H = tail \ T, and "live" = unfetched tail entries
with 0 < r_i < 1,

    B_hat - B = -beta(k) + Y,   Y = sum_live c_i (Z_i - r_i),   beta(k) = sum_{T \ F} w_i (1 - r_i)  (known).

Failure means |B_hat - B| > eps B. With u+ = eps B + beta and u- = eps B - beta,

    P(fail) = P(Y > u+) + P(Y < -u-).

These are two DISJOINT events, so adding them loses nothing. Write S = sum_live c_i Z_i >= 0 and
mu = sum_live c_i r_i. Then the upper fail is S > a := mu + u+ and the lower fail is S < b := mu - u-.

## 1. hx: exact heavy part plus Chernoff light part

**Why Chernoff is loose here.** Take one unfetched HT token with c = w / r much larger than eps B. The decision fails
exactly when that token is sampled, so the true probability is r. Chernoff sees only its moment-generating function
and returns exp(-KL(q || r)) with q = a / c = r + u+ / c. For r = 0.01 and q = 0.3, that is 0.46 against the true 0.01: 46x
too pessimistic. Unfetched heavy tokens with small r are exactly the situation Vkv1 is in, which is why chern_grad sits
19x above the Monte Carlo ideal (mc_grad) at eps 0.5.

**Bound.** Split the live set into the m entries with the largest c (the heavy set M; it may depend on k, w and r, not
on Z) and the rest L. S = S_M + S_L, with S_M and S_L independent.

*Upper tail.*
- Round each heavy jump UP onto a grid of step h_a: c_i^up = h_a ceil(c_i / h_a) >= c_i. Then S_M^up >= S_M for
  every draw, so P(S > a) <= P(S_M^up + S_L > a).
- N = S_M^up / h_a is integer-valued. Its distribution is computed exactly by dynamic programming over the m terms
  (multiplying the polynomials (1 - r_i) + r_i x^{s_i}, with s_i = c_i^up / h_a).
- Conditioning on N and using independence:

      P(S > a) <= sum_N P(N) * min(1, Ch_L+(a - h_a N)),   Ch_L+(t) = inf_{lam >= 0} exp(K_L(lam) - lam t),

  where K_L(lam) = sum_L log(1 - r_i + r_i e^{lam c_i}). This is Chernoff for S_L alone, valid for every t.
- With N_b bins and h_a = a / (N_b - 1), the bins N <= N_b - 2 have t >= h_a > 0. The last bin collects all
  N >= N_b - 1, and its mass counts as failure with probability 1. This only adds mass, so the bound stays valid.

*Lower tail.*
- Round heavy jumps DOWN: c_i^dn = h_b floor(c_i / h_b) <= c_i. Then S_M^dn <= S_M, so
  P(S < b) <= P(S_M^dn + S_L < b).
- Because S_L >= 0, the bins with h_b N >= b contribute 0:

      P(S < b) <= sum_{N < b / h_b} P(N) * min(1, Ch_L-(b - h_b N)),   Ch_L-(t) = inf_{lam >= 0} exp(K_L(-lam) + lam t).

  If b <= 0 the probability is 0. If u- <= 0, then b >= mu and the formula still applies; no special case is needed.

*Properties.*
- Validity: every step is an exact identity, an inequality between random variables that holds for every draw
  (rounding), or Chernoff applied to S_L alone. It is a rigorous upper bound, not an estimate.
- Tightness:
  - m = 0 reduces exactly to the current two-point Chernoff (diag_lib.chernoff_curves; uncentred form, the same lam
    grid).
  - As h -> 0 with every live term heavy, it is the exact failure probability.
  - The rounding slack is at most h per sampled heavy token, and h <= a / (N_b - 1) is about 1.5% of eps B at
    N_b = 512, eps = 0.3.
- Choosing k: any k that is a function of (w, r) alone keeps the per-decision guarantee, because the bound holds at
  that k for the fixed weights. So searching over k is free of selection bias:
  - evaluate the bound on the 2^(1/4) ladder;
  - take the first passing ladder point;
  - refine with up to 8 evenly spaced k between it and the previous ladder point, taking the smallest that passes.
  The bound is not monotone in k (neither is chern's), so "first passing" is a choice, not an optimum. Any passing k is
  valid.
- Cost per (row, k): an m x N_b dynamic programme plus |grid| x N_b for the light Chernoff, on the ladder only.

## 2. store: capped persistence charged to the budget

**Memory.** Per (layer, KV head), the Bernoulli draw keeps (kappa - rho) n entries in expectation, using the same
global c as before. A store of at most rho n slots holds K and V of entries fetched earlier in this answer. Total GPU
memory is therefore at most kappa n, plus one forward's fetch buffer, exactly as for no persistence. rho is charged in
full even when some slots hold already-sampled entries, so the accounting is conservative. needle_final_c does the
same with rho = 1%.

**Policy (draw-free).**
- Each entry has a priority pi_i = max over the rows of this answer so far (question rows, then each decode step) of
  w_i / B_row, for entries that were in some row's fetch set F.
- After every forward: store := the top rho n entries by pi among (store) united with (F of this forward), restricted
  to the tail.
- The store is emptied at a new question.
- Store entries are exact for that forward: weight w, no bias. That is, they leave the random part.

**Why the guarantee survives.** The store at any forward is a deterministic function of the queries and keys
(through F and pi), and these are independent of Z. Conditional on the store, the Z_i of the remaining tail are still
independent Bernoulli(r_i), so Section 1 (or any certificate) applies verbatim at each forward.

**What must NOT be done.** The memory occupied is |store and not Z|, since sampled entries are already resident. It
would be tempting to fill the cap counting only unsampled entries ("free slots" for sampled ones). Then the store
depends on Z: an entry left out because the cap was hit is known to have Z = 0, so the tail is no longer Bernoulli(r).
The policy above ignores Z entirely, and this is the reason it charges rho in full.

**Trade-off.** Lowering the draw from kappa to kappa - rho raises every c = w / r by kappa / (kappa - rho), which is
1.19x at rho = 1%. That costs some question-pass fetch. The store removes repeated decode fetches of the same heavy
tokens, which were about half of chern_grad's traffic without persistence at eps 0.5.

## 3. A caveat that predates this note (applies to every Vkv1 variant)

In REAL generation the decode query at step t depends on the tokens generated so far, and these depend on earlier
attention estimates, which used the same layer's draw Z. So at decode, w is not independent of Z, and the per-decision
guarantee is exact only for the question pass. The question pass is fine: layer l's queries depend only on layers < l,
whose draws are independent. The offline diagnostic replays fixed full-cache queries, so there it is exact. Making
decode rigorous would need fresh draws per step or a sequential (martingale) argument. Not addressed here; recorded so
that it is not forgotten.
