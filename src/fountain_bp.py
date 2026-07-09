"""
Fountain (LT / low-density generator-matrix) code with soft belief-propagation
decoding --- the order-agnostic core of MOSAIC.

Unlike a convolutional code + Viterbi (which read coded bits in sequence along a
trellis), here the message is recovered from an UNORDERED bag of noisy parity
samples. Each "check" is XOR(m[i] for i in subset) observed through a noisy
channel; belief propagation on the bipartite (message-bit <-> check) graph
infers the message regardless of the order the checks arrived in. That order-
independence is what makes the watermark robust to token rearrangement.

A check is described by:
  - subset: tuple of message-bit indices it XORs together
  - llr:    soft observation of the parity value (sign = observed value,
            magnitude = confidence). +inf-ish for a clean read, ~0 for an
            unreliable one.

This module is pure numpy and has no dependency on any language model, so it can
be unit-tested on simulated channels.
"""

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------------

def robust_soliton(k: int, c: float = 0.1, delta: float = 0.5) -> np.ndarray:
    """
    Robust Soliton distribution over degrees 1..k (Luby, 2002). Returns a
    probability vector p where p[d-1] = P(degree = d).

    For the short block lengths in watermarking (k <= 64) we also clamp very
    high degrees, which only hurt at short length.
    """
    k = int(k)
    rho = np.zeros(k)
    rho[0] = 1.0 / k
    for d in range(2, k + 1):
        rho[d - 1] = 1.0 / (d * (d - 1))

    R = c * np.log(k / delta) * np.sqrt(k)
    tau = np.zeros(k)
    kr = max(int(round(k / R)), 1)
    for d in range(1, min(kr, k + 1)):
        tau[d - 1] = R / (d * k)
    if 1 <= kr <= k:
        tau[kr - 1] = R * np.log(max(R / delta, np.e)) / k

    mu = rho + tau
    mu /= mu.sum()
    return mu


def sample_degree(rng: np.random.Generator, dist: np.ndarray) -> int:
    """Sample a degree in 1..len(dist) from a probability vector."""
    return int(rng.choice(len(dist), p=dist)) + 1


# ---------------------------------------------------------------------------
# Belief propagation decoder (LDGM / LT over a soft channel)
# ---------------------------------------------------------------------------

def bp_decode(
    k: int,
    checks: Sequence[Tuple[Sequence[int], float]],
    max_iters: int = 80,
    damping: float = 0.0,
) -> np.ndarray:
    """
    Sum-product belief propagation.

    Args:
        k:        number of message bits to recover
        checks:   list of (subset, llr). subset is the message-bit indices the
                  check XORs; llr is the soft observation of that parity
                  (>0 => parity observed as 1).
        max_iters: BP iterations
        damping:  optional message damping in [0,1) for stability

    Returns:
        hard-decision message estimate, shape (k,), dtype int8
    """
    m = len(checks)
    if m == 0:
        return np.zeros(k, dtype=np.int8)

    # Build adjacency
    var_to_checks: List[List[int]] = [[] for _ in range(k)]
    check_vars: List[np.ndarray] = []
    check_llr = np.empty(m)
    # API convention: input llr > 0 means the observed parity is 1. The
    # sum-product box-plus rule below is written in the L = log P(0)/P(1)
    # convention (L > 0 => bit 0), so negate on the way in and flip the final
    # decision accordingly.
    for ci, (subset, llr) in enumerate(checks):
        sub = np.asarray(sorted(set(int(s) for s in subset)), dtype=np.int64)
        check_vars.append(sub)
        check_llr[ci] = -llr
        for v in sub:
            var_to_checks[v].append(ci)

    # Messages: msg_vc[ci] is an array aligned with check_vars[ci] (var->check),
    # msg_cv[ci] likewise (check->var). Initialize var->check to 0 (no prior).
    msg_vc = [np.zeros(len(cv)) for cv in check_vars]
    msg_cv = [np.zeros(len(cv)) for cv in check_vars]

    tanh_clip = 0.999999

    for _ in range(max_iters):
        # ---- check -> variable (tanh / box-plus rule) ----
        for ci in range(m):
            cv = check_vars[ci]
            L = len(cv)
            if L == 0:
                continue
            # incorporate the check's own observed parity llr as a constant term
            t = np.tanh(np.clip(msg_vc[ci] * 0.5, -20, 20))
            t = np.clip(t, -tanh_clip, tanh_clip)
            t_obs = np.tanh(np.clip(check_llr[ci] * 0.5, -20, 20))
            t_obs = float(np.clip(t_obs, -tanh_clip, tanh_clip))
            # leave-one-out product via prefix/suffix (zero-safe, no division)
            prefix = np.ones(L + 1)
            for a in range(L):
                prefix[a + 1] = prefix[a] * t[a]
            suffix = np.ones(L + 1)
            for a in range(L - 1, -1, -1):
                suffix[a] = suffix[a + 1] * t[a]
            new_cv = np.empty(L)
            for a in range(L):
                prod_excl = t_obs * prefix[a] * suffix[a + 1]
                prod_excl = float(np.clip(prod_excl, -tanh_clip, tanh_clip))
                new_cv[a] = 2.0 * np.arctanh(prod_excl)
            if damping > 0:
                msg_cv[ci] = damping * msg_cv[ci] + (1 - damping) * new_cv
            else:
                msg_cv[ci] = new_cv

        # ---- variable -> check (sum rule) ----
        # variable belief = sum of all incoming check->var messages
        var_belief = np.zeros(k)
        # gather
        for ci in range(m):
            cv = check_vars[ci]
            for a, v in enumerate(cv):
                var_belief[v] += msg_cv[ci][a]
        # outgoing var->check = belief minus this check's contribution
        for ci in range(m):
            cv = check_vars[ci]
            for a, v in enumerate(cv):
                msg_vc[ci][a] = var_belief[v] - msg_cv[ci][a]

    # Final decision (internal convention is L = log P(0)/P(1), so belief < 0
    # means bit 1).
    var_belief = np.zeros(k)
    for ci in range(m):
        cv = check_vars[ci]
        for a, v in enumerate(cv):
            var_belief[v] += msg_cv[ci][a]
    return (var_belief < 0).astype(np.int8)
