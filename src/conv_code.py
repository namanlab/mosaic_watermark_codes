"""
Rate-1/2 convolutional code (K=7, generators 133/171 octal) with
soft-decision Viterbi decoding.

This is the same code used in GPS, IEEE 802.11, and DVB — chosen because
the watermark channel (token-level green/red votes) is a noisy BSC-like
channel, and soft Viterbi gives ~5 dB of coding gain over uncoded
transmission. This is what converts ~95% bit accuracy into ~100% exact
message match.

All operations are vectorized numpy over the 64 trellis states.
"""

import numpy as np

# Industry-standard generators for K=7, rate 1/2
G1 = 0o133  # 1011011
G2 = 0o171  # 1111001
K = 7                 # constraint length
NUM_STATES = 1 << (K - 1)   # 64
TAIL_BITS = K - 1     # zero-tail termination


def _parity(x: np.ndarray) -> np.ndarray:
    """Bitwise parity of each element (vectorized popcount mod 2)."""
    x = x.copy()
    result = np.zeros_like(x)
    while np.any(x):
        result ^= x & 1
        x >>= 1
    return result


# Precompute trellis: for each state s (6 bits of history) and input bit b,
# the register is (b << 6) | s viewed as [b, s5..s0]; outputs are parities
# against G1, G2; next state is the top 6 bits of the shifted register.
_states = np.arange(NUM_STATES)
_TRELLIS_OUT = np.zeros((NUM_STATES, 2, 2), dtype=np.int8)   # [state, input] -> (out1, out2)
_TRELLIS_NEXT = np.zeros((NUM_STATES, 2), dtype=np.int64)    # [state, input] -> next state
for _b in (0, 1):
    _reg = (_b << (K - 1)) | _states          # 7-bit register contents
    _TRELLIS_OUT[:, _b, 0] = _parity(_reg & G1)
    _TRELLIS_OUT[:, _b, 1] = _parity(_reg & G2)
    _TRELLIS_NEXT[:, _b] = _reg >> 1          # shift right: input becomes MSB of state


def encode(message_bits) -> np.ndarray:
    """
    Convolutionally encode message bits (rate 1/2, zero-tail terminated).

    Args:
        message_bits: iterable of 0/1, length k

    Returns:
        coded bits, length 2*(k + 6)
    """
    bits = list(message_bits) + [0] * TAIL_BITS
    out = np.zeros(2 * len(bits), dtype=np.int8)
    state = 0
    for i, b in enumerate(bits):
        out[2 * i] = _TRELLIS_OUT[state, b, 0]
        out[2 * i + 1] = _TRELLIS_OUT[state, b, 1]
        state = _TRELLIS_NEXT[state, b]
    return out


def coded_length(num_message_bits: int) -> int:
    """Number of coded bits for a k-bit message."""
    return 2 * (num_message_bits + TAIL_BITS)


def viterbi_decode(llrs: np.ndarray, num_message_bits: int) -> np.ndarray:
    """
    Soft-decision Viterbi decoding.

    Args:
        llrs: array of length 2*(k+6); llrs[i] > 0 means coded bit i is
              more likely 1, magnitude = confidence. Zero = erasure.
        num_message_bits: k

    Returns:
        decoded message bits, length k
    """
    n_steps = num_message_bits + TAIL_BITS
    assert len(llrs) == 2 * n_steps, f"expected {2 * n_steps} llrs, got {len(llrs)}"

    # Branch metric for emitting bit o given llr L: contribute +L if o=1, -L...
    # Use metric = (2o-1) * L so higher total metric = more consistent path.
    path_metric = np.full(NUM_STATES, -np.inf)
    path_metric[0] = 0.0   # encoder starts in state 0
    backptr = np.zeros((n_steps, NUM_STATES), dtype=np.int8)  # chosen input bit per state

    # Precompute, for each (state, input), branch metric coefficients
    out_sign = 2.0 * _TRELLIS_OUT.astype(np.float64) - 1.0   # [state, input, 2] in {-1,+1}

    prev_state = np.zeros((n_steps, NUM_STATES), dtype=np.int64)

    for t in range(n_steps):
        l1, l2 = llrs[2 * t], llrs[2 * t + 1]
        # branch metric for each (state, input)
        bm = out_sign[:, :, 0] * l1 + out_sign[:, :, 1] * l2   # [64, 2]
        cand = path_metric[:, None] + bm                        # [64, 2] metric arriving at _TRELLIS_NEXT

        # During the zero-tail, only input bit 0 is allowed
        allowed_inputs = (0,) if t >= num_message_bits else (0, 1)

        new_metric = np.full(NUM_STATES, -np.inf)
        new_back = np.zeros(NUM_STATES, dtype=np.int8)
        new_prev = np.zeros(NUM_STATES, dtype=np.int64)
        # scatter-max into next states
        for b in allowed_inputs:
            nxt = _TRELLIS_NEXT[:, b]
            np.maximum.at(new_metric, nxt, cand[:, b])
        # second pass to record argmax (which (prev, b) achieved the max)
        for b in allowed_inputs:
            nxt = _TRELLIS_NEXT[:, b]
            hit = (cand[:, b] == new_metric[nxt]) & np.isfinite(cand[:, b])
            new_back[nxt[hit]] = b
            new_prev[nxt[hit]] = _states[hit]

        path_metric = new_metric
        backptr[t] = new_back
        prev_state[t] = new_prev

    # Zero-tail: encoder ends in state 0
    state = 0
    decoded = np.zeros(n_steps, dtype=np.int8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = backptr[t, state]
        state = prev_state[t, state]

    return decoded[:num_message_bits]
