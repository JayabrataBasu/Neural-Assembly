#!/usr/bin/env python3
"""
tests for LSTM and GRU recurrent layers.

validates construction, forward pass shapes, numerical properties,
batched operation, initial state handling, and parameter accessors.
all backed by C implementations in rnn.c.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyneural.rnn import LSTM, GRU

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1


# ── helpers ──────────────────────────────────────────────────────

def make_input(batch, seq_len, input_size, offset=0.0):
    """Generate a deterministic input: sin-wave shifted by index."""
    data = []
    for b in range(batch):
        for t in range(seq_len):
            for i in range(input_size):
                idx = b * seq_len * input_size + t * input_size + i
                data.append(math.sin((idx + offset) * 0.1))
    return data


def all_finite(xs):
    return all(math.isfinite(x) for x in xs)


def any_nonzero(xs, tol=1e-12):
    return any(abs(x) > tol for x in xs)


# ── Section 1: LSTM construction ─────────────────────────────────

print("\n--- LSTM construction ---")

lstm = LSTM(input_size=4, hidden_size=8)
check(lstm._ptr is not None and lstm._ptr != 0, "LSTM creates valid pointer")
check(lstm.input_size == 4, "input_size stored correctly")
check(lstm.hidden_size == 8, "hidden_size stored correctly")
check(repr(lstm) == "LSTM(input_size=4, hidden_size=8)", "LSTM repr")


# ── Section 2: LSTM forward — basic shapes ───────────────────────

print("\n--- LSTM forward shapes ---")

inp = make_input(batch=1, seq_len=5, input_size=4)
output, (h_n, c_n) = lstm(inp, batch=1, seq_len=5)
check(len(output) == 1 * 5 * 8, f"output length = {1*5*8}")
check(len(h_n) == 1 * 8, f"h_n length = {1*8}")
check(len(c_n) == 1 * 8, f"c_n length = {1*8}")
check(all_finite(output), "output is finite")
check(all_finite(h_n), "h_n is finite")
check(all_finite(c_n), "c_n is finite")


# ── Section 3: LSTM — last time step matches h_n ─────────────────

print("\n--- LSTM h_n matches last output ---")

# the last hidden state should be the same as the output at t=seq_len-1
last_output = output[(5 - 1) * 8 : 5 * 8]  # batch=0, t=4 (last)
for j in range(8):
    check(abs(last_output[j] - h_n[j]) < 1e-12,
          f"h_n[{j}] matches output[-1][{j}]")


# ── Section 4: LSTM — hidden state bounded by tanh ───────────────

print("\n--- LSTM hidden state bounds ---")

# h = o * tanh(c), so |h| <= 1
check(all(abs(x) <= 1.0 + 1e-10 for x in h_n), "h_n values in [-1, 1]")
check(any_nonzero(h_n), "h_n is not all zeros")


# ── Section 5: LSTM — batched operation ──────────────────────────

print("\n--- LSTM batched ---")

lstm_b = LSTM(input_size=3, hidden_size=5)
inp_b = make_input(batch=4, seq_len=6, input_size=3)
out_b, (h_b, c_b) = lstm_b(inp_b, batch=4, seq_len=6)
check(len(out_b) == 4 * 6 * 5, "batched output length correct")
check(len(h_b) == 4 * 5, "batched h_n length correct")
check(len(c_b) == 4 * 5, "batched c_n length correct")
check(all_finite(out_b), "batched output is finite")

# different batch elements should give different results
h_batch0 = h_b[0:5]
h_batch1 = h_b[5:10]
check(any(abs(h_batch0[j] - h_batch1[j]) > 1e-6 for j in range(5)),
      "different batches produce different h_n")


# ── Section 6: LSTM — initial state ──────────────────────────────

print("\n--- LSTM initial state ---")

lstm_h0 = LSTM(input_size=2, hidden_size=3)
inp_h0 = make_input(batch=1, seq_len=4, input_size=2)

# run with zero init
out_zero, (h_zero, _) = lstm_h0(inp_h0, batch=1, seq_len=4)

# run with non-zero init
h_init = [0.5, -0.3, 0.1]
c_init = [0.2, 0.0, -0.4]
out_init, (h_init_out, _) = lstm_h0(inp_h0, batch=1, seq_len=4,
                                     h_0=h_init, c_0=c_init)

# results should differ
check(any(abs(out_zero[j] - out_init[j]) > 1e-6 for j in range(len(out_zero))),
      "non-zero initial state changes output")
check(any(abs(h_zero[j] - h_init_out[j]) > 1e-6 for j in range(3)),
      "non-zero initial state changes h_n")


# ── Section 7: LSTM — seq_len auto-infer ─────────────────────────

print("\n--- LSTM seq_len inference ---")

lstm_auto = LSTM(input_size=2, hidden_size=4)
inp_auto = make_input(batch=1, seq_len=7, input_size=2)
out_auto, _ = lstm_auto(inp_auto, batch=1)  # no explicit seq_len
check(len(out_auto) == 1 * 7 * 4, "auto-inferred seq_len gives correct output length")


# ── Section 8: LSTM — parameters ─────────────────────────────────

print("\n--- LSTM parameters ---")

lstm_p = LSTM(input_size=3, hidden_size=4)
params = lstm_p.parameters()
check(len(params) == 4, "parameters() returns 4 arrays")
check(len(params[0]) == 4 * 4 * 3, "W_ih shape = [4*H, I]")
check(len(params[1]) == 4 * 4 * 4, "W_hh shape = [4*H, H]")
check(len(params[2]) == 4 * 4, "b_ih shape = [4*H]")
check(len(params[3]) == 4 * 4, "b_hh shape = [4*H]")

# weights should be non-zero (Xavier init)
check(any_nonzero(params[0]), "W_ih is not all zeros")
check(any_nonzero(params[1]), "W_hh is not all zeros")

# forget gate bias should be 1.0 (indices 4..7 for hidden=4)
fg_bias = params[2][4:8]
check(all(abs(b - 1.0) < 1e-10 for b in fg_bias), "forget gate bias = 1.0")


# ── Section 9: LSTM — wrong input size raises ────────────────────

print("\n--- LSTM input validation ---")

try:
    lstm(make_input(1, 5, 4) + [0.0], batch=1, seq_len=5)  # one extra element
    check(False, "wrong input length should raise ValueError")
except ValueError:
    check(True, "wrong input length raises ValueError")


# ── Section 10: LSTM — deterministic (same input → same output) ──

print("\n--- LSTM determinism ---")

lstm_det = LSTM(input_size=2, hidden_size=3)
inp_det = make_input(1, 3, 2)
out1, _ = lstm_det(inp_det, batch=1, seq_len=3)
out2, _ = lstm_det(inp_det, batch=1, seq_len=3)
check(all(abs(out1[i] - out2[i]) < 1e-14 for i in range(len(out1))),
      "same input produces identical output")


# ══════════════════════════════════════════════════════════════════
# GRU
# ══════════════════════════════════════════════════════════════════

# ── Section 11: GRU construction ─────────────────────────────────

print("\n--- GRU construction ---")

gru = GRU(input_size=4, hidden_size=8)
check(gru._ptr is not None and gru._ptr != 0, "GRU creates valid pointer")
check(gru.input_size == 4, "input_size stored correctly")
check(gru.hidden_size == 8, "hidden_size stored correctly")
check(repr(gru) == "GRU(input_size=4, hidden_size=8)", "GRU repr")


# ── Section 12: GRU forward — basic shapes ───────────────────────

print("\n--- GRU forward shapes ---")

inp_g = make_input(batch=1, seq_len=5, input_size=4)
out_g, h_g = gru(inp_g, batch=1, seq_len=5)
check(len(out_g) == 1 * 5 * 8, f"output length = {1*5*8}")
check(len(h_g) == 1 * 8, f"h_n length = {1*8}")
check(all_finite(out_g), "output is finite")
check(all_finite(h_g), "h_n is finite")


# ── Section 13: GRU — last output matches h_n ────────────────────

print("\n--- GRU h_n matches last output ---")

last_g = out_g[(5 - 1) * 8 : 5 * 8]
for j in range(8):
    check(abs(last_g[j] - h_g[j]) < 1e-12,
          f"h_n[{j}] matches output[-1][{j}]")


# ── Section 14: GRU — batched operation ──────────────────────────

print("\n--- GRU batched ---")

gru_b = GRU(input_size=3, hidden_size=5)
inp_gb = make_input(batch=4, seq_len=6, input_size=3)
out_gb, h_gb = gru_b(inp_gb, batch=4, seq_len=6)
check(len(out_gb) == 4 * 6 * 5, "batched output length correct")
check(len(h_gb) == 4 * 5, "batched h_n length correct")
check(all_finite(out_gb), "batched output is finite")

h_g0 = h_gb[0:5]
h_g1 = h_gb[5:10]
check(any(abs(h_g0[j] - h_g1[j]) > 1e-6 for j in range(5)),
      "different batches produce different h_n")


# ── Section 15: GRU — initial state ──────────────────────────────

print("\n--- GRU initial state ---")

gru_h0 = GRU(input_size=2, hidden_size=3)
inp_gh0 = make_input(batch=1, seq_len=4, input_size=2)

out_gzero, h_gzero = gru_h0(inp_gh0, batch=1, seq_len=4)
out_ginit, h_ginit = gru_h0(inp_gh0, batch=1, seq_len=4, h_0=[0.5, -0.3, 0.1])

check(any(abs(out_gzero[j] - out_ginit[j]) > 1e-6 for j in range(len(out_gzero))),
      "non-zero initial state changes output")


# ── Section 16: GRU — parameters ─────────────────────────────────

print("\n--- GRU parameters ---")

gru_p = GRU(input_size=3, hidden_size=4)
gparams = gru_p.parameters()
check(len(gparams) == 4, "parameters() returns 4 arrays")
check(len(gparams[0]) == 3 * 4 * 3, "W_ih shape = [3*H, I]")
check(len(gparams[1]) == 3 * 4 * 4, "W_hh shape = [3*H, H]")
check(len(gparams[2]) == 3 * 4, "b_ih shape = [3*H]")
check(len(gparams[3]) == 3 * 4, "b_hh shape = [3*H]")
check(any_nonzero(gparams[0]), "W_ih is not all zeros")


# ── Section 17: GRU — seq_len auto-infer ─────────────────────────

print("\n--- GRU seq_len inference ---")

gru_auto = GRU(input_size=2, hidden_size=4)
inp_gauto = make_input(batch=1, seq_len=7, input_size=2)
out_gauto, _ = gru_auto(inp_gauto, batch=1)
check(len(out_gauto) == 1 * 7 * 4, "auto-inferred seq_len gives correct output length")


# ── Section 18: GRU — deterministic ──────────────────────────────

print("\n--- GRU determinism ---")

gru_det = GRU(input_size=2, hidden_size=3)
inp_gdet = make_input(1, 3, 2)
out_g1, _ = gru_det(inp_gdet, batch=1, seq_len=3)
out_g2, _ = gru_det(inp_gdet, batch=1, seq_len=3)
check(all(abs(out_g1[i] - out_g2[i]) < 1e-14 for i in range(len(out_g1))),
      "same input produces identical output")


# ── Section 19: GRU — hidden bounded ─────────────────────────────

print("\n--- GRU hidden bounds ---")

# GRU: h' = (1-z)*tanh(...) + z*h, starting from h=0
# after first step with zero init, h is a convex combination of tanh and 0
# so |h| <= 1 should hold, but in general multiple steps can accumulate.
# However GRU is designed to keep |h| bounded by the tanh/sigmoid gates.
# Let's just check the first run didn't blow up.
check(all(abs(x) < 10.0 for x in h_g), "h_n values are bounded")
check(any_nonzero(h_g), "h_n is not all zeros")


# ── Section 20: cross-check — LSTM vs GRU produce different outputs ─

print("\n--- LSTM vs GRU differ ---")

lstm_cmp = LSTM(input_size=4, hidden_size=8)
gru_cmp = GRU(input_size=4, hidden_size=8)
inp_cmp = make_input(1, 5, 4, offset=42.0)

out_l, _ = lstm_cmp(inp_cmp, batch=1, seq_len=5)
out_r, _ = gru_cmp(inp_cmp, batch=1, seq_len=5)
check(any(abs(out_l[i] - out_r[i]) > 1e-6 for i in range(min(len(out_l), len(out_r)))),
      "LSTM and GRU produce different outputs")


# ── Section 21: larger hidden sizes ──────────────────────────────

print("\n--- larger hidden sizes ---")

lstm_big = LSTM(input_size=32, hidden_size=64)
inp_big = make_input(2, 10, 32)
out_big, (h_big, c_big) = lstm_big(inp_big, batch=2, seq_len=10)
check(len(out_big) == 2 * 10 * 64, "large LSTM output shape correct")
check(all_finite(out_big), "large LSTM output is finite")
check(all(abs(x) <= 1.0 + 1e-10 for x in h_big), "large LSTM h_n bounded by 1")

gru_big = GRU(input_size=32, hidden_size=64)
out_gbig, h_gbig = gru_big(inp_big, batch=2, seq_len=10)
check(len(out_gbig) == 2 * 10 * 64, "large GRU output shape correct")
check(all_finite(out_gbig), "large GRU output is finite")


# ── Summary ──────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"RNN tests:  {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
