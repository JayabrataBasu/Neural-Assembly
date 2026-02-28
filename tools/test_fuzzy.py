#!/usr/bin/env python3
"""
Tests for the fuzzy logic engine — membership functions, operators,
defuzzification methods, and the full FuzzySystem inference pipeline.
"""
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ===================================================================
print("=== Triangular membership ===")

# Classic triangle: a=0, b=5, c=10
check("tri peak at centre", approx(pn.triangular(5, 0, 5, 10), 1.0))
check("tri left shoulder", approx(pn.triangular(0, 0, 5, 10), 0.0))
check("tri right shoulder", approx(pn.triangular(10, 0, 5, 10), 0.0))
check("tri midpoint left", approx(pn.triangular(2.5, 0, 5, 10), 0.5))
check("tri midpoint right", approx(pn.triangular(7.5, 0, 5, 10), 0.5))
check("tri outside left", approx(pn.triangular(-1, 0, 5, 10), 0.0))
check("tri outside right", approx(pn.triangular(11, 0, 5, 10), 0.0))

# Asymmetric triangle
check("tri asymmetric peak", approx(pn.triangular(2, 0, 2, 10), 1.0))
check("tri asymmetric 75%", approx(pn.triangular(1, 0, 2, 10), 0.5))

# ===================================================================
print("\n=== Trapezoidal membership ===")

# Trapezoid: a=0, b=2, c=8, d=10
check("trap plateau left", approx(pn.trapezoidal(2, 0, 2, 8, 10), 1.0))
check("trap plateau middle", approx(pn.trapezoidal(5, 0, 2, 8, 10), 1.0))
check("trap plateau right", approx(pn.trapezoidal(8, 0, 2, 8, 10), 1.0))
check("trap ramp up midpoint", approx(pn.trapezoidal(1, 0, 2, 8, 10), 0.5))
check("trap ramp down midpoint", approx(pn.trapezoidal(9, 0, 2, 8, 10), 0.5))
check("trap outside left", approx(pn.trapezoidal(-1, 0, 2, 8, 10), 0.0))
check("trap outside right", approx(pn.trapezoidal(11, 0, 2, 8, 10), 0.0))

# ===================================================================
print("\n=== Gaussian membership ===")

check("gauss peak at mean", approx(pn.gaussian(5, 5, 2), 1.0))
check("gauss at ±1σ", approx(pn.gaussian(7, 5, 2), math.exp(-0.5)))
check("gauss symmetry", approx(pn.gaussian(3, 5, 2), pn.gaussian(7, 5, 2)))
check("gauss far from mean → ~0", pn.gaussian(100, 5, 2) < 1e-10)
check("gauss sigma=0 → 0", approx(pn.gaussian(5, 5, 0), 0.0))

# ===================================================================
print("\n=== Fuzzy operators ===")

check("AND(0.3, 0.7) = 0.3", approx(pn.fuzzy_and(0.3, 0.7), 0.3))
check("AND(1.0, 0.5) = 0.5", approx(pn.fuzzy_and(1.0, 0.5), 0.5))
check("AND(0.0, 0.9) = 0.0", approx(pn.fuzzy_and(0.0, 0.9), 0.0))

check("OR(0.3, 0.7) = 0.7", approx(pn.fuzzy_or(0.3, 0.7), 0.7))
check("OR(0.0, 0.0) = 0.0", approx(pn.fuzzy_or(0.0, 0.0), 0.0))
check("OR(1.0, 0.5) = 1.0", approx(pn.fuzzy_or(1.0, 0.5), 1.0))

check("NOT(0.0) = 1.0", approx(pn.fuzzy_not(0.0), 1.0))
check("NOT(1.0) = 0.0", approx(pn.fuzzy_not(1.0), 0.0))
check("NOT(0.3) = 0.7", approx(pn.fuzzy_not(0.3), 0.7))

# De Morgan's law: NOT(AND(a, b)) == OR(NOT(a), NOT(b))
a, b = 0.4, 0.6
lhs = pn.fuzzy_not(pn.fuzzy_and(a, b))
rhs = pn.fuzzy_or(pn.fuzzy_not(a), pn.fuzzy_not(b))
check("De Morgan's law holds", approx(lhs, rhs))

# ===================================================================
print("\n=== Defuzzification — centroid ===")

# Symmetric distribution → centroid at the middle
vals = [0.0, 1.0, 2.0, 3.0, 4.0]
mems = [0.0, 0.5, 1.0, 0.5, 0.0]
c = pn.defuzz_centroid(vals, mems)
check("symmetric → centroid at 2.0", approx(c, 2.0))

# Skewed left
mems_left = [1.0, 0.5, 0.0, 0.0, 0.0]
c_left = pn.defuzz_centroid(vals, mems_left)
check("skewed left → centroid < 1.0", c_left < 1.0)

# All zeros → should return 0
c_zero = pn.defuzz_centroid(vals, [0.0, 0.0, 0.0, 0.0, 0.0])
check("all-zero memberships → centroid = 0", approx(c_zero, 0.0))

# ===================================================================
print("\n=== Defuzzification — bisector ===")

b_sym = pn.defuzz_bisector(vals, mems)
check("symmetric → bisector at 2.0", approx(b_sym, 2.0))

# Skewed right → bisector should shift right
mems_right = [0.0, 0.0, 0.0, 0.5, 1.0]
b_right = pn.defuzz_bisector(vals, mems_right)
check("skewed right → bisector > 3.0", b_right >= 3.0)

# ===================================================================
print("\n=== Defuzzification — mean of maximum ===")

# Peak is clearly at index 2 (value = 2.0)
m = pn.defuzz_mom(vals, mems)
check("single peak → MOM at 2.0", approx(m, 2.0))

# Two equal peaks at 1.0 and 3.0 → mean should be 2.0
mems_two = [0.0, 1.0, 0.5, 1.0, 0.0]
m2 = pn.defuzz_mom(vals, mems_two)
check("two equal peaks → MOM = average", approx(m2, 2.0))

# ===================================================================
print("\n=== FuzzySystem construction ===")

fis = pn.FuzzySystem(n_inputs=2, output_range=(0, 30), resolution=200)
check("n_inputs = 2", fis.n_inputs == 2)
check("n_rules = 0 initially", fis.n_rules == 0)
check("defuzz_method = centroid", fis.defuzz_method == "centroid")

# ===================================================================
print("\n=== FuzzySystem — tipping example ===")

# Classic Mamdani tipping controller: food + service → tip %
tip = pn.FuzzySystem(n_inputs=2, output_range=(0, 30), resolution=200)

# Input 0: food quality [0, 10]
tip.add_input_mf(0, "poor",      "triangular", 0, 0, 5,   range=(0, 10))
tip.add_input_mf(0, "average",   "triangular", 0, 5, 10)
tip.add_input_mf(0, "excellent", "triangular", 5, 10, 10)

# Input 1: service quality [0, 10]
tip.add_input_mf(1, "poor",      "triangular", 0, 0, 5,   range=(0, 10))
tip.add_input_mf(1, "average",   "triangular", 0, 5, 10)
tip.add_input_mf(1, "excellent", "triangular", 5, 10, 10)

# Output: tip percentage
tip.add_output_mf("low",    "triangular", 0, 5, 10)
tip.add_output_mf("medium", "triangular", 10, 15, 20)
tip.add_output_mf("high",   "triangular", 20, 25, 30)

check("input term names populated",
      len(tip.input_term_names[0]) == 3)
check("output term names populated",
      len(tip.output_term_names) == 3)

# Rules: poor food OR poor service → low tip, etc.
tip.add_rule([(0, "poor")], "low")
tip.add_rule([(1, "poor")], "low")
tip.add_rule([(0, "average")], "medium")
tip.add_rule([(1, "average")], "medium")
tip.add_rule([(0, "excellent"), (1, "excellent")], "high")

check("5 rules added", tip.n_rules == 5)

# Excellent food + excellent service → should tip well (> 15%)
t1 = tip.evaluate([9.0, 9.0])
check("great food + great service → high tip",
      t1 > 15.0)

# Poor food + poor service → low tip (< 12%)
t2 = tip.evaluate([1.0, 1.0])
check("poor food + poor service → low tip",
      t2 < 12.0)

# Average inputs → somewhere in the middle
t3 = tip.evaluate([5.0, 5.0])
check("average inputs → mid-range tip",
      8.0 < t3 < 22.0)

# Good food, bad service — should still be somewhere reasonable
t4 = tip.evaluate([9.0, 1.0])
check("mixed inputs → reasonable output",
      0.0 < t4 < 30.0)

# ===================================================================
print("\n=== FuzzySystem — different defuzzification methods ===")

tip_b = pn.FuzzySystem(n_inputs=2, output_range=(0, 30),
                        resolution=200, defuzz_method="bisector")
tip_b.add_input_mf(0, "low", "triangular", 0, 0, 5, range=(0, 10))
tip_b.add_input_mf(0, "high", "triangular", 5, 10, 10)
tip_b.add_input_mf(1, "low", "triangular", 0, 0, 5, range=(0, 10))
tip_b.add_input_mf(1, "high", "triangular", 5, 10, 10)
tip_b.add_output_mf("small", "triangular", 0, 5, 15)
tip_b.add_output_mf("large", "triangular", 15, 25, 30)
tip_b.add_rule([(0, "low")], "small")
tip_b.add_rule([(0, "high"), (1, "high")], "large")

rb = tip_b.evaluate([8.0, 8.0])
check("bisector defuzz gives valid output", 0.0 < rb < 30.0)

tip_m = pn.FuzzySystem(n_inputs=2, output_range=(0, 30),
                        resolution=200, defuzz_method="mom")
tip_m.add_input_mf(0, "low", "triangular", 0, 0, 5, range=(0, 10))
tip_m.add_input_mf(0, "high", "triangular", 5, 10, 10)
tip_m.add_input_mf(1, "low", "triangular", 0, 0, 5, range=(0, 10))
tip_m.add_input_mf(1, "high", "triangular", 5, 10, 10)
tip_m.add_output_mf("small", "triangular", 0, 5, 15)
tip_m.add_output_mf("large", "triangular", 15, 25, 30)
tip_m.add_rule([(0, "low")], "small")
tip_m.add_rule([(0, "high"), (1, "high")], "large")

rm = tip_m.evaluate([8.0, 8.0])
check("MOM defuzz gives valid output", 0.0 < rm < 30.0)

# ===================================================================
print("\n=== FuzzySystem — trapezoidal and gaussian MFs ===")

fis2 = pn.FuzzySystem(n_inputs=1, output_range=(0, 100), resolution=300)
fis2.add_input_mf(0, "cold", "trapezoidal", 0, 0, 10, 20, range=(0, 50))
fis2.add_input_mf(0, "warm", "gaussian", 25, 5)
fis2.add_input_mf(0, "hot",  "trapezoidal", 30, 40, 50, 50)

fis2.add_output_mf("off",  "trapezoidal", 0, 0, 10, 30)
fis2.add_output_mf("half", "gaussian", 50, 10)
fis2.add_output_mf("full", "trapezoidal", 70, 90, 100, 100)

fis2.add_rule([(0, "cold")], "off")
fis2.add_rule([(0, "warm")], "half")
fis2.add_rule([(0, "hot")],  "full")

# Cold input → low output
r_cold = fis2.evaluate([5.0])
check("cold → output near 'off' range", r_cold < 40.0)

# Hot input → high output
r_hot = fis2.evaluate([45.0])
check("hot → output near 'full' range", r_hot > 60.0)

# Warm input → mid range
r_warm = fis2.evaluate([25.0])
check("warm → output in middle", 20.0 < r_warm < 80.0)

# ===================================================================
print("\n=== FuzzySystem — rule weights ===")

fw = pn.FuzzySystem(n_inputs=1, output_range=(0, 10), resolution=200)
fw.add_input_mf(0, "a", "triangular", 0, 5, 10, range=(0, 10))
fw.add_output_mf("lo", "triangular", 0, 2, 5)
fw.add_output_mf("hi", "triangular", 5, 8, 10)

# Low weight rule vs full weight rule
fw.add_rule([(0, "a")], "lo", weight=0.1)
fw.add_rule([(0, "a")], "hi", weight=1.0)

r_w = fw.evaluate([5.0])
check("high-weight rule dominates", r_w > 5.0)

# ===================================================================
print("\n=== FuzzySystem — parameter validation ===")

try:
    pn.FuzzySystem(n_inputs=0)
    check("n_inputs=0 raises ValueError", False)
except ValueError:
    check("n_inputs=0 raises ValueError", True)

try:
    pn.FuzzySystem(n_inputs=1, resolution=1)
    check("resolution=1 raises ValueError", False)
except ValueError:
    check("resolution=1 raises ValueError", True)

try:
    pn.FuzzySystem(n_inputs=1, defuzz_method="banana")
    check("bad defuzz_method raises ValueError", False)
except ValueError:
    check("bad defuzz_method raises ValueError", True)

# Wrong number of inputs to evaluate
try:
    fis.evaluate([1.0, 2.0, 3.0])
    check("wrong input count raises ValueError", False)
except ValueError:
    check("wrong input count raises ValueError", True)

# Unknown MF type
try:
    fis_bad = pn.FuzzySystem(n_inputs=1, output_range=(0, 1))
    fis_bad.add_input_mf(0, "x", "sigmoid", 0.5, range=(0, 1))
    check("unknown MF type raises ValueError", False)
except ValueError:
    check("unknown MF type raises ValueError", True)

# Unknown term name in rule
try:
    tip.add_rule([(0, "nonexistent")], "low")
    check("unknown term name raises KeyError", False)
except KeyError:
    check("unknown term name raises KeyError", True)

# Unknown output term in rule
try:
    tip.add_rule([(0, "poor")], "nonexistent")
    check("unknown output term raises KeyError", False)
except KeyError:
    check("unknown output term raises KeyError", True)

# ===================================================================
print("\n=== FuzzySystem — repr ===")

r = repr(tip)
check("repr contains 'FuzzySystem'", "FuzzySystem" in r)
check("repr shows n_inputs", "n_inputs=2" in r)
check("repr shows n_rules", "n_rules=5" in r)
check("repr shows defuzz method", "centroid" in r)

# ===================================================================
print(f"\n{'='*50}")
print(f"Fuzzy logic tests: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
