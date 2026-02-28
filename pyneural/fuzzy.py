"""
pyneural.fuzzy — Fuzzy logic inference system.

Provides a Pythonic wrapper around the C fuzzy engine (fuzzy.c).
You define input/output variables with membership functions, write
IF-THEN rules, and then evaluate crisp inputs to get a defuzzified
crisp output.  Suitable for neuro-fuzzy hybrids, control systems,
and anywhere you need interpretable approximate reasoning.

Quick example — a tipping controller::

    import pyneural as pn

    # Two inputs (food quality, service quality), one output (tip %)
    fis = pn.FuzzySystem(n_inputs=2, output_range=(0, 30))

    # Define membership functions for input 0 (food)
    fis.add_input_mf(0, "poor",      "triangular", 0, 0, 5, range=(0, 10))
    fis.add_input_mf(0, "average",   "triangular", 0, 5, 10)
    fis.add_input_mf(0, "excellent", "triangular", 5, 10, 10)

    # Define membership functions for input 1 (service)
    fis.add_input_mf(1, "poor",      "triangular", 0, 0, 5, range=(0, 10))
    fis.add_input_mf(1, "average",   "triangular", 0, 5, 10)
    fis.add_input_mf(1, "excellent", "triangular", 5, 10, 10)

    # Output terms
    fis.add_output_mf("low",    "triangular", 0, 5, 10)
    fis.add_output_mf("medium", "triangular", 10, 15, 20)
    fis.add_output_mf("high",   "triangular", 20, 25, 30)

    # Rules
    fis.add_rule([(0, "poor"), (1, "poor")], "low")
    fis.add_rule([(0, "average"),], "medium")
    fis.add_rule([(0, "excellent"), (1, "excellent")], "high")

    tip = fis.evaluate([7.5, 8.0])
    print(f"Suggested tip: {tip:.1f}%")
"""

from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .core import _lib


# ── Standalone membership / operator helpers ─────────────────────────

def triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership: ramps a→b→c, peak at b."""
    return _lib.fuzzy_triangular(x, a, b, c)


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: ramp a→b, flat b→c, ramp c→d."""
    return _lib.fuzzy_trapezoidal(x, a, b, c, d)


def gaussian(x: float, mean: float, sigma: float) -> float:
    """Gaussian membership: bell curve centred at *mean*."""
    return _lib.fuzzy_gaussian(x, mean, sigma)


def fuzzy_and(a: float, b: float) -> float:
    """Fuzzy AND (min)."""
    return _lib.fuzzy_and(a, b)


def fuzzy_or(a: float, b: float) -> float:
    """Fuzzy OR (max)."""
    return _lib.fuzzy_or(a, b)


def fuzzy_not(a: float) -> float:
    """Fuzzy NOT (1 − a)."""
    return _lib.fuzzy_not(a)


def defuzz_centroid(
    values: Sequence[float], memberships: Sequence[float]
) -> float:
    """Centroid (centre-of-gravity) defuzzification."""
    n = len(values)
    v_arr = (ctypes.c_double * n)(*values)
    m_arr = (ctypes.c_double * n)(*memberships)
    return _lib.fuzzy_defuzz_centroid(v_arr, m_arr, n)


def defuzz_bisector(
    values: Sequence[float], memberships: Sequence[float]
) -> float:
    """Bisector defuzzification — splits the area in half."""
    n = len(values)
    v_arr = (ctypes.c_double * n)(*values)
    m_arr = (ctypes.c_double * n)(*memberships)
    return _lib.fuzzy_defuzz_bisector(v_arr, m_arr, n)


def defuzz_mom(
    values: Sequence[float], memberships: Sequence[float]
) -> float:
    """Mean-of-Maximum defuzzification."""
    n = len(values)
    v_arr = (ctypes.c_double * n)(*values)
    m_arr = (ctypes.c_double * n)(*memberships)
    return _lib.fuzzy_defuzz_mom(v_arr, m_arr, n)


# ── Membership function type codes ──────────────────────────────────

_MF_TYPES = {"triangular": 0, "trapezoidal": 1, "gaussian": 2}

# Defuzzification method codes
_DEFUZZ_METHODS = {"centroid": 0, "bisector": 1, "mom": 2}


# ── FuzzySystem — the main high-level class ─────────────────────────

class FuzzySystem:
    """
    Mamdani-style fuzzy inference system.

    Wraps the C engine for fast evaluation while providing a readable
    Python interface for defining variables, terms, and rules.

    Args:
        n_inputs:       Number of input variables.
        output_range:   (lo, hi) universe of discourse for the output.
        resolution:     Number of sample points for defuzzification.
                        More = more accurate but slower.  200 is usually
                        plenty; bump it up for very narrow MFs.
        defuzz_method:  One of 'centroid', 'bisector', 'mom'.
    """

    def __init__(
        self,
        n_inputs: int,
        output_range: Tuple[float, float] = (0.0, 1.0),
        resolution: int = 200,
        defuzz_method: str = "centroid",
    ):
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")
        if defuzz_method not in _DEFUZZ_METHODS:
            raise ValueError(
                f"defuzz_method must be one of {list(_DEFUZZ_METHODS)}, "
                f"got '{defuzz_method}'"
            )

        dm = _DEFUZZ_METHODS[defuzz_method]
        self._ptr = _lib.fuzzy_system_create(n_inputs, resolution, dm)
        if not self._ptr:
            raise RuntimeError("Failed to create FuzzySystem (C allocation failed)")

        self._n_inputs = n_inputs
        self._defuzz_method = defuzz_method

        # Set the output range immediately
        lo, hi = output_range
        rc = _lib.fuzzy_system_set_output_range(self._ptr, lo, hi)
        if rc != 0:
            raise ValueError(f"Invalid output range ({lo}, {hi})")

        # We keep human-readable names so rules can be written as strings
        # rather than raw integer indices.
        # _input_mf_names[var_idx] = {"name": term_index, ...}
        self._input_mf_names: List[Dict[str, int]] = [{} for _ in range(n_inputs)]
        self._output_mf_names: Dict[str, int] = {}

        # Track whether input ranges have been set
        self._input_ranges_set: List[bool] = [False] * n_inputs

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.fuzzy_system_free(self._ptr)
            self._ptr = None

    # ── Input variable configuration ────────────────────────────────

    def set_input_range(self, var_idx: int, lo: float, hi: float):
        """Set the universe of discourse for input variable *var_idx*."""
        rc = _lib.fuzzy_system_set_input_range(self._ptr, var_idx, lo, hi)
        if rc != 0:
            raise ValueError(
                f"set_input_range failed (var_idx={var_idx}, lo={lo}, hi={hi})"
            )
        self._input_ranges_set[var_idx] = True

    def add_input_mf(
        self,
        var_idx: int,
        name: str,
        mf_type: str,
        *params: float,
        range: Optional[Tuple[float, float]] = None,
    ) -> int:
        """
        Add a membership function to input variable *var_idx*.

        Args:
            var_idx:  Which input variable (0-indexed).
            name:     Human-readable name like "low", "medium", "high".
            mf_type:  One of 'triangular', 'trapezoidal', 'gaussian'.
            *params:  MF parameters — (a, b, c) for tri, (a, b, c, d)
                      for trap, (mean, sigma) for gaussian.
            range:    Optionally set the variable's universe in one shot.

        Returns:
            The integer term index (used internally by the C engine).
        """
        if mf_type not in _MF_TYPES:
            raise ValueError(f"Unknown mf_type '{mf_type}', must be one of {list(_MF_TYPES)}")

        if range is not None:
            self.set_input_range(var_idx, range[0], range[1])

        # Pad params to 4 values (the C API always takes 4 doubles)
        p = list(params) + [0.0] * (4 - len(params))

        mt = _MF_TYPES[mf_type]
        idx = _lib.fuzzy_system_add_input_mf(
            self._ptr, var_idx, mt, p[0], p[1], p[2], p[3]
        )
        if idx < 0:
            raise RuntimeError(
                f"Failed to add input MF '{name}' to var {var_idx}"
            )

        self._input_mf_names[var_idx][name] = idx
        return idx

    # ── Output variable configuration ───────────────────────────────

    def add_output_mf(
        self,
        name: str,
        mf_type: str,
        *params: float,
    ) -> int:
        """
        Add a membership function to the output variable.

        Args:
            name:     Human-readable name for this output term.
            mf_type:  One of 'triangular', 'trapezoidal', 'gaussian'.
            *params:  MF parameters (same conventions as input MFs).

        Returns:
            The integer term index.
        """
        if mf_type not in _MF_TYPES:
            raise ValueError(f"Unknown mf_type '{mf_type}'")

        p = list(params) + [0.0] * (4 - len(params))
        mt = _MF_TYPES[mf_type]
        idx = _lib.fuzzy_system_add_output_mf(
            self._ptr, mt, p[0], p[1], p[2], p[3]
        )
        if idx < 0:
            raise RuntimeError(f"Failed to add output MF '{name}'")

        self._output_mf_names[name] = idx
        return idx

    # ── Rules ───────────────────────────────────────────────────────

    def add_rule(
        self,
        antecedents: List[Tuple[int, str]],
        consequent: str,
        weight: float = 1.0,
    ):
        """
        Add a fuzzy IF-THEN rule.

        Args:
            antecedents: List of (var_idx, term_name) pairs.
                         e.g. [(0, "high"), (1, "low")]
            consequent:  The output term name that fires.
            weight:      Rule weight (default 1.0).

        Example::
            fis.add_rule([(0, "hot"), (1, "humid")], "high_fan_speed")
        """
        n = len(antecedents)
        if n < 1:
            raise ValueError("Rules must have at least one antecedent")

        # Resolve term names to integer indices
        vars_arr = (ctypes.c_int * n)()
        terms_arr = (ctypes.c_int * n)()
        for i, (vi, tname) in enumerate(antecedents):
            if vi < 0 or vi >= self._n_inputs:
                raise ValueError(f"var_idx {vi} out of range [0, {self._n_inputs})")
            if tname not in self._input_mf_names[vi]:
                avail = list(self._input_mf_names[vi].keys())
                raise KeyError(
                    f"Term '{tname}' not found for input var {vi}. "
                    f"Available: {avail}"
                )
            vars_arr[i] = vi
            terms_arr[i] = self._input_mf_names[vi][tname]

        if consequent not in self._output_mf_names:
            avail = list(self._output_mf_names.keys())
            raise KeyError(
                f"Output term '{consequent}' not found. Available: {avail}"
            )
        ct = self._output_mf_names[consequent]

        rc = _lib.fuzzy_system_add_rule(
            self._ptr, vars_arr, terms_arr, n, ct, weight
        )
        if rc != 0:
            raise RuntimeError("Failed to add rule (C engine returned error)")

    # ── Evaluation ──────────────────────────────────────────────────

    def evaluate(self, inputs: Sequence[float]) -> float:
        """
        Run the fuzzy inference engine on crisp inputs.

        Args:
            inputs: A sequence of float values, one per input variable.

        Returns:
            The defuzzified crisp output.

        Raises:
            ValueError: If the number of inputs doesn't match n_inputs.
            RuntimeError: If evaluation fails (e.g. no rules defined).
        """
        if len(inputs) != self._n_inputs:
            raise ValueError(
                f"Expected {self._n_inputs} inputs, got {len(inputs)}"
            )

        in_arr = (ctypes.c_double * self._n_inputs)(*inputs)
        out_val = ctypes.c_double(0.0)

        rc = _lib.fuzzy_system_evaluate(
            self._ptr, in_arr, ctypes.byref(out_val)
        )
        if rc != 0:
            raise RuntimeError("Fuzzy evaluation failed (check rules and MFs)")

        return out_val.value

    # ── Properties ──────────────────────────────────────────────────

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_rules(self) -> int:
        return _lib.fuzzy_system_n_rules(self._ptr)

    @property
    def defuzz_method(self) -> str:
        return self._defuzz_method

    @property
    def input_term_names(self) -> List[Dict[str, int]]:
        """Name→index mapping for each input variable's MFs."""
        return self._input_mf_names

    @property
    def output_term_names(self) -> Dict[str, int]:
        """Name→index mapping for the output variable's MFs."""
        return self._output_mf_names

    def __repr__(self) -> str:
        return (
            f"FuzzySystem(n_inputs={self._n_inputs}, "
            f"n_rules={self.n_rules}, "
            f"defuzz='{self._defuzz_method}')"
        )
