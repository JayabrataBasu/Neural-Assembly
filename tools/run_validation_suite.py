#!/usr/bin/env python3
"""Unified validation suite for Neural-Assembly.

Phases:
1) Build + dataset prep
2) Smoke tests
3) Regression tests (Python + training)
4) Compatibility checks (header/API symbol surface)

Exit code is non-zero when mandatory checks fail.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class CheckResult:
    name: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    critical: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_cmd(name: str, command: str, critical: bool = True) -> CheckResult:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return CheckResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        critical=critical,
    )


def _venv_python() -> str:
    venv_python = os.path.join(PROJECT_ROOT, ".neuasm", "bin", "python")
    if os.path.exists(venv_python):
        return shlex.quote(venv_python)
    return shlex.quote(sys.executable)


def run_compatibility_checks(results: List[CheckResult]) -> None:
    # 1) Ensure required core symbols exist in shared lib.
    required_symbols = {
        "neural_init",
        "neural_shutdown",
        "neural_version",
        "neural_get_simd_level",
        "neural_get_simd_name",
        "neural_tensor_create",
        "neural_tensor_free",
        "neural_tensor_data",
        "neural_linear_create",
        "neural_linear_forward",
        "neural_relu",
        "neural_sigmoid",
        "neural_softmax",
        "neural_optimizer_free",
        "neural_sgd_create",
        "neural_adam_create",
        "neural_adamw_create",
        "neural_dataset_load_csv",
        "neural_dataset_get_batch",
    }

    nm = run_cmd(
        "compat:nm_symbols",
        "nm -D --defined-only libneural.so",
        critical=True,
    )
    results.append(nm)

    if nm.ok:
        exported = set()
        for line in nm.stdout.splitlines():
            parts = line.split()
            if parts:
                exported.add(parts[-1])

        missing = sorted(required_symbols - exported)
        if missing:
            results.append(
                CheckResult(
                    name="compat:required_symbols",
                    command="required_symbols in libneural.so",
                    returncode=1,
                    stdout="",
                    stderr="Missing symbols: " + ", ".join(missing),
                    critical=True,
                )
            )
        else:
            results.append(
                CheckResult(
                    name="compat:required_symbols",
                    command="required_symbols in libneural.so",
                    returncode=0,
                    stdout="All required symbols present",
                    stderr="",
                    critical=True,
                )
            )

    # 2) Compare neural_api.h declarations to exported symbols.
    header_path = os.path.join(PROJECT_ROOT, "neural_api.h")
    if not os.path.exists(header_path):
        results.append(
            CheckResult(
                name="compat:header_exists",
                command="neural_api.h present",
                returncode=1,
                stdout="",
                stderr="neural_api.h missing",
                critical=True,
            )
        )
        return

    with open(header_path, "r", encoding="utf-8") as f:
        header_text = f.read()

    declared = set(re.findall(r"\b(neural_[A-Za-z0-9_]+)\s*\(", header_text))

    if nm.ok:
        exported = set(line.split()[-1] for line in nm.stdout.splitlines() if line.split())
        missing_declared = sorted(x for x in declared if x not in exported)
        if missing_declared:
            # Warning-level check (non-critical) because header may include planned APIs.
            results.append(
                CheckResult(
                    name="compat:header_vs_exports",
                    command="declared symbols exported",
                    returncode=1,
                    stdout="",
                    stderr=(
                        "Declared in header but not exported (first 20): "
                        + ", ".join(missing_declared[:20])
                    ),
                    critical=False,
                )
            )
        else:
            results.append(
                CheckResult(
                    name="compat:header_vs_exports",
                    command="declared symbols exported",
                    returncode=0,
                    stdout="Header/API symbol surface consistent",
                    stderr="",
                    critical=False,
                )
            )


def build_test_plan(tier: str) -> List[Tuple[str, str, bool]]:
    py = _venv_python()
    adamw_simple_retry = f"for i in 1 2 3; do {py} test_adamw_simple.py && exit 0; done; exit 1"
    adamw_minimal_retry = f"for i in 1 2 3; do {py} test_adamw_minimal.py && exit 0; done; exit 1"

    plan: List[Tuple[str, str, bool]] = [
        ("build:exe", "make -j1", True),
        ("build:shared", "make lib -j1", True),
        ("data:generate", f"{py} tools/gen_datasets.py", True),
        ("smoke:framework_test", "./neural_framework test", True),
        ("smoke:verify_simple", f"{py} tools/verify_simple.py", False),
        ("smoke:verify_full", f"{py} tools/verify_correctness.py", False),
    ]

    if tier in {"regression", "full"}:
        plan.extend(
            [
                ("py:autograd", f"{py} tools/test_autograd.py", True),
                ("py:dataloader", f"{py} tools/test_dataloader.py", True),
                ("py:grad_clip", f"{py} tools/test_gradient_clipping.py", False),
                (
                    "py:grad_clip_comprehensive",
                    f"{py} tools/test_gradient_clipping_comprehensive.py",
                    False,
                ),
                (
                    "py:grad_clip_integration",
                    f"{py} tools/test_gradient_clipping_integration.py",
                    False,
                ),
                ("py:adamw_simple", adamw_simple_retry, False),
                ("py:adamw_minimal", adamw_minimal_retry, True),
                ("py:convergence", f"{py} test_convergence.py", True),
                (
                    "train:xor",
                    "./neural_framework train configs/xor_config.ini /tmp/na_xor_model.bin",
                    True,
                ),
                (
                    "train:sine",
                    "./neural_framework train configs/sine_config.ini /tmp/na_sine_model.bin",
                    True,
                ),
                (
                    "infer:xor",
                    "./neural_framework infer configs/xor_config.ini /tmp/na_xor_model.bin",
                    True,
                ),
                (
                    "train:mnist",
                    "./neural_framework train configs/mnist_config.ini /tmp/na_mnist_model.bin",
                    False,
                ),
                (
                    "infer:mnist",
                    "./neural_framework infer configs/mnist_config.ini /tmp/na_mnist_model.bin",
                    False,
                ),
                (
                    "train:wine",
                    "./neural_framework train configs/wine_quality_config.ini /tmp/na_wine_model.bin",
                    False,
                ),
            ]
        )

    return plan


def print_result(res: CheckResult) -> None:
    status = "PASS" if res.ok else "FAIL"
    crit = "critical" if res.critical else "non-critical"
    print(f"[{status}] {res.name} ({crit})")
    if not res.ok:
        if res.stderr.strip():
            print("  stderr:")
            for line in res.stderr.strip().splitlines()[:8]:
                print(f"    {line}")
        elif res.stdout.strip():
            print("  output:")
            for line in res.stdout.strip().splitlines()[:8]:
                print(f"    {line}")


def summarize(results: List[CheckResult]) -> int:
    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    critical_failed = [r for r in results if (not r.ok and r.critical)]

    print("\n" + "=" * 72)
    print("Validation Summary")
    print("=" * 72)
    print(f"Total checks: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Critical failures: {len(critical_failed)}")

    if critical_failed:
        print("\nCritical failures:")
        for r in critical_failed:
            print(f"- {r.name}")

    return 1 if critical_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Neural-Assembly validation suite")
    parser.add_argument(
        "--tier",
        choices=["smoke", "regression", "full"],
        default="regression",
        help="Validation breadth",
    )
    args = parser.parse_args()

    results: List[CheckResult] = []
    plan = build_test_plan(args.tier)

    print(f"Running validation tier: {args.tier}")
    print(f"Project root: {PROJECT_ROOT}\n")

    for name, cmd, critical in plan:
        res = run_cmd(name, cmd, critical=critical)
        results.append(res)
        print_result(res)

    run_compatibility_checks(results)

    # Print compatibility results at the end as well.
    for res in results:
        if res.name.startswith("compat:"):
            print_result(res)

    return summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
