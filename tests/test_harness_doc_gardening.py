"""Tests for harness_doc_gardening script.

Validates Golden Set: script output on fixtures matches expected.txt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_harness_doc_gardening_golden_set() -> None:
    """Run doc-gardening on fixtures and assert output matches expected.txt."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "harness_doc_gardening.py"
    fixture_dir = root / "tests" / "fixtures" / "doc-gardening"
    expected_file = fixture_dir / "expected.txt"

    assert script.exists(), f"Script not found: {script}"
    assert fixture_dir.exists(), f"Fixture dir not found: {fixture_dir}"
    assert expected_file.exists(), f"Expected file not found: {expected_file}"

    result = subprocess.run(
        [sys.executable, str(script), "--path", str(fixture_dir)],
        capture_output=True,
        text=True,
        cwd=str(root),
    )

    # Script should find violations (exit 1)
    assert result.returncode == 1, (
        f"Expected exit 1 (violations), got {result.returncode}. "
        f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
    )

    expected_lines = [
        line.strip()
        for line in expected_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    actual_lines = [
        line.strip()
        for line in result.stdout.strip().splitlines()
        if line.strip()
    ]

    assert sorted(actual_lines) == sorted(expected_lines), (
        f"Output mismatch.\n"
        f"Expected ({len(expected_lines)} lines):\n"
        + "\n".join(sorted(expected_lines))
        + f"\n\nActual ({len(actual_lines)} lines):\n"
        + "\n".join(sorted(actual_lines))
    )


def test_harness_doc_gardening_exit_code() -> None:
    """Run doc-gardening on project docs; exit 0 or 1 (no path error)."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "harness_doc_gardening.py"
    assert script.exists(), f"Script not found: {script}"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        capture_output=True,
        text=True,
        cwd=str(root),
    )

    # 0 = no issues, 1 = issues, 2 = path error
    assert result.returncode in (0, 1), (
        f"Unexpected exit code {result.returncode}. "
        f"stderr: {result.stderr!r}"
    )
