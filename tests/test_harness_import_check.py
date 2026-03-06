"""Tests for harness_import_check script.

Validates that the import direction checker runs and respects exit code convention:
- 0 = pass (no violations)
- 1 = violations found
- 2 = path error
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_harness_import_check_exit_code() -> None:
    """Run harness_import_check and assert exit 0 (no violations)."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "harness_import_check.py"
    assert script.exists(), f"Script not found: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert result.returncode == 0, (
        f"harness_import_check failed (exit {result.returncode}). "
        f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
    )
