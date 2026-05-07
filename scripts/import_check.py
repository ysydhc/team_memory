#!/usr/bin/env python3
"""Check that all daemon modules can be imported without errors.

Catches missing imports, syntax errors, and broken references
that unit tests (with mocks) might miss.

Usage: python scripts/import_check.py
Exit 0 = all OK, 1 = failures found.
"""
import importlib
import sys

MODULES = [
    "daemon.app",
    "daemon.config",
    "daemon.pipeline",
    "daemon.draft_buffer",
    "daemon.draft_refiner",
    "daemon.refinement_worker",
    "daemon.response_buffer",
    "daemon.search_log_writer",
    "daemon.tm_sink",
    "daemon.tm_stats",
    "daemon.watcher",
    "daemon.convergence_detector",
    "daemon.faithfulness_judge",
    "daemon.faithfulness_batch",
    "daemon.entity_dedup",
    "daemon.reloader",
    "daemon.conversation_buffer",
    "daemon.personal_memory_extractor",
]

failed = []
for mod in MODULES:
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append(f"  {mod}: {e}")

if failed:
    print(f"Import check FAILED ({len(failed)} errors):")
    for f in failed:
        print(f)
    sys.exit(1)

# Also verify that required top-level imports exist in source code
# (catches accidentally deleted imports like "import asyncpg")
REQUIRED_SOURCE_IMPORTS = {
    "daemon.search_log_writer": ["import asyncpg"],
    "daemon.response_buffer": ["import asyncpg"],
    "daemon.faithfulness_batch": ["import asyncpg"],
}

for mod_name, required_lines in REQUIRED_SOURCE_IMPORTS.items():
    mod = importlib.import_module(mod_name)
    source_file = getattr(mod, "__file__", "")
    if not source_file:
        continue
    with open(source_file) as f:
        source = f.read()
    for line in required_lines:
        # Check for standalone import (not commented out)
        import re
        pattern = rf"^\s*{re.escape(line)}"
        if not re.search(pattern, source, re.MULTILINE):
            failed.append(f"  {mod_name}: missing '{line}' in source")

if failed:
    print(f"Import check FAILED ({len(failed)} errors):")
    for f in failed:
        print(f)
    sys.exit(1)
else:
    print(f"Import check OK ({len(MODULES)} modules, {len(REQUIRED_SOURCE_IMPORTS)} source checks)")
