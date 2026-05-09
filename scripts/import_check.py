#!/usr/bin/env python3
"""Check that all daemon modules can be imported without errors.

Catches missing imports, syntax errors, and broken references
that unit tests (with mocks) might miss.

Usage: PYTHONPATH=src python scripts/import_check.py
Exit 0 = all OK, 1 = failures found.
"""
import importlib
import sys

MODULES = [
    "team_memory.daemon.app",
    "team_memory.daemon.config",
    "team_memory.daemon.pipeline",
    "team_memory.daemon.draft_buffer",
    "team_memory.daemon.draft_refiner",
    "team_memory.daemon.refinement_worker",
    "team_memory.daemon.response_buffer",
    "team_memory.daemon.search_log_writer",
    "team_memory.daemon.tm_sink",
    "team_memory.daemon.tm_stats",
    "team_memory.daemon.watcher",
    "team_memory.daemon.convergence_detector",
    "team_memory.daemon.faithfulness_judge",
    "team_memory.daemon.faithfulness_batch",
    "team_memory.daemon.entity_dedup",
    "team_memory.daemon.reloader",
    "team_memory.daemon.conversation_buffer",
    "team_memory.daemon.personal_memory_extractor",
    "team_memory.daemon.intent_classifier",
    "team_memory.daemon.entity_backfill",
    "team_memory.daemon.embedding_backfill",
    "team_memory.daemon.search_eval",
    "team_memory.daemon.test_llm",
    "team_memory.daemon.solution_backfill",
    "team_memory.daemon.wiki_cli",
    "team_memory.daemon.wiki_compiler",
    "team_memory.daemon.llm_wiki",
    "team_memory.daemon.auto_dream",
    "team_memory.daemon.markdown_indexer",
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
    "team_memory.daemon.search_log_writer": ["import asyncpg"],
    "team_memory.daemon.response_buffer": ["import asyncpg"],
    "team_memory.daemon.faithfulness_batch": ["import asyncpg"],
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
