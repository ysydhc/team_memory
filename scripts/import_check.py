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
    # runtime
    "team_memory.daemon.runtime.app",
    "team_memory.daemon.runtime.config",
    "team_memory.daemon.runtime.reloader",
    "team_memory.daemon.runtime.health_check",
    "team_memory.daemon.runtime.structured_logger",
    # buffers
    "team_memory.daemon.buffers.conversation_buffer",
    "team_memory.daemon.buffers.draft_buffer",
    "team_memory.daemon.buffers.response_buffer",
    "team_memory.daemon.buffers.pending_queue",
    "team_memory.daemon.buffers.payload_capture",
    # refinement
    "team_memory.daemon.refinement.pipeline",
    "team_memory.daemon.refinement.draft_refiner",
    "team_memory.daemon.refinement.refinement_worker",
    # backfill
    "team_memory.daemon.backfill.embedding_backfill",
    "team_memory.daemon.backfill.entity_backfill",
    "team_memory.daemon.backfill.solution_backfill",
    "team_memory.daemon.backfill.entity_dedup",
    # quality
    "team_memory.daemon.quality.faithfulness_batch",
    "team_memory.daemon.quality.faithfulness_judge",
    "team_memory.daemon.quality.search_eval",
    "team_memory.daemon.quality.convergence_detector",
    # wiki
    "team_memory.daemon.wiki.llm_wiki",
    "team_memory.daemon.wiki.wiki_cli",
    "team_memory.daemon.wiki.wiki_compiler",
    "team_memory.daemon.wiki.markdown_indexer",
    # observe
    "team_memory.daemon.observe.metrics_collector",
    "team_memory.daemon.observe.search_log_writer",
    "team_memory.daemon.observe.tm_stats",
    "team_memory.daemon.observe.watcher",
    # integration
    "team_memory.daemon.integration.tm_sink",
    "team_memory.daemon.integration.auto_dream",
    "team_memory.daemon.integration.personal_memory_extractor",
    "team_memory.daemon.integration.intent_classifier",
    "team_memory.daemon.integration.test_llm",
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
    "team_memory.daemon.observe.search_log_writer": ["import asyncpg"],
    "team_memory.daemon.buffers.response_buffer": ["import asyncpg"],
    "team_memory.daemon.quality.faithfulness_batch": ["import asyncpg"],
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
