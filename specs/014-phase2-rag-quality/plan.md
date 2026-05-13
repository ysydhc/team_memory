# Implementation Plan: Phase 2 — RAG 质量提升

**Branch**: `014-phase2-rag-quality` | **Date**: 2026-05-13 | **Spec**: `specs/014-phase2-rag-quality/spec.md`
**Input**: Feature specification from `/specs/014-phase2-rag-quality/spec.md`

## Summary

Enhance the RAG pipeline quality and reliability by removing the legacy `[mem:]` evaluation marker, executing cross-encoder predictions asynchronously, securing FTS PostgreSQL queries against syntax errors, optimizing the reranking and adaptive filtering order, and implementing the missing `grouped` and `top_k_children` functionality.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: asyncio, asyncpg, SentenceTransformers (Cross-Encoder)
**Storage**: PostgreSQL (pgvector, FTS)
**Testing**: pytest
**Target Platform**: Linux/macOS daemon & web server
**Project Type**: team-memory backend service
**Performance Goals**: Improve concurrent RPS by unblocking the event loop during reranking
**Constraints**: Must maintain existing memory structure while augmenting the search API output format when grouping is requested.
**Scale/Scope**: Impacts the core search pipeline, evaluation service, and LLM hooks.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Layered Architecture**: PASS. The changes occur entirely within the `Services` layer (search pipeline, analysis) and `Hooks` layer.
- **Async First**: PASS. The core driver of US3 is to enforce "Async First" by offloading CPU-bound ML predictions to `asyncio.to_thread`.
- **Type Safety**: PASS. Will update `SearchRequest` and pipeline return types appropriately.
- **Observability**: PASS. Pipeline stages log their latency.

## Project Structure

### Documentation (this feature)

```text
specs/014-phase2-rag-quality/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── search-api-grouped.md
└── tasks.md
```

### Source Code (repository root)

```text
src/team_memory/
├── services/
│   ├── analysis/
│   │   └── evaluation.py       # Remove [mem:] markers
│   ├── search/
│   │   ├── search_orchestrator.py # Grouping logic, remove marker injection
│   │   └── search_pipeline.py     # Reranker async, adaptive filter order
│   └── memory/
│       └── memory_operations.py   # Remove [mem:] hint
├── storage/
│   └── query_builders.py       # TS query escape
├── reranker/
│   └── cross_encoder_provider.py # asyncio.to_thread
└── hooks/
    └── platforms/claude/       # Remove [mem:] context prefix

tests/
├── test_evaluation.py
├── test_search_pipeline.py
├── test_query_builders.py
└── test_search_orchestrator.py
```

**Structure Decision**: The changes modify existing files across `services/search`, `services/analysis`, `storage/`, `reranker/`, and `hooks/`. No new major modules are created.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

N/A
