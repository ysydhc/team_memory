# TeamMemory Constitution

## Core Principles

### I. Layered Architecture (分层架构)
Dependencies flow strictly downward: Server/Web → Services → Storage → Models.
Never import upward. Never import sideways across the same layer.
Each layer has a single responsibility and clear boundaries.

### II. Async First (异步优先)
All DB operations must use async/await.
All I/O-bound operations (HTTP, file, embedding) must be async.
Never block the event loop with synchronous calls.

### III. Type Safety (类型安全)
All function signatures must have type annotations.
External input must be validated via Pydantic models.
Internal data transfer uses typed dicts or Pydantic, not raw dicts.

### IV. Project Isolation (项目隔离)
Every experience/memory belongs to a project (project field).
All queries must filter by project to prevent cross-contamination.
Project name is inferred from .tm.toml > git repo > directory name.

### V. Bilingual (双语)
All comments, docstrings, and documentation in Chinese + English.
Commit messages in Chinese + English.
Error messages in Chinese + English.

### VI. Observability (可观测性)
Use project logger, never bare print().
Log key decisions and state changes at INFO level.
Log errors with full context at ERROR level.
Health checks must be lightweight (no model loading).

## Development Guidelines

### Testing
- Unit tests: pytest with @pytest.mark.unit (L1)
- Integration tests: pytest with @pytest.mark.integration (L2)
- Coverage targets: MCP ≥90%, Services ≥80%
- Run: `make test` (all), `make verify` (lint + test)
- New features must have tests before merge

### Code Style
- Ruff for linting (line-length=120)
- Run: `make lint` before commit
- No hardcoded secrets, no bare print()
- Import order: stdlib → third-party → local

### Documentation
- README.md: user-facing guide
- AGENTS.md: AI agent instructions
- docs/: design docs, ADRs, guides
- Update docs when changing architecture/APIs

### Git Workflow
- Feature branches from master
- Commit messages: type(scope): description
- Types: feat, fix, refactor, docs, test, chore
- `make verify` must pass before commit

## Non-Negotiables

1. NEVER hardcode API keys, passwords, or secrets in code
2. NEVER import upward in the layer hierarchy
3. NEVER skip project filtering in database queries
4. NEVER use bare print() — always use logger
5. NEVER commit code that breaks `make test`
6. NEVER merge without `make verify` passing
7. NEVER change model schema without Alembic migration
8. NEVER add MCP tools without updating README + tests
