# Operations Runbook

## Rollback Procedure

### Application Rollback
1. Stop the current deployment
2. Revert to previous Docker image tag
3. Restart the application

### Database Rollback
**Important:** Migration 002 (mvp_cleanup) is one-way and cannot be downgraded.

To rollback to a specific migration:
```bash
# Check current revision
alembic current

# Downgrade one step
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>
```

**Revision chain:**
- 010_indexes_and_constraints (latest)
- 009_background_tasks
- 008_archive_raw_conversation
- 007_attachment_source_path
- 006_archive_knowledge_fields
- 005_archive_upload_failures
- 004_personal_memory_profile_kind
- 003_personal_memories_if_missing
- 002_mvp_cleanup <- **ONE-WAY: cannot downgrade past this point**
- 001_initial_mvp

### Emergency: Database Backup & Restore
```bash
# Backup
docker compose exec postgres pg_dump -U developer team_memory > backup.sql

# Restore
docker compose exec -T postgres psql -U developer team_memory < backup.sql
```

## Health Check Endpoints
- `/health` -- Full system health (DB + Ollama + Embedding + LLM + Cache)
- `/ready` -- Readiness probe (DB only)

## Common Issues
| Issue | Solution |
|-------|----------|
| Port 9111 occupied | `make release-9111` |
| Embedding fails | `ollama pull nomic-embed-text` |
| DB connection refused | `docker compose up -d postgres` |
| Cache degraded | Check Redis: `docker compose logs redis` |
