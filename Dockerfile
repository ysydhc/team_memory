# Multi-stage Dockerfile for team_memory
# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir hatchling

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY config.yaml ./

# Build the wheel
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy migrations, config, and static files
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY config.yaml ./
COPY src/team_memory/web/static/ ./src/team_memory/web/static/

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose port for web server
EXPOSE 9111

ENTRYPOINT ["docker-entrypoint.sh"]

# Default: run the web server
# Override with: docker run ... team-memory (for MCP server)
CMD ["team-memory-web"]
