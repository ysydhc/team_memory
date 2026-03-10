# GitNexus Bridge

HTTP bridge for GitNexus — exposes context, clusters, impact, and graph APIs for the TeamMemory architecture provider.

## Overview

The Bridge connects to GitNexus via CLI (`npx gitnexus`) and exposes a simple HTTP API. TM backend configures `architecture.gitnexus.bridge_url` and calls these endpoints.

## Endpoints

| Method | Path | Query | Response |
|--------|------|-------|----------|
| GET | `/context` | `repo` (optional) | `{ available, repo_name?, symbols?, relationships?, processes?, stale?, reason? }` |
| GET | `/clusters` | `repo` (optional) | `[{ name, cohesion?, member_count? }]` |
| GET | `/cluster/:name` | `repo` (optional) | `{ name, members: [{ id, path?, kind? }] }` |
| GET | `/node-clusters` | `node` (required), `repo` | `{ clusters: [...] }` — clusters containing the node |
| GET | `/impact` | `path` (required), `depth`, `repo` | `{ upstream: [...], downstream: [...] }` |
| GET | `/graph` | `repo`, `cluster`, `file_path` (optional) | `{ nodes, edges, focus_node_id? }` |
| GET | `/repo-root` | — | `{ repo_root }` — absolute path for open-in-editor |

## Prerequisites

- Node.js 18+
- GitNexus index: run `npx gitnexus analyze` in the repo root before using the bridge

## Installation

```bash
cd tools/gitnexus-bridge
npm install
```

## Start

From the **repo root** (or any directory with `.gitnexus/meta.json` above it):

```bash
# From team_doc root
node tools/gitnexus-bridge/server.js

# Or from bridge directory (auto-detects repo root)
cd tools/gitnexus-bridge && npm start
```

**Environment variables:**

- `PORT` — HTTP port (default: 9321)
- `GITNEXUS_REPO_PATH` — Repo root path (default: auto-detect by walking up to find `.gitnexus`)

## Verification

```bash
# Context (overview)
curl -s http://127.0.0.1:9321/context

# Clusters
curl -s "http://127.0.0.1:9321/clusters"

# Cluster members
curl -s "http://127.0.0.1:9321/cluster/Tests"

# Impact
curl -s "http://127.0.0.1:9321/impact?path=get_current_user&depth=2"

# Graph (with cluster filter)
curl -s "http://127.0.0.1:9321/graph?cluster=Services"
```

## TM Integration

1. Set `architecture.gitnexus.bridge_url: "http://127.0.0.1:9321"` in `config.yaml`
2. Start the bridge before TM Web
3. `GET /api/v1/architecture/context` will return real data when bridge is running
