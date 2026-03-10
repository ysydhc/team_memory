#!/usr/bin/env node
/**
 * GitNexus Bridge — HTTP API for TM architecture provider.
 *
 * Connects to GitNexus via CLI (npx gitnexus) and exposes:
 *   GET /context, /clusters, /cluster/:name, /impact, /graph, /search
 *
 * Port: 9321 (configurable via PORT env)
 * Repo root: GITNEXUS_REPO_PATH or process.cwd()
 */

import express from "express";
import { spawn } from "child_process";
import { readFile } from "fs/promises";
import { existsSync } from "fs";
import { join, resolve } from "path";
import { execSync } from "child_process";

const PORT = parseInt(process.env.PORT || "9321", 10);

/** Find repo root by walking up to find .gitnexus */
function findRepoRoot() {
  if (process.env.GITNEXUS_REPO_PATH) return process.env.GITNEXUS_REPO_PATH;
  let dir = process.cwd();
  for (let i = 0; i < 5; i++) {
    if (existsSync(join(dir, ".gitnexus", "meta.json"))) return dir;
    const parent = join(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

const REPO_ROOT = findRepoRoot();
const META_PATH = join(REPO_ROOT, ".gitnexus", "meta.json");

/** GitNexus meta-nodes — not source code, exclude from all node responses.
 *  - comm_<num>: Community (clustering)
 *  - proc_<num>_<name>: Process (execution flow)
 *  - Folder:<path>: Folder (directory structure)
 */
function isMetaNode(id) {
  const s = String(id || "");
  return /^comm_\d+$/.test(s) || /^proc_\d+_/.test(s) || /^Folder:/.test(s);
}

/** Process-only meta nodes — used for bridging (A->proc->B => A->B). */
function isProcessNode(id) {
  const s = String(id || "");
  return /^proc_\d+_/.test(s);
}

/** Bridge edges through Process nodes: when A->proc->B, add A->B so connectivity is preserved after filtering proc. */
function bridgeEdgesThroughProcess(edges) {
  const incoming = new Map(); // procId -> [source ids]
  const outgoing = new Map(); // procId -> [target ids]
  for (const e of edges) {
    if (isProcessNode(e.source) && !isMetaNode(e.target)) {
      if (!outgoing.has(e.source)) outgoing.set(e.source, []);
      outgoing.get(e.source).push(e.target);
    } else if (isProcessNode(e.target) && !isMetaNode(e.source)) {
      if (!incoming.has(e.target)) incoming.set(e.target, []);
      incoming.get(e.target).push(e.source);
    }
  }
  const bridged = [];
  for (const [procId, targets] of outgoing) {
    const sources = incoming.get(procId) || [];
    for (const a of sources) {
      for (const b of targets) {
        if (a !== b) bridged.push({ source: a, target: b, type: "STEP_IN_PROCESS" });
      }
    }
  }
  const directEdges = edges.filter((e) => !isMetaNode(e.source) && !isMetaNode(e.target));
  const directKeySet = new Set(directEdges.map((e) => `${e.source}\0${e.target}`));
  for (const e of bridged) {
    if (!directKeySet.has(`${e.source}\0${e.target}`)) {
      directEdges.push(e);
      directKeySet.add(`${e.source}\0${e.target}`);
    }
  }
  return directEdges;
}

const app = express();
app.use(express.json());

/** Run npx gitnexus <cmd> <args> and return parsed JSON. GitNexus CLI outputs JSON to stderr. */
function runGitnexus(cmd, args = [], opts = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn("npx", ["-y", "gitnexus", cmd, ...args], {
      cwd: REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      ...opts,
    });
    let out = "";
    let err = "";
    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.stderr.on("data", (d) => (err += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(err || out || `gitnexus ${cmd} exited ${code}`));
        return;
      }
      const src = err.trim() || out.trim();
      try {
        resolve(JSON.parse(src));
      } catch {
        reject(new Error(src || "Empty response from gitnexus"));
      }
    });
  });
}

/** Run git rev-parse HEAD. */
function getCurrentCommit() {
  try {
    return execSync("git rev-parse HEAD", { cwd: REPO_ROOT, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

/** Read meta.json if exists. */
async function readMeta() {
  if (!existsSync(META_PATH)) return null;
  const raw = await readFile(META_PATH, "utf8");
  return JSON.parse(raw);
}

/** Resolve repo name from meta or path. */
function getRepoName(meta) {
  if (meta?.repoPath) {
    const parts = meta.repoPath.replace(/\/$/, "").split("/");
    return parts[parts.length - 1] || "unknown";
  }
  return "team_doc";
}

// --- Routes ---

/** GET /context — codebase overview and staleness */
app.get("/context", async (req, res) => {
  const repo = req.query.repo;
  try {
    const meta = await readMeta();
    if (!meta) {
      return res.json({
        available: false,
        reason: "No GitNexus index (.gitnexus/meta.json not found)",
      });
    }

    const stats = meta.stats || {};
    const currentCommit = getCurrentCommit();
    const stale = !!(
      currentCommit &&
      meta.lastCommit &&
      currentCommit !== meta.lastCommit
    );

    res.json({
      available: true,
      repo_name: repo || getRepoName(meta),
      symbols: stats.nodes ?? stats.symbols ?? null,
      relationships: stats.edges ?? null,
      processes: stats.processes ?? null,
      stale,
      provider: "gitnexus",
    });
  } catch (e) {
    res.status(500).json({
      available: false,
      reason: String(e.message),
    });
  }
});

/** GET /repo-root — absolute path for open-in-editor (vscode://file/...) */
app.get("/repo-root", (_req, res) => {
  res.json({ repo_root: resolve(REPO_ROOT) });
});

/** GET /clusters — list clusters (communities) */
app.get("/clusters", async (req, res) => {
  const repo = req.query.repo;
  try {
    const result = await runGitnexus("cypher", [
      "MATCH (c:Community) RETURN c.heuristicLabel as name, c.cohesion as cohesion, c.symbolCount as member_count ORDER BY c.symbolCount DESC",
      "-r",
      repo || "team_doc",
    ]);
    if (result.error) throw new Error(result.error);
    if (!result.markdown) throw new Error("No markdown in cypher result");

    const rows = parseCypherMarkdown(result.markdown);
    // Aggregate by name (multiple Community nodes can share heuristicLabel)
    const byName = new Map();
    for (const r of rows) {
      const n = r.name || String(r.name);
      if (!byName.has(n)) {
        byName.set(n, { name: n, cohesion: r.cohesion, member_count: 0 });
      }
      const ent = byName.get(n);
      ent.member_count += r.member_count != null ? Number(r.member_count) : 0;
    }
    const clusters = Array.from(byName.values())
      .sort((a, b) => (b.member_count || 0) - (a.member_count || 0))
      .map((c) => ({
        name: c.name,
        cohesion: c.cohesion != null ? Math.round(Number(c.cohesion) * 100) / 100 : null,
        member_count: c.member_count || null,
      }));

    res.json(clusters);
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** GET /cluster/:name — cluster members */
app.get("/cluster/:name", async (req, res) => {
  const { name } = req.params;
  const repo = req.query.repo;
  const safeName = name.replace(/"/g, '\\"');
  try {
    const result = await runGitnexus("cypher", [
      `MATCH (f)-[]->(c:Community) WHERE c.heuristicLabel = "${safeName}" RETURN f.id as id, f.filePath as path LIMIT 500`,
      "-r",
      repo || "team_doc",
    ]);
    if (result.error) throw new Error(result.error);

    const rows = parseCypherMarkdown(result.markdown);
    const members = rows
      .filter((r) => r.id && !isMetaNode(r.id))
      .map((r) => {
        const id = r.id || "";
        const kind = id.includes(":") ? id.split(":")[0] : null;
        return { id, path: r.path ?? null, kind };
      });

    res.json({ name, members });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** GET /node-clusters — clusters that contain the given node (by node id) */
app.get("/node-clusters", async (req, res) => {
  const node = req.query.node;
  const repo = req.query.repo || "team_doc";
  if (!node || typeof node !== "string") {
    return res.status(400).json({ error: "node is required" });
  }
  if (!/^[a-zA-Z0-9_.\/\-: ]+$/.test(node)) {
    return res.status(400).json({ error: "node contains invalid characters" });
  }
  try {
    const safeId = String(node).replace(/\\/g, "\\\\").replace(/'/g, "''");
    let result = await runGitnexus("cypher", [
      `MATCH (n)-[]->(c:Community) WHERE n.id = '${safeId}' RETURN c.heuristicLabel as name`,
      "-r",
      repo,
    ]);
    let rows = result.markdown ? parseCypherMarkdown(result.markdown) : [];
    if (rows.length === 0 && node.includes(":")) {
      const path = node.split(":").slice(1, -1).join(":");
      if (path) {
        const safePath = String(path).replace(/\\/g, "\\\\").replace(/'/g, "''");
        result = await runGitnexus("cypher", [
          `MATCH (n)-[]->(c:Community) WHERE n.filePath = '${safePath}' RETURN DISTINCT c.heuristicLabel as name LIMIT 20`,
          "-r",
          repo,
        ]);
        rows = result.markdown ? parseCypherMarkdown(result.markdown) : [];
      }
    }
    const names = [...new Set((rows || []).filter((r) => r.name).map((r) => r.name))];
    res.json({ clusters: names });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** Extract symbol name from full node id (Kind:path:name -> name). GitNexus impact expects symbol name. */
function impactTargetFromPath(path) {
  const s = String(path || "").trim();
  if (!s) return s;
  const parts = s.split(":");
  if (parts.length >= 3) return parts[parts.length - 1];
  return s;
}

/** GET /impact — upstream/downstream impact */
app.get("/impact", async (req, res) => {
  const path = req.query.path;
  const depth = parseInt(req.query.depth || "2", 10);
  const repo = req.query.repo || "team_doc";

  if (!path) {
    return res.status(400).json({ error: "path is required" });
  }

  const target = impactTargetFromPath(path);

  try {
    const [upRes, downRes] = await Promise.all([
      runGitnexus("impact", [target, "-d", "upstream", "--depth", String(depth), "-r", repo]),
      runGitnexus("impact", [target, "-d", "downstream", "--depth", String(depth), "-r", repo]),
    ]);

    let upstream = flattenByDepth(upRes.byDepth);
    let downstream = flattenByDepth(downRes.byDepth);
    upstream = upstream.filter((i) => !isMetaNode(i.id));
    downstream = downstream.filter((i) => !isMetaNode(i.id));

    res.json({ upstream, downstream });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** Normalize cluster param to array (Express gives string or string[]). */
function normalizeClusters(cluster) {
  if (cluster == null) return [];
  if (Array.isArray(cluster)) return cluster.filter(Boolean);
  return [cluster];
}

const MAX_CLUSTERS = 30;

/** Normalize ensure_nodes param to array of non-empty strings. */
function normalizeEnsureNodes(param) {
  if (param == null) return [];
  const arr = Array.isArray(param) ? param : [param];
  return arr.filter((x) => x && String(x).trim());
}

/** GET /graph — nodes and edges. cluster(s): filter by cluster(s). Accepts cluster or clusters param. ensure_nodes: ids to include when using clusters. */
app.get("/graph", async (req, res) => {
  const clusterParam = req.query.cluster ?? req.query.clusters;
  let clusters = normalizeClusters(clusterParam);
  if (clusters.length > MAX_CLUSTERS) clusters = clusters.slice(0, MAX_CLUSTERS);
  const ensureNodes = normalizeEnsureNodes(req.query.ensure_nodes);
  const file_path = req.query.file_path;
  const repo = req.query.repo || "team_doc";
  const max_depth = Math.min(Math.max(parseInt(req.query.max_depth || "1", 10) || 1, 1), 4);
  const max_nodes = 600;

  try {
    let nodes = [];
    let edges = [];
    let focus_node_id = null;

    if (file_path && clusters.length === 0) {
      // Load nodes from specific file + 1-hop neighbors (callers/callees from other files)
      const safePath = String(file_path).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
      const fileRes = await runGitnexus("cypher", [
        `MATCH (n) WHERE n.filePath = "${safePath}" RETURN n.id as id, n.name as label, n.filePath as path LIMIT 200`,
        "-r",
        repo,
      ]);
      if (!fileRes.error && fileRes.markdown) {
        const rows = parseCypherMarkdown(fileRes.markdown);
        const nodeMap = new Map();
        for (const r of rows || []) {
          if (!r.id || isMetaNode(r.id)) continue;
          nodeMap.set(r.id, {
            id: r.id,
            label: r.label ?? null,
            kind: r.id?.includes(":") ? r.id.split(":")[0] : null,
            path: r.path ?? null,
          });
        }
        const fileNodeIds = new Set(nodeMap.keys());
        if (fileNodeIds.size > 0) {
          const inList = [...fileNodeIds].slice(0, 300).map((id) => `'${String(id).replace(/'/g, "''")}'`).join(",");
          // Edges from file nodes to any target (including external); edges from any source to file nodes
          const [outRes, inRes] = await Promise.all([
            runGitnexus("cypher", [
              `MATCH (a)-[r:CodeRelation]->(b) WHERE a.id IN [${inList}] RETURN a.id as source, b.id as target, r.type as type LIMIT 400`,
              "-r",
              repo,
            ]),
            runGitnexus("cypher", [
              `MATCH (a)<-[r:CodeRelation]-(b) WHERE a.id IN [${inList}] RETURN b.id as source, a.id as target, r.type as type LIMIT 400`,
              "-r",
              repo,
            ]),
          ]);
          const edgeRows = [
            ...(outRes.markdown ? parseCypherMarkdown(outRes.markdown) : []),
            ...(inRes.markdown ? parseCypherMarkdown(inRes.markdown) : []),
          ];
          for (const r of edgeRows || []) {
            if (!r.source || !r.target) continue;
            if (!isMetaNode(r.source) && !isMetaNode(r.target)) {
              edges.push({ source: r.source, target: r.target, type: r.type || "" });
              for (const id of [r.source, r.target]) {
                if (!nodeMap.has(id) && nodeMap.size < max_nodes) {
                  const path = id.includes(":") ? id.split(":")[1] : id;
                  const label = id.includes(":") ? id.split(":").pop() : id.split("/").pop();
                  nodeMap.set(id, {
                    id,
                    label: label || id,
                    kind: id?.includes(":") ? id.split(":")[0] : null,
                    path: path || null,
                  });
                }
              }
            }
          }
          nodes = [...nodeMap.values()];
        }
      }
    } else if (clusters.length > 0) {
      const PRIORITY_KINDS = ["File:", "Class:", "Interface:"];
      const priorityCond = PRIORITY_KINDS.map((k) => `f.id STARTS WITH '${k}'`).join(" OR ");
      const othersLimitPerCluster = Math.max(20, Math.floor(max_nodes / clusters.length));
      const priorityPromises = clusters.map((c) => {
        const safe = String(c).replace(/"/g, '\\"');
        return runGitnexus("cypher", [
          `MATCH (f)-[]->(c:Community) WHERE c.heuristicLabel = "${safe}" AND (${priorityCond}) RETURN f.id as id, f.name as label, f.filePath as path`,
          "-r",
          repo,
        ]);
      });
      const othersPromises = clusters.map((c) => {
        const safe = String(c).replace(/"/g, '\\"');
        return runGitnexus("cypher", [
          `MATCH (f)-[]->(c:Community) WHERE c.heuristicLabel = "${safe}" AND NOT (${priorityCond}) RETURN f.id as id, f.name as label, f.filePath as path LIMIT ${othersLimitPerCluster}`,
          "-r",
          repo,
        ]);
      });
      const [priorityResults, othersResults] = await Promise.all([
        Promise.all(priorityPromises),
        Promise.all(othersPromises),
      ]);
      const nodeMap = new Map();
      for (const membersRes of priorityResults) {
        if (membersRes.error) continue;
        const rows = parseCypherMarkdown(membersRes.markdown);
        for (const r of rows || []) {
          if (!r.id || isMetaNode(r.id)) continue;
          nodeMap.set(r.id, {
            id: r.id,
            label: r.label ?? null,
            kind: r.id?.includes(":") ? r.id.split(":")[0] : null,
            path: r.path ?? null,
          });
        }
      }
      const priorityCount = nodeMap.size;
      const othersBudget = Math.max(0, max_nodes - priorityCount);
      const othersLimitTotal = Math.min(othersBudget, othersLimitPerCluster * clusters.length);
      let othersAdded = 0;
      for (const membersRes of othersResults) {
        if (membersRes.error || othersAdded >= othersLimitTotal) continue;
        const rows = parseCypherMarkdown(membersRes.markdown);
        for (const r of rows || []) {
          if (!r.id || isMetaNode(r.id) || nodeMap.has(r.id)) continue;
          if (othersAdded >= othersLimitTotal) break;
          nodeMap.set(r.id, {
            id: r.id,
            label: r.label ?? null,
            kind: r.id?.includes(":") ? r.id.split(":")[0] : null,
            path: r.path ?? null,
          });
          othersAdded++;
        }
      }
      for (const nodeId of ensureNodes) {
        if (isMetaNode(nodeId) || nodeMap.has(nodeId)) continue;
        const safeId = String(nodeId).replace(/'/g, "''");
        const ensureRes = await runGitnexus("cypher", [
          `MATCH (n) WHERE n.id = "${safeId}" RETURN n.id as id, n.name as label, n.filePath as path LIMIT 1`,
          "-r",
          repo,
        ]);
        if (!ensureRes.error && ensureRes.markdown) {
          const rows = parseCypherMarkdown(ensureRes.markdown);
          const r = rows?.[0];
          if (r?.id) {
            nodeMap.set(r.id, {
              id: r.id,
              label: r.label ?? null,
              kind: r.id?.includes(":") ? r.id.split(":")[0] : null,
              path: r.path ?? null,
            });
          }
        }
      }
      nodes = [...nodeMap.values()];
      if (nodes.length > 0) {
        const nodeIds = new Set(nodes.map((n) => n.id));
        const ensureSet = new Set(ensureNodes);
        const orderedIds = [
          ...ensureNodes.filter((id) => nodeIds.has(id)),
          ...[...nodeIds].filter((id) => !ensureSet.has(id)),
        ];
        const inList = orderedIds.slice(0, 300).map((id) => `'${String(id).replace(/'/g, "''")}'`).join(",");
        const edgesRes = await runGitnexus("cypher", [
          `MATCH (a)-[r:CodeRelation]->(b) WHERE a.id IN [${inList}] AND b.id IN [${inList}] RETURN a.id as source, b.id as target, r.type as type LIMIT 500`,
          "-r",
          repo,
        ]);
        if (!edgesRes.error && edgesRes.markdown) {
          const edgeRows = parseCypherMarkdown(edgesRes.markdown);
          edges = edgeRows
            .filter((r) => r.source && r.target && nodeIds.has(r.source) && nodeIds.has(r.target))
            .map((r) => ({ source: r.source, target: r.target, type: r.type || "" }));
        }
      }
    } else {
      const clustersRes = await runGitnexus("cypher", [
        "MATCH (c:Community) RETURN c.heuristicLabel as name, c.symbolCount as cnt ORDER BY c.symbolCount DESC LIMIT 5",
        "-r",
        repo,
      ]);
      if (clustersRes.error) throw new Error(clustersRes.error);
      const clusterRows = parseCypherMarkdown(clustersRes.markdown);
      const clusterNames = (clusterRows || []).filter((r) => r.name).map((r) => r.name);
      const nodeMap = new Map();
      const edgeSet = new Set();
      let frontier = [];
      const membersPromises = clusterNames.map((cname) => {
        const safe = cname.replace(/"/g, '\\"');
        return runGitnexus("cypher", [
          `MATCH (f)-[]->(c:Community) WHERE c.heuristicLabel = "${safe}" RETURN f.id as id, f.name as label, f.filePath as path LIMIT 15`,
          "-r",
          repo,
        ]);
      });
      const membersResults = await Promise.all(membersPromises);
      for (const membersRes of membersResults) {
        if (membersRes.error) continue;
        const rows = parseCypherMarkdown(membersRes.markdown);
        for (const r of rows || []) {
          if (!r.id || nodeMap.has(r.id)) continue;
          nodeMap.set(r.id, {
            id: r.id,
            label: r.label ?? null,
            kind: r.id?.includes(":") ? r.id.split(":")[0] : null,
            path: r.path ?? null,
          });
          frontier.push(r.id);
        }
      }
      for (let d = 0; d < max_depth && frontier.length > 0 && nodeMap.size < max_nodes; d++) {
        const batch = frontier.slice(0, 100);
        frontier = [];
        const inList = batch.map((id) => `'${String(id).replace(/'/g, "''")}'`).join(",");
        const [outRes, inRes] = await Promise.all([
          runGitnexus("cypher", [
            `MATCH (a)-[r:CodeRelation]->(b) WHERE a.id IN [${inList}] RETURN a.id as source, b.id as target, r.type as type LIMIT 200`,
            "-r",
            repo,
          ]),
          runGitnexus("cypher", [
            `MATCH (a)<-[r:CodeRelation]-(b) WHERE a.id IN [${inList}] RETURN b.id as source, a.id as target, r.type as type LIMIT 200`,
            "-r",
            repo,
          ]),
        ]);
        const edgeRows = [
          ...(outRes.markdown ? parseCypherMarkdown(outRes.markdown) : []),
          ...(inRes.markdown ? parseCypherMarkdown(inRes.markdown) : []),
        ];
        if (edgeRows.length === 0) break;
        for (const r of edgeRows || []) {
          if (!r.source || !r.target) continue;
          edgeSet.add(`${r.source}\0${r.target}\0${r.type || ""}`);
          for (const id of [r.source, r.target]) {
            if (!nodeMap.has(id) && nodeMap.size < max_nodes) {
              frontier.push(id);
              const label = id.includes(":") ? id.split(":").pop() : id.split("/").pop();
              nodeMap.set(id, {
                id,
                label: label || id,
                kind: id?.includes(":") ? id.split(":")[0] : null,
                path: id?.includes(":") ? id.split(":")[1] : id,
              });
            }
          }
        }
      }
      nodes = [...nodeMap.values()];
      edges = [...edgeSet].map((k) => {
        const [source, target, type] = k.split("\0");
        return { source, target, type: type || "" };
      });
    }

    // Bridge through Process: A->proc->B => add A->B so connectivity is preserved
    edges = bridgeEdgesThroughProcess(edges);

    // Filter meta-nodes: comm_ (no bridge needed), proc_ (bridged above), Folder (structural only)
    nodes = nodes.filter((n) => !isMetaNode(n.id));
    let nodeIdSet = new Set(nodes.map((n) => n.id));
    edges = edges.filter((e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target));

    // Filter out orphan nodes (degree 0) — they appear as isolated "single nodes" with no edges,
    // often caused by meta-node filtering (e.g. edges via Process/Community were removed)
    const connectedIds = new Set();
    for (const e of edges) {
      connectedIds.add(e.source);
      connectedIds.add(e.target);
    }
    nodes = nodes.filter((n) => connectedIds.has(n.id));
    nodeIdSet = new Set(nodes.map((n) => n.id));
    edges = edges.filter((e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target));

    if (file_path && nodes.length > 0) {
      const match = nodes.find((n) => n.path === file_path);
      if (match) focus_node_id = match.id;
    }

    res.json({ nodes, edges, focus_node_id });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** Whitelist for search params: only [a-zA-Z0-9_./\-: ] allowed. Returns false if invalid. */
const SEARCH_WHITELIST = /^[a-zA-Z0-9_.\/\-: ]+$/;

function isSearchParamValid(val) {
  if (val == null || typeof val !== "string") return false;
  return SEARCH_WHITELIST.test(val);
}

/** GET /search — search nodes by name or filePath (substring match). clusters: single or array. */
app.get("/search", async (req, res) => {
  const q = req.query.q;
  const scope = req.query.scope || "global";
  const clusterParam = req.query.cluster ?? req.query.clusters;
  const clusters = Array.isArray(clusterParam) ? clusterParam : (clusterParam ? [clusterParam] : []);
  const repo = req.query.repo || "team_doc";

  if (!q || typeof q !== "string") {
    return res.status(400).json({ error: "q is required" });
  }
  if (!isSearchParamValid(q)) {
    return res.status(400).json({ error: "q contains invalid characters (allowed: a-zA-Z0-9_./\\-: )" });
  }
  if (scope !== "global" && scope !== "cluster") {
    return res.status(400).json({ error: "scope must be global or cluster" });
  }
  if (scope === "cluster") {
    if (clusters.length === 0) {
      return res.status(400).json({ error: "cluster or clusters is required when scope=cluster" });
    }
    for (const c of clusters) {
      if (!isSearchParamValid(c)) {
        return res.status(400).json({ error: "cluster contains invalid characters (allowed: a-zA-Z0-9_./\\-: )" });
      }
    }
  }

  const safeQ = String(q).replace(/\\/g, "\\\\").replace(/'/g, "''");
  const cond = `(n.filePath CONTAINS '${safeQ}' OR n.name CONTAINS '${safeQ}' OR n.id CONTAINS '${safeQ}')`;
  try {
    let query;
    if (scope === "global") {
      query = `MATCH (n) WHERE ${cond} RETURN n.id as id, n.name as name, n.filePath as path LIMIT 50`;
    } else {
      const clusterList = clusters.slice(0, 20).map((c) => `'${String(c).replace(/'/g, "''")}'`).join(",");
      query = `MATCH (n)-[]->(c:Community) WHERE c.heuristicLabel IN [${clusterList}] AND ${cond} RETURN n.id as id, n.name as name, n.filePath as path LIMIT 50`;
    }

    const result = await runGitnexus("cypher", [query, "-r", repo]);
    if (result.error) throw new Error(result.error);

    const rows = parseCypherMarkdown(result.markdown);
    const nodes = rows
      .filter((r) => r.id != null && !isMetaNode(r.id))
      .map((r) => ({
        id: r.id,
        label: r.name ?? null,
        path: r.path ?? null,
        kind: r.id && String(r.id).includes(":") ? String(r.id).split(":")[0] : null,
      }));

    res.json({ nodes });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

/** Parse markdown table from cypher result into array of objects */
function parseCypherMarkdown(md) {
  if (!md || typeof md !== "string") return [];
  const lines = md.trim().split("\n").filter((l) => l.trim());
  if (lines.length < 2) return [];
  const headerLine = lines[0];
  const header = headerLine.replace(/^\|?\s*/, "").replace(/\s*\|?$/, "").split("|").map((h) => h.trim()).filter(Boolean);
  const rows = [];
  for (let i = 2; i < lines.length; i++) {
    const cellLine = lines[i];
    const cells = cellLine.replace(/^\|?\s*/, "").replace(/\s*\|?$/, "").split("|").map((c) => c.trim());
    const obj = {};
    header.forEach((h, j) => {
      let v = cells[j];
      if (v === "null" || v === undefined || v === "") v = null;
      else if (typeof v === "string" && v.startsWith("{") && v.endsWith("}")) {
        try {
          const parsed = JSON.parse(v);
          if (parsed && typeof parsed === "object" && "id" in parsed) v = parsed.id ?? parsed.name ?? parsed.filePath ?? v;
          else v = parsed;
        } catch {
          // keep v as-is
        }
      }
      obj[h] = v;
    });
    rows.push(obj);
  }
  return rows;
}

/** Flatten byDepth {"1": [...], "2": [...]} into [{id, path, depth}, ...] */
function flattenByDepth(byDepth) {
  if (!byDepth || typeof byDepth !== "object") return [];
  const out = [];
  for (const [d, arr] of Object.entries(byDepth)) {
    const depth = parseInt(d, 10);
    if (!Array.isArray(arr)) continue;
    for (const item of arr) {
      out.push({
        id: item.id ?? item.name ?? "",
        path: item.filePath ?? item.path ?? null,
        depth,
      });
    }
  }
  return out;
}

// --- Start ---

app.listen(PORT, "127.0.0.1", () => {
  console.log(`GitNexus Bridge listening on http://127.0.0.1:${PORT}`);
  console.log(`Repo root: ${REPO_ROOT}`);
});
