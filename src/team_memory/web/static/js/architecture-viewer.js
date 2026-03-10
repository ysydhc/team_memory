/**
 * Architecture viewer: graph (Cytoscape), clusters, node sidebar (impact + experiences).
 * T7 of code-arch-viz-gitnexus plan.
 */

import { state } from './store.js';

function api(...args) {
    return window.__api(...args);
}

let _archCy = null;
let _archGraphLoaded = false;
let _archClustersLoaded = false;
let _archSidebarJustOpened = false;

/** Setup click-outside to close panels. Call after architecture graph DOM is ready. */
export function setupArchitectureClickOutside() {
    const wrapper = document.querySelector('.architecture-graph-wrapper');
    if (!wrapper) return;
    const handler = (e) => {
        if (_archSidebarJustOpened) return;
        const searchPanel = document.getElementById('architecture-search-panel');
        const searchBtn = document.getElementById('architecture-search-float-btn');
        const sidebar = document.getElementById('architecture-node-sidebar');
        const sidebarBtn = document.getElementById('architecture-sidebar-float-btn');
        if (!searchPanel) return;
        if (!searchPanel.classList.contains('collapsed') && !searchPanel.contains(e.target) && !(searchBtn && searchBtn.contains(e.target))) {
            searchPanel.classList.add('collapsed');
            if (searchBtn) searchBtn.classList.remove('hidden');
        }
        const pinnedPanels = document.getElementById('architecture-pinned-panels');
        const clickOnPinned = pinnedPanels && pinnedPanels.contains(e.target);
        if (sidebar && !state.architectureSidebarPinned && !sidebar.classList.contains('hidden') && !sidebar.classList.contains('collapsed') && !sidebar.contains(e.target) && !(sidebarBtn && sidebarBtn.contains(e.target)) && !clickOnPinned) {
            if (_archCy) _archCy.elements().unselect();
            sidebar.classList.add('hidden');
            if (sidebarBtn) sidebarBtn.classList.add('hidden');
        }
    };
    wrapper.addEventListener('click', handler);
}

/** GitNexus-inspired node colors by kind (Function, Class, Method, File, etc.) */
const KIND_COLORS = {
    Function: { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd' },
    Class: { bg: '#2e1f4e', border: '#8b5cf6', text: '#c4b5fd' },
    Method: { bg: '#134e4a', border: '#14b8a6', text: '#5eead4' },
    Interface: { bg: '#422006', border: '#f59e0b', text: '#fcd34d' },
    File: { bg: '#1e293b', border: '#64748b', text: '#94a3b8' },
    Community: { bg: '#064e3b', border: '#10b981', text: '#6ee7b7' },
    Process: { bg: '#431407', border: '#f97316', text: '#fdba74' },
    default: { bg: '#1e293b', border: '#475569', text: '#cbd5e1' },
};

function kindStyle(k) {
    const c = KIND_COLORS[k] || KIND_COLORS.default;
    return { 'background-color': c.bg, 'border-color': c.border, color: c.text };
}

/** Short label for node: prefer name, add class/file context when name is generic (e.g. rank, format). */
function nodeLabel(n) {
    const id = n.id || '';
    const raw = n.label || n.path || id || '';
    const kind = n.kind || '';
    let name = raw.includes('/') ? raw.split('/').pop() : raw;
    const path = n.path || (id.includes(':') ? id.split(':').slice(1, -1).join(':') : null);
    const fileBase = path ? path.split('/').pop() : null;
    const lastPart = id.includes(':') ? id.split(':').pop() : '';
    if ((kind === 'Method' || kind === 'Function') && name && !name.includes('.') && !name.includes('(')) {
        if (lastPart.includes('.')) {
            name = lastPart;
        } else if (fileBase) {
            name = `${name} (${fileBase})`;
        }
    } else if (kind === 'Class' && lastPart && lastPart !== name) {
        name = lastPart;
    } else if (fileBase && name.length < 12 && !name.includes('.') && !name.includes('(')) {
        name = `${name} (${fileBase})`;
    }
    return name.length > 80 ? name.slice(0, 77) + '…' : name;
}

const _archLog = (msg, data) => {
    if (typeof console !== 'undefined' && console.log) {
        console.log('[Arch]', msg, data !== undefined ? data : '');
    }
};

const NODE_MAX_WIDTH = 240;
const NODE_TEXT_MAX_WIDTH = 212;
const NODE_MIN_WIDTH = 80;
const NODE_MIN_HEIGHT = 32;
const NODE_MAX_HEIGHT = 100;

/** Estimate node dimensions from label for adaptive sizing. */
function nodeDimensionsFromLabel(label) {
    const len = (label || '').length;
    const pxPerChar = 6.5;
    const charsPerLine = 32;
    const lineHeight = 20;
    const paddingX = 10;
    const paddingY = 6;
    const w = Math.min(Math.max(len * pxPerChar + paddingX, NODE_MIN_WIDTH), NODE_MAX_WIDTH);
    const lines = Math.max(1, Math.ceil(len / charsPerLine));
    const h = Math.min(Math.max(lines * lineHeight + paddingY, NODE_MIN_HEIGHT), NODE_MAX_HEIGHT);
    return { w, h };
}

/** Render architecture graph with Cytoscape (GitNexus-inspired styling). */
function renderArchitectureGraph(containerId, graph) {
    _archLog('renderArchitectureGraph start', { containerId, hasGraph: !!graph });
    if (typeof window.cytoscape === 'undefined') {
        _archLog('ERROR: Cytoscape 未加载');
        throw new Error('Cytoscape 未加载，请刷新页面后重试');
    }
    const container = document.getElementById(containerId);
    if (!container) {
        _archLog('ERROR: container not found', containerId);
        return null;
    }
    if (_archCy) {
        _archCy.destroy();
        _archCy = null;
        _archLog('destroyed previous cy');
    }
    const nodes = graph.nodes || [];
    const nodeIds = new Set(nodes.map((n) => n.id));
    const rawEdges = graph.edges || [];
    const edges = rawEdges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));
    if (edges.length < rawEdges.length) {
        _archLog('filtered invalid edges', { removed: rawEdges.length - edges.length });
    }
    _archLog('graph data', { nodesCount: nodes.length, edgesCount: edges.length });
    if (nodes.length === 0 && edges.length === 0) {
        _archLog('empty graph, showing placeholder');
        container.innerHTML = '<div style="padding:48px;text-align:center;color:var(--text-muted)">暂无节点数据</div>';
        return null;
    }
    container.innerHTML = '';
    const elements = [
        ...nodes.map((n) => ({
            data: {
                id: n.id,
                label: nodeLabel(n),
                fullLabel: n.label || n.path || n.id,
                path: n.path || n.id,
                kind: n.kind || '',
            },
        })),
        ...edges.map((e, i) => ({
            data: {
                id: 'e' + i,
                source: e.source,
                target: e.target,
                type: (e.type || '').toUpperCase(),
            },
        })),
    ];
    const kindSelectors = ['Function', 'Class', 'Method', 'Interface', 'File', 'Community', 'Process']
        .map((k) => ({ selector: `node[kind="${k}"]`, style: kindStyle(k) }));
    const cy = window.cytoscape({
        container,
        elements,
        style: [
            {
                selector: 'node',
                style: {
                    shape: 'round-rectangle',
                    width: (ele) => nodeDimensionsFromLabel(ele.data('label')).w,
                    height: (ele) => nodeDimensionsFromLabel(ele.data('label')).h,
                    padding: '2px 4px',
                    'background-color': KIND_COLORS.default.bg,
                    'border-width': 1.5,
                    'border-color': KIND_COLORS.default.border,
                    label: 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    color: KIND_COLORS.default.text,
                    'font-size': '13px',
                    'font-weight': '500',
                    'text-wrap': 'wrap',
                    'text-max-width': `${NODE_TEXT_MAX_WIDTH}px`,
                },
            },
            ...kindSelectors,
            {
                selector: 'node:selected',
                style: { 'border-color': '#3b82f6', 'border-width': 2, 'background-opacity': 1 },
            },
            {
                selector: 'node.hover',
                style: { 'border-width': 2, 'border-color': '#94a3b8' },
            },
            {
                selector: 'node.highlight-pulse',
                style: { 'border-width': 4, 'border-color': '#3b82f6', 'background-opacity': 1 },
            },
            {
                selector: 'node.arch-filtered, edge.arch-filtered',
                style: { display: 'none' },
            },
            {
                selector: 'edge',
                style: {
                    'line-color': '#475569',
                    'target-arrow-color': '#64748b',
                    'curve-style': 'unbundled-bezier',
                    'control-point-distance': '40px',
                    'control-point-weight': '0.5',
                    'target-arrow-shape': 'triangle',
                    width: 1.2,
                },
            },
            {
                selector: 'edge[type="CALLS"]',
                style: { 'line-color': '#3b82f6', 'target-arrow-color': '#60a5fa', width: 1.5 },
            },
            {
                selector: 'edge[type="IMPORTS"]',
                style: { 'line-color': '#64748b', 'target-arrow-color': '#94a3b8', width: 1 },
            },
        ],
        minZoom: 0.2,
        maxZoom: 4,
    });
    cy.on('tap', 'node', (ev) => {
        const node = ev.target;
        const path = node.data('path') || node.id();
        showArchitectureNodeSidebar(node.id(), path);
    });
    cy.on('mouseover', 'node', (e) => {
        e.target.addClass('hover');
        e.target.data('fullLabel') && (e.target.cy().container().title = e.target.data('fullLabel'));
    });
    cy.on('mouseout', 'node', (e) => {
        e.target.removeClass('hover');
        e.target.cy().container().title = '';
    });
    _archLog('cytoscape created, running layout', { elements: cy.elements().length });
    runArchitectureLayout(cy);
    _archCy = cy;
    container._cy = cy;
    return cy;
}

function runArchitectureLayout(cy) {
    _archLog('runArchitectureLayout start', { hasCy: !!cy, elementsCount: cy?.elements().length ?? 0 });
    if (!cy || cy.elements().length === 0) {
        _archLog('skip layout: no cy or empty');
        return;
    }
    if (typeof cy.layout !== 'function') {
        _archLog('ERROR: cy.layout is not a function');
        return;
    }
    const tryLayout = (name, opts) => {
        try {
            _archLog('layout try', name);
            const layout = cy.layout({ name, ...opts });
            layout.on('layoutstop', () => _archLog('layoutstop', name));
            layout.on('layouterror', (e) => _archLog('layouterror', { name, error: e.error?.message || e }));
            layout.run();
            _archLog('layout.run() called', name);
            return true;
        } catch (e) {
            _archLog('layout catch', { name, error: String(e?.message || e) });
            return false;
        }
    };
    if (tryLayout('fcose', { animate: 'end', animationDuration: 500, randomize: true, quality: 'default', nodeSeparation: 50, nodeRepulsion: 8000, idealEdgeLength: 60, gravity: 0.45, gravityRange: 2, padding: 25 })) return;
    if (tryLayout('concentric', { animate: 'end', animationDuration: 500, concentric: (n) => n.degree(), levelWidth: () => 1, minNodeSpacing: 40, padding: 25 })) return;
    if (tryLayout('cola', { avoidOverlap: true, handleDisconnected: true, nodeSpacing: 8, animate: 'end', animationDuration: 500, padding: 25 })) return;
    if (tryLayout('dagre', { rankDir: 'TB', nodeSep: 35, rankSep: 40, acyclicer: 'greedy', nodeDimensionsIncludeLabels: true, animate: 'end', animationDuration: 500, padding: 25 })) return;
    if (tryLayout('cose', { animate: 'end', animationDuration: 500, nodeRepulsion: 12000, idealEdgeLength: 80, nodeOverlap: 20, padding: 25 })) return;
    _archLog('all layouts failed, using cy.fit');
    cy.fit(undefined, 30);
}

const ARCH_FILTER_KEY = 'architecture_filter_pattern';

/** Apply node filter: hide nodes matching pattern (prefix match on label/id). */
export function applyArchitectureFilter() {
    const input = document.getElementById('architecture-filter-input');
    const pattern = (input ? input.value.trim() : '') || '';
    state.architectureFilterPattern = pattern;
    try {
        if (pattern) localStorage.setItem(ARCH_FILTER_KEY, pattern);
        else localStorage.removeItem(ARCH_FILTER_KEY);
    } catch (_) {}
    if (!_archCy) return;
    const lower = pattern.toLowerCase();
    _archCy.batch(() => {
        _archCy.nodes().removeClass('arch-filtered');
        _archCy.edges().removeClass('arch-filtered');
        if (lower) {
            const toHide = _archCy.nodes().filter((node) => {
                const label = (node.data('label') || '').toLowerCase();
                const full = (node.data('fullLabel') || '').toLowerCase();
                const idParts = (node.id() || '').split(':');
                const namePart = idParts.length ? idParts[idParts.length - 1].toLowerCase() : '';
                return label.startsWith(lower) || full.startsWith(lower) || namePart.startsWith(lower);
            });
            toHide.addClass('arch-filtered');
            toHide.connectedEdges().addClass('arch-filtered');
        }
    });
    const count = _archCy.nodes('.arch-filtered').length;
    if (typeof window.__toast === 'function' && pattern) {
        window.__toast(count ? `已隐藏 ${count} 个节点` : '无匹配节点', 'info');
    }
}

/** Setup filter input: restore state, bind apply on Enter. */
export function setupArchitectureFilter() {
    const input = document.getElementById('architecture-filter-input');
    if (!input) return;
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem(ARCH_FILTER_KEY) : null;
    const pattern = saved || state.architectureFilterPattern || '';
    input.value = pattern;
    state.architectureFilterPattern = pattern;
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            applyArchitectureFilter();
        }
    });
}

/** Push current view state to back stack (before switching main node). nodeId/path = node to center when returning. */
function pushArchitectureBackState(nodeId, path) {
    const clusters = state.architectureCurrentClusters;
    const filePath = state.architectureCurrentFilePath;
    state.architectureBackStack.push({ clusters: clusters ? [...clusters] : null, filePath, nodeId: nodeId || null, path: path || null });
}

/** Pop from back stack and restore that view. Centers on the stored node. */
export async function architectureBack() {
    const stack = state.architectureBackStack;
    if (!stack.length) return;
    const prev = stack.pop();
    updateArchitectureBackButtonVisibility();
    const { clusters, filePath, nodeId, path } = prev;
    await loadArchitectureGraph(clusters, filePath, null);
    if (nodeId) {
        await highlightArchitectureNode(nodeId, path, true, 'auto');
    }
}

/** Show/hide back button based on stack. */
export function updateArchitectureBackButtonVisibility() {
    const btn = document.getElementById('architecture-back-float-btn');
    if (!btn) return;
    btn.classList.toggle('hidden', !state.architectureBackStack.length);
}

/** Clear search results and return to initial state. */
export function clearArchitectureSearch() {
    const input = document.getElementById('architecture-search-input');
    if (input) input.value = '';
    const tab = document.getElementById('architecture-tab-graph');
    if (tab) {
        const body = tab.querySelector('.architecture-search-results-body');
        if (body) body.innerHTML = '';
        if (_archCy) _archCy.elements().unselect();
    }
    state.architectureCurrentClusters = null;
    updateClusterLabel();
}

/** Update cluster label display (全部 = global). Truncate when many to avoid overflow. */
function updateClusterLabel() {
    const label = document.getElementById('architecture-cluster-label');
    if (!label) return;
    const c = state.architectureCurrentClusters;
    if (!c || c.length === 0) {
        label.textContent = '全部';
        label.title = '';
        return;
    }
    const maxShow = 3;
    const text = c.length > maxShow
        ? `当前: 已选 ${c.length} 个集群 (${c.slice(0, maxShow).join(', ')}…)`
        : `当前: ${c.join(', ')}`;
    label.textContent = text;
    label.title = c.join('\n');
}

/** Run architecture node search (click-triggered). Scope from UI selector (全局/当前集群). */
export async function runArchitectureSearch() {
    const input = document.getElementById('architecture-search-input');
    const btn = document.getElementById('architecture-search-btn');
    const tab = document.getElementById('architecture-tab-graph');
    if (!input || !btn || !tab) return;
    const q = (input.value || '').trim();
    if (!q) {
        if (typeof window.__toast === 'function') window.__toast('请输入搜索关键词', 'error');
        return;
    }
    const clusters = state.architectureCurrentClusters;
    const scope = getArchitectureSearchScope();
    if (scope === 'cluster' && (!clusters || clusters.length === 0)) {
        if (typeof window.__toast === 'function') window.__toast('当前集群模式需先选择集群', 'error');
        return;
    }
    const panel = document.getElementById('architecture-search-panel');
    const floatBtn = document.getElementById('architecture-search-float-btn');
    const body = tab.querySelector('.architecture-search-results-body');
    if (!body) return;
    btn.disabled = true;
    body.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    if (panel) {
        panel.classList.remove('collapsed');
        if (floatBtn) floatBtn.classList.add('hidden');
    }
    try {
        const params = new URLSearchParams({ q, scope });
        if (scope === 'cluster' && clusters?.length) {
            clusters.forEach((c) => params.append('clusters', c));
            params.set('cluster', clusters[0]);
        }
        const data = await api('GET', `/api/v1/architecture/search?${params.toString()}`);
        const nodes = data.nodes || [];
        if (nodes.length === 0) {
            body.innerHTML = '<div style="padding:24px;color:var(--text-muted);font-size:13px">未找到匹配节点</div>';
        } else {
            body.innerHTML = nodes
                .map(
                    (n) =>
                        `<div class="architecture-search-result-item" data-node-id="${esc(n.id)}" data-node-path="${esc(n.path || n.id)}" style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;margin-bottom:6px;border:1px solid var(--border);border-radius:var(--radius);font-size:12px;word-break:break-all;background:var(--bg-tertiary)">
                            <span class="architecture-search-result-label" style="flex:1;cursor:pointer;min-width:0" role="button" tabindex="0">${esc(n.label || n.path || n.id)}</span>
                            <span class="architecture-search-result-actions" style="flex-shrink:0;display:flex;gap:4px">
                                <button type="button" class="btn btn-secondary btn-sm architecture-search-open-main" style="padding:2px 6px;font-size:11px">主节点</button>
                                <button type="button" class="btn btn-secondary btn-sm architecture-search-open-current" style="padding:2px 6px;font-size:11px">当前</button>
                            </span>
                        </div>`
                )
                .join('');
            body.querySelectorAll('.architecture-search-result-item').forEach((el) => {
                const nodeId = el.dataset.nodeId;
                const nodePath = el.dataset.nodePath || null;
                const label = el.querySelector('.architecture-search-result-label');
                const mainBtn = el.querySelector('.architecture-search-open-main');
                const currentBtn = el.querySelector('.architecture-search-open-current');
                const go = (mode) => (e) => {
                    e.stopPropagation();
                    highlightArchitectureNode(nodeId, nodePath, false, mode);
                };
                if (label) {
                    label.addEventListener('click', () => highlightArchitectureNode(nodeId, nodePath));
                    label.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            highlightArchitectureNode(nodeId, nodePath);
                        }
                    });
                }
                if (mainBtn) mainBtn.addEventListener('click', go('main'));
                if (currentBtn) currentBtn.addEventListener('click', go('current'));
            });
        }
    } catch (e) {
        const msg = String(e.message || e);
        const isProviderUnavailable = msg.includes('503') || msg.includes('Architecture provider not configured') || msg.includes('not configured');
        const displayMsg = isProviderUnavailable ? '架构服务未配置或 Bridge 未启动' : msg;
        body.innerHTML = `<div style="padding:24px;color:var(--text-muted);font-size:13px">${esc(displayMsg)}</div>`;
        if (typeof window.__toast === 'function') {
            if (isProviderUnavailable && !state.architectureProviderUnavailableShown) {
                state.architectureProviderUnavailableShown = true;
                window.__toast(displayMsg, 'info');
            } else if (!isProviderUnavailable) {
                window.__toast('搜索失败: ' + displayMsg, 'error');
            }
        }
    } finally {
        btn.disabled = false;
    }
}

/** Extract file path from node id (e.g. Class:src/foo.py:Bar -> src/foo.py). */
function pathFromNodeId(nodeId) {
    if (!nodeId || typeof nodeId !== 'string') return null;
    const parts = nodeId.split(':');
    if (parts.length >= 3) return parts.slice(1, -1).join(':');
    if (parts.length === 2 && parts[0] === 'File') return parts[1];
    return null;
}

/** Extract node name from nodeId (e.g. Function:path:name -> name). */
function nodeNameFromNodeId(nodeId) {
    if (!nodeId || typeof nodeId !== 'string') return null;
    const parts = nodeId.split(':');
    if (parts.length >= 3) return parts[parts.length - 1];
    if (parts.length === 2 && parts[0] !== 'File') return parts[1];
    return null;
}

/** Get node kind from graph or nodeId (e.g. Class:src/foo.py:Bar -> Class). */
function kindFromNode(nodeId) {
    if (_archCy) {
        const node = _archCy.getElementById(nodeId);
        if (node.length) {
            const k = node.data('kind');
            if (k) return k;
        }
    }
    if (nodeId && String(nodeId).includes(':')) return String(nodeId).split(':')[0];
    return null;
}

/** Highlight and center a node in the graph; clear previous selection.
 * @param nodeId - node id
 * @param nodePath - file path for reload
 * @param skipReload - when true, do not attempt reload (used after a reload to avoid infinite retry)
 * @param openMode - 'main' | 'current' | 'auto': main=reload by file; current=only if in graph; auto=smart default */
async function highlightArchitectureNode(nodeId, nodePath, skipReload, openMode = 'auto') {
    const path = skipReload ? null : (nodePath || pathFromNodeId(nodeId));
    if (!_archCy) {
        if (path && openMode !== 'current') {
            if (typeof window.__toast === 'function') window.__toast('正在加载节点所在文件…', 'info');
            await loadArchitectureGraph(null, path);
            highlightArchitectureNode(nodeId, null, true, 'auto');
            return;
        }
        if (typeof window.__toast === 'function') window.__toast('请先选择集群或等待图加载完成', 'error');
        return;
    }
    _archCy.elements().unselect();
    const node = _archCy.getElementById(nodeId);
    if (node.length) {
        if (openMode === 'current' || openMode === 'auto') {
            node.select();
            const toFit = node.closedNeighborhood();
            _archCy.animate({ fit: { eles: toFit, padding: 140 }, duration: 400 });
            node.addClass('highlight-pulse');
            let pulseCount = 0;
            const pulseInterval = setInterval(() => {
                pulseCount++;
                if (pulseCount >= 6) {
                    clearInterval(pulseInterval);
                    node.removeClass('highlight-pulse');
                    return;
                }
                node.toggleClass('highlight-pulse');
            }, 200);
            showArchitectureNodeSidebar(nodeId, node.data('path') || nodeId);
            return;
        }
        if (openMode === 'main' && path) {
            if (typeof window.__toast === 'function') window.__toast('正在加载节点所在文件…', 'info');
            await loadArchitectureGraph(null, path);
            highlightArchitectureNode(nodeId, null, true, 'auto');
            return;
        }
    }
    if (openMode === 'current') {
        const clusters = state.architectureCurrentClusters;
        if (clusters && clusters.length > 0) {
            if (typeof window.__toast === 'function') window.__toast('正在加载节点到当前图…', 'info');
            await loadArchitectureGraph(clusters, null, [nodeId]);
            highlightArchitectureNode(nodeId, null, true, 'auto');
            return;
        }
        if (typeof window.__toast === 'function') window.__toast('该节点不在当前视图中', 'error');
        return;
    }
    if (path && (openMode === 'main' || openMode === 'auto')) {
        if (typeof window.__toast === 'function') window.__toast('正在加载节点所在文件…', 'info');
        await loadArchitectureGraph(null, path);
        highlightArchitectureNode(nodeId, null, true, 'auto');
    } else {
        if (typeof window.__toast === 'function') window.__toast('该节点不在当前视图中，可尝试选择对应集群后搜索', 'error');
    }
}

/** Load graph from API and render. clusters (array) or cluster (string) or null; file_path optional; ensureNodes: ids to include when using clusters. */
export async function loadArchitectureGraph(clustersOrCluster, filePath, ensureNodes) {
    const clusters = Array.isArray(clustersOrCluster)
        ? (clustersOrCluster.length ? clustersOrCluster : null)
        : (clustersOrCluster ? [clustersOrCluster] : null);
    _archLog('loadArchitectureGraph start', { clusters, filePath, ensureNodes });
    state.architectureCurrentClusters = clusters;
    state.architectureCurrentFilePath = filePath || null;
    updateClusterLabel();
    const tab = document.getElementById('architecture-tab-graph');
    if (!tab) {
        _archLog('ERROR: architecture-tab-graph not found');
        return;
    }
    const container = document.getElementById('architecture-graph-container');
    if (!container) {
        _archLog('ERROR: architecture-graph-container not found');
        return;
    }
    const loading = tab.querySelector('.architecture-graph-loading');
    const hasClusters = clusters && clusters.length > 0;
    if (loading) {
        loading.classList.remove('hidden');
        if (hasClusters) loading.innerHTML = '<div class="spinner"></div>';
        // else: keep initial content (matches default "加载全项目图约 5 秒") to avoid flash
    }
    try {
        let url = '/api/v1/architecture/graph?';
        const params = new URLSearchParams();
        if (hasClusters) clusters.forEach((c) => params.append('clusters', c));
        else params.set('max_depth', '1');
        if (filePath) params.set('file_path', filePath);
        if (hasClusters && ensureNodes?.length) ensureNodes.forEach((id) => params.append('ensure_nodes', id));
        url += params.toString();
        _archLog('fetching graph', url);
        const graph = await api('GET', url);
        _archLog('graph API response', { nodesCount: graph?.nodes?.length ?? 0, edgesCount: graph?.edges?.length ?? 0 });
        renderArchitectureGraph('architecture-graph-container', graph);
        if (loading) loading.classList.add('hidden');
        _archGraphLoaded = true;
        state.architectureProviderUnavailableShown = false;
        applyArchitectureFilter();
        _archLog('loadArchitectureGraph done');
    } catch (e) {
        const msg = String(e?.message || e);
        const isProviderUnavailable = msg.includes('503') || msg.includes('Architecture provider not configured') || msg.includes('not configured');
        _archLog(isProviderUnavailable ? 'loadArchitectureGraph: provider unavailable' : 'loadArchitectureGraph ERROR', { message: msg });
        if (loading) loading.classList.add('hidden');
        if (isProviderUnavailable) {
            container.innerHTML = `
                <div style="padding:48px 24px;text-align:center;color:var(--text-muted)">
                    <div style="font-size:32px;margin-bottom:12px">🏗️</div>
                    <p style="font-size:14px;margin-bottom:8px">架构服务暂不可用</p>
                    <p style="font-size:12px;max-width:360px;margin:0 auto">请检查 <code>config.yaml</code> 中的 <code>architecture.gitnexus.bridge_url</code>，并确保 Bridge 已启动。</p>
                </div>
            `;
            if (!state.architectureProviderUnavailableShown && typeof window.__toast === 'function') {
                state.architectureProviderUnavailableShown = true;
                window.__toast('架构服务未配置或 Bridge 未启动', 'info');
            }
        } else {
            container.innerHTML = `<div style="padding:24px;color:var(--red)">加载失败: ${msg}</div>`;
        }
    }
}

/** Show sidebar with impact and experiences for the selected node. */
export async function showArchitectureNodeSidebar(nodeId, path) {
    const tab = document.getElementById('architecture-tab-graph');
    if (!tab) return;
    const sidebar = document.getElementById('architecture-node-sidebar');
    const floatBtn = document.getElementById('architecture-sidebar-float-btn');
    if (!sidebar) return;
    sidebar.classList.remove('hidden');
    sidebar.classList.remove('collapsed');
    if (floatBtn) floatBtn.classList.remove('hidden');
    _archSidebarJustOpened = true;
    setTimeout(() => { _archSidebarJustOpened = false; }, 150);
    const titleEl = sidebar.querySelector('.architecture-sidebar-title');
    const impactEl = sidebar.querySelector('.architecture-sidebar-impact');
    const expEl = sidebar.querySelector('.architecture-sidebar-experiences');
    if (titleEl) titleEl.textContent = path || nodeId;
    if (impactEl) impactEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    if (expEl) expEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    const nodeKey = path || nodeId;
    // GitNexus impact expects symbol name (e.g. "showDetail"), not full node id (Function:path:name)
    const impactPath = nodeId.includes(':')
        ? (nodeNameFromNodeId(nodeId) || path || nodeId)
        : (path || nodeId);
    const filePath = path || pathFromNodeId(nodeId);
    try {
        const [impact, experiences, clustersRes] = await Promise.all([
            api('GET', `/api/v1/architecture/impact?path=${encodeURIComponent(impactPath)}&depth=2`).catch(() => ({ upstream: [], downstream: [] })),
            api('GET', `/api/v1/architecture/experiences?node=${encodeURIComponent(nodeKey)}`).catch(() => []),
            api('GET', `/api/v1/architecture/node-clusters?node=${encodeURIComponent(nodeId)}`).catch(() => ({ clusters: [] })),
        ]);
        const clusters = clustersRes?.clusters || [];
        if (impactEl) {
            const seenUp = new Set();
            const seenDown = new Set();
            const pathFromImpactId = (id) => (id && String(id).includes(':') ? String(id).split(':').slice(1, -1).join(':') : null);
            const dedupeKey = (i) => i.path || pathFromImpactId(i.id) || i.id || '';
            const displayLabel = (i) => (i.path || i.id || '').split('/').pop() || (i.path || i.id || '');
            const up = (impact.upstream || [])
                .filter((i) => {
                    const key = dedupeKey(i);
                    if (!key || seenUp.has(key)) return false;
                    seenUp.add(key);
                    return true;
                })
                .map((i) => `<div class="architecture-impact-item">${esc(displayLabel(i))}</div>`).join('');
            const down = (impact.downstream || [])
                .filter((i) => {
                    const key = dedupeKey(i);
                    if (!key || seenDown.has(key)) return false;
                    seenDown.add(key);
                    return true;
                })
                .map((i) => `<div class="architecture-impact-item">${esc(displayLabel(i))}</div>`).join('');
            const actionBtns = [];
            if (filePath) {
                actionBtns.push(`<button type="button" class="btn btn-primary btn-sm architecture-switch-main-btn" data-switch-path="${esc(filePath)}" data-switch-node="${esc(nodeId)}" title="以此节点为中心重新加载图" style="margin-bottom:8px">切换主节点</button>`);
                actionBtns.push(`<button type="button" class="btn btn-secondary btn-sm architecture-jump-btn" data-jump-path="${esc(filePath)}" title="在 Cursor/VS Code 中打开文件" style="margin-bottom:12px">在编辑器中打开</button>`);
            }
            const kind = kindFromNode(nodeId);
            const kindDisplay = kind || '未知';
            const nodeName = nodeNameFromNodeId(nodeId);
            const fileBase = filePath ? filePath.split('/').pop() : null;
            const nodeNameDisplay = nodeName ? (fileBase ? `${nodeName} (${fileBase})` : nodeName) : null;
            const nodeNameHtml = nodeNameDisplay
                ? `<div class="architecture-sidebar-section">
                    <div class="architecture-sidebar-section-title">节点名称</div>
                    <div class="architecture-impact-list" style="display:flex;align-items:center;gap:6px">
                        <code class="architecture-node-name-copy" data-copy="${esc(nodeNameDisplay)}" style="flex:1;min-width:0;padding:4px 8px;background:var(--bg-tertiary);border-radius:var(--radius);font-size:12px;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="点击复制">${esc(nodeNameDisplay)}</code>
                    </div>
                </div>`
                : '';
            const clustersPart = clusters.length
                ? clusters.map((c) => `<button type="button" class="architecture-impact-item architecture-cluster-link" data-cluster="${esc(c)}" title="切换到此集群视图">${esc(c)}</button>`).join('')
                : '';
            const metaText = `类型: ${kindDisplay}${clustersPart ? ' · 集群: ' + clusters.join(', ') : ' · 集群: 无'}`;
            const kindClustersHtml = `<div class="architecture-sidebar-section">
                    <div class="architecture-meta-row" style="display:flex;flex-wrap:wrap;align-items:center;font-size:12px;line-height:1.5;max-width:100%;gap:4px 6px" title="${esc(metaText)}">
                        <span style="color:var(--text-muted)">类型:</span> ${esc(kindDisplay)}
                        <span style="color:var(--text-muted)">·</span>
                        <span style="color:var(--text-muted)">集群:</span>
                        ${clustersPart ? `<span style="display:inline-flex;align-items:center;gap:4px">${clustersPart}</span>` : '无'}
                    </div>
                </div>`;
            impactEl.innerHTML = (actionBtns.length ? '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">' + actionBtns.join('') + '</div>' : '') + nodeNameHtml + kindClustersHtml + `
                <div class="architecture-sidebar-section">
                    <div class="architecture-sidebar-section-title" title="谁调用了它、谁依赖它">上游 <span class="architecture-sidebar-hint">谁依赖此节点</span></div>
                    <div class="architecture-impact-list">${up || '<span class="architecture-empty">无</span>'}</div>
                </div>
                <div class="architecture-sidebar-section">
                    <div class="architecture-sidebar-section-title" title="它调用了谁、它依赖谁">下游 <span class="architecture-sidebar-hint">此节点依赖谁</span></div>
                    <div class="architecture-impact-list">${down || '<span class="architecture-empty">无</span>'}</div>
                </div>
            `;
        }
        if (expEl) {
            const items = (experiences || []).map((e) =>
                `<a href="#" class="architecture-exp-item" data-exp-id="${esc(e.experience_id)}">${esc(e.title || e.experience_id)}</a>`
            ).join('');
            const mountBtn = `<button type="button" class="btn btn-primary btn-sm architecture-mount-btn" data-mount-node="${esc(nodeKey)}" style="width:100%;text-align:center">新建经验并挂载到此节点</button>`;
            expEl.innerHTML = (items ? `<div class="architecture-exp-list">${items}</div>` : '<div class="architecture-empty">暂无关联经验</div>') + mountBtn;
        }
        bindArchitectureHandlersToContainer(impactEl, expEl, nodeKey);
    } catch (e) {
        if (impactEl) impactEl.innerHTML = `<div style="color:var(--text-muted)">加载失败: ${esc(e.message || '')}</div>`;
        if (expEl) expEl.innerHTML = '<div style="color:var(--text-muted)">加载失败</div>';
    }
}

/** Bind click handlers for switch main, jump, copy, cluster links, mount, experience links. Reusable for sidebar and pinned cards. */
function bindArchitectureHandlersToContainer(impactEl, expEl, nodeKey) {
    if (!impactEl) return;
    const jumpBtn = impactEl.querySelector('.architecture-jump-btn');
    if (jumpBtn) {
        jumpBtn.onclick = async () => {
                const p = jumpBtn.dataset.jumpPath;
                if (!p) return;
                try {
                    const res = await api('GET', `/api/v1/architecture/open-in-editor?path=${encodeURIComponent(p)}`);
                    if (res?.open_url) {
                        const a = document.createElement('a');
                        a.href = res.open_url;
                        a.style.display = 'none';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        if (typeof window.__toast === 'function') window.__toast('已在编辑器中打开', 'info');
                    } else {
                        await navigator.clipboard?.writeText(p);
                        if (typeof window.__toast === 'function') window.__toast('已复制路径，可在 Cursor 中 Cmd+P 打开', 'info');
                    }
                } catch (e) {
                    if (typeof window.__toast === 'function') window.__toast('跳转失败: ' + (e.message || ''), 'error');
                }
            };
        }
        const switchMainBtn = impactEl?.querySelector('.architecture-switch-main-btn');
        if (switchMainBtn) {
            switchMainBtn.onclick = async () => {
                const p = switchMainBtn.dataset.switchPath;
                const nodeIdToHighlight = switchMainBtn.dataset.switchNode;
                if (!p) return;
                pushArchitectureBackState(nodeIdToHighlight, p);
                updateArchitectureBackButtonVisibility();
                if (typeof window.__toast === 'function') window.__toast('正在切换主节点…', 'info');
                await loadArchitectureGraph(null, p);
                if (nodeIdToHighlight) {
                    await highlightArchitectureNode(nodeIdToHighlight, p, true);
                }
            };
        }
        impactEl?.querySelectorAll('.architecture-node-name-copy').forEach((el) => {
            el.onclick = () => {
                const t = el.dataset.copy || el.textContent || '';
                navigator.clipboard?.writeText(t).then(() => {
                    if (typeof window.__toast === 'function') window.__toast('已复制', 'info');
                }).catch(() => {});
            };
        });
    impactEl?.querySelectorAll('.architecture-cluster-link').forEach((btn) => {
        btn.onclick = () => {
            const name = btn.dataset.cluster;
            if (name) switchToGraphWithCluster(name);
        };
    });
    if (expEl) {
        const mountBtn = expEl.querySelector('[data-mount-node]');
        if (mountBtn && nodeKey) {
            mountBtn.onclick = () => {
                state.architectureMountNode = mountBtn.dataset.mountNode || nodeKey;
                if (typeof window.openCreateModal === 'function') window.openCreateModal();
            };
        }
        expEl.querySelectorAll('.architecture-exp-item').forEach((a) => {
            a.addEventListener('click', (ev) => {
                ev.preventDefault();
                if (typeof window.showDetail === 'function') window.showDetail(a.dataset.expId);
            });
        });
    }
}

/** Pin current node detail: add to pinned panels and keep sidebar open on click-outside. */
export function pinArchitectureNodeDetail() {
    const sidebar = document.getElementById('architecture-node-sidebar');
    const container = document.getElementById('architecture-pinned-panels');
    if (!sidebar || !container || sidebar.classList.contains('hidden')) return;
    const titleEl = sidebar.querySelector('.architecture-sidebar-title');
    const impactEl = sidebar.querySelector('.architecture-sidebar-impact');
    const expEl = sidebar.querySelector('.architecture-sidebar-experiences');
    const title = (titleEl?.textContent || '').trim() || '节点详情';
    if (!impactEl) return;
    state.architectureSidebarPinned = true;
    const card = document.createElement('div');
    card.className = 'architecture-pinned-card';
    card.innerHTML = `
        <div class="architecture-pinned-card-header">
            <span class="architecture-pinned-card-drag" title="拖动">⋮⋮</span>
            <span class="architecture-pinned-card-title" title="${esc(title)}">${esc(title.length > 50 ? title.slice(0, 47) + '…' : title)}</span>
            <button type="button" class="architecture-pinned-card-close" title="关闭" aria-label="关闭">×</button>
        </div>
        <div class="architecture-pinned-card-body"></div>
    `;
    const body = card.querySelector('.architecture-pinned-card-body');
    const impactClone = impactEl.cloneNode(true);
    const expClone = expEl ? expEl.cloneNode(true) : null;
    body.appendChild(impactClone);
    if (expClone) body.appendChild(expClone);
    const pathFromBtn = impactClone.querySelector('.architecture-jump-btn')?.dataset?.jumpPath;
    const nodeIdFromBtn = impactClone.querySelector('.architecture-switch-main-btn')?.dataset?.switchNode;
    const nodeKeyForCard = expClone?.querySelector('[data-mount-node]')?.dataset?.mountNode || pathFromBtn || nodeIdFromBtn || title;
    bindArchitectureHandlersToContainer(impactClone, expClone, nodeKeyForCard);
    container.appendChild(card);
    container.classList.remove('hidden');
    setupPinnedCardDrag(card);
    const closeBtn = card.querySelector('.architecture-pinned-card-close');
    if (closeBtn) {
        closeBtn.onclick = () => {
            card.remove();
            if (container.children.length === 0) container.classList.add('hidden');
        };
    }
    const pinBtn = sidebar.querySelector('.architecture-sidebar-pin-btn');
    if (pinBtn) pinBtn.classList.add('pinned');
}

/** Setup drag for pinned card. Header acts as drag handle. */
function setupPinnedCardDrag(card) {
    const header = card.querySelector('.architecture-pinned-card-header');
    const closeBtn = card.querySelector('.architecture-pinned-card-close');
    if (!header) return;
    card.style.position = 'absolute';
    const container = card.parentElement;
    const existingCount = container ? [...container.children].filter((c) => c.classList.contains('architecture-pinned-card')).length - 1 : 0;
    const offset = Math.min(existingCount * 24, 120);
    card.style.left = `${offset}px`;
    card.style.top = `${offset}px`;
    card.style.width = '260px';
    card.style.maxWidth = '260px';
    header.style.cursor = 'grab';
    header.addEventListener('mousedown', (e) => {
        if (closeBtn && closeBtn.contains(e.target)) return;
        e.preventDefault();
        const startX = e.clientX;
        const startY = e.clientY;
        const startLeft = parseInt(card.style.left || '0', 10);
        const startTop = parseInt(card.style.top || '0', 10);
        header.style.cursor = 'grabbing';
        const onMove = (ev) => {
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;
            card.style.left = `${Math.max(0, startLeft + dx)}px`;
            card.style.top = `${Math.max(0, startTop + dy)}px`;
        };
        const onUp = () => {
            header.style.cursor = 'grab';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}

/** Unpin: allow click-outside to close sidebar again. */
export function unpinArchitectureSidebar() {
    state.architectureSidebarPinned = false;
    const pinBtn = document.querySelector('.architecture-sidebar-pin-btn');
    if (pinBtn) pinBtn.classList.remove('pinned');
}

/** Toggle pin: if already pinned, unpin; otherwise add current to pinned panels. */
export function toggleArchitecturePin() {
    if (state.architectureSidebarPinned) {
        unpinArchitectureSidebar();
    } else {
        pinArchitectureNodeDetail();
    }
}

function esc(s) {
    if (s == null) return '';
    const t = String(s);
    return t
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Load clusters and render list. */
export async function loadArchitectureClusters() {
    const tab = document.getElementById('architecture-tab-clusters');
    if (!tab) return;
    const listEl = tab.querySelector('.architecture-clusters-list');
    const membersEl = tab.querySelector('.architecture-cluster-members');
    if (!listEl) return;
    listEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    if (membersEl) membersEl.innerHTML = '';
    try {
        const clusters = await api('GET', '/api/v1/architecture/clusters');
        _archClustersLoaded = true;
        if (!clusters || clusters.length === 0) {
            listEl.innerHTML = '<div style="padding:24px;color:var(--text-muted)">暂无集群数据</div>';
            return;
        }
        listEl.innerHTML = clusters
            .map(
                (c) => `
            <div class="architecture-cluster-item" data-name="${esc(c.name)}" style="padding:10px 12px;margin-bottom:8px;border:1px solid var(--border);border-radius:var(--radius);cursor:pointer;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span class="cluster-name" style="font-weight:500;flex:1">${esc(c.name)}</span>
                ${c.member_count != null ? `<span class="cluster-count" style="font-size:12px;color:var(--text-muted)">${c.member_count} 成员</span>` : ''}
                <button type="button" class="btn btn-sm" data-action="view-graph" data-cluster="${esc(c.name)}">在图上看</button>
            </div>
        `
            )
            .join('');
        listEl.querySelectorAll('.architecture-cluster-item').forEach((el) => {
            const name = el.dataset.name;
            el.addEventListener('click', (ev) => {
                if (ev.target.dataset.action === 'view-graph') {
                    ev.stopPropagation();
                    switchToGraphWithCluster(name);
                    return;
                }
                loadArchitectureClusterMembers(name, membersEl);
            });
        });
    } catch (e) {
        listEl.innerHTML = `<div style="padding:24px;color:var(--red)">加载失败: ${esc(e.message || '')}</div>`;
    }
}

/** Load members of one cluster. */
async function loadArchitectureClusterMembers(name, membersEl) {
    if (!membersEl) return;
    membersEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', `/api/v1/architecture/cluster/${encodeURIComponent(name)}`);
        const members = data.members || [];
        membersEl.innerHTML = `
            <h4 style="margin:0 0 12px;font-size:14px">${esc(name)} 成员</h4>
            <ul class="architecture-members-list">
                ${members.map((m) => `<li>${esc(m.path || m.id)}</li>`).join('')}
            </ul>
        `;
    } catch (e) {
        membersEl.innerHTML = `<div style="color:var(--red)">加载失败: ${esc(e.message || '')}</div>`;
    }
}

export const ARCH_DEFAULT_CLUSTERS_KEY = 'architecture_default_clusters';
export const ARCH_SEARCH_SCOPE_KEY = 'architecture_search_scope';

/** Setup scope selector (全局/当前集群) with localStorage persistence. */
export function setupArchitectureScopeSelector() {
    const seg = document.querySelector('.architecture-search-scope-segmented');
    if (!seg) return;
    const opts = seg.querySelectorAll('.architecture-scope-option');
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem(ARCH_SEARCH_SCOPE_KEY) : null;
    const initial = saved === 'cluster' ? 'cluster' : 'global';
    opts.forEach((btn) => {
        const v = btn.dataset.scope;
        const active = v === initial;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', String(active));
    });
    opts.forEach((btn) => {
        btn.addEventListener('click', () => {
            const v = btn.dataset.scope;
            opts.forEach((b) => {
                const on = b.dataset.scope === v;
                b.classList.toggle('active', on);
                b.setAttribute('aria-pressed', String(on));
            });
            if (typeof localStorage !== 'undefined') localStorage.setItem(ARCH_SEARCH_SCOPE_KEY, v);
        });
    });
}

/** Get current search scope from UI. */
export function getArchitectureSearchScope() {
    const active = document.querySelector('.architecture-scope-option.active');
    return active?.dataset.scope === 'cluster' ? 'cluster' : 'global';
}

/** Open cluster selector popup; multi-select with checkboxes, 设为默认, 确定. */
export async function openArchitectureClusterSelector() {
    const overlay = document.createElement('div');
    overlay.className = 'architecture-cluster-popup-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    const popup = document.createElement('div');
    popup.className = 'architecture-cluster-popup';
    popup.style.cssText = 'background:var(--bg-primary);border:1px solid var(--border);border-radius:12px;padding:24px;box-shadow:0 12px 40px rgba(0,0,0,0.4);';
    popup.onclick = (e) => e.stopPropagation();
    popup.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    overlay.appendChild(popup);
    document.body.appendChild(overlay);
    try {
        const clusterList = await api('GET', '/api/v1/architecture/clusters');
        const current = state.architectureCurrentClusters || [];
        const saved = (typeof localStorage !== 'undefined' && localStorage.getItem(ARCH_DEFAULT_CLUSTERS_KEY))
            ? JSON.parse(localStorage.getItem(ARCH_DEFAULT_CLUSTERS_KEY) || '[]')
            : null;
        const items = (clusterList || []).map(
            (c) => {
                const checked = current.includes(c.name);
                const count = c.member_count != null ? ` <span class="cluster-count">(${c.member_count})</span>` : '';
                return `<label class="architecture-cluster-check-item">
                    <input type="checkbox" data-cluster="${esc(c.name)}" ${checked ? 'checked' : ''}>
                    <span>${esc(c.name)}${count}</span>
                </label>`;
            }
        ).join('');
        popup.innerHTML = `
            <h4>选择集群（可多选）</h4>
            <div class="architecture-cluster-popup-actions">
                <button type="button" class="btn btn-secondary" data-action="all">全部</button>
                <button type="button" class="btn btn-secondary" data-action="default">恢复默认</button>
                <button type="button" class="btn btn-secondary" data-action="save-default">设为默认</button>
            </div>
            <div class="architecture-cluster-list">${items}</div>
            <div class="architecture-cluster-popup-footer" style="margin-top:20px;text-align:center">
                <button type="button" class="btn btn-primary btn-apply" data-action="apply">确定</button>
            </div>
        `;
        const getChecked = () => [...popup.querySelectorAll('input[type=checkbox]:checked')].map((el) => el.dataset.cluster);
        const checkboxes = () => popup.querySelectorAll('input[type=checkbox]');
        popup.querySelector('[data-action="all"]').onclick = () => {
            const cbs = checkboxes();
            const allChecked = cbs.length > 0 && [...cbs].every((cb) => cb.checked);
            cbs.forEach((cb) => { cb.checked = !allChecked; });
        };
        popup.querySelector('[data-action="default"]').onclick = () => {
            const def = saved || state.architectureDefaultClusters || [];
            popup.querySelectorAll('input[type=checkbox]').forEach((cb) => {
                cb.checked = def.includes(cb.dataset.cluster);
            });
        };
        popup.querySelector('[data-action="save-default"]').onclick = () => {
            const sel = getChecked();
            if (typeof localStorage !== 'undefined') {
                localStorage.setItem(ARCH_DEFAULT_CLUSTERS_KEY, JSON.stringify(sel));
                if (typeof window.__toast === 'function') window.__toast('已设为默认展示', 'success');
            }
        };
        const MAX_CLUSTERS = 30;
        popup.querySelector('[data-action="apply"]').onclick = () => {
            const sel = getChecked();
            if (sel.length > MAX_CLUSTERS) {
                if (typeof window.__toast === 'function') {
                    window.__toast(`最多选择 ${MAX_CLUSTERS} 个集群，已截取前 ${MAX_CLUSTERS} 个`, 'warning');
                }
            }
            overlay.remove();
            loadArchitectureGraph(sel.length ? sel.slice(0, MAX_CLUSTERS) : null, null);
        };
    } catch (e) {
        popup.innerHTML = `<div style="color:var(--red);font-size:13px">加载失败: ${esc(e.message || '')}</div>`;
    }
}

/** Switch to graph tab and load graph with cluster filter. */
export function switchToGraphWithCluster(clusterName) {
    const content = document.getElementById('architecture-content');
    if (!content) return;
    const graphTabBtn = content.querySelector('.mode-tab[onclick*="graph"]');
    if (graphTabBtn) graphTabBtn.click();
    setTimeout(() => loadArchitectureGraph(clusterName, null), 100);
}

export function isArchitectureGraphLoaded() {
    return _archGraphLoaded;
}

export function isArchitectureClustersLoaded() {
    return _archClustersLoaded;
}

/** Reset loaded state when architecture content is replaced (e.g. refresh). */
export function resetArchitectureViewerState() {
    _archGraphLoaded = false;
    _archClustersLoaded = false;
    _archCy = null;
    state.architectureCurrentClusters = null;
}
