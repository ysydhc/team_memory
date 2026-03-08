/**
 * Workflow viewer: YAML parse, $ref resolve, Cytoscape render, drag-drop.
 * Plan: workflow-visualization-v1-implementation-plan.md
 */
const REF_DEPTH_MAX = 10;

function parseYaml(text) {
    if (typeof window.jsyaml === 'undefined') throw new Error('js-yaml 未加载，请刷新页面后重试');
    return window.jsyaml.load(text);
}

function resolveStepRef(step, fileMap, basePath, visitedPaths, depth) {
    if (depth > REF_DEPTH_MAX) {
        throw new Error('$ref 引用层级过深（超过 10 层）');
    }
    if (!step || typeof step !== 'object') return step;
    const keys = Object.keys(step);
    if (keys.length !== 1 || keys[0] !== '$ref') return step;
    const refPath = step.$ref;
    const normalizedPath = basePath ? (basePath + '/' + refPath).replace(/\/+/g, '/') : refPath;
    if (visitedPaths.has(normalizedPath)) {
        throw new Error('检测到循环引用：' + normalizedPath);
    }
    visitedPaths.add(normalizedPath);
    try {
        const content = fileMap[normalizedPath];
        if (!content) throw new Error('Step ref not found: ' + normalizedPath);
        const resolved = typeof content === 'string' ? parseYaml(content) : content;
        if (!resolved || typeof resolved !== 'object') throw new Error('Invalid step file: ' + normalizedPath);
        const refBase = normalizedPath.replace(/\/[^/]+$/, '') || '.';
        const nextResolved = resolveStepRef(resolved, fileMap, refBase, new Set(visitedPaths), depth + 1);
        visitedPaths.delete(normalizedPath);
        return nextResolved;
    } catch (e) {
        visitedPaths.delete(normalizedPath);
        throw e;
    }
}

function resolveWorkflow(data, fileMap, basePath) {
    if (!data || !Array.isArray(data.steps)) return data;
    const resolved = [];
    for (const step of data.steps) {
        if (step && typeof step === 'object' && step.$ref) {
            resolved.push(resolveStepRef(step, fileMap, basePath || '', new Set(), 0));
        } else {
            resolved.push(step);
        }
    }
    return { ...data, steps: resolved };
}

function findMainWorkflow(fileMap) {
    const candidates = Object.keys(fileMap)
        .filter((p) => /\.(yaml|yml)$/i.test(p))
        .sort();
    for (const path of candidates) {
        try {
            const raw = fileMap[path];
            const data = typeof raw === 'string' ? parseYaml(raw) : raw;
            if (data && data.meta && Array.isArray(data.steps)) return { path, data };
        } catch (_) {}
    }
    return null;
}

async function collectFilesFromEntry(entry, prefix, fileMap) {
    if (entry.isFile) {
        const file = await entry.file();
        const ext = (file.name || '').toLowerCase();
        if (ext.endsWith('.yaml') || ext.endsWith('.yml')) {
            const text = await file.text();
            const key = prefix ? prefix + '/' + file.name : file.name;
            fileMap[key] = text;
        }
    } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let entries = [];
        let batch = await reader.readEntries();
        while (batch.length) {
            entries = entries.concat(Array.from(batch));
            batch = await reader.readEntries();
        }
        for (const e of entries) {
            const name = e.name || 'unknown';
            const nextPrefix = prefix ? prefix + '/' + name : name;
            await collectFilesFromEntry(e, nextPrefix, fileMap);
        }
    }
}

/** Build fileMap from FileList with webkitRelativePath (from input[webkitdirectory]). */
async function collectFilesFromWebkitFiles(files) {
    const fileMap = {};
    for (const file of Array.from(files)) {
        if (!/\.(yaml|yml)$/i.test(file.name)) continue;
        const path = file.webkitRelativePath || file.name;
        fileMap[path] = await file.text();
    }
    return fileMap;
}

/** Build fileMap from File System Access API (showDirectoryPicker). Avoids native folder picker issues. */
async function collectFilesFromDirHandle(dirHandle, prefix, fileMap) {
    for await (const entry of dirHandle.values()) {
        const name = entry.name || 'unknown';
        const path = prefix ? prefix + '/' + name : name;
        if (entry.kind === 'file') {
            if (/\.(yaml|yml)$/i.test(name)) {
                const file = await entry.getFile();
                fileMap[path] = await file.text();
            }
        } else if (entry.kind === 'directory') {
            await collectFilesFromDirHandle(entry, path, fileMap);
        }
    }
    return fileMap;
}

/** Wrap long labels with \n so Cytoscape renders them within node bounds. */
function wrapLabel(text, maxPerLine = 14) {
    if (!text || text.length <= maxPerLine) return text;
    const parts = [];
    let rest = text;
    while (rest.length > maxPerLine) {
        let split = maxPerLine;
        const chunk = rest.slice(0, maxPerLine);
        const atParen = chunk.lastIndexOf('(');
        const atComma = chunk.lastIndexOf('，');
        const atCommaEn = chunk.lastIndexOf(',');
        if (atParen > maxPerLine * 0.5) split = atParen;
        else if (atComma > maxPerLine * 0.5) split = atComma + 1;
        else if (atCommaEn > maxPerLine * 0.5) split = atCommaEn + 1;
        parts.push(rest.slice(0, split));
        rest = rest.slice(split).replace(/^\s+/, '');
    }
    if (rest) parts.push(rest);
    return parts.join('\n');
}

/** Estimate node dimensions so box tightly wraps text (CJK ~19px/char at 18px font). */
function estimateNodeSize(wrappedLabel) {
    const lines = wrappedLabel.split('\n');
    const maxLen = Math.max(...lines.map((l) => l.length), 1);
    return { w: Math.min(400, maxLen * 19 + 40), h: lines.length * 30 + 36 };
}

function workflowToGraph(data) {
    const nodes = [];
    const edges = [];
    const steps = data?.steps || [];
    const idSet = new Set();
    steps.forEach((s, idx) => {
        const id = s?.id;
        if (!id) return;
        idSet.add(id);
        const rawLabel = s.name || id;
        const label = wrapLabel(rawLabel);
        const { w, h } = estimateNodeSize(label);
        const detail = {
            action: s.action || '',
            acceptance_criteria: s.acceptance_criteria || '',
            when: s.when,
            allowed_next: s.allowed_next,
            optional: s.optional,
            checkpoint: s.checkpoint,
        };
        nodes.push({ id, label, width: w, height: h, order: idx, detail });
    });
    for (const s of steps) {
        const from = s?.id;
        if (!from) continue;
        const when = s.when;
        if (when && Array.isArray(when)) {
            for (const b of when) {
                const to = b?.next;
                if (to && idSet.has(to)) {
                    edges.push({ from, to, label: b.condition || '' });
                }
            }
        } else {
            const allowed = s.allowed_next;
            if (Array.isArray(allowed)) {
                for (const to of allowed) {
                    if (to && idSet.has(to)) edges.push({ from, to, label: '' });
                }
            }
        }
    }
    return { nodes, edges };
}

let _workflowCy = null;
let _layoutOrientation = 'cose'; // 'cose' | 'horizontal' | 'vertical'
let _workflowButtonsBound = false;
let _workflowDetailBtn = null;
let _workflowDetailHideTimer = null;

// 调试日志：默认开启，控制台执行 window._workflowDebug=false 可关闭
const _wfLog = (tag, ...args) => {
    if (typeof window !== 'undefined' && window._workflowDebug !== false) {
        console.log('[WF-DEBUG]', tag, ...args);
    }
};

function renderGraph(containerId, graph) {
    if (typeof window.cytoscape === 'undefined') throw new Error('Cytoscape 未加载，请刷新页面后重试');
    const container = document.getElementById(containerId);
    if (!container) return null;
    const elements = [
        ...graph.nodes.map((n) => ({
            data: {
                id: n.id,
                label: n.label,
                width: n.width,
                height: n.height,
                order: n.order ?? 0,
                detail: n.detail || {},
            },
        })),
        ...graph.edges.map((e, i) => ({ data: { id: 'e' + i, source: e.from, target: e.to, label: e.label } })),
    ];
    // n8n/Dify-style: card nodes, rounded corners, subtle borders, clean typography
    const cy = window.cytoscape({
        container,
        elements,
        style: [
            {
                selector: 'node',
                style: {
                    shape: 'round-rectangle',
                    'corner-radius': '8px',
                    width: (el) => (el.data('width') || 0) + 'px',
                    height: (el) => (el.data('height') || 0) + 'px',
                    padding: '8px 12px',
                    'background-color': '#1e293b',
                    'border-width': 1,
                    'border-color': '#334155',
                    label: 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    color: '#f1f5f9',
                    'font-size': '18px',
                    'font-weight': '500',
                    'text-wrap': 'wrap',
                    'text-max-width': (el) => Math.max(100, (el.data('width') || 200) - 24) + 'px',
                    'text-overflow-wrap': 'anywhere',
                    'text-margin-y': 0,
                    'line-height': 1.5,
                },
            },
            {
                selector: 'node:selected',
                style: { 'border-color': '#3b82f6', 'border-width': 2 },
            },
            {
                selector: 'node.hover',
                style: { 'background-color': '#334155', 'border-color': '#475569' },
            },
            {
                selector: 'edge',
                style: {
                    'line-color': '#475569',
                    'target-arrow-color': '#64748b',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    width: 1.5,
                },
            },
            {
                selector: 'edge[label]',
                style: {
                    label: 'data(label)',
                    'font-size': '12px',
                    color: '#94a3b8',
                    'text-rotation': 'autorotate',
                    'text-background-color': '#1e293b',
                    'text-background-opacity': 0.9,
                    'text-background-padding': '3px',
                    'text-background-shape': 'round-rectangle',
                },
            },
        ],
        minZoom: 0.2,
        maxZoom: 4,
    });
    runWorkflowLayout(cy, _layoutOrientation);
    _wfLog('renderGraph', 'cy 创建完成，绑定事件');
    cy.on('mouseover', 'node', (e) => {
        _wfLog('mouseover', 'node', e.target.id());
        e.target.addClass('hover');
        showWorkflowDetailButton(cy, e.target);
    });
    cy.on('mouseout', 'node', (e) => {
        _wfLog('mouseout', 'node', e.target.id());
        e.target.removeClass('hover');
        scheduleHideWorkflowDetailButton(e.target);
    });
    cy.on('tap', 'node', (e) => {
        const target = e.target;
        _wfLog('tap', 'node', target?.id?.());
        openWorkflowStepDetailModal(target);
    });
    container._cy = cy;
    _workflowCy = cy;
    return cy;
}

function ensureWorkflowDetailButton() {
    const overlay = document.getElementById('workflow-node-overlay');
    _wfLog('ensureWorkflowDetailButton', 'overlay', !!overlay, 'overlayId', overlay?.id);
    if (!overlay) return null;
    if (_workflowDetailBtn) {
        _wfLog('ensureWorkflowDetailButton', '复用已有按钮');
        return _workflowDetailBtn;
    }
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm';
    btn.textContent = '显示详情';
    btn.style.cssText = 'position:absolute;pointer-events:auto;background:var(--accent);color:#fff;border:none;padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.3);display:none;z-index:60;white-space:nowrap';
    btn.onclick = (ev) => {
        ev.stopPropagation();
        const nodeId = btn.dataset.nodeId;
        _wfLog('btn.onclick', 'nodeId', nodeId, '_workflowCy', !!_workflowCy);
        if (nodeId && _workflowCy) {
            const node = _workflowCy.getElementById(nodeId);
            _wfLog('btn.onclick', 'node.length', node?.length?.());
            if (node.length) openWorkflowStepDetailModal(node);
        }
    };
    btn.onmouseenter = () => {
        if (_workflowDetailHideTimer) clearTimeout(_workflowDetailHideTimer);
        _workflowDetailHideTimer = null;
    };
    btn.onmouseleave = () => hideWorkflowDetailButton();
    overlay.appendChild(btn);
    _workflowDetailBtn = btn;
    _wfLog('ensureWorkflowDetailButton', '新建按钮并 append 到 overlay');
    return btn;
}

function showWorkflowDetailButton(cy, node) {
    const btn = ensureWorkflowDetailButton();
    const overlay = document.getElementById('workflow-node-overlay');
    _wfLog('showWorkflowDetailButton', 'entry', 'nodeId', node?.id?.(), 'btn', !!btn, 'overlay', !!overlay, 'node.length', node?.length?.());
    if (!btn) {
        _wfLog('showWorkflowDetailButton', 'early return: !btn');
        return;
    }
    if (!overlay) {
        _wfLog('showWorkflowDetailButton', 'early return: !overlay');
        return;
    }
    if (!node.length()) {
        _wfLog('showWorkflowDetailButton', 'early return: !node.length()');
        return;
    }
    const pos = node.renderedPosition();
    const w = node.renderedWidth();
    const h = node.renderedHeight();
    const container = cy.container();
    _wfLog('showWorkflowDetailButton', 'pos', pos, 'w', w, 'h', h, 'container', !!container);
    if (!container) {
        _wfLog('showWorkflowDetailButton', 'early return: !container');
        return;
    }
    const containerRect = container.getBoundingClientRect();
    const overlayRect = overlay.getBoundingClientRect();
    const offsetX = containerRect.left - overlayRect.left + (container.clientLeft || 0);
    const offsetY = containerRect.top - overlayRect.top + (container.clientTop || 0);
    const btnX = offsetX + pos.x + w / 2 - 36;
    const btnY = offsetY + pos.y - 28;
    const maxX = overlayRect.width - 90;
    const maxY = overlayRect.height - 36;
    const finalLeft = Math.max(8, Math.min(maxX, btnX));
    const finalTop = Math.max(8, Math.min(maxY, btnY));
    btn.dataset.nodeId = node.id();
    btn.style.left = finalLeft + 'px';
    btn.style.top = finalTop + 'px';
    btn.style.display = 'block';
    _wfLog('showWorkflowDetailButton', '按钮已显示', 'left', finalLeft, 'top', finalTop, 'overlayRect', { w: overlayRect.width, h: overlayRect.height });
}

function scheduleHideWorkflowDetailButton(node) {
    if (_workflowDetailHideTimer) clearTimeout(_workflowDetailHideTimer);
    _workflowDetailHideTimer = setTimeout(() => {
        const btn = _workflowDetailBtn;
        if (btn) btn.style.display = 'none';
        _workflowDetailHideTimer = null;
    }, 150);
}

function hideWorkflowDetailButton() {
    if (_workflowDetailHideTimer) clearTimeout(_workflowDetailHideTimer);
    _workflowDetailHideTimer = null;
    const btn = _workflowDetailBtn;
    if (btn) btn.style.display = 'none';
}

function openWorkflowStepDetailModal(node) {
    _wfLog('openWorkflowStepDetailModal', 'entry', 'node', node?.id?.());
    const detail = node?.data?.('detail') || {};
    const title = node?.data?.('label') || node?.id?.() || '步骤详情';
    const titleEl = document.getElementById('workflow-step-detail-title');
    const bodyEl = document.getElementById('workflow-step-detail-body');
    const modal = document.getElementById('workflow-step-detail-modal');
    _wfLog('openWorkflowStepDetailModal', 'titleEl', !!titleEl, 'bodyEl', !!bodyEl, 'modal', !!modal);
    if (!titleEl || !bodyEl || !modal) {
        _wfLog('openWorkflowStepDetailModal', 'early return: 缺少 DOM 元素');
        return;
    }
    titleEl.textContent = title.replace(/\n/g, ' ');
    const parts = [];
    if (detail.action) parts.push('【执行说明】\n' + detail.action.trim());
    if (detail.acceptance_criteria) parts.push('\n【验收标准】\n' + detail.acceptance_criteria.trim());
    if (detail.when && detail.when.length) {
        const whenStr = detail.when.map((w) => `  ${w.condition || ''} → ${w.next || ''}`).join('\n');
        parts.push('\n【条件分支】\n' + whenStr);
    }
    if (detail.allowed_next && detail.allowed_next.length) {
        parts.push('\n【下一步】\n  ' + detail.allowed_next.join(', '));
    }
    if (detail.optional != null) parts.push('\n【可选】' + (detail.optional ? '是' : '否'));
    if (detail.checkpoint != null) parts.push('\n【检查点】' + (detail.checkpoint ? '是' : '否'));
    bodyEl.textContent = parts.length ? parts.join('\n') : '暂无详情';
    modal.classList.remove('hidden');
    const cs = window.getComputedStyle?.(modal);
    _wfLog('openWorkflowStepDetailModal', 'modal.classList.remove(hidden) 已执行', 'display', cs?.display, 'visibility', cs?.visibility, 'zIndex', cs?.zIndex);
}

function closeWorkflowStepDetailModal() {
    const modal = document.getElementById('workflow-step-detail-modal');
    if (modal) modal.classList.add('hidden');
}

function runWorkflowLayout(cy, orientation) {
    if (!cy || (typeof cy.destroyed === 'function' ? cy.destroyed() : cy.destroyed)) return;
    const opts = {
        padding: 40,
        avoidOverlap: true,
        nodeDimensionsIncludeLabels: true,
        animate: false,
        fit: false,
    };
    if (orientation === 'horizontal') {
        opts.name = 'dagre';
        opts.rankDir = 'LR';
        opts.acyclicer = 'greedy';
        opts.rankSep = 60;
        opts.nodeSep = 50;
        opts.edgeSep = 20;
        opts.ranker = 'network-simplex';
        opts.sort = (a, b) => (a.data('order') ?? 0) - (b.data('order') ?? 0);
    } else if (orientation === 'vertical') {
        opts.name = 'dagre';
        opts.rankDir = 'TB';
        opts.acyclicer = 'greedy';
        opts.rankSep = 60;
        opts.nodeSep = 50;
        opts.edgeSep = 20;
        opts.ranker = 'network-simplex';
        opts.sort = (a, b) => (a.data('order') ?? 0) - (b.data('order') ?? 0);
    } else {
        // Prefer Dagre for layered left-to-right layout (like target fig 2)
        opts.name = 'dagre';
        opts.rankDir = 'LR';
        opts.acyclicer = 'greedy';
        opts.rankSep = 60;
        opts.nodeSep = 50;
        opts.edgeSep = 20;
        opts.ranker = 'network-simplex';
        opts.sort = (a, b) => (a.data('order') ?? 0) - (b.data('order') ?? 0);
    }
    let layout;
    try {
        layout = cy.layout(opts);
        if (opts.name === 'dagre') {
            console.info('[workflow-viewer] 使用 Dagre 层次布局');
        }
    } catch (e) {
        if (opts.name === 'dagre') {
            console.warn('[workflow-viewer] Dagre 不可用，降级为 Cola:', e?.message || e);
            opts.name = 'cola';
            opts.flow = { axis: 'y', minSeparation: 50 };
            opts.nodeSpacing = () => 60;
            opts.edgeLength = () => 120;
            opts.randomize = false;
            opts.maxSimulationTime = 5000;
            delete opts.rankDir;
            delete opts.acyclicer;
            delete opts.rankSep;
            delete opts.nodeSep;
            delete opts.edgeSep;
            delete opts.ranker;
            try {
                layout = cy.layout(opts);
                console.info('[workflow-viewer] 已降级为 Cola 布局');
            } catch (e2) {
                opts.name = 'fcose';
                opts.quality = 'proof';
                opts.randomize = true;
                opts.nodeRepulsion = () => 8000;
                opts.idealEdgeLength = () => 100;
                opts.edgeElasticity = () => 0.35;
                opts.nodeSeparation = 80;
                opts.numIter = 3000;
                delete opts.flow;
                delete opts.nodeSpacing;
                delete opts.edgeLength;
                delete opts.maxSimulationTime;
                layout = cy.layout(opts);
                console.info('[workflow-viewer] 已降级为 fCoSE 布局');
            }
        } else if (opts.name === 'cola') {
            console.warn('[workflow-viewer] Cola 不可用，降级为 fCoSE:', e?.message || e);
            opts.name = 'fcose';
            opts.quality = 'proof';
            opts.randomize = true;
            opts.nodeRepulsion = () => 8000;
            opts.idealEdgeLength = () => 100;
            opts.edgeElasticity = () => 0.35;
            opts.nodeSeparation = 80;
            opts.numIter = 3000;
            delete opts.flow;
            delete opts.nodeSpacing;
            delete opts.edgeLength;
            delete opts.maxSimulationTime;
            layout = cy.layout(opts);
            console.info('[workflow-viewer] 已降级为 fCoSE 布局');
        } else if (opts.name === 'fcose') {
            opts.name = 'cose';
            opts.idealEdgeLength = 120;
            opts.nodeRepulsion = 16000;
            opts.nodeOverlap = 50;
            opts.randomize = true;
            delete opts.quality;
            delete opts.edgeElasticity;
            delete opts.nodeSeparation;
            delete opts.numIter;
            layout = cy.layout(opts);
        } else {
            throw e;
        }
    }
    const effectiveLayout = opts.name;
    const rankDir = opts.rankDir || 'LR';
        layout.run();
        layout.on('layoutstop', () => {
            _wfLog('layoutstop', '布局完成，可交互');
        // Dagre: use taxi (orthogonal) edges for 90-degree bends, direction matches rankDir
        if (effectiveLayout === 'dagre') {
            const taxiDir = rankDir === 'TB' ? 'downward' : 'rightward';
            cy.edges().style({ 'curve-style': 'taxi', 'taxi-direction': taxiDir });
        } else {
            cy.edges().style('curve-style', 'bezier');
        }
        cy.fit(undefined, 40);
    });
}

function fitWorkflowView(cyOrNull) {
    const container = document.getElementById('workflow-graph-container');
    const cy = (container && container._cy) || cyOrNull || _workflowCy;
    if (!cy || (typeof cy.destroyed === 'function' ? cy.destroyed() : cy.destroyed)) return;
    requestAnimationFrame(() => {
        cy.resize();
        cy.fit(cy.elements(), 40);
    });
}

function showError(msg) {
    const el = document.getElementById('workflow-error');
    if (el) {
        el.textContent = msg;
        el.classList.remove('hidden');
    }
}

function hideError() {
    const el = document.getElementById('workflow-error');
    if (el) el.classList.add('hidden');
}

function processWorkflowData(data, fileMap, basePath) {
    const hasUnresolvedRef = (steps) => steps?.some((s) => s && typeof s === 'object' && s.$ref);
    if (fileMap && Object.keys(fileMap).length > 1 && hasUnresolvedRef(data?.steps)) {
        return resolveWorkflow(data, fileMap, basePath);
    }
    if (hasUnresolvedRef(data?.steps)) {
        throw new Error('请拖入包含工作流的文件夹以解析 $ref');
    }
    return data;
}

function handleWorkflowResult(data, fileMap, mainPath) {
    hideError();
    // basePath = directory containing main file. "a/b.yaml" -> "a", "root.yaml" -> ""
    const basePath = mainPath && mainPath.includes('/') ? mainPath.replace(/\/[^/]+$/, '') : '';
    let resolved;
    try {
        resolved = processWorkflowData(data, fileMap, basePath);
    } catch (e) {
        showError(e.message || '解析失败');
        return;
    }
    if (!resolved?.steps?.length) {
        showError('该 YAML 缺少 steps 字段');
        return;
    }
    const graph = workflowToGraph(resolved);
    _wfLog('handleWorkflowResult', 'graph 节点数', graph?.nodes?.length, '边数', graph?.edges?.length);
    const dropZone = document.getElementById('workflow-drop-zone');
    const graphContainer = document.getElementById('workflow-graph-container');
    const overlay = document.getElementById('workflow-node-overlay');
    _wfLog('handleWorkflowResult', 'graphContainer', !!graphContainer, 'overlay', !!overlay);
    if (dropZone) dropZone.classList.add('hidden');
    const wrapper = document.getElementById('workflow-graph-wrapper');
    if (graphContainer && wrapper) {
        wrapper.classList.remove('hidden');
        graphContainer._cy = null;
        graphContainer.innerHTML = '';
        hideWorkflowDetailButton();
        _wfLog('handleWorkflowResult', '开始 renderGraph');
        try {
            renderGraph('workflow-graph-container', graph);
            _wfLog('handleWorkflowResult', 'renderGraph 完成');
        } catch (e) {
            _wfLog('handleWorkflowResult', 'renderGraph 异常', e);
            showError(e.message || '渲染失败');
        }
    } else {
        _wfLog('handleWorkflowResult', '跳过渲染: graphContainer 或 wrapper 缺失');
    }
}

function handleFileList(files, isFolder) {
    const yamlFiles = Array.from(files).filter((f) => /\.(yaml|yml)$/i.test(f.name));
    if (yamlFiles.length === 0) {
        showError('请选择 .yaml 或 .yml 工作流文件');
        return;
    }
    if (isFolder || yamlFiles.length > 1) {
        showError('单文件模式：请拖入单个 YAML 文件，或拖入文件夹以解析 $ref');
        return;
    }
    const file = yamlFiles[0];
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const data = parseYaml(reader.result);
            if (!data?.steps) {
                showError('该 YAML 缺少 steps 字段');
                return;
            }
            handleWorkflowResult(data, null, null);
        } catch (e) {
            showError('解析失败：' + (e.message || e));
        }
    };
    reader.readAsText(file);
}

export function initWorkflowViewer() {
    _wfLog('init', '工作流可视化初始化，调试日志已开启（window._workflowDebug=false 可关闭）');
    window.closeWorkflowStepDetailModal = closeWorkflowStepDetailModal;
    const dropZone = document.getElementById('workflow-drop-zone');
    const fileInput = document.getElementById('workflow-file-input');
    const dirInput = document.getElementById('workflow-dir-input');
    const btnFile = document.getElementById('workflow-btn-file');
    const btnDir = document.getElementById('workflow-btn-dir');
    const graphContainer = document.getElementById('workflow-graph-container');

    if (!dropZone) return;

    if (btnFile) btnFile.onclick = (e) => { e.stopPropagation(); fileInput?.click(); };
    if (btnDir) {
        btnDir.onclick = async (e) => {
            e.stopPropagation();
            if (typeof window.showDirectoryPicker === 'function') {
                try {
                    const dirHandle = await window.showDirectoryPicker();
                    const fileMap = {};
                    await collectFilesFromDirHandle(dirHandle, '', fileMap);
                    if (Object.keys(fileMap).length === 0) {
                        showError('该文件夹内无 .yaml 或 .yml 文件');
                        return;
                    }
                    const main = findMainWorkflow(fileMap);
                    if (!main) {
                        showError('未找到有效工作流（需含 meta 与 steps）');
                        return;
                    }
                    const data = typeof main.data === 'string' ? parseYaml(main.data) : main.data;
                    handleWorkflowResult(data, fileMap, main.path);
                } catch (err) {
                    if (err?.name !== 'AbortError') showError(err?.message || '解析失败');
                }
            } else {
                dirInput?.click();
            }
        };
    }

    dirInput.onchange = async (e) => {
        const files = e.target?.files;
        e.target.value = '';
        if (!files?.length) return;
        const fileMap = await collectFilesFromWebkitFiles(files);
        if (Object.keys(fileMap).length === 0) {
            showError('该文件夹内无 .yaml 或 .yml 文件');
            return;
        }
        const main = findMainWorkflow(fileMap);
        if (!main) {
            showError('未找到有效工作流（需含 meta 与 steps）');
            return;
        }
        try {
            const data = typeof main.data === 'string' ? parseYaml(main.data) : main.data;
            handleWorkflowResult(data, fileMap, main.path);
        } catch (err) {
            showError(err.message || '解析失败');
        }
    };

    dropZone.ondragover = (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = 'var(--accent)';
        dropZone.style.background = 'var(--accent-glow)';
    };
    dropZone.ondragleave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
    };
    dropZone.ondrop = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
        const items = e.dataTransfer?.items;
        if (items && items.length > 0) {
            const item = items[0];
            const getEntry = item.webkitGetAsEntry || item.getAsEntry;
            if (getEntry) {
                const entry = getEntry.call(item);
                if (entry?.isDirectory) {
                    const fileMap = {};
                    await collectFilesFromEntry(entry, '', fileMap);
                    const main = findMainWorkflow(fileMap);
                    if (!main) {
                        showError('未找到有效工作流（需含 meta 与 steps）');
                        return;
                    }
                    try {
                        const data = typeof main.data === 'string' ? parseYaml(main.data) : main.data;
                        handleWorkflowResult(data, fileMap, main.path);
                    } catch (err) {
                        showError(err.message || '解析失败');
                    }
                    return;
                }
            }
        }
        const files = e.dataTransfer?.files;
        if (files?.length) handleFileList(files, false);
    };

    dropZone.onclick = (e) => { if (!e.target?.closest?.('button')) fileInput?.click(); };
    fileInput.onchange = (e) => {
        const files = e.target?.files;
        if (files?.length) handleFileList(files, false);
        e.target.value = '';
    };

    const wrapper = document.getElementById('workflow-graph-wrapper');
    if (wrapper) wrapper.classList.add('hidden');
    hideError();

    if (!_workflowButtonsBound) {
        const wrapperForDelegation = document.getElementById('workflow-graph-wrapper');
        if (wrapperForDelegation) {
            wrapperForDelegation.addEventListener('click', (e) => {
                const t = e.target?.closest?.('button');
                if (!t) return;
                if (t.id === 'workflow-btn-reset') {
                    e.stopPropagation();
                    e.preventDefault();
                    fitWorkflowView();
                } else if (t.id === 'workflow-btn-orient') {
                    e.stopPropagation();
                    e.preventDefault();
                    if (!_workflowCy || (typeof _workflowCy.destroyed === 'function' ? _workflowCy.destroyed() : _workflowCy.destroyed)) return;
                    _layoutOrientation = _layoutOrientation === 'horizontal' ? 'vertical' : 'horizontal';
                    runWorkflowLayout(_workflowCy, _layoutOrientation);
                }
            });
            _workflowButtonsBound = true;
        }
    }
}
