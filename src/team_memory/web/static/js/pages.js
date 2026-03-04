/**
 * Page rendering functions: list, detail, dashboard, drafts, reviews, settings.
 */

import { state, defaultTypeIcons } from './store.js';
import { esc, formatDate, timeAgo } from './utils.js';
import { resolveProjectInput, loadSchemaAndPopulateFilters, applyProjectPlaceholders } from './schema.js';

function api(...args) {
    return window.__api(...args);
}

function toast(msg, type = 'info') {
    return window.__toast(msg, type);
}

function navigate(page) {
    return window.__navigate(page);
}

// ===== Stale Check Helper =====
export function isStaleDate(dateStr) {
    if (!dateStr) return false;
    const d = new Date(dateStr);
    const now = new Date();
    const diffMonths = (now.getFullYear() - d.getFullYear()) * 12 + (now.getMonth() - d.getMonth());
    return diffMonths >= 6;
}

const typeIcons = {
    general: '📝', feature: '🚀', bugfix: '🐛', tech_design: '📐',
    incident: '🔥', best_practice: '✨', learning: '📚',
};

/** Copy text to clipboard; works in non-secure context (no HTTPS) via execCommand fallback. */
function copyTextToClipboard(text) {
    if (typeof navigator !== 'undefined' && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        return navigator.clipboard.writeText(text).then(() => true).catch(() => false);
    }
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
        ok = document.execCommand('copy');
    } finally {
        document.body.removeChild(ta);
    }
    return Promise.resolve(ok);
}

/** Build basic info text from data attributes (ID, title, tags, type, project, createdBy, created). */
function getExpBasicText(attrs) {
    const id = attrs.expId ?? '';
    const title = attrs.expTitle ?? '';
    const tags = attrs.expTags ?? '';
    const created = attrs.expCreated ?? '';
    const createdBy = attrs.expCreatedBy ?? '';
    const type = attrs.expType ?? '';
    const project = attrs.expProject ?? '';
    const lines = [
        `ID: ${id}`,
        `标题: ${title}`,
        ...(tags ? [`标签: ${tags}`] : []),
        ...(type ? [`类型: ${type}`] : []),
        ...(project ? [`项目: ${project}`] : []),
        ...(createdBy ? [`创建者: ${createdBy}`] : []),
        ...(created ? [`创建时间: ${created}`] : []),
    ];
    return lines.filter(Boolean).join('\n');
}

/** Build full experience text (problem, solution, code, etc.) from API experience object. */
function formatExpFullText(exp) {
    const parts = [];
    parts.push(`# ${exp.title || ''}`);
    parts.push(`ID: ${exp.id}`);
    if (exp.tags && exp.tags.length) parts.push(`标签: ${exp.tags.join(', ')}`);
    if (exp.experience_type) parts.push(`类型: ${exp.experience_type}`);
    if (exp.project) parts.push(`项目: ${exp.project}`);
    if (exp.created_by) parts.push(`创建者: ${exp.created_by}`);
    if (exp.created_at) parts.push(`创建时间: ${exp.created_at}`);
    parts.push('');
    parts.push('## 问题描述');
    parts.push(exp.description || '');
    if (exp.solution) {
        parts.push('');
        parts.push('## 解决方案');
        parts.push(exp.solution);
    }
    if (exp.summary) {
        parts.push('');
        parts.push('## 摘要');
        parts.push(exp.summary);
    }
    if (exp.code_snippets) {
        parts.push('');
        parts.push('## 代码示例');
        parts.push(exp.code_snippets);
    }
    const sd = exp.structured_data || {};
    const sdKeys = Object.keys(sd).filter((k) => sd[k] !== null && sd[k] !== undefined && sd[k] !== '');
    if (sdKeys.length) {
        parts.push('');
        parts.push('## 结构化数据');
        sdKeys.forEach((k) => {
            const v = sd[k];
            parts.push(`${k}: ${Array.isArray(v) ? v.join(', ') : String(v)}`);
        });
    }
    if (exp.children && exp.children.length) {
        parts.push('');
        parts.push('## 子经验');
        exp.children.forEach((child, i) => {
            parts.push(`### ${i + 1}. ${child.title || ''}`);
            if (child.description) parts.push(child.description);
            if (child.solution) parts.push(child.solution);
        });
    }
    return parts.join('\n');
}

/** Copy dropdown HTML (for list/draft/review cards). Pass copyAttrs string to put on the wrapper div. */
function getCopyDropdownHtml(copyAttrs) {
    const svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    return `<div class="exp-copy-dropdown" ${copyAttrs}>
      <button type="button" class="exp-copy-btn" title="复制" aria-haspopup="true" aria-expanded="false"><span class="exp-copy-icon" aria-hidden="true">${svg}</span></button>
      <div class="exp-copy-dropdown-menu" role="menu">
        <button type="button" role="menuitem" data-copy-option="id">复制经验ID</button>
        <button type="button" role="menuitem" data-copy-option="title">复制经验名称</button>
        <button type="button" role="menuitem" data-copy-option="basic">复制基础信息</button>
        <button type="button" role="menuitem" data-copy-option="full">复制全部信息</button>
      </div>
    </div>`;
}

/** Copy dropdown HTML for detail page (uses state.currentDetail, no data-exp-*). */
function getCopyDropdownDetailHtml() {
    const svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    return `<div class="exp-copy-dropdown exp-copy-dropdown-detail">
      <button type="button" class="exp-copy-btn" title="复制" aria-haspopup="true" aria-expanded="false"><span class="exp-copy-icon" aria-hidden="true">${svg}</span></button>
      <div class="exp-copy-dropdown-menu" role="menu">
        <button type="button" role="menuitem" data-copy-option="id">复制经验ID</button>
        <button type="button" role="menuitem" data-copy-option="title">复制经验名称</button>
        <button type="button" role="menuitem" data-copy-option="basic">复制基础信息</button>
        <button type="button" role="menuitem" data-copy-option="full">复制全部信息</button>
      </div>
    </div>`;
}

const COPY_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

/** Copy dropdown for task group card (data-copy-type="taskgroup", data-group-*). */
function getTaskGroupCopyDropdownHtml(attrs) {
    return `<div class="exp-copy-dropdown" data-copy-type="taskgroup" ${attrs}>
      <button type="button" class="exp-copy-btn" title="复制" aria-haspopup="true" aria-expanded="false"><span class="exp-copy-icon" aria-hidden="true">${COPY_SVG}</span></button>
      <div class="exp-copy-dropdown-menu" role="menu">
        <button type="button" role="menuitem" data-copy-option="id">复制任务组ID</button>
        <button type="button" role="menuitem" data-copy-option="title">复制任务组名称</button>
        <button type="button" role="menuitem" data-copy-option="basic">复制基础信息</button>
        <button type="button" role="menuitem" data-copy-option="full">复制全部信息</button>
      </div>
    </div>`;
}

/** Copy dropdown for task card / slideout (data-copy-type="task", data-task-*). */
function getTaskCopyDropdownHtml(attrs) {
    return `<div class="exp-copy-dropdown" data-copy-type="task" ${attrs}>
      <button type="button" class="exp-copy-btn" title="复制" aria-haspopup="true" aria-expanded="false"><span class="exp-copy-icon" aria-hidden="true">${COPY_SVG}</span></button>
      <div class="exp-copy-dropdown-menu" role="menu">
        <button type="button" role="menuitem" data-copy-option="id">复制任务ID</button>
        <button type="button" role="menuitem" data-copy-option="title">复制任务名称</button>
        <button type="button" role="menuitem" data-copy-option="basic">复制基础信息</button>
        <button type="button" role="menuitem" data-copy-option="full">复制全部信息</button>
      </div>
    </div>`;
}

function getTaskGroupBasicText(attrs) {
    const id = attrs.groupId ?? '';
    const title = attrs.groupTitle ?? '';
    const archived = attrs.groupArchived === 'true' ? '是' : '否';
    const total = attrs.groupTotal ?? '';
    const completed = attrs.groupCompleted ?? '';
    const lines = [
        `ID: ${id}`,
        `名称: ${title}`,
        `归档: ${archived}`,
        ...(total ? [`任务数: ${completed}/${total}`] : []),
    ];
    return lines.filter(Boolean).join('\n');
}

function formatTaskGroupFullText(g) {
    const parts = [];
    parts.push(`# ${g.title || ''}`);
    parts.push(`ID: ${g.id}`);
    parts.push(`归档: ${g.archived ? '是' : '否'}`);
    const tasks = g.tasks || [];
    parts.push(`任务数: ${tasks.length}`);
    parts.push('');
    if (tasks.length) {
        parts.push('## 子任务');
        tasks.forEach((t, i) => {
            parts.push(`${i + 1}. [${t.status || ''}] ${t.title || ''}`);
            if (t.description) parts.push(`   ${String(t.description).slice(0, 120)}${t.description.length > 120 ? '...' : ''}`);
        });
    }
    return parts.join('\n');
}

function getTaskBasicText(attrs) {
    const id = attrs.taskId ?? '';
    const title = attrs.taskTitle ?? '';
    const status = attrs.taskStatus ?? '';
    const priority = attrs.taskPriority ?? '';
    const groupId = attrs.taskGroupId ?? '';
    const lines = [
        `ID: ${id}`,
        `标题: ${title}`,
        ...(status ? [`状态: ${status}`] : []),
        ...(priority ? [`优先级: ${priority}`] : []),
        ...(groupId ? [`任务组ID: ${groupId}`] : []),
    ];
    return lines.filter(Boolean).join('\n');
}

function formatTaskFullText(t) {
    const parts = [];
    parts.push(`# ${t.title || ''}`);
    parts.push(`ID: ${t.id}`);
    if (t.group_id) parts.push(`任务组ID: ${t.group_id}`);
    if (t.status) parts.push(`状态: ${t.status}`);
    if (t.priority) parts.push(`优先级: ${t.priority}`);
    if (t.importance != null) parts.push(`重要度: ${t.importance}`);
    if (t.due_date) parts.push(`截止: ${t.due_date}`);
    if (t.labels && t.labels.length) parts.push(`标签: ${t.labels.join(', ')}`);
    parts.push('');
    if (t.description) {
        parts.push('## 描述');
        parts.push(t.description);
    }
    if (t.acceptance_criteria) {
        parts.push('');
        parts.push('## 验收标准');
        parts.push(t.acceptance_criteria);
    }
    return parts.join('\n');
}

async function copyTaskGroupOption(option, dropdownEl) {
    let text = '';
    if (option === 'id') {
        text = dropdownEl.dataset.groupId ?? '';
    } else if (option === 'title') {
        text = dropdownEl.dataset.groupTitle ?? '';
    } else if (option === 'basic') {
        text = getTaskGroupBasicText(dropdownEl.dataset);
    } else if (option === 'full') {
        const id = dropdownEl.dataset.groupId;
        if (!id) {
            toast('无法获取任务组ID', 'error');
            return;
        }
        try {
            const g = await api('GET', `/api/v1/task-groups/${id}`);
            text = formatTaskGroupFullText(g);
        } catch (e) {
            toast('获取任务组失败: ' + e.message, 'error');
            return;
        }
    }
    const ok = await copyTextToClipboard(text);
    if (ok) toast('已复制', 'success');
    else toast('复制失败', 'error');
}

async function copyTaskOption(option, dropdownEl) {
    let text = '';
    if (option === 'id') {
        text = dropdownEl.dataset.taskId ?? '';
    } else if (option === 'title') {
        text = dropdownEl.dataset.taskTitle ?? '';
    } else if (option === 'basic') {
        text = getTaskBasicText(dropdownEl.dataset);
    } else if (option === 'full') {
        const id = dropdownEl.dataset.taskId;
        if (!id) {
            toast('无法获取任务ID', 'error');
            return;
        }
        try {
            const t = await api('GET', `/api/v1/tasks/${id}?with_context=true`);
            text = formatTaskFullText(t);
        } catch (e) {
            toast('获取任务失败: ' + e.message, 'error');
            return;
        }
    }
    const ok = await copyTextToClipboard(text);
    if (ok) toast('已复制', 'success');
    else toast('复制失败', 'error');
}

/** Handle copy option from dropdown (card: dropdown has data-exp-*; detail: use state.currentDetail). */
async function copyExpOption(option, dropdownEl) {
    const isDetail = dropdownEl.classList.contains('exp-copy-dropdown-detail');
    let text = '';
    if (option === 'id') {
        text = isDetail ? String(state.currentDetail?.id ?? '') : (dropdownEl.dataset.expId ?? '');
    } else if (option === 'title') {
        text = isDetail ? (state.currentDetail?.title ?? '') : (dropdownEl.dataset.expTitle ?? '');
    } else if (option === 'basic') {
        const attrs = isDetail
            ? {
                expId: state.currentDetail?.id,
                expTitle: state.currentDetail?.title,
                expTags: (state.currentDetail?.tags || []).join(', '),
                expType: state.currentDetail?.experience_type,
                expProject: state.currentDetail?.project,
                expCreatedBy: state.currentDetail?.created_by,
                expCreated: state.currentDetail?.created_at,
            }
            : dropdownEl.dataset;
        text = getExpBasicText(attrs);
    } else if (option === 'full') {
        if (isDetail && state.currentDetail) {
            text = formatExpFullText(state.currentDetail);
        } else {
            const id = dropdownEl.dataset.expId;
            if (!id) {
                toast('无法获取经验ID', 'error');
                return;
            }
            try {
                const exp = await api('GET', `/api/v1/experiences/${id}`);
                text = formatExpFullText(exp);
            } catch (e) {
                toast('获取经验失败: ' + e.message, 'error');
                return;
            }
        }
    }
    const ok = await copyTextToClipboard(text);
    if (ok) toast('已复制', 'success');
    else toast('复制失败', 'error');
}

/** Close all copy dropdowns. */
function closeAllCopyDropdowns() {
    document.querySelectorAll('.exp-copy-dropdown.open').forEach((el) => el.classList.remove('open'));
}

/** Bind copy dropdown: toggle on trigger, option click copies and closes, click outside closes. */
function bindCopyDropdowns(container) {
    if (!container) return;
    container.querySelectorAll('.exp-copy-dropdown').forEach((dropdown) => {
        const trigger = dropdown.querySelector('.exp-copy-btn');
        const menu = dropdown.querySelector('.exp-copy-dropdown-menu');
        if (!trigger || !menu) return;
        trigger.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const isOpen = dropdown.classList.toggle('open');
            trigger.setAttribute('aria-expanded', isOpen);
        });
        menu.querySelectorAll('[data-copy-option]').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const option = btn.dataset.copyOption;
                const copyType = dropdown.dataset.copyType || 'exp';
                if (copyType === 'taskgroup') {
                    copyTaskGroupOption(option, dropdown);
                } else if (copyType === 'task') {
                    copyTaskOption(option, dropdown);
                } else {
                    copyExpOption(option, dropdown);
                }
                dropdown.classList.remove('open');
                trigger.setAttribute('aria-expanded', 'false');
            });
        });
    });
    if (!container._copyDropdownOutside) {
        container._copyDropdownOutside = true;
        document.addEventListener('click', () => closeAllCopyDropdowns());
    }
}

// ===== Render Experience Cards =====
export function renderExpList(containerId, experiences) {
    const container = document.getElementById(containerId);
    if (!experiences || experiences.length === 0) {
        container.innerHTML = `<div class="empty-state"><div class="icon">📚</div><h3>暂无经验记录</h3><p>点击右上角"新建经验"添加第一条</p></div>`;
        return;
    }
    const tierBadge = (v) => {
        const tier = v.quality_tier || 'bronze';
        const score = v.quality_score ?? 100;
        const colors = { gold: '#FFD700', silver: '#C0C0C0', bronze: '#CD7F32', outdated: '#888' };
        const bg = { gold: '#FFF8E1', silver: '#F5F5F5', bronze: '#FFF3E0', outdated: '#F5F5F5' };
        return `<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:${bg[tier]||bg.bronze};color:${colors[tier]||colors.bronze};font-weight:600;margin-left:4px">${tier.charAt(0).toUpperCase()+tier.slice(1)} ${score}</span>`;
    };
    const pinBadge = (v) => v.pinned ? '<span style="font-size:10px;margin-left:4px" title="已置顶">📌</span>' : '';
    container.innerHTML = experiences
        .map((exp) => {
            const view = exp.parent || exp;
            const cardId = exp.group_id || view.id || exp.id || '';
            const isStale = view.last_used_at && isStaleDate(view.last_used_at);
            const typeIcon = typeIcons[view.experience_type] || defaultTypeIcons[view.experience_type] || '📝';
            const matchedNodes = (exp.matched_nodes || [])
                .slice(0, 2)
                .map((n) => `<span class="tag" style="background:var(--accent-glow);color:var(--accent)">#${esc(n.path || '')} ${esc(n.node_title || '')}</span>`)
                .join('');
            const treeScore =
                exp.tree_score !== undefined
                    ? `<span class="tag" style="background:var(--accent-glow);color:var(--accent)">tree ${(Number(exp.tree_score) * 100).toFixed(0)}%</span>`
                    : '';
            const projectTag = view.project
                ? `<span class="tag" style="background:var(--bg-input);color:var(--text-muted);font-size:11px">📁 ${esc(view.project)}</span>`
                : '';
            const viewCount = view.view_count || 0;
            const useCount = view.use_count || 0;
            const avgRating = view.avg_rating || 0;
            const ratingDisplay = avgRating > 0 ? `★ ${avgRating.toFixed(1)}` : '★ -';
            const metricsHtml = `<div class="card-metrics"><span>${ratingDisplay}</span><span>👁 ${viewCount}</span><span>📊 ${useCount}</span></div>`;
            const tagsStr = (view.tags || []).join(', ');
            const copyId = String(cardId ?? '');
            const copyAttrs = `data-exp-id="${esc(copyId)}" data-exp-title="${esc(view.title || '')}" data-exp-tags="${esc(tagsStr)}" data-exp-created="${esc(view.created_at || '')}" data-exp-created-by="${esc(view.created_by || '')}" data-exp-type="${esc(view.experience_type || 'general')}" data-exp-project="${esc(view.project || '')}"`;
            return `
    <div class="exp-card" onclick="showDetail('${cardId}')">
      <div class="exp-card-header">
        <div class="exp-card-title">
          <span class="type-icon">${typeIcon}</span>${esc(view.title)}${tierBadge(view)}${pinBadge(view)}${isStale ? '<span class="stale-badge">疑似过时</span>' : ''}${view.status === 'draft' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--accent-glow);color:var(--accent);margin-left:6px">草稿</span>' : ''}${view.status === 'review' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--yellow-bg);color:var(--yellow);margin-left:6px">审核中</span>' : ''}${view.status === 'rejected' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--red-bg);color:var(--red);margin-left:6px">已拒绝</span>' : ''}${view.visibility === 'private' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--green-bg,#e8f5e9);color:var(--green,#2e7d32);margin-left:6px">仅自己</span>' : ''}${view.visibility === 'global' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:#e0f2fe;color:#0369a1;margin-left:6px">全局</span>' : ''}
        </div>
        <div class="exp-card-meta">
          ${projectTag}
          ${exp.similarity !== undefined ? `<span class="similarity-badge">${(exp.similarity * 100).toFixed(0)}%</span>` : ''}
          <span>${timeAgo(view.created_at)}</span>
          ${getCopyDropdownHtml(copyAttrs)}
        </div>
      </div>
      <div class="exp-card-desc">${esc(view.description || '')}</div>
      ${matchedNodes || treeScore ? `<div style="margin-bottom:8px;display:flex;gap:6px;flex-wrap:wrap">${treeScore}${matchedNodes}</div>` : ''}
      <div class="exp-card-footer">
        <div class="exp-card-tags">${view.visibility === 'global' ? '<span class="tag" style="background:#e0f2fe;color:#0369a1;font-weight:600">全局</span>' : ''}${view.visibility === 'private' ? '<span class="tag" style="background:#f3e8ff;color:#7c3aed;font-weight:600">仅自己</span>' : ''}${(view.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}${view.category ? `<span class="tag" style="background:var(--bg-input);color:var(--text-muted)">${esc(view.category)}</span>` : ''}${exp.children_count > 0 || exp.total_children > 0 ? `<span class="children-badge">${exp.children_count || exp.total_children} steps</span>` : ''}</div>
        <div style="display:flex;align-items:center;gap:12px">
          ${metricsHtml}
          <span style="font-size:12px;color:var(--text-muted)">${esc(view.created_by || '')}</span>
        </div>
      </div>
    </div>
  `;
        })
        .join('');
    bindCopyDropdowns(container);
}

// ===== Dashboard (merged into list page) =====
export async function loadDashboard() {
    loadList(1);
}

// ===== List Sub-tab State =====
let _listSubTab = 'all'; // 'all' | 'draft' | 'review'

export function switchListSubTab(tab) {
    _listSubTab = tab;
    document.querySelectorAll('#page-list .mode-tab').forEach((el) => el.classList.remove('active'));
    const tabEl = document.getElementById(`list-tab-${tab}`);
    if (tabEl) tabEl.classList.add('active');
    const statusFilter = document.getElementById('list-status-filter');
    if (statusFilter) {
        if (tab === 'draft') statusFilter.value = 'draft';
        else if (tab === 'review') statusFilter.value = 'review';
        else statusFilter.value = '';
    }
    loadList(1);
}

// ===== Experience List =====
export async function loadList(page = 1) {
    state.listPage = page;
    const container = document.getElementById('list-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    const multiProjects = window.getSelectedProjects ? window.getSelectedProjects('list') : [];
    const projectFilter = multiProjects.length > 0
        ? multiProjects.join(',')
        : (state.activeProject || state.defaultProject || 'default');

    // Load stats for the top cards
    try {
        const stats = await api(
            'GET',
            `/api/v1/stats?project=${encodeURIComponent(projectFilter)}`
        );
        const el = (id) => document.getElementById(id);
        if (el('stat-total')) el('stat-total').textContent = stats.total_experiences || 0;
        if (el('stat-recent')) el('stat-recent').textContent = stats.recent_7days || 0;
        if (el('stat-pending')) el('stat-pending').textContent = stats.pending_reviews || 0;
        const tags = stats.tag_distribution || {};
        if (el('stat-tags')) el('stat-tags').textContent = Object.keys(tags).length;
    } catch (_) { /* stats load failure is non-blocking */ }

    try {
        const statusFilter = document.getElementById('list-status-filter')?.value || '';
        const typeFilter = document.getElementById('list-type-filter')?.value || '';
        const tierFilter = document.getElementById('list-tier-filter')?.value || '';
        const visibilityFilter = document.getElementById('list-visibility-filter')?.value || '';
        let url = `/api/v1/experiences?page=${page}&page_size=15`;
        if (projectFilter) url += `&project=${encodeURIComponent(projectFilter)}`;
        if (visibilityFilter) url += `&visibility=${encodeURIComponent(visibilityFilter)}`;
        if (statusFilter) url += `&status=${statusFilter}`;
        if (state.selectedTag) url += `&tag=${encodeURIComponent(state.selectedTag)}`;
        if (typeFilter) url += `&experience_type=${encodeURIComponent(typeFilter)}`;
        if (tierFilter) url += `&quality_tier=${encodeURIComponent(tierFilter)}`;

        let tagUrl = `/api/v1/tags?project=${encodeURIComponent(projectFilter)}`;
        if (visibilityFilter) tagUrl += `&visibility=${encodeURIComponent(visibilityFilter)}`;
        const [data, tagData] = await Promise.all([
            api('GET', url),
            api('GET', tagUrl),
        ]);
        renderExpList('list-content', data.experiences);
        renderPagination(data);

        const tags = tagData.tags || {};
        state.allTags = tags;
        const bar = document.getElementById('list-tags-bar');
        if (Object.keys(tags).length > 0) {
            const tagEntries = Object.entries(tags).sort((a, b) => b[1] - a[1]);
            bar.innerHTML =
                '<span class="tag-label">标签筛选:</span>' +
                `<span class="tag" onclick="filterByTag(null)" style="${!state.selectedTag ? 'background:var(--accent);color:#fff' : ''}">全部</span>` +
                tagEntries
                    .map(
                        ([tag, cnt]) =>
                            `<span class="tag" onclick="filterByTag('${tag}')" style="${state.selectedTag === tag ? 'background:var(--accent);color:#fff' : ''}">${tag} (${cnt})</span>`
                    )
                    .join('');
        } else {
            bar.innerHTML = '';
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${e.message}</p></div>`;
    }
}

export function filterByTag(tag) {
    state.selectedTag = tag;
    state.listPage = 1;
    navigate('list');
}

function renderPagination(data) {
    const el = document.getElementById('list-pagination');
    if (data.total_pages <= 1) {
        el.innerHTML = '';
        return;
    }
    el.innerHTML = `
    <button class="btn btn-secondary btn-sm" onclick="loadList(${data.page - 1})" ${data.page <= 1 ? 'disabled' : ''}>上一页</button>
    <span class="page-info">${data.page} / ${data.total_pages}</span>
    <button class="btn btn-secondary btn-sm" onclick="loadList(${data.page + 1})" ${data.page >= data.total_pages ? 'disabled' : ''}>下一页</button>
  `;
}

// ===== Detail View =====
export const viewDetail = (id) => showDetail(id);

/** Back from current detail: either to previous experience (from stack) or to referrer page (list/search etc.). */
export function backToPreviousDetail() {
    if ((state.detailBackStack || []).length > 0) {
        const id = state.detailBackStack.pop();
        showDetail(id, { isBack: true });
    } else {
        navigate(state.detailReferrer || 'list');
    }
}

export async function showDetail(id, opts = {}) {
    if (state.currentPage !== 'detail') {
        state.detailBackStack = [];
    } else if (!opts.isBack && state.currentDetail?.id && state.currentDetail.id !== id) {
        (state.detailBackStack = state.detailBackStack || []).push(state.currentDetail.id);
    }
    state.detailReferrer = state.currentPage || 'list';
    state.currentPage = 'detail';
    if (location.hash !== '#detail/' + id) {
        history.pushState(null, '', location.pathname + '#detail/' + id);
    }
    document.querySelectorAll('.page').forEach((p) => p.classList.add('hidden'));
    document.querySelectorAll('.topbar-nav a').forEach((a) => a.classList.remove('active'));
    if (state.detailReferrer === 'search') {
        document.querySelectorAll('.topbar-nav a[data-page="search"]').forEach((a) => a.classList.add('active'));
    }
    const page = document.getElementById('page-detail');
    page.classList.remove('hidden');
    page.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const [exp, linksRes] = await Promise.all([
            api('GET', `/api/v1/experiences/${id}`),
            api('GET', `/api/v1/experiences/${id}/links`).catch(() => ({ links: [] })),
        ]);
        const links = linksRes.links || [];
        state.currentDetail = exp;
        const typeIcon = typeIcons[exp.experience_type] || defaultTypeIcons[exp.experience_type] || '📝';
        const dTier = exp.quality_tier || 'bronze';
        const dScore = exp.quality_score ?? 100;
        const tierColors = { gold: '#FFD700', silver: '#C0C0C0', bronze: '#CD7F32', outdated: '#888' };
        const tierBg = { gold: '#FFF8E1', silver: '#F5F5F5', bronze: '#FFF3E0', outdated: '#F5F5F5' };
        const tierLabel = `<span style="font-size:12px;padding:2px 8px;border-radius:4px;background:${tierBg[dTier]||tierBg.bronze};color:${tierColors[dTier]||tierColors.bronze};font-weight:600;margin-left:8px">${dTier.charAt(0).toUpperCase()+dTier.slice(1)} ${dScore}</span>`;
        const pinnedLabel = exp.pinned ? '<span style="margin-left:4px" title="已置顶，不衰减">📌</span>' : '';
        const typeBadges = `<span class="type-icon" style="font-size:20px">${typeIcon}</span>${tierLabel}${pinnedLabel}`;
        const compBar =
            exp.completeness_score != null
                ? `<div class="completeness-bar" style="max-width:120px" title="完整度 ${exp.completeness_score}%"><div class="completeness-bar-fill" style="width:${exp.completeness_score}%"></div></div>`
                : '';

        const sd = exp.structured_data || {};
        const sdKeys = Object.keys(sd).filter((k) => sd[k] !== null && sd[k] !== undefined && sd[k] !== '');
        const sdHtml =
            sdKeys.length > 0
                ? `
      <div class="detail-section">
        <h3 class="detail-collapsible expanded" onclick="this.parentElement.querySelector('.detail-collapsible-content').classList.toggle('hidden');this.classList.toggle('expanded')">结构化数据 <span class="toggle-arrow">▸</span></h3>
        <div class="detail-collapsible-content">
          ${sdKeys
              .map((k) => {
                  const v = sd[k];
                  const disp = Array.isArray(v) ? v.join('\n') : String(v);
                  return `<div class="form-group" style="margin-bottom:12px"><strong style="font-size:12px;color:var(--text-muted)">${esc(k)}:</strong><div class="content" style="margin-top:4px;white-space:pre-wrap">${esc(disp)}</div></div>`;
              })
              .join('')}
        </div>
      </div>`
                : '';

        const gitRefsHtml =
            exp.git_refs && exp.git_refs.length > 0
                ? `
      <div class="detail-section">
        <h3>Git 引用</h3>
        <div class="content">
          ${exp.git_refs
              .map((r) => {
                  const url = r.url || (r.hash ? `#${r.hash}` : '');
                  const label = r.description || r.hash || url || r.type;
                  return url ? `<a href="${esc(url)}" target="_blank" rel="noopener" style="color:var(--accent);margin-right:12px">${esc(label)}</a>` : `<span style="margin-right:12px">${esc(label)}</span>`;
              })
              .join('')}
        </div>
      </div>`
                : '';

        const relatedLinksHtml =
            exp.related_links && exp.related_links.length > 0
                ? `
      <div class="detail-section">
        <h3>相关链接</h3>
        <div class="content">
          ${exp.related_links
              .map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener" style="color:var(--accent);display:block;margin-bottom:4px">${esc(l.title || l.url)}</a>`)
              .join('')}
        </div>
      </div>`
                : '';

        const linkTypeLabels = { related: '相关', supersedes: '取代', derived_from: '衍生自' };
        const experienceLinksHtml =
            links.length > 0
                ? `
      <div class="detail-section">
        <h3>关联经验 (${links.length})</h3>
        <ul class="related-exp-list">
          ${links
                .map((l) => {
                    const typeLabel = linkTypeLabels[l.link_type] || l.link_type;
                    const title = esc(l.other_title || l.other_id);
                    const titleAttr = esc((l.other_title || l.other_id).toString().replace(/"/g, '&quot;'));
                    return `<li><a class="related-exp-pill" href="#detail/${esc(l.other_id)}" onclick="showDetail('${esc(l.other_id)}');return false" title="${titleAttr}">${title}<span class="related-exp-pill-type">${typeLabel}</span></a></li>`;
                })
                .join('')}
        </ul>
      </div>`
                : '';

        const backPage = state.detailReferrer || 'list';
        const backLabels = { reviews: '审核队列', drafts: '草稿箱', list: '经验列表', search: '语义搜索', dashboard: '仪表盘' };
        const backLabel = backLabels[backPage] || '列表';
        const hasBackStack = (state.detailBackStack || []).length > 0;
        const backBtnLabel = hasBackStack ? '返回上一经验' : `返回${backLabel}`;
        const backBtnOnclick = hasBackStack ? 'backToPreviousDetail()' : `navigate('${backPage}')`;
        page.innerHTML = `
      <button type="button" class="back-btn" onclick="${backBtnOnclick}">← ${backBtnLabel}</button>
      <div class="detail-view">
        <div class="detail-header">
          <h1>${typeBadges} ${esc(exp.title)}
            ${exp.status === 'draft' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--accent-glow);color:var(--accent);margin-left:12px;vertical-align:middle">草稿</span>' : ''}
            ${exp.status === 'review' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--yellow-bg);color:var(--yellow);margin-left:12px;vertical-align:middle">审核中</span>' : ''}
            ${exp.status === 'published' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--green-bg,#e8f5e9);color:var(--green,#2e7d32);margin-left:12px;vertical-align:middle">已发布</span>' : ''}
            ${exp.status === 'rejected' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--red-bg);color:var(--red);margin-left:12px;vertical-align:middle">已拒绝</span>' : ''}
            ${exp.visibility === 'private' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#f3e8ff;color:#7c3aed;margin-left:6px;vertical-align:middle">仅自己</span>' : ''}
            ${exp.visibility === 'global' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#e0f2fe;color:#0369a1;margin-left:6px;vertical-align:middle">全局</span>' : ''}
            ${exp.visibility === 'project' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#fef3c7;color:#92400e;margin-left:6px;vertical-align:middle">项目内</span>' : ''}
          </h1>
          <div class="detail-meta" style="align-items:center">
            ${compBar ? `<span>${compBar}</span>` : ''}
            <span>👤 ${esc(exp.created_by)}</span>
            <span>📅 ${formatDate(exp.created_at)}</span>
            <span>👁 ${exp.view_count} 次查看</span>
            <span>📊 ${exp.use_count} 次引用</span>
            ${exp.avg_rating > 0 ? `<span>★ ${exp.avg_rating.toFixed(1)} 评分</span>` : ''}
            ${exp.programming_language ? `<span>🔧 ${esc(exp.programming_language)}</span>` : ''}
            ${exp.framework ? `<span>📦 ${esc(exp.framework)}</span>` : ''}
            ${getCopyDropdownDetailHtml()}
          </div>
          <div style="margin-top:12px">${(exp.tags || []).map((t) => `<span class="tag" onclick="filterByTag('${esc(t)}')">${esc(t)}</span>`).join('')}</div>
        </div>
        <div class="detail-body">
          ${exp.summary ? `
          <div class="detail-section" style="background:var(--accent-glow);border:1px solid rgba(59,130,246,0.2);border-radius:var(--radius);padding:12px 16px;margin-bottom:12px">
            <h3 style="color:var(--accent);font-size:13px;margin-bottom:4px">摘要</h3>
            <div class="content" style="font-size:14px">${esc(exp.summary)}</div>
          </div>
          ` : ''}
          <div class="detail-section">
            <h3>问题描述</h3>
            <div class="content">${esc(exp.description)}</div>
          </div>
          ${exp.solution ? `
          <div class="detail-section">
            <h3>解决方案</h3>
            <div class="content">${esc(exp.solution)}</div>
          </div>
          ` : ''}
          ${sdHtml}
          ${gitRefsHtml}
          ${relatedLinksHtml}
          ${exp.code_snippets ? `
          <div class="detail-section">
            <h3>代码示例</h3>
            <div class="code-block">${esc(exp.code_snippets)}</div>
          </div>
          ` : ''}
          ${exp.feedbacks && exp.feedbacks.length > 0 ? `
          <div class="detail-section">
            <h3>反馈 (${exp.feedbacks.length})</h3>
            <div class="feedback-list">
              ${exp.feedbacks
                  .map(
                      (fb) => `
                <div class="feedback-item">
                  <div class="fb-header">
                    <span class="fb-stars">${'★'.repeat(fb.rating || 0)}${'☆'.repeat(5 - (fb.rating || 0))}</span>
                    <span style="color:var(--text-muted);font-size:12px">${esc(fb.feedback_by)} · ${timeAgo(fb.created_at)}</span>
                  </div>
                  ${fb.comment ? `<div class="fb-comment">${esc(fb.comment)}</div>` : ''}
                </div>
              `
                  )
                  .join('')}
            </div>
          </div>
          ` : ''}
          ${exp.children && exp.children.length > 0 ? `
          <div class="detail-section">
            <h3>子经验 (${exp.children.length})</h3>
            <div class="children-list">
              ${exp.children
                  .map(
                      (child, idx) => `
                <div class="child-item">
                  <div class="child-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <span class="child-idx">${idx + 1}</span>
                    <span class="child-title">${esc(child.title)}</span>
                    <span class="child-toggle">▸</span>
                  </div>
                  <div class="child-body">
                    <div class="child-field"><strong>问题：</strong>${esc(child.description || '')}</div>
                    ${child.solution ? `<div class="child-field"><strong>方案：</strong>${esc(child.solution)}</div>` : ''}
                    ${child.code_snippets ? `<div class="child-field"><strong>代码：</strong><pre>${esc(child.code_snippets)}</pre></div>` : ''}
                  </div>
                </div>
              `
                  )
                  .join('')}
            </div>
          </div>
          ` : ''}
          ${experienceLinksHtml}
          <div class="detail-section">
            <h3 style="cursor:pointer" onclick="toggleVersionHistory('${exp.id}')">版本历史 <span id="version-toggle-arrow" style="font-size:11px">▸</span></h3>
            <div id="version-history-panel" class="hidden">
              <div id="version-list" class="version-list">
                <div class="loading"><div class="spinner"></div></div>
              </div>
            </div>
          </div>
        </div>
        <div class="detail-actions">
          ${exp.status === 'draft' ? `
            <button class="btn btn-sm" style="background:var(--yellow);color:#fff" onclick="changeExpStatus('${exp.id}','review')">提交审核</button>
            <button class="btn btn-sm" style="background:var(--green);color:#fff;margin-left:4px" onclick="changeExpStatus('${exp.id}','published')">直接发布</button>` : ''}
          ${exp.status === 'review' ? `
            <button class="btn btn-sm" style="background:var(--green);color:#fff" onclick="changeExpStatus('${exp.id}','published')">批准发布</button>
            <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red);margin-left:4px" onclick="changeExpStatus('${exp.id}','rejected')">拒绝</button>` : ''}
          ${exp.status === 'rejected' ? `
            <button class="btn btn-sm" style="background:var(--accent);color:#fff" onclick="changeExpStatus('${exp.id}','draft')">退回草稿</button>` : ''}
          ${exp.status === 'published' ? `
            <button class="btn btn-sm" style="background:var(--accent-glow);color:var(--accent)" onclick="changeExpStatus('${exp.id}','draft')">撤回到草稿</button>` : ''}
          <button class="btn btn-primary btn-sm" onclick="openEditModal('${exp.id}')">✏️ 编辑</button>
          <button class="btn btn-primary btn-sm" onclick="openFeedbackModal('${exp.id}')">💬 提交反馈</button>
          ${!exp.summary ? `<button class="btn btn-sm" style="background:var(--accent-glow);color:var(--accent)" onclick="generateSummary('${exp.id}')">📝 生成摘要</button>` : ''}
          <button class="btn btn-danger btn-sm" onclick="deleteExp('${exp.id}')">🗑 删除</button>
          <div style="flex:1"></div>
          <span style="font-size:12px;color:var(--text-muted)">ID: ${exp.id}</span>
        </div>
      </div>
    `;
        bindCopyDropdowns(page);
    } catch (e) {
        page.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${e.message}</p></div>`;
    }
}

// ===== Drafts =====
export async function loadDrafts() {
    const container = document.getElementById('drafts-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const project = state.activeProject || state.defaultProject || 'default';
        const data = await api('GET', `/api/v1/experiences/drafts?page=1&page_size=50&project=${encodeURIComponent(project)}`);
        if (!data.experiences || data.experiences.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="icon">📝</div><h3>暂无草稿</h3><p>创建经验时勾选"保存为草稿"即可</p></div>';
            return;
        }
        container.innerHTML = data.experiences
            .map(
                (exp) => {
                    const tagsStr = (exp.tags || []).join(', ');
                    const copyAttrs = `data-exp-id="${esc(exp.id)}" data-exp-title="${esc(exp.title)}" data-exp-tags="${esc(tagsStr)}" data-exp-created="${esc(exp.created_at || '')}" data-exp-created-by="${esc(exp.created_by || '')}" data-exp-type="${esc(exp.experience_type || 'general')}" data-exp-project="${esc(exp.project || '')}"`;
                    return `
      <div class="exp-card" onclick="viewDetail('${exp.id}')">
        <div class="exp-card-header">
          <div class="exp-card-title">${esc(exp.title)} <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--accent-glow);color:var(--accent)">草稿</span></div>
          <div class="exp-card-meta"><span>${timeAgo(exp.created_at)}</span>${getCopyDropdownHtml(copyAttrs)}</div>
        </div>
        <div class="exp-card-desc">${esc((exp.description || '').substring(0, 120))}${(exp.description || '').length > 120 ? '...' : ''}</div>
        <div class="exp-card-footer">
          <div class="exp-card-tags">${(exp.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="display:flex;gap:6px">
            <button class="btn btn-sm" style="background:var(--green);color:#fff;font-size:11px;padding:2px 10px" onclick="event.stopPropagation();changeExpStatus('${exp.id}','published')">发布</button>
            <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red);font-size:11px;padding:2px 10px" onclick="event.stopPropagation();deleteExp('${exp.id}')">删除</button>
          </div>
        </div>
      </div>
    `;
                }
            )
            .join('');
        bindCopyDropdowns(container);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载草稿失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export async function changeExpStatus(id, newStatus, newVisibility = null) {
    const labels = { draft: '草稿', review: '审核中', published: '已发布', rejected: '已拒绝' };
    const label = labels[newStatus] || newStatus;
    if (!confirm(`确定要将状态改为「${label}」吗？`)) return;
    try {
        const body = { status: newStatus };
        if (newVisibility) body.visibility = newVisibility;
        const res = await api('POST', `/api/v1/experiences/${id}/status`, body);
        toast(res.message || '操作成功', 'success');
        showDetail(id);
    } catch (e) {
        toast('状态变更失败: ' + e.message, 'error');
    }
}

export async function publishDraft(id, target = 'personal') {
    const newStatus = target === 'team' ? 'review' : 'published';
    const newVis = target === 'team' ? 'project' : 'private';
    await changeExpStatus(id, newStatus, newVis);
}

// ===== Reviews =====
export async function loadReviews() {
    const container = document.getElementById('reviews-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', '/api/v1/reviews/pending');
        if (!data.experiences || data.experiences.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon"></div><div class="empty-text">暂无待审核经验</div></div>';
            return;
        }
        container.innerHTML = data.experiences
            .map(
                (exp) => {
                    const tagsStr = (exp.tags || []).join(', ');
                    const copyAttrs = `data-exp-id="${esc(exp.id)}" data-exp-title="${esc(exp.title)}" data-exp-tags="${esc(tagsStr)}" data-exp-created="${esc(exp.created_at || '')}" data-exp-created-by="${esc(exp.created_by || '')}" data-exp-type="${esc(exp.experience_type || 'general')}" data-exp-project="${esc(exp.project || '')}"`;
                    return `
      <div class="exp-card" onclick="viewDetail('${exp.id}')" style="cursor:pointer">
        <div class="exp-card-header">
          <div class="exp-card-title">${esc(exp.title)} <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--yellow-bg);color:var(--yellow)">待审核</span></div>
          <div class="exp-card-meta"><span>来源: ${exp.source || 'unknown'}</span><span>${timeAgo(exp.created_at)}</span>${getCopyDropdownHtml(copyAttrs)}</div>
        </div>
        <div class="exp-card-desc">${esc((exp.description || '').substring(0, 200))}</div>
        <div style="display:flex;gap:8px;margin-top:8px" onclick="event.stopPropagation()">
          <button class="btn btn-sm" style="background:var(--green);color:#fff;font-size:12px;padding:4px 16px"
            onclick="changeExpStatus('${exp.id}', 'published')">批准并发布</button>
          <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red);font-size:12px;padding:4px 16px"
            onclick="changeExpStatus('${exp.id}', 'rejected')">退回</button>
        </div>
        <div class="exp-card-footer" style="margin-top:8px">
          <div class="exp-card-tags">${(exp.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <span style="font-size:12px;color:var(--text-muted)">${esc(exp.created_by || '')}</span>
        </div>
      </div>
    `;
                }
            )
            .join('');
        bindCopyDropdowns(container);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载审核队列失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export async function reviewExperience(id, status) {
    const newStatus = status === 'approved' ? 'published' : 'rejected';
    await changeExpStatus(id, newStatus);
}

// ===== Version History =====
export async function toggleVersionHistory(expId) {
    const panel = document.getElementById('version-history-panel');
    const arrow = document.getElementById('version-toggle-arrow');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        arrow.textContent = '▾';
        await loadVersionHistory(expId);
    } else {
        panel.classList.add('hidden');
        arrow.textContent = '▸';
    }
}

async function loadVersionHistory(expId) {
    const container = document.getElementById('version-list');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', `/api/v1/experiences/${expId}/versions`);
        if (!data.versions || data.versions.length === 0) {
            container.innerHTML = '<div style="padding:12px;color:var(--text-muted);font-size:13px">暂无版本历史</div>';
            return;
        }
        container.innerHTML = data.versions
            .map(
                (v) => `
      <div class="version-item" onclick="toggleVersionSnapshot(this)">
        <div class="version-info">
          <span class="ver-num">v${v.version_number}</span>
          <span class="ver-meta">${esc(v.changed_by)} · ${timeAgo(v.created_at)}${v.change_summary ? ' · ' + esc(v.change_summary) : ''}</span>
        </div>
        <div class="version-actions">
          <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation();viewVersionSnapshot('${v.id}')">查看</button>
          <button class="btn btn-primary btn-sm" onclick="event.stopPropagation();rollbackVersion('${expId}','${v.id}',${v.version_number})">回滚</button>
        </div>
      </div>
      <div class="version-snapshot hidden" id="snap-${v.id}"></div>
    `
            )
            .join('');
    } catch (e) {
        container.innerHTML = '<div style="padding:12px;color:var(--red);font-size:13px">加载版本历史失败</div>';
    }
}

export async function viewVersionSnapshot(versionId) {
    const snapEl = document.getElementById('snap-' + versionId);
    if (!snapEl) return;
    if (!snapEl.classList.contains('hidden')) {
        snapEl.classList.add('hidden');
        return;
    }
    snapEl.classList.remove('hidden');
    snapEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', `/api/v1/experiences/x/versions/${versionId}`);
        const s = data.snapshot || {};
        let html = '';
        if (s.title) html += `<div class="snap-field"><span class="snap-label">标题</span><div>${esc(s.title)}</div></div>`;
        if (s.description) html += `<div class="snap-field"><span class="snap-label">问题描述</span><div style="white-space:pre-wrap">${esc(s.description)}</div></div>`;
        if (s.solution) html += `<div class="snap-field"><span class="snap-label">解决方案</span><div style="white-space:pre-wrap">${esc(s.solution)}</div></div>`;
        if (s.root_cause) html += `<div class="snap-field"><span class="snap-label">根因</span><div>${esc(s.root_cause)}</div></div>`;
        if (s.tags && s.tags.length > 0) html += `<div class="snap-field"><span class="snap-label">标签</span><div>${s.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div></div>`;
        if (s.code_snippets) html += `<div class="snap-field"><span class="snap-label">代码</span><div class="code-block">${esc(s.code_snippets)}</div></div>`;
        if (s.children && s.children.length > 0) {
            html += `<div class="snap-field"><span class="snap-label">子经验 (${s.children.length})</span>`;
            s.children.forEach((c, i) => {
                html += `<div style="margin:8px 0;padding:8px;border:1px solid var(--border);border-radius:6px"><strong>${i + 1}. ${esc(c.title)}</strong><div style="font-size:13px;color:var(--text-secondary);margin-top:4px">${esc(c.solution || '')}</div></div>`;
            });
            html += '</div>';
        }
        snapEl.innerHTML = html || '<div style="color:var(--text-muted)">快照为空</div>';
    } catch (e) {
        snapEl.innerHTML = `<div style="color:var(--red)">加载快照失败: ${e.message}</div>`;
    }
}

export function toggleVersionSnapshot(el) {
    const next = el.nextElementSibling;
    if (next && next.classList.contains('version-snapshot')) {
        next.classList.toggle('hidden');
    }
}

export async function rollbackVersion(expId, versionId, verNum) {
    if (!confirm(`确定要回滚到版本 v${verNum} 吗？当前内容将被替换为该版本的快照。`)) return;
    try {
        await api('POST', `/api/v1/experiences/${expId}/rollback/${versionId}`);
        toast('回滚成功', 'success');
        showDetail(expId);
    } catch (e) {
        toast('回滚失败: ' + e.message, 'error');
    }
}

// ===== Usage Stats =====
export async function loadUsageStats() {
    const container = document.getElementById('usage-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const selectedProjects = typeof window.getSelectedProjects === 'function' ? window.getSelectedProjects('usage') : [];
        const projectForSkills = selectedProjects.length ? selectedProjects[0] : (state.activeProject || state.defaultProject || 'default');
        const projectQs = selectedProjects.length ? selectedProjects.map((p) => 'project=' + encodeURIComponent(p)).join('&') : '';
        const suffix = projectQs ? '&' + projectQs : '';
        const results = await Promise.allSettled([
            api('GET', projectQs ? '/api/v1/analytics/tool-usage/summary?' + projectQs : '/api/v1/analytics/tool-usage/summary'),
            api('GET', '/api/v1/analytics/tool-usage?group_by=tool' + (suffix ? suffix : '')),
            api('GET', '/api/v1/analytics/tool-usage?group_by=user' + (suffix ? suffix : '')),
            api('GET', '/api/v1/analytics/tool-usage?group_by=api_key' + (suffix ? suffix : '')),
            api('GET', `/api/v1/analytics/skills-rules?project=${encodeURIComponent(projectForSkills)}`),
        ]);
        const val = (r, fallback) => r.status === 'fulfilled' ? r.value : fallback;
        const summary = val(results[0], { top_tools: [], total_calls: 0 });
        const byTool = val(results[1], { data: [] });
        const byUser = val(results[2], { data: [] });
        const byApiKey = val(results[3], { data: [] });
        const skillsRules = val(results[4], { categories: {}, total_files: 0, workspace: '' });
        const maxCount = Math.max(...(byTool.data || []).map(t => t.count), 1);
        const toolRows = (byTool.data || []).slice(0, 15).map(t => {
            const pct = Math.round((t.count / maxCount) * 100);
            const typeLabel = t.tool_type === 'skill' ? '🎯' : '🔧';
            return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <span style="min-width:24px">${typeLabel}</span>
                <span style="min-width:160px;font-size:13px">${esc(t.tool_name)}</span>
                <div style="flex:1;background:var(--bg-secondary);border-radius:4px;height:20px;overflow:hidden">
                    <div style="width:${pct}%;background:var(--accent);height:100%;border-radius:4px;transition:width .3s"></div>
                </div>
                <span style="min-width:60px;text-align:right;font-size:13px;font-weight:500">${t.count} 次</span>
                <span style="min-width:80px;text-align:right;font-size:12px;color:var(--text-muted)">${t.avg_duration_ms ?? 0}ms</span>
                ${(t.errors ?? 0) > 0 ? `<span style="color:var(--red);font-size:12px">${t.errors} 错误</span>` : ''}
            </div>`;
        }).join('');

        const userRows = (byUser.data || []).map(u => `
            <tr><td>${esc(u.user ?? '匿名')}</td><td style="text-align:right">${u.count}</td><td style="text-align:right">${u.avg_duration_ms ?? 0}ms</td></tr>
        `).join('');

        const apiKeyRows = (byApiKey.data || []).map(k => `
            <tr><td>${esc(k.api_key_name)}</td><td style="text-align:right">${k.count}</td><td style="text-align:right">${k.avg_duration_ms ?? 0}ms</td><td style="text-align:right">${k.errors ?? 0}</td></tr>
        `).join('');

        const catLabels = {
            claude_skills: { icon: '🎯', label: 'Claude Skills (项目)' },
            cursor_rules: { icon: '📏', label: 'Cursor Rules (项目)' },
            cursor_prompts: { icon: '💬', label: 'Cursor Prompts (项目)' },
            cursor_skills: { icon: '🔧', label: 'Cursor Skills (项目)' },
            user_claude_skills: { icon: '🎯', label: 'Claude Skills (用户级)' },
            user_cursor_skills: { icon: '🔧', label: 'Cursor Skills (用户级)' },
        };
        const cats = skillsRules.categories || {};
        const skillsHtml = Object.entries(cats).map(([catKey, cat]) => {
            const meta = catLabels[catKey] || { icon: '📄', label: catKey };
            if (cat.count === 0) return '';
            const files = cat.files || [];
            const visible = files.slice(0, 8);
            const rest = files.slice(8);
            const fileCard = (f) => {
                const isEnabled = f.enabled !== false;
                const attrEsc = (s) => String(s || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                const fullPath = attrEsc(f.full_path);
                const dirPath = attrEsc(f.dir_path);
                const summary = (f.summary || '').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                return `<div class="sr-card${!isEnabled ? ' disabled' : ''}" data-enabled="${isEnabled}">
                    <div class="sr-card-header">
                        <span class="sr-card-name">${esc(f.name)}</span>
                        <label class="sr-toggle" onclick="event.stopPropagation();toggleSkillFile('${esc(catKey)}','${esc(f.path)}',this)">
                            <span class="sr-toggle-bg" style="background:${isEnabled ? 'var(--green)' : 'var(--border)'}"></span>
                            <span class="sr-toggle-knob" style="${isEnabled ? 'left:18px' : 'left:2px'}"></span>
                        </label>
                    </div>
                    ${summary ? `<div class="sr-card-summary" title="${summary}">${summary}</div>` : ''}
                    <div class="sr-card-actions">
                        <button type="button" class="sr-file-btn" title="查看内容" data-full-path="${fullPath}">👁</button>
                        <button type="button" class="sr-file-btn" title="复制路径" data-dir-path="${dirPath}">📂</button>
                    </div>
                </div>`;
            };
            const visibleHtml = visible.map(fileCard).join('');
            const restHtml = rest.length ? rest.map(fileCard).join('') : '';
            return `<div class="sr-category-card">
                <div class="sr-category-header">
                    <span class="sr-category-icon">${meta.icon}</span>
                    <span class="sr-category-label">${meta.label}</span>
                    <span class="sr-category-count">${cat.count}</span>
                </div>
                <div class="sr-category-grid">${visibleHtml}</div>
                ${rest.length ? `<div class="sr-category-more hidden">${restHtml}</div><button type="button" class="btn btn-secondary btn-sm sr-more-btn" data-rest-count="${rest.length}" onclick="toggleSrCategoryMore(this)">更多 (${rest.length})</button>` : ''}
            </div>`;
        }).join('');

        container.innerHTML = `
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px">
                <div class="stat-card"><div class="stat-value">${summary.total_calls || 0}</div><div class="stat-label">近30天 MCP 调用</div></div>
                <div class="stat-card"><div class="stat-value">${(summary.top_tools || []).length}</div><div class="stat-label">活跃工具数</div></div>
                <div class="stat-card"><div class="stat-value">${skillsRules.total_files || 0}</div><div class="stat-label">Skills / Rules 总数</div></div>
            </div>

            <div class="mode-tabs" style="margin-bottom:16px">
                <button class="mode-tab active" onclick="switchUsageTab('skills',this)">Skills & Rules</button>
                <button class="mode-tab" onclick="switchUsageTab('mcp',this)">MCP 工具调用</button>
                <button class="mode-tab" onclick="switchUsageTab('team',this)">团队成员</button>
                <button class="mode-tab" onclick="switchUsageTab('apikey',this)">按 API Key</button>
            </div>

            <div id="usage-tab-skills">
                ${skillsHtml || '<p style="color:var(--text-muted)">未扫描到 Skills/Rules 文件</p>'}
                <div style="font-size:11px;color:var(--text-muted);margin-top:8px">工作区: ${esc(skillsRules.workspace || '')}</div>
            </div>

            <div id="usage-tab-mcp" class="hidden">
                ${toolRows || '<p style="color:var(--text-muted)">暂无调用数据</p>'}
            </div>

            <div id="usage-tab-team" class="hidden">
                ${userRows ? `<table class="data-table"><thead><tr><th>用户</th><th style="text-align:right">调用次数</th><th style="text-align:right">平均耗时</th></tr></thead><tbody>${userRows}</tbody></table>` : '<p style="color:var(--text-muted)">暂无数据</p>'}
            </div>
            <div id="usage-tab-apikey" class="hidden">
                ${apiKeyRows ? `<table class="data-table"><thead><tr><th>API Key</th><th style="text-align:right">调用次数</th><th style="text-align:right">平均耗时</th><th style="text-align:right">错误数</th></tr></thead><tbody>${apiKeyRows}</tbody></table>` : '<p style="color:var(--text-muted)">暂无数据（MCP 调用时可设置 TEAM_MEMORY_API_KEY_NAME 关联到 Key）</p>'}
            </div>
        `;
        if (!container._srDelegation) {
            container._srDelegation = true;
            container.addEventListener('click', (e) => {
                const el = e.target.nodeType === 1 ? e.target : e.target.parentElement;
                if (!el) return;
                const fullPathBtn = el.closest('button[data-full-path]');
                if (fullPathBtn && fullPathBtn.dataset.fullPath) {
                    e.stopPropagation();
                    previewSkillContent(fullPathBtn.dataset.fullPath);
                    return;
                }
                const dirPathBtn = el.closest('button[data-dir-path]');
                if (dirPathBtn && dirPathBtn.dataset.dirPath) {
                    e.stopPropagation();
                    copyToClipboard(dirPathBtn.dataset.dirPath);
                }
            });
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

window.toggleSrCategoryMore = function(btn) {
    const card = btn.closest('.sr-category-card');
    const moreEl = card && card.querySelector('.sr-category-more');
    if (!moreEl) return;
    const isHidden = moreEl.classList.toggle('hidden');
    const n = moreEl.querySelectorAll('.sr-card').length;
    btn.textContent = isHidden ? `更多 (${n})` : '收起';
};

// ===== Usage Sub-tabs =====
window.switchUsageTab = function(tab, btn) {
    ['skills', 'mcp', 'team', 'apikey'].forEach(t => {
        const el = document.getElementById('usage-tab-' + t);
        if (el) el.classList.toggle('hidden', t !== tab);
    });
    if (btn) {
        btn.closest('.mode-tabs').querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
};

window.previewSkillContent = async function(fullPath) {
    try {
        const data = await api('GET', `/api/v1/analytics/skills-rules/preview?path=${encodeURIComponent(fullPath)}`);
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
        overlay.innerHTML = `<div class="modal" style="max-width:720px">
            <div class="modal-header"><h2>${esc(data.name || 'Preview')}</h2>
            <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button></div>
            <div class="modal-body"><pre style="white-space:pre-wrap;word-break:break-word;font-family:var(--font-mono);font-size:12px;max-height:60vh;overflow:auto;background:var(--bg-input);padding:16px;border-radius:var(--radius);border:1px solid var(--border)">${esc(data.content || '')}</pre></div>
        </div>`;
        document.body.appendChild(overlay);
    } catch (e) {
        toast('预览失败: ' + e.message, 'error');
    }
};

window.copyToClipboard = function(text) {
    copyTextToClipboard(text).then((ok) => {
        if (ok) toast('路径已复制: ' + text, 'success');
        else toast(text, 'info');
    });
};

/** Copy experience ID, title, tags, etc. from a list card (called by exp-copy-btn). */
window.copyExpInfoFromCard = function(btn) {
    const card = btn.closest('.exp-card');
    if (!card) return;
    const id = btn.dataset.expId || '';
    const title = btn.dataset.expTitle || '';
    const tags = btn.dataset.expTags || '';
    const created = btn.dataset.expCreated || '';
    const createdBy = btn.dataset.expCreatedBy || '';
    const type = btn.dataset.expType || '';
    const project = btn.dataset.expProject || '';
    const lines = [
        `ID: ${id}`,
        `标题: ${title}`,
        ...(tags ? [`标签: ${tags}`] : []),
        ...(type ? [`类型: ${type}`] : []),
        ...(project ? [`项目: ${project}`] : []),
        ...(createdBy ? [`创建者: ${createdBy}`] : []),
        ...(created ? [`创建时间: ${created}`] : []),
    ];
    const text = lines.filter(Boolean).join('\n');
    copyTextToClipboard(text).then((ok) => {
        if (ok) toast('已复制经验信息', 'success');
        else toast('复制失败', 'error');
    });
};

// ===== Skill Toggle =====
window.toggleSkillFile = async function(category, filePath, toggleEl) {
    const bg = toggleEl.querySelector('.sr-toggle-bg');
    const knob = toggleEl.querySelector('.sr-toggle-knob');
    if (!bg || !knob) return;
    const isCurrentlyEnabled = knob.style.left === '18px';
    const newEnabled = !isCurrentlyEnabled;
    try {
        const selectedProjects = typeof window.getSelectedProjects === 'function' ? window.getSelectedProjects('usage') : [];
        const project = selectedProjects.length ? selectedProjects[0] : (state.activeProject || state.defaultProject || 'default');
        await api('POST', `/api/v1/analytics/skills-rules/toggle?project=${encodeURIComponent(project)}`, {
            category, file_path: filePath, enabled: newEnabled,
        });
        bg.style.background = newEnabled ? 'var(--green)' : 'var(--border)';
        knob.style.left = newEnabled ? '18px' : '2px';
        const card = toggleEl.closest('.sr-card');
        if (card) {
            card.classList.toggle('disabled', !newEnabled);
            card.dataset.enabled = newEnabled ? 'true' : 'false';
        }
        toast(newEnabled ? '已启用' : '已禁用', 'success');
    } catch (e) {
        toast('操作失败: ' + e.message, 'error');
    }
};

// ===== Dedup =====
export async function reembedGroups() {
    try {
        const data = await api('POST', '/api/v1/lifecycle/reembed-groups');
        const n = data.updated ?? 0;
        const errs = data.errors ?? [];
        if (errs.length) {
            toast(`已重算 ${n} 个经验组向量，${errs.length} 个失败`, 'info');
        } else {
            toast(`已重算 ${n} 个经验组向量`, 'success');
        }
        loadDuplicates();
    } catch (e) {
        toast('刷新失败: ' + e.message, 'error');
    }
}

export async function loadDuplicates() {
    const threshold = parseFloat(document.getElementById('dedup-threshold').value) || 0.92;
    const container = document.getElementById('dedup-results');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', `/api/v1/lifecycle/duplicates?threshold=${threshold}&limit=20`);
        if (!data.duplicates || data.duplicates.length === 0) {
            container.innerHTML = `<div class="empty-state"><div class="icon">✅</div><h3>没有发现重复经验</h3><p>所有经验的相似度均低于阈值 ${threshold}</p></div>`;
            return;
        }
        container.innerHTML = data.duplicates
            .map(
                (pair, idx) => `
      <div class="dup-pair">
        <div class="dup-card">
          <h4>${esc(pair.exp_a.title)}</h4>
          <p>${esc((pair.exp_a.description || '').substring(0, 120))}...</p>
          <div style="margin-top:8px">${(pair.exp_a.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">评分: ${(pair.exp_a.avg_rating || 0).toFixed(1)} · 引用: ${pair.exp_a.use_count || 0} · 查看: ${pair.exp_a.view_count || 0}</div>
          <div style="margin-top:8px;display:flex;gap:6px">
            <button class="btn btn-primary btn-sm" onclick="doMerge('${pair.exp_a.id}','${pair.exp_b.id}')">✓ 保留此项</button>
            <button class="btn btn-secondary btn-sm" onclick="showDetail('${pair.exp_a.id}')">详情</button>
          </div>
        </div>
        <div class="dup-vs">
          <div class="sim-score">${(pair.similarity * 100).toFixed(1)}%</div>
          <div style="font-size:11px;color:var(--text-muted)">相似度</div>
          <button class="btn btn-secondary btn-sm" style="margin-top:8px;font-size:11px" onclick="toggleDupDiff(${idx})">对比差异</button>
        </div>
        <div class="dup-card">
          <h4>${esc(pair.exp_b.title)}</h4>
          <p>${esc((pair.exp_b.description || '').substring(0, 120))}...</p>
          <div style="margin-top:8px">${(pair.exp_b.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">评分: ${(pair.exp_b.avg_rating || 0).toFixed(1)} · 引用: ${pair.exp_b.use_count || 0} · 查看: ${pair.exp_b.view_count || 0}</div>
          <div style="margin-top:8px;display:flex;gap:6px">
            <button class="btn btn-primary btn-sm" onclick="doMerge('${pair.exp_b.id}','${pair.exp_a.id}')">✓ 保留此项</button>
            <button class="btn btn-secondary btn-sm" onclick="showDetail('${pair.exp_b.id}')">详情</button>
          </div>
        </div>
      </div>
      <div id="dup-diff-${idx}" class="hidden" style="margin-top:-8px;margin-bottom:16px">
        <div class="diff-container" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
          <div class="diff-pane" style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:12px;overflow:auto;max-height:400px">
            <h4 style="margin:0 0 8px;color:var(--red,#ef4444)">A: ${esc(pair.exp_a.title)}</h4>
            <div style="font-size:12px">${_renderDiffHighlight(pair.exp_a.description || '', pair.exp_b.description || '', 'a')}</div>
            ${pair.exp_a.solution ? `<div style="margin-top:8px;font-size:12px"><strong>方案:</strong><br>${_renderDiffHighlight(pair.exp_a.solution, pair.exp_b.solution || '', 'a')}</div>` : ''}
          </div>
          <div class="diff-pane" style="background:var(--bg-secondary);border:1px solid var(--accent);border-radius:var(--radius);padding:12px;overflow:auto;max-height:400px">
            <h4 style="margin:0 0 8px;color:var(--accent)">合并预览</h4>
            <div id="merge-preview-${idx}" style="font-size:12px;color:var(--text-muted)">
              <button class="btn btn-secondary btn-sm" onclick="loadMergePreview(${idx},'${pair.exp_a.id}','${pair.exp_b.id}')">生成预览</button>
            </div>
          </div>
          <div class="diff-pane" style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:12px;overflow:auto;max-height:400px">
            <h4 style="margin:0 0 8px;color:var(--green,#22c55e)">B: ${esc(pair.exp_b.title)}</h4>
            <div style="font-size:12px">${_renderDiffHighlight(pair.exp_b.description || '', pair.exp_a.description || '', 'b')}</div>
            ${pair.exp_b.solution ? `<div style="margin-top:8px;font-size:12px"><strong>方案:</strong><br>${_renderDiffHighlight(pair.exp_b.solution, pair.exp_a.solution || '', 'b')}</div>` : ''}
          </div>
        </div>
      </div>
    `
            )
            .join('');
        toast(`发现 ${data.total} 组重复经验`, 'info');
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>扫描失败</h3><p>${e.message}</p></div>`;
    }
}

export async function doMerge(primaryId, secondaryId) {
    if (!confirm('确认合并？次要经验将被合并到主经验中（标签合并、反馈迁移），次要经验将被删除。')) return;
    try {
        await api('POST', '/api/v1/lifecycle/merge', { primary_id: primaryId, secondary_id: secondaryId });
        toast('合并成功', 'success');
        loadDuplicates();
    } catch (e) {
        toast('合并失败: ' + e.message, 'error');
    }
}

// ===== Diff Highlight + Merge Preview =====
function _renderDiffHighlight(text, otherText, side) {
    const linesA = text.split('\n');
    const linesB = new Set(otherText.split('\n').map(l => l.trim()));
    return linesA.map(line => {
        const trimmed = line.trim();
        if (!trimmed) return '';
        const isCommon = linesB.has(trimmed);
        if (isCommon) {
            return `<div style="background:rgba(255,213,79,0.15);padding:1px 4px;border-radius:2px">${esc(line)}</div>`;
        }
        const color = side === 'a' ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)';
        return `<div style="background:${color};padding:1px 4px;border-radius:2px">${esc(line)}</div>`;
    }).join('');
}

export async function loadMergePreview(idx, primaryId, secondaryId) {
    const container = document.getElementById('merge-preview-' + idx);
    if (!container) return;
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('POST', '/api/v1/lifecycle/merge-preview', {
            primary_id: primaryId,
            secondary_id: secondaryId,
        });
        const m = data.merged;
        container.innerHTML = `
            <div style="margin-bottom:8px"><strong>${esc(m.title)}</strong></div>
            <div style="white-space:pre-wrap;margin-bottom:8px">${esc(m.description || '')}</div>
            ${m.solution ? `<div style="margin-top:8px"><strong>方案:</strong><br><div style="white-space:pre-wrap">${esc(m.solution)}</div></div>` : ''}
            <div style="margin-top:8px">${(m.tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div>
        `;
    } catch (e) {
        container.innerHTML = `<p style="color:var(--red)">${esc(e.message)}</p>`;
    }
}

// ===== Stale Scan =====
export async function scanStale() {
    try {
        const data = await api('POST', '/api/v1/lifecycle/scan-stale');
        const panel = document.getElementById('stale-results');
        const list = document.getElementById('stale-list');
        if (data.stale_experiences && data.stale_experiences.length > 0) {
            panel.classList.remove('hidden');
            renderExpList('stale-list', data.stale_experiences);
            toast(`发现 ${data.total} 条疑似过时的经验（超过 ${data.threshold_months} 个月未使用）`, 'info');
        } else {
            panel.classList.add('hidden');
            toast('没有发现过时的经验', 'success');
        }
    } catch (e) {
        toast('过期扫描失败: ' + e.message, 'error');
    }
}

// ===== Settings / Installables =====
export async function toggleInstallablePreview(itemIdEncoded, sourceEncoded, btn) {
    const container = btn.parentElement?.querySelector('.inline-preview');
    if (container && container.innerHTML) {
        container.innerHTML = '';
        btn.textContent = '预览';
        return;
    }
    const itemId = decodeURIComponent(itemIdEncoded || '');
    const source = decodeURIComponent(sourceEncoded || '');
    try {
        const data = await api('GET', `/api/v1/installables/preview?id=${encodeURIComponent(itemId)}&source=${encodeURIComponent(source)}`);
        const target = btn.parentElement;
        if (!target) return;
        let previewEl = target.querySelector('.inline-preview');
        if (!previewEl) {
            previewEl = document.createElement('div');
            previewEl.className = 'inline-preview';
            previewEl.style.cssText =
                'margin-top:8px;padding:10px;background:var(--bg-primary);border:1px solid var(--border);border-radius:6px;font-size:12px;white-space:pre-wrap;max-height:300px;overflow-y:auto;font-family:var(--font-mono);width:100%;flex-basis:100%';
            target.appendChild(previewEl);
        }
        previewEl.textContent = data.content || 'No content';
        btn.textContent = '收起';
    } catch (e) {
        toast('预览失败: ' + e.message, 'error');
    }
}

export function renderInstallables(items) {
    const el = document.getElementById('installables-list');
    if (!el) return;
    if (!items || items.length === 0) {
        el.innerHTML = '<div class="empty-state"><h3>未找到可安装项</h3><p>可尝试切换来源或检查 manifest 配置</p></div>';
        return;
    }
    const canInstall = state.currentUser && state.currentUser.role === 'admin';
    el.innerHTML = items
        .map(
            (item) => {
                const id = encodeURIComponent(item.id || '');
                const src = encodeURIComponent(item.source || '');
                return `
    <div class="exp-card" style="cursor:default">
      <div class="exp-card-header">
        <div class="exp-card-title">
          ${item.type === 'rule' ? '📐' : '🧠'} ${esc(item.name)}
          <span class="tag" style="margin-left:8px">${esc(item.type)}</span>
          <span class="tag">${esc(item.source)}</span>
        </div>
        <div class="exp-card-meta">
          <span>${esc(item.version || 'unknown')}</span>
        </div>
      </div>
      <div class="exp-card-desc">${esc(item.description || '')}</div>
      <div class="installable-item-actions" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start">
        <button class="btn btn-secondary btn-sm" onclick="toggleInstallablePreview('${id}','${src}',this)">预览</button>
        ${canInstall ? `<button class="btn btn-primary btn-sm" onclick="installInstallable('${id}','${src}')">安装</button>` : '<span class="hint">仅 admin 可安装</span>'}
      </div>
    </div>
  `;
            }
        )
        .join('');
}

export async function loadInstallables() {
    const source = document.getElementById('installables-source-filter')?.value || '';
    const type = document.getElementById('installables-type-filter')?.value || '';
    const listEl = document.getElementById('installables-list');
    if (listEl) {
        listEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    }
    try {
        const params = new URLSearchParams();
        if (source) params.set('source', source);
        if (type) params.set('type', type);
        const qs = params.toString();
        const data = await api('GET', `/api/v1/installables${qs ? '?' + qs : ''}`);
        state.cachedInstallables = data.items || [];
        renderInstallables(state.cachedInstallables);
    } catch (e) {
        if (listEl) {
            listEl.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
        }
    }
}

export async function previewInstallable(itemIdEncoded, sourceEncoded) {
    toast('请使用行内预览按钮查看内容', 'info');
}

export async function installInstallable(itemIdEncoded, sourceEncoded) {
    const id = decodeURIComponent(itemIdEncoded || '');
    const source = decodeURIComponent(sourceEncoded || '');
    if (!confirm(`确认安装 ${id} 到项目目录？`)) return;
    try {
        const result = await api('POST', '/api/v1/installables/install', { id, source });
        toast('安装成功: ' + (result.target_path || ''), 'success');
    } catch (e) {
        toast('安装失败: ' + e.message, 'error');
    }
}

// ===== Settings Config =====
export async function loadAllConfig() {
    try {
        const all = await api('GET', '/api/v1/config/all');
        state.defaultProject = all.default_project || state.defaultProject || 'default';
        populateSettingsProjectDropdown();
        applyProjectPlaceholders();
        const r = all.retrieval || {};
        state.cachedRetrievalConfig = r;
        document.getElementById('cfg-max-tokens').value = r.max_tokens != null ? r.max_tokens : '';
        document.getElementById('cfg-max-count').value = r.max_count;
        document.getElementById('cfg-trim-strategy').value = r.trim_strategy;
        document.getElementById('cfg-top-k-children').value = r.top_k_children;
        document.getElementById('cfg-min-avg-rating').value = r.min_avg_rating;
        document.getElementById('cfg-rating-weight').value = r.rating_weight;
        document.getElementById('cfg-summary-model').value = r.summary_model || '';
        const pg = all.pageindex_lite || {};
        document.getElementById('cfg-pg-enabled').value = pg.enabled !== false ? 'true' : 'false';
        document.getElementById('cfg-pg-only-long').value = pg.only_long_docs !== false ? 'true' : 'false';
        document.getElementById('cfg-pg-min-doc-chars').value = pg.min_doc_chars || 800;
        document.getElementById('cfg-pg-max-depth').value = pg.max_tree_depth || 4;
        document.getElementById('cfg-pg-max-nodes').value = pg.max_nodes_per_doc || 40;
        document.getElementById('cfg-pg-max-node-chars').value = pg.max_node_chars || 1200;
        document.getElementById('cfg-pg-weight').value = pg.tree_weight != null ? pg.tree_weight : 0.15;
        document.getElementById('cfg-pg-min-score').value = pg.min_node_score != null ? pg.min_node_score : 0.01;
        document.getElementById('cfg-pg-include-nodes').value = pg.include_matched_nodes !== false ? 'true' : 'false';
        const s = all.search || {};
        document.getElementById('cfg-search-mode').value = s.mode || 'hybrid';
        document.getElementById('cfg-rrf-k').value = s.rrf_k || 60;
        document.getElementById('cfg-vector-weight').value = s.vector_weight || 0.7;
        document.getElementById('cfg-fts-weight').value = s.fts_weight || 0.3;
        document.getElementById('cfg-adaptive-filter').value = s.adaptive_filter !== false ? 'true' : 'false';
        document.getElementById('cfg-score-gap').value = s.score_gap_threshold || 0.15;
        document.getElementById('cfg-min-confidence').value = s.min_confidence_ratio || 0.6;
        const rr = all.reranker || {};
        document.getElementById('cfg-reranker-provider').value = rr.provider || 'none';
        const c = all.cache || {};
        document.getElementById('cfg-cache-enabled').value = c.enabled !== false ? 'true' : 'false';
        document.getElementById('cfg-cache-ttl').value = c.ttl_seconds || 300;
        document.getElementById('cfg-cache-max-size').value = c.max_size || 100;
        document.getElementById('cfg-cache-embedding-size').value = c.embedding_cache_size || 200;
        document.getElementById('settings-save-status').textContent = '';
        await Promise.all([loadInstallables(), loadScanDirsConfig()]);
    } catch (e) {
        toast('加载配置失败: ' + e.message, 'error');
    }
}

export async function loadRetrievalConfig() {
    await loadAllConfig();
}

/** Load and render /health for ops; no auth required. */
export async function loadHealthStatus() {
    const loadingEl = document.getElementById('health-status-loading');
    const bodyEl = document.getElementById('health-status-body');
    const errorEl = document.getElementById('health-status-error');
    if (!loadingEl || !bodyEl || !errorEl) return;

    loadingEl.classList.remove('hidden');
    loadingEl.textContent = '正在检测…';
    bodyEl.classList.add('hidden');
    errorEl.classList.add('hidden');
    errorEl.textContent = '';

    try {
        const base = window.__apiBaseUrl || '';
        const r = await fetch(base ? `${base}/health` : '/health', { method: 'GET', credentials: 'same-origin' });
        const data = await r.json().catch(() => ({}));

        loadingEl.classList.add('hidden');

        const status = data.status || 'unknown';
        const checks = data.checks || {};
        const version = data.version || '';
        const timestamp = data.timestamp || '';

        const summaryEl = document.getElementById('health-summary');
        if (summaryEl) {
            const statusLabel = status === 'healthy' ? '正常' : status === 'degraded' ? '降级' : '异常';
            const statusClass = status === 'healthy' ? 'status-healthy' : status === 'degraded' ? 'status-degraded' : 'status-unhealthy';
            summaryEl.innerHTML = `
        <span class="status-badge ${statusClass}">${statusLabel}</span>
        <span>${version ? `v${version}` : ''}</span>
        <span class="hint">${timestamp ? new Date(timestamp).toLocaleString('zh-CN') : ''}</span>
      `;
        }

        const checksEl = document.getElementById('health-checks');
        if (checksEl) {
            const order = ['database', 'ollama', 'cache', 'dashboard_stats', 'embedding_provider', 'migration', 'event_bus', 'embedding_queue'];
            const labels = {
                database: '数据库',
                ollama: 'Ollama',
                cache: '缓存',
                dashboard_stats: '仪表盘',
                embedding_provider: 'Embedding',
                migration: '迁移',
                event_bus: '事件总线',
                embedding_queue: '嵌入队列',
            };
            checksEl.innerHTML = order.filter((k) => k in checks).map((key) => {
                const c = checks[key];
                const st = (c && c.status) || 'unknown';
                const name = labels[key] || key;
                let detail = '';
                if (st === 'down' && c) {
                    if (c.error) detail += c.error;
                    if (c.ops_hint) detail += (detail ? ' ' : '') + c.ops_hint;
                    if (c.latency_ms != null) detail = (detail || '') + ` (${c.latency_ms}ms)`;
                } else if (c && c.latency_ms != null) {
                    detail = `${c.latency_ms}ms`;
                }
                return `<div class="health-check-item ${st}">
          <div class="check-name">${esc(name)}</div>
          ${detail ? `<div class="check-detail">${esc(detail)}</div>` : ''}
        </div>`;
            }).join('');
        }

        bodyEl.classList.remove('hidden');

        const refreshBtn = document.getElementById('health-btn-refresh');
        const copyBtn = document.getElementById('health-btn-copy-cmd');
        if (refreshBtn) refreshBtn.onclick = () => loadHealthStatus();
        if (copyBtn) {
            copyBtn.onclick = () => {
                copyTextToClipboard('make health');
                toast('已复制 make health', 'success');
            };
        }
    } catch (e) {
        loadingEl.classList.add('hidden');
        errorEl.textContent = '获取健康状态失败: ' + (e.message || String(e));
        errorEl.classList.remove('hidden');
    }
}

export async function saveRetrievalConfig() {
    const maxTokensVal = document.getElementById('cfg-max-tokens').value.trim();
    const body = {
        max_tokens: maxTokensVal === '' ? null : parseInt(maxTokensVal, 10),
        max_count: parseInt(document.getElementById('cfg-max-count').value, 10) || 20,
        trim_strategy: document.getElementById('cfg-trim-strategy').value,
        top_k_children: parseInt(document.getElementById('cfg-top-k-children').value, 10) || 3,
        min_avg_rating: parseFloat(document.getElementById('cfg-min-avg-rating').value) || 0.0,
        rating_weight: parseFloat(document.getElementById('cfg-rating-weight').value) || 0.3,
        summary_model: document.getElementById('cfg-summary-model').value.trim() || null,
    };
    try {
        const result = await api('PUT', '/api/v1/config/retrieval', body);
        state.cachedRetrievalConfig = result.config;
        toast('检索参数已保存', 'success');
        document.getElementById('settings-save-status').textContent = '已保存 ' + new Date().toLocaleTimeString('zh-CN');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function saveDefaultProjectConfig() {
    const val = (document.getElementById('cfg-default-project')?.value || '').trim();
    if (!val) {
        toast('默认项目不能为空', 'error');
        return;
    }
    try {
        const result = await api('PUT', '/api/v1/config/project', { default_project: val });
        state.defaultProject = result.default_project || val;
        applyProjectPlaceholders();
        toast('默认项目已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function loadScanDirsConfig() {
    try {
        const data = await api('GET', '/api/v1/config/scan-dirs');
        const paths = data.project_paths || {};
        const el = document.getElementById('cfg-project-paths');
        if (el) {
            el.value = Object.entries(paths).map(([k, v]) => `${k}=${v}`).join('\n');
        }
        const extras = data.extra_scan_dirs || [];
        const el2 = document.getElementById('cfg-extra-scan-dirs');
        if (el2) {
            el2.value = extras.map(d => `${d.label}=${d.path}=${d.pattern}`).join('\n');
        }
        const customContainer = document.getElementById('custom-scan-paths');
        if (customContainer) {
            customContainer.innerHTML = '';
            for (const d of extras) {
                const row = document.createElement('div');
                row.className = 'scan-path-row custom';
                row.innerHTML = `<input class="scan-path-val" type="text" value="${d.path || ''}">` +
                    `<span class="scan-path-del" onclick="this.parentElement.remove()">✕</span>`;
                customContainer.appendChild(row);
            }
        }
    } catch (_) { /* non-blocking */ }
}

export async function saveScanDirsConfig() {
    const pathsText = document.getElementById('cfg-project-paths')?.value || '';
    const project_paths = {};
    for (const line of pathsText.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.includes('=')) continue;
        const idx = trimmed.indexOf('=');
        const key = trimmed.slice(0, idx).trim();
        const val = trimmed.slice(idx + 1).trim();
        if (key && val) project_paths[key] = val;
    }
    const extrasText = document.getElementById('cfg-extra-scan-dirs')?.value || '';
    const extra_scan_dirs = [];
    for (const line of extrasText.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const parts = trimmed.split('=');
        if (parts.length >= 2) {
            extra_scan_dirs.push({
                label: parts[0].trim(),
                path: parts[1].trim(),
                pattern: parts[2]?.trim() || '*',
            });
        }
    }
    const customPaths = typeof window.getCustomScanPaths === 'function'
        ? window.getCustomScanPaths() : [];
    for (const cp of customPaths) {
        const label = cp.replace(/[/\\]/g, '_').replace(/^_+|_+$/g, '');
        extra_scan_dirs.push({ label, path: cp, pattern: '*' });
    }
    try {
        await api('PUT', '/api/v1/config/scan-dirs', { project_paths, extra_scan_dirs });
        toast('扫描目录配置已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function saveSearchConfig() {
    const body = {
        mode: document.getElementById('cfg-search-mode').value,
        rrf_k: parseInt(document.getElementById('cfg-rrf-k').value, 10) || 60,
        vector_weight: parseFloat(document.getElementById('cfg-vector-weight').value) || 0.7,
        fts_weight: parseFloat(document.getElementById('cfg-fts-weight').value) || 0.3,
        adaptive_filter: document.getElementById('cfg-adaptive-filter').value === 'true',
        score_gap_threshold: parseFloat(document.getElementById('cfg-score-gap').value) || 0.15,
        min_confidence_ratio: parseFloat(document.getElementById('cfg-min-confidence').value) || 0.6,
    };
    try {
        await api('PUT', '/api/v1/config/search', body);
        toast('搜索管线配置已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function saveRerankerConfig() {
    const body = { provider: document.getElementById('cfg-reranker-provider').value };
    try {
        const result = await api('PUT', '/api/v1/config/reranker', body);
        toast(result.message || 'Reranker 配置已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function saveCacheConfig() {
    const body = {
        enabled: document.getElementById('cfg-cache-enabled').value === 'true',
        ttl_seconds: parseInt(document.getElementById('cfg-cache-ttl').value, 10) || 300,
        max_size: parseInt(document.getElementById('cfg-cache-max-size').value, 10) || 100,
        embedding_cache_size: parseInt(document.getElementById('cfg-cache-embedding-size').value, 10) || 200,
    };
    try {
        await api('PUT', '/api/v1/config/cache', body);
        toast('缓存配置已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function savePageIndexLiteConfig() {
    const body = {
        enabled: document.getElementById('cfg-pg-enabled').value === 'true',
        only_long_docs: document.getElementById('cfg-pg-only-long').value === 'true',
        min_doc_chars: parseInt(document.getElementById('cfg-pg-min-doc-chars').value, 10) || 800,
        max_tree_depth: parseInt(document.getElementById('cfg-pg-max-depth').value, 10) || 4,
        max_nodes_per_doc: parseInt(document.getElementById('cfg-pg-max-nodes').value, 10) || 40,
        max_node_chars: parseInt(document.getElementById('cfg-pg-max-node-chars').value, 10) || 1200,
        tree_weight: parseFloat(document.getElementById('cfg-pg-weight').value) || 0.15,
        min_node_score: parseFloat(document.getElementById('cfg-pg-min-score').value) || 0.01,
        include_matched_nodes: document.getElementById('cfg-pg-include-nodes').value === 'true',
    };
    try {
        await api('PUT', '/api/v1/config/pageindex-lite', body);
        toast('PageIndex-Lite 配置已保存', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function clearCache() {
    if (!confirm('确认清除所有搜索缓存？')) return;
    try {
        await api('POST', '/api/v1/cache/clear');
        toast('缓存已清除', 'success');
    } catch (e) {
        toast('清除失败: ' + e.message, 'error');
    }
}

// ===== Schema Management =====
export async function switchPreset() {
    const preset = document.getElementById('schema-preset-select').value;
    try {
        await api('PUT', '/api/v1/config/schema', { preset });
        toast('已切换到预设: ' + preset, 'success');
        loadSchemaAndPopulateFilters(api);
        loadCurrentSchema();
    } catch (e) {
        toast('切换失败: ' + e.message, 'error');
    }
}

export async function generateSchemaFromDoc() {
    const content = document.getElementById('schema-gen-input').value.trim();
    if (!content || content.length < 10) {
        toast('请输入至少 10 个字符的文档内容', 'error');
        return;
    }
    try {
        toast('正在分析文档...', 'info');
        const r = await api('POST', '/api/v1/schema/generate', { content });
        state.generatedSchemaData = r;
        document.getElementById('schema-yaml-preview').textContent = r.yaml_preview || '(empty)';
        document.getElementById('schema-preview').style.display = 'block';
        toast('分析完成: ' + (r.analysis_summary || ''), 'success');
    } catch (e) {
        toast('生成失败: ' + e.message, 'error');
    }
}

export async function applyGeneratedSchema() {
    if (!state.generatedSchemaData) return;
    const config = {};
    if (state.generatedSchemaData.types_found) config.experience_types = state.generatedSchemaData.types_found;
    if (state.generatedSchemaData.categories_found) config.categories = state.generatedSchemaData.categories_found;
    try {
        await api('PUT', '/api/v1/config/schema', config);
        toast('Schema 配置已应用', 'success');
        document.getElementById('schema-preview').style.display = 'none';
        loadSchemaAndPopulateFilters(api);
        loadCurrentSchema();
    } catch (e) {
        toast('应用失败: ' + e.message, 'error');
    }
}

export async function loadCurrentSchema() {
    try {
        const schema = await api('GET', '/api/v1/schema');
        const el = document.getElementById('schema-current-types');
        if (!el) return;
        let html = '<h3 style="margin-bottom:8px">当前类型体系 (preset: ' + esc(schema.preset || 'software-dev') + ')</h3>';
        html += '<div style="display:flex;flex-wrap:wrap;gap:6px">';
        (schema.experience_types || []).forEach((t) => {
            html += `<span class="badge" style="font-size:12px;padding:3px 8px">${esc(t.id)} — ${esc(t.label || t.id)}</span>`;
        });
        html += '</div>';
        if (schema.categories && schema.categories.length) {
            html += '<div style="margin-top:8px"><strong>分类:</strong> ' + schema.categories.map((c) => esc(c.label || c.id)).join(', ') + '</div>';
        }
        if (schema.severity_levels && schema.severity_levels.length) {
            html += '<div style="margin-top:4px"><strong>严重等级:</strong> ' + schema.severity_levels.join(', ') + '</div>';
        }
        el.innerHTML = html;
        const sel = document.getElementById('schema-preset-select');
        if (sel && schema.preset) sel.value = schema.preset;
    } catch (_) {}
}

// ===== Summary =====
export async function generateSummary(id) {
    try {
        toast('正在生成摘要（需要 LLM 服务）...', 'info');
        await api('POST', `/api/v1/experiences/${id}/summarize`);
        toast('摘要已生成', 'success');
        viewDetail(id);
    } catch (e) {
        const msg = e.message || '';
        if (msg.includes('summary generation failed') || msg.includes('500') || msg.includes('Connection')) {
            toast('摘要生成失败: 请确认 LLM 服务已启动且配置了对话模型（非 embedding 模型）。可在设置 > 检索参数中配置 summary_model', 'error');
        } else {
            toast('摘要生成失败: ' + msg, 'error');
        }
    }
}

export async function batchSummarize() {
    try {
        toast('正在批量生成摘要...', 'info');
        const result = await api('POST', '/api/v1/experiences/batch-summarize?limit=10');
        toast(`摘要生成完成: ${result.generated}/${result.total_candidates} 条`, 'success');
    } catch (e) {
        toast('批量摘要失败: ' + e.message, 'error');
    }
}

// ===== Key / User Management (admin) =====

export async function loadKeyManagement() {
    const card = document.getElementById('settings-key-mgmt');
    if (!card) return;
    if (!state.currentUser || state.currentUser.role !== 'admin') {
        card.style.display = 'none';
        return;
    }
    card.style.display = 'block';

    try {
        const data = await api('GET', '/api/v1/keys');
        const keys = data.keys || [];
        const pending = keys.filter(k => !k.is_active && !k.has_api_key);
        const active = keys.filter(k => k.is_active || k.has_api_key);

        const pendSec = document.getElementById('keys-pending-section');
        const pendList = document.getElementById('keys-pending-list');
        if (pending.length > 0) {
            pendSec.style.display = 'block';
            pendList.innerHTML = pending.map(k => `
                <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:8px;">
                    <div>
                        <strong>${_esc(k.user_name)}</strong>
                        <span style="color:var(--text-secondary); font-size:12px; margin-left:8px;">注册于 ${_fmtDate(k.created_at)}</span>
                    </div>
                    <div>
                        <button class="btn btn-primary" onclick="approveUser(${k.id})" style="font-size:12px; padding:4px 12px; margin-right:6px;">通过</button>
                        <button class="btn" onclick="rejectUser(${k.id})" style="font-size:12px; padding:4px 12px; color:var(--danger, #ef4444);">拒绝</button>
                    </div>
                </div>
            `).join('');
        } else {
            pendSec.style.display = 'none';
        }

        const activeList = document.getElementById('keys-active-list');
        if (active.length === 0) {
            activeList.innerHTML = '<p style="color:var(--text-secondary);">暂无用户</p>';
        } else {
            activeList.innerHTML = `
                <table style="width:100%; border-collapse:collapse; font-size:13px;">
                    <thead>
                        <tr style="text-align:left; border-bottom:2px solid var(--border);">
                            <th style="padding:8px 6px;">用户名</th>
                            <th style="padding:8px 6px;">角色</th>
                            <th style="padding:8px 6px;">状态</th>
                            <th style="padding:8px 6px;">API Key</th>
                            <th style="padding:8px 6px;">创建时间</th>
                            <th style="padding:8px 6px;">操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${active.map(k => `
                            <tr style="border-bottom:1px solid var(--border);">
                                <td style="padding:8px 6px; font-weight:500;">${_esc(k.user_name)}</td>
                                <td style="padding:8px 6px;">
                                    <select onchange="updateUserRole(${k.id}, this.value)" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border);border-radius:4px;padding:3px 6px;font-size:12px;">
                                        <option value="admin" ${k.role === 'admin' ? 'selected' : ''}>admin</option>
                                        <option value="editor" ${k.role === 'editor' ? 'selected' : ''}>editor</option>
                                        <option value="viewer" ${k.role === 'viewer' ? 'selected' : ''}>viewer</option>
                                    </select>
                                </td>
                                <td style="padding:8px 6px;">
                                    <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500;${k.is_active ? 'background:#dcfce7;color:#166534;' : 'background:#fee2e2;color:#991b1b;'}">${k.is_active ? '活跃' : '停用'}</span>
                                </td>
                                <td style="padding:8px 6px;">
                                    <span style="font-size:11px; color:var(--text-secondary);">${k.has_api_key ? '已分配' : '无'}</span>
                                </td>
                                <td style="padding:8px 6px; font-size:12px; color:var(--text-secondary);">${_fmtDate(k.created_at)}</td>
                                <td style="padding:8px 6px;">
                                    ${k.is_active
                                        ? `<button class="btn" onclick="toggleUserActive(${k.id}, false)" style="font-size:11px;padding:3px 8px;color:var(--danger,#ef4444);">停用</button>`
                                        : `<button class="btn" onclick="toggleUserActive(${k.id}, true)" style="font-size:11px;padding:3px 8px;color:#16a34a;">激活</button>`
                                    }
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    } catch (e) {
        toast('加载用户列表失败: ' + e.message, 'error');
    }
}

export async function approveUser(id) {
    try {
        const result = await api('PUT', `/api/v1/keys/${id}`, { is_active: true });
        if (result.api_key) {
            const msg = `用户 ${result.user_name} 已审批通过。\n\nAPI Key（仅显示一次，请复制保存）：\n${result.api_key}`;
            prompt('审批成功 - 请复制 API Key 分发给用户', result.api_key);
            toast('审批成功，API Key 已生成', 'success');
        } else {
            toast('审批成功', 'success');
        }
        loadKeyManagement();
    } catch (e) {
        toast('审批失败: ' + e.message, 'error');
    }
}

export async function rejectUser(id) {
    if (!confirm('确定要拒绝此注册申请？将删除该记录。')) return;
    try {
        await api('DELETE', `/api/v1/keys/${id}`);
        toast('已拒绝', 'success');
        loadKeyManagement();
    } catch (e) {
        toast('操作失败: ' + e.message, 'error');
    }
}

export async function createUserAdmin() {
    const username = document.getElementById('admin-new-username').value.trim();
    const role = document.getElementById('admin-new-role').value;
    const password = document.getElementById('admin-new-password').value;
    if (!username) { toast('请输入用户名', 'error'); return; }

    try {
        const body = { user_name: username, role };
        if (password) body.password = password;
        const result = await api('POST', '/api/v1/keys', body);
        if (result.api_key) {
            prompt('用户创建成功 - 请复制 API Key 分发给用户', result.api_key);
        }
        toast('用户创建成功', 'success');
        document.getElementById('admin-create-user-form').style.display = 'none';
        document.getElementById('admin-new-username').value = '';
        document.getElementById('admin-new-password').value = '';
        loadKeyManagement();
    } catch (e) {
        toast('创建失败: ' + e.message, 'error');
    }
}

export async function updateUserRole(id, newRole) {
    try {
        await api('PUT', `/api/v1/keys/${id}`, { role: newRole });
        toast('角色已更新', 'success');
    } catch (e) {
        toast('更新失败: ' + e.message, 'error');
        loadKeyManagement();
    }
}

export async function toggleUserActive(id, active) {
    const action = active ? '激活' : '停用';
    if (!confirm(`确定要${action}此用户？`)) return;
    try {
        await api('PUT', `/api/v1/keys/${id}`, { is_active: active });
        toast(`用户已${action}`, 'success');
        loadKeyManagement();
    } catch (e) {
        toast(`${action}失败: ` + e.message, 'error');
        loadKeyManagement();
    }
}

// ===== Tasks (Kanban Board) =====

const WIP_LIMIT = 5;
const KANBAN_COLS = [
    { status: 'wait', label: '待处理', icon: '⏳' },
    { status: 'plan', label: '计划中', icon: '📋' },
    { status: 'in_progress', label: '进行中', icon: '🔧' },
    { status: 'completed', label: '已完成', icon: '✅' },
];
const UNGROUPED_ID = '__ungrouped__';
const VISIBLE_STORAGE_PREFIX = 'tm_tasks_visible_';

function _visibleStorageKey() {
    const proj = (state.activeProject || state.defaultProject || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
    return VISIBLE_STORAGE_PREFIX + (_tasksShowArchived ? 'archived_' : '') + proj;
}

function _loadVisibleGroupsFromStorage() {
    try {
        const raw = localStorage.getItem(_visibleStorageKey());
        if (!raw) return null;
        const arr = JSON.parse(raw);
        return Array.isArray(arr) ? new Set(arr) : null;
    } catch {
        return null;
    }
}

function _saveVisibleGroupsToStorage() {
    try {
        localStorage.setItem(_visibleStorageKey(), JSON.stringify(Array.from(_kanbanVisibleGroups)));
    } catch (e) {
        if (typeof console !== 'undefined' && console.warn) console.warn('Failed to persist visible groups:', e);
    }
}

const _kanbanVisibleGroups = new Set();
let _kanbanInitialized = false;
let _tasksShowArchived = false;
const PRIORITY_COLORS = { urgent: 'priority-urgent', high: 'priority-high', medium: 'priority-medium', low: 'priority-low' };

export function switchTasksSubTab(mode) {
    _tasksShowArchived = mode === 'archived';
    if (_tasksShowArchived) {
        _kanbanVisibleGroups.clear();
    }
    const groupSelect = document.getElementById('tasks-group-filter');
    if (groupSelect) groupSelect.value = '';
    const statusSelect = document.getElementById('tasks-status-filter');
    if (statusSelect) statusSelect.value = mode;
    const titleEl = document.getElementById('page-tasks-title');
    if (titleEl) titleEl.textContent = _tasksShowArchived ? '归档任务' : '任务列表';
    loadTasks();
}

function _daysUntilDue(dueDate) {
    if (!dueDate) return null;
    const diff = (new Date(dueDate) - new Date()) / (1000 * 60 * 60 * 24);
    return Math.ceil(diff);
}

function _renderTaskCard(t) {
    const priClass = PRIORITY_COLORS[t.priority] || 'priority-medium';
    const stars = '★'.repeat(t.importance || 3) + '☆'.repeat(5 - (t.importance || 3));
    const days = _daysUntilDue(t.due_date);
    const dueTag = days !== null
        ? (days < 0 ? `<span style="color:var(--red);font-weight:600">已逾期${-days}天</span>`
            : days <= 3 ? `<span style="color:#f59e0b">${days}天后截止</span>`
            : `<span>${days}天后</span>`)
        : '';
    const labels = (t.labels || []).map(l => `<span class="tag" style="font-size:10px">${esc(l)}</span>`).join('');
    const sediment = t.sediment_experience_id
        ? `<a onclick="event.stopPropagation();showDetail('${t.sediment_experience_id}')" style="font-size:10px;color:var(--accent);cursor:pointer">沉淀经验</a>`
        : '';
    const taskCopyAttrs = `data-task-id="${esc(t.id)}" data-task-title="${esc((t.title || '').replace(/"/g, '&quot;'))}" data-task-status="${esc(t.status || '')}" data-task-priority="${esc(t.priority || '')}" data-task-group-id="${esc(t.group_id || '')}"`;
    const taskCopyBtn = getTaskCopyDropdownHtml(taskCopyAttrs);
    return `
    <div class="task-card" onclick="showTaskDetail('${t.id}')">
      <div class="task-card-title"><span class="priority-dot ${priClass}"></span> ${esc(t.title)}<span onclick="event.stopPropagation()" style="margin-left:4px;vertical-align:middle">${taskCopyBtn}</span></div>
      <div class="task-card-meta">
        <span class="importance-stars">${stars}</span>
        ${dueTag}
        ${labels}
        ${sediment}
      </div>
    </div>`;
}

export async function loadTasks() {
    const board = document.getElementById('tasks-board');
    const groupsContainer = document.getElementById('tasks-groups');
    if (!board) return;
    const statusSelect = document.getElementById('tasks-status-filter');
    if (statusSelect) statusSelect.value = _tasksShowArchived ? 'archived' : 'active';
    const titleEl = document.getElementById('page-tasks-title');
    if (titleEl) titleEl.textContent = _tasksShowArchived ? '归档任务' : '任务列表';
    board.innerHTML = '<div class="loading" style="grid-column:1/-1"><div class="spinner"></div></div>';
    try {
        const project = state.activeProject || state.defaultProject || 'default';
        const groupFilter = document.getElementById('tasks-group-filter')?.value || '';
        let url = `/api/v1/tasks?project=${encodeURIComponent(project)}`;
        if (groupFilter && groupFilter !== UNGROUPED_ID) url += `&group_id=${encodeURIComponent(groupFilter)}`;
        const includeArchived = _tasksShowArchived;
        const groupUrl = `/api/v1/task-groups?project=${encodeURIComponent(project)}${includeArchived ? '&include_archived=true' : ''}`;
        const [taskData, groupData] = await Promise.all([
            api('GET', url),
            api('GET', groupUrl),
        ]);
        const tasks = taskData.tasks || [];
        let groups = (groupData.groups || []).filter(g => includeArchived ? g.archived : !g.archived);
        const ungroupedTasks = tasks.filter(t => !t.group_id);
        if (ungroupedTasks.length > 0) {
            groups = [
                { id: UNGROUPED_ID, title: '无任务组', tasks: ungroupedTasks, archived: false, isVirtual: true },
                ...groups,
            ];
        }

        // Fetch workflow progress per group (4.2 group progress)
        await Promise.all(groups.map(async (g) => {
            if (g.isVirtual) return;
            try {
                const d = await api('GET', `/api/v1/task-groups/${g.id}/workflow-progress`);
                g.workflowProgress = d;
            } catch {
                g.workflowProgress = null;
            }
        }));

        // Restore visible groups from localStorage when available (filter stale IDs)
        const groupIds = new Set(groups.map((g) => g.id));
        const stored = _loadVisibleGroupsFromStorage();
        if (stored && stored.size > 0) {
            _kanbanVisibleGroups.clear();
            stored.forEach((id) => {
                if (groupIds.has(id)) _kanbanVisibleGroups.add(id);
            });
            _kanbanInitialized = true;
        }

        // Initialize visible groups on first load (active mode only); archived mode defaults to all closed
        // Only run when no stored preference exists
        if (stored === null && !_tasksShowArchived && !_kanbanInitialized && _kanbanVisibleGroups.size === 0 && groups.length > 0) {
            groups.forEach(g => _kanbanVisibleGroups.add(g.id));
            _kanbanInitialized = true;
        }

        window.__tasksCurrentGroups = groups;

        const groupSelect = document.getElementById('tasks-group-filter');
        if (groupSelect) {
            const cur = groupSelect.value;
            groupSelect.innerHTML = '<option value="">全部任务</option>' +
                groups.map(g => `<option value="${g.id}"${g.id === cur ? ' selected' : ''}>${esc(g.title)}</option>`).join('');
        }

        // Filter tasks by visible groups (only when not filtering by a specific group)
        const filteredTasks = groupFilter
            ? (groupFilter === UNGROUPED_ID ? tasks.filter(t => !t.group_id) : tasks)
            : _tasksShowArchived
                ? tasks.filter(t =>
                    !t.group_id
                        ? _kanbanVisibleGroups.has(UNGROUPED_ID)
                        : (groupIds.has(t.group_id) && _kanbanVisibleGroups.has(t.group_id))
                )
                : tasks.filter(t => {
                    if (!t.group_id) return _kanbanVisibleGroups.has(UNGROUPED_ID);
                    return _kanbanVisibleGroups.has(t.group_id);
                });

        let html = '';
        if (groupFilter) {
            const g = groups.find(gr => gr.id === groupFilter);
            const gName = g?.title || (groupFilter === UNGROUPED_ID ? '无任务组' : '任务组');
            html += `<div class="tasks-back-row" style="grid-column:1/-1">
              <button class="back-btn" type="button" onclick="document.getElementById('tasks-group-filter').value='';loadTasks()">
                ← 返回全部任务</button>
              <span style="font-size:13px;color:var(--text-muted);margin-left:8px">${esc(gName)}</span>
            </div>`;
            if (g) {
                const isArchived = _tasksShowArchived && g.archived;
                const wp = g.workflowProgress;
                const prog = wp ? { total: wp.total, completed: wp.completed } : (g.progress || { total: (g.tasks || []).length, completed: (g.tasks || []).filter(t => t.status === 'completed' || t.status === 'cancelled').length });
                const pct = prog.total ? Math.round(prog.completed / prog.total * 100) : 0;
                const groupCompleted = g.group_completed === true;
                const hasSediment = g.has_sediment === true;
                const needsRetro = !g.isVirtual && groupCompleted && !hasSediment && !isArchived;
                const sedimentBtn = !g.isVirtual && !isArchived && groupCompleted ? `<button class="sediment-btn" title="保存为组经验" onclick="sedimentTaskGroup('${g.id}')">${needsRetro ? '组复盘' : '经验'}</button>` : '';
                const archiveBtn = !g.isVirtual && !isArchived && pct === 100 ? `<button class="archive-btn" onclick="archiveGroup('${g.id}')">📦</button>` : '';
                const unarchiveBtn = isArchived ? `<button class="unarchive-btn" title="恢复到此任务列表" onclick="unarchiveGroup('${g.id}')">恢复</button>` : '';
                const archiveBadge = isArchived ? `<span class="archive-badge">已归档</span>` : '';
                const circleColor = pct === 100 ? 'var(--green)' : pct >= 50 ? 'var(--accent)' : 'var(--yellow)';
                const stepSummary = (() => {
                    if (!wp || !wp.tasks || wp.tasks.length === 0) return '';
                    const byStep = {};
                    wp.tasks.filter(t => t.status !== 'completed' && t.status !== 'cancelled').forEach(t => {
                        const sid = t.current_step_id || 'unknown';
                        byStep[sid] = (byStep[sid] || 0) + 1;
                    });
                    const parts = Object.entries(byStep).map(([sid, n]) => `${WORKFLOW_STEP_LABELS[sid] || sid} ${n}`).filter(Boolean);
                    return parts.length > 0 ? `<div class="group-step-summary">${parts.join('，')}</div>` : '';
                })();
                const retroBlock = needsRetro ? `<div class="group-retro-prompt">
                        <div class="group-retro-prompt-text">待执行组复盘：本组任务已全部完成，请执行 tm_save_group 完成组级经验沉淀（总-分或总-分-分），再选活下一任务。</div>
                        <button class="btn btn-primary btn-sm group-retro-btn" onclick="sedimentTaskGroup('${g.id}')">组复盘 / 保存为组经验</button>
                       </div>` : '';
                const groupDetailCopyAttrs = `data-group-id="${esc(g.id)}" data-group-title="${esc(g.title || '').replace(/"/g, '&quot;')}" data-group-archived="${isArchived}" data-group-total="${prog.total || 0}" data-group-completed="${prog.completed || 0}"`;
                const groupDetailCopyBtn = getTaskGroupCopyDropdownHtml(groupDetailCopyAttrs);
                html += `<div class="group-mode-header" style="grid-column:1/-1;margin-bottom:12px;padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;position:relative">
                  ${archiveBadge}
                  <div class="task-group-header">
                    <div style="display:flex;align-items:center;gap:12px;flex:1;min-width:0">
                      <svg class="circular-progress" viewBox="0 0 36 36">
                        <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"></path>
                        <path class="circle-fill" style="stroke:${circleColor}" stroke-dasharray="${pct}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"></path>
                        <text x="18" y="21" class="pct-text">${pct}%</text>
                      </svg>
                      <div style="flex:1;min-width:0">
                        <div style="font-weight:500;display:flex;align-items:center;gap:6px;min-width:0">
                          <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(g.title).replace(/"/g, '&quot;')}">${esc(g.title)}</span>
                        </div>
                        <div style="font-size:11px;color:var(--text-muted)">${prog.completed}/${prog.total} 任务完成</div>
                        ${stepSummary}
                      </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;flex-shrink:0">
                      ${groupDetailCopyBtn}
                      ${sedimentBtn}
                      ${unarchiveBtn}
                      ${archiveBtn}
                    </div>
                  </div>
                  ${retroBlock}
                </div>`;
            }
        }
        for (const col of KANBAN_COLS) {
            const colTasks = filteredTasks.filter(t => t.status === col.status);
            const isWip = col.status === 'in_progress';
            const wipWarn = isWip && colTasks.length >= WIP_LIMIT
                ? `<span class="wip-warn">WIP ${colTasks.length}/${WIP_LIMIT}</span>` : '';
            html += `<div class="kanban-col">
              <div class="kanban-col-header">
                <span>${col.icon} ${col.label} <span class="col-count">(${colTasks.length})</span></span>
                ${wipWarn}
              </div>
              ${colTasks.length === 0 ? '<div style="text-align:center;color:var(--text-muted);font-size:12px;padding:20px 0">暂无</div>' : colTasks.map(_renderTaskCard).join('')}
            </div>`;
        }
        board.innerHTML = html;
        bindCopyDropdowns(board);

        const TASK_GROUP_VISIBLE = 3;
        if (groups.length > 0 && !groupFilter) {
            const cardsHtml = groups.map((g) => {
                const isArchived = _tasksShowArchived && g.archived;
                const wp = g.workflowProgress;
                const prog = wp ? { total: wp.total, completed: wp.completed } : (g.progress || { total: (g.tasks || []).length, completed: (g.tasks || []).filter(t => t.status === 'completed' || t.status === 'cancelled').length });
                const pct = prog.total
                    ? Math.round(prog.completed / prog.total * 100) : 0;
                const groupCompleted = g.group_completed === true;
                const hasSediment = g.has_sediment === true;
                const needsRetro = !g.isVirtual && groupCompleted && !hasSediment && !isArchived;
                const sedimentBtn = !g.isVirtual && !isArchived && groupCompleted
                    ? `<button class="sediment-btn" title="保存为组经验" onclick="event.stopPropagation();sedimentTaskGroup('${g.id}')">${needsRetro ? '组复盘' : '经验'}</button>`
                    : '';
                const archiveBtn = !g.isVirtual && !isArchived && pct === 100
                    ? `<button class="archive-btn" onclick="event.stopPropagation();archiveGroup('${g.id}')">📦</button>`
                    : '';
                const unarchiveBtn = isArchived
                    ? `<button class="unarchive-btn" title="恢复到此任务列表" onclick="event.stopPropagation();unarchiveGroup('${g.id}')">恢复</button>`
                    : '';
                const archiveBadge = isArchived ? `<span class="archive-badge">已归档</span>` : '';
                const circleColor = pct === 100
                    ? 'var(--green)' : pct >= 50
                        ? 'var(--accent)' : 'var(--yellow)';
                const isVisible = _kanbanVisibleGroups.has(g.id);
                const eyeIcon = isVisible ? '👁' : '👁‍🗨';
                const eyeCls = isVisible ? 'active' : '';
                const stepSummary = (() => {
                    if (!wp || !wp.tasks || wp.tasks.length === 0) return '';
                    const byStep = {};
                    wp.tasks.filter(t => t.status !== 'completed' && t.status !== 'cancelled').forEach(t => {
                        const sid = t.current_step_id || 'unknown';
                        byStep[sid] = (byStep[sid] || 0) + 1;
                    });
                    const parts = Object.entries(byStep).map(([sid, n]) => `${WORKFLOW_STEP_LABELS[sid] || sid} ${n}`).filter(Boolean);
                    return parts.length > 0 ? `<div class="group-step-summary">${parts.join('，')}</div>` : '';
                })();
                const taskStepMap = wp && wp.tasks ? Object.fromEntries(wp.tasks.map(t => [t.task_id, t.current_step_id])) : {};
                const subtasks = (g.tasks || []).slice(0, 5);
                const moreCount = (g.tasks || []).length - 5;
                const subtaskHtml = subtasks.length > 0
                    ? `<div class="group-subtask-list">${subtasks.map(t => {
                        const stepLabel = taskStepMap[t.id] ? (WORKFLOW_STEP_LABELS[taskStepMap[t.id]] || taskStepMap[t.id]) : '';
                        return `<div class="group-subtask-item"><span class="status-dot ${t.status || 'wait'}"></span><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(t.title)}</span>${stepLabel ? `<span class="group-subtask-step">${esc(stepLabel)}</span>` : ''}</div>`;
                      }).join('')}${moreCount > 0 ? `<div style="font-size:11px;color:var(--text-muted)">+${moreCount} more</div>` : ''}</div>`
                    : '';
                const retroBlock = needsRetro
                    ? `<div class="group-retro-prompt" onclick="event.stopPropagation()">
                        <div class="group-retro-prompt-text">待执行组复盘：本组任务已全部完成，请执行 tm_save_group 完成组级经验沉淀（总-分或总-分-分），再选活下一任务。</div>
                        <button class="btn btn-primary btn-sm group-retro-btn" onclick="event.stopPropagation();sedimentTaskGroup('${g.id}')">组复盘 / 保存为组经验</button>
                       </div>`
                    : '';
                const pinBtn = !g.isVirtual && !isArchived
                    ? `<button type="button" class="task-group-pin-btn" title="置顶到最前" onclick="event.stopPropagation();pinTaskGroupToTop('${g.id}')">置顶</button>`
                    : '';
                const groupCopyAttrs = `data-group-id="${esc(g.id)}" data-group-title="${esc(g.title || '').replace(/"/g, '&quot;')}" data-group-archived="${isArchived}" data-group-total="${prog.total || 0}" data-group-completed="${prog.completed || 0}"`;
                const groupCopyBtn = getTaskGroupCopyDropdownHtml(groupCopyAttrs);
                const dragAttrs = !g.isVirtual ? ` draggable="true" data-group-id="${g.id}"` : ` data-group-id="${g.id}"`;
                const actionRow = `<div class="task-group-actions" onclick="event.stopPropagation()">${pinBtn}<span>${groupCopyBtn}</span>${unarchiveBtn}<button class="group-eye-btn ${eyeCls}" title="${isVisible ? '隐藏看板任务' : '显示看板任务'}" onclick="event.stopPropagation();toggleGroupVisibility('${g.id}')">${eyeIcon}</button>${archiveBtn}</div>`;
                return `<div class="task-group-card task-group-collapsible${isArchived ? ' archived' : ''}${g.isVirtual ? ' ungrouped' : ''}"${dragAttrs} style="min-width:320px;max-width:360px;flex-shrink:0;scroll-snap-align:start;position:relative"
                  onclick="document.getElementById('tasks-group-filter').value='${g.id}';loadTasks()">
                  ${archiveBadge}
                  <div class="task-group-header">
                    <div style="display:flex;align-items:center;gap:12px;flex:1;min-width:0">
                      <svg class="circular-progress" viewBox="0 0 36 36">
                        <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"></path>
                        <path class="circle-fill" style="stroke:${circleColor}" stroke-dasharray="${pct}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"></path>
                        <text x="18" y="21" class="pct-text">${pct}%</text>
                      </svg>
                      <div style="flex:1;min-width:0">
                        <div style="font-weight:500;min-width:0">
                          <span style="display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(g.title).replace(/"/g, '&quot;')}">${esc(g.title)}</span>
                        </div>
                        <div style="font-size:11px;color:var(--text-muted)">${prog.completed}/${prog.total} 任务完成</div>
                        ${stepSummary}
                      </div>
                    </div>
                  </div>
                  ${actionRow}
                  ${retroBlock}
                  ${subtaskHtml}
                  ${sedimentBtn ? `<div class="task-group-footer" style="margin-top:12px;display:flex;justify-content:flex-end"><div onclick="event.stopPropagation()">${sedimentBtn}</div></div>` : ''}
                </div>`;
            }).join('');
            const sectionTitle = _tasksShowArchived ? '归档任务组' : '任务组';
            const showAllBtn = `<button type="button" class="task-group-quick-btn" onclick="showAllTaskGroups()" title="全部显示">全部显示</button>`;
            const hideAllBtn = `<button type="button" class="task-group-quick-btn" onclick="hideAllTaskGroups()" title="全部不显示">全部不显示</button>`;
            groupsContainer.innerHTML =
                `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:8px">` +
                `<h3 style="font-size:14px;font-weight:600;color:var(--text-secondary);margin:0">${sectionTitle}</h3>` +
                `<div style="display:flex;gap:6px">${showAllBtn}${hideAllBtn}</div>` +
                `</div>` +
                `<div id="task-groups-grid" class="task-groups-grid" style="display:flex;gap:16px;overflow-x:auto;padding-bottom:18px;scroll-snap-type: x mandatory">${cardsHtml}</div>`;

            const grid = document.getElementById('task-groups-grid');
            if (grid) {
                const findCard = (el) => el?.closest?.('.task-group-card') || null;
                const clearDragOver = () => grid.querySelectorAll('.task-group-card.drag-over').forEach((c) => c.classList.remove('drag-over'));

                // Mouse wheel over task groups: vertical wheel scrolls horizontally for easier sliding
                grid.addEventListener('wheel', (e) => {
                    const canScrollH = grid.scrollWidth > grid.clientWidth;
                    if (!canScrollH) return;
                    const deltaY = e.deltaY;
                    const deltaX = e.deltaX;
                    if (deltaY !== 0 || deltaX !== 0) {
                        e.preventDefault();
                        grid.scrollLeft += deltaX !== 0 ? deltaX : deltaY;
                    }
                }, { passive: false });

                grid.addEventListener('dragstart', (e) => {
                    const card = findCard(e.target);
                    if (card && card.dataset.groupId && !card.classList.contains('ungrouped')) {
                        e.dataTransfer?.setData?.('text/plain', card.dataset.groupId);
                        e.dataTransfer.effectAllowed = 'move';
                    }
                });
                grid.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    const card = findCard(e.target);
                    if (card && card.dataset.groupId) {
                        clearDragOver();
                        card.classList.add('drag-over');
                    }
                });
                grid.addEventListener('dragleave', (e) => {
                    const card = findCard(e.target);
                    if (card && e.relatedTarget && !card.contains(e.relatedTarget)) card.classList.remove('drag-over');
                });
                grid.addEventListener('drop', async (e) => {
                    e.preventDefault();
                    clearDragOver();
                    const card = findCard(e.target);
                    const targetGroupId = card?.dataset?.groupId;
                    const draggedId = e.dataTransfer?.getData?.('text/plain');
                    if (!draggedId || !targetGroupId) return;

                    const currentGroups = window.__tasksCurrentGroups || [];
                    const realGroups = currentGroups.filter((g) => !g.isVirtual);
                    const fromIdx = realGroups.findIndex((g) => g.id === draggedId);
                    const toIdx = targetGroupId === UNGROUPED_ID ? 0 : realGroups.findIndex((g) => g.id === targetGroupId);
                    if (fromIdx === -1 || toIdx === -1 || fromIdx === toIdx) return;

                    const arr = [...realGroups];
                    const [moved] = arr.splice(fromIdx, 1);
                    arr.splice(toIdx, 0, moved);
                    await reorderTaskGroups(arr.map((g) => g.id));
                });
                grid.addEventListener('dragend', clearDragOver);
            }
            bindCopyDropdowns(groupsContainer);
        } else {
            /* Only show "暂无归档任务组" when on full list (no group filter) and archived tab; hide in group detail view */
            if (!groupFilter && _tasksShowArchived) {
                groupsContainer.innerHTML = `<div class="empty-state" style="padding:32px;text-align:center;color:var(--text-muted)">
                    <div style="font-size:32px;margin-bottom:12px;opacity:0.6">📦</div>
                    <div style="font-size:14px;font-weight:500;margin-bottom:6px">暂无归档任务组</div>
                    <div style="font-size:12px">已完成并归档的任务组将显示在此处</div>
                  </div>`;
            } else {
                groupsContainer.innerHTML = '';
            }
        }
    } catch (e) {
        board.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><h3>加载任务失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

// Step id -> display label for task-execution-workflow (UX: 中文映射)
const WORKFLOW_STEP_LABELS = {
    'step-coldstart': '冷启动',
    'step-claim': '认领',
    'step-execute': '执行',
    'step-complete': '完成并沉淀',
    'step-verify': '验收',
    'step-retro': '组级复盘',
};

function renderWorkflowSteps(steps) {
    return steps.map(s => {
        const label = WORKFLOW_STEP_LABELS[s.step_id] || s.step_id;
        const isCurrent = s.status === 'current';
        const statusIcon = isCurrent ? '<span class="workflow-step-current">进行中</span>' : '<span class="workflow-step-done">✓</span>';
        const summary = (s.last_summary || '').trim();
        const timeStr = s.last_at ? timeAgo(s.last_at) : '';
        return `<div class="workflow-step-item ${isCurrent ? 'current' : ''}">
          <div class="workflow-step-head">${statusIcon} <span class="workflow-step-label">${esc(label)}</span></div>
          ${summary ? `<div class="workflow-step-summary">${esc(summary)}</div>` : ''}
          ${timeStr ? `<div class="workflow-step-time">${esc(timeStr)}</div>` : ''}
        </div>`;
    }).join('');
}

export function showTaskDetail(taskId) {
    const overlay = document.getElementById('task-slideout-overlay');
    const panel = document.getElementById('task-slideout');
    const content = document.getElementById('task-slideout-content');
    if (!content) return;

    content.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    overlay?.classList.add('open');
    panel?.classList.add('open');

    const project = state.activeProject || state.defaultProject || 'default';
    Promise.all([
        api('GET', `/api/v1/tasks/${taskId}?with_context=true`),
        api('GET', `/api/v1/task-groups?project=${encodeURIComponent(project)}`),
    ]).then(([data, groupData]) => {
        const t = data;
        const groups = (groupData.groups || []).filter(g => !g.archived);
        const groupOptions = groups.map(g =>
            `<option value="${g.id}"${g.id === t.group_id ? ' selected' : ''}>${esc(g.title)}</option>`
        ).join('');
        const groupSection = `
          <div class="field-group">
            <div class="field-label">所属任务组</div>
            <div style="display:flex;gap:8px;align-items:center">
              <select id="sl-group" style="flex:1">
                <option value="">无任务组</option>
                ${groupOptions}
              </select>
              <button class="btn btn-sm" style="font-size:11px;padding:4px 10px;white-space:nowrap"
                onclick="createTaskGroupFromSlideout('${t.id}')">+ 新建</button>
            </div>
          </div>`;
        const slideoutTaskCopyAttrs = `data-task-id="${esc(t.id)}" data-task-title="${esc((t.title || '').replace(/"/g, '&quot;'))}" data-task-status="${esc(t.status || '')}" data-task-priority="${esc(t.priority || '')}" data-task-group-id="${esc(t.group_id || '')}"`;
        const slideoutTaskCopyBtn = getTaskCopyDropdownHtml(slideoutTaskCopyAttrs);
        content.innerHTML = `
        <h2 style="display:flex;align-items:center;gap:8px">任务详情<span>${slideoutTaskCopyBtn}</span></h2>
        ${groupSection}
        <div class="field-group">
          <div class="field-label">标题</div>
          <input id="sl-title" value="${esc(t.title || '')}" />
        </div>
        <div class="field-group">
          <div class="field-label">描述</div>
          <textarea id="sl-desc">${esc(t.description || '')}</textarea>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          <div class="field-group">
            <div class="field-label">状态</div>
            <select id="sl-status">
              ${['wait', 'plan', 'in_progress', 'completed', 'cancelled'].map(s =>
                `<option value="${s}"${s === t.status ? ' selected' : ''}>${s}</option>`
              ).join('')}
            </select>
          </div>
          <div class="field-group">
            <div class="field-label">优先级</div>
            <select id="sl-priority">
              ${['low', 'medium', 'high', 'urgent'].map(p =>
                `<option value="${p}"${p === t.priority ? ' selected' : ''}>${p}</option>`
              ).join('')}
            </select>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          <div class="field-group">
            <div class="field-label">重要度 (1-5)</div>
            <input id="sl-importance" type="number" min="1" max="5" value="${t.importance || 3}" />
          </div>
          <div class="field-group">
            <div class="field-label">截止日期</div>
            <input id="sl-due" type="date" value="${(t.due_date || '').slice(0, 10)}" />
          </div>
        </div>
        <div class="field-group">
          <div class="field-label">验收标准</div>
          <textarea id="sl-acceptance-criteria" placeholder="与当前 step 的 acceptance_criteria 关联">${esc(t.acceptance_criteria || '')}</textarea>
        </div>
        <div class="field-group">
          <div class="field-label">已验收</div>
          <select id="sl-acceptance-met">
            <option value="">未设置</option>
            <option value="true"${t.acceptance_met === true ? ' selected' : ''}>是</option>
            <option value="false"${t.acceptance_met === false ? ' selected' : ''}>否</option>
          </select>
        </div>
        ${t.sediment_experience_id ? `<div class="field-group"><div class="field-label">沉淀经验</div><a onclick="closeTaskSlideout();showDetail('${t.sediment_experience_id}')" style="color:var(--accent);cursor:pointer;font-size:13px">查看关联经验</a></div>` : ''}
        <div class="slideout-actions">
          <button class="btn btn-primary btn-sm" onclick="saveTaskFromSlideout('${t.id}')">保存</button>
          <button class="btn btn-sm" style="background:var(--accent-glow);color:var(--accent)" onclick="generateTaskPrompt('${t.id}')">AI Prompt</button>
          <button class="btn btn-danger btn-sm" onclick="deleteTaskFromSlideout('${t.id}')">删除</button>
          <div style="flex:1"></div>
          <button class="btn btn-sm" onclick="closeTaskSlideout()">关闭</button>
        </div>
        <div class="field-group workflow-progress-block" id="sl-workflow-progress-block">
          <div class="field-label">工作流进度</div>
          <div id="sl-workflow-steps"><div class="workflow-steps-loading">加载中...</div></div>
        </div>
        <div class="msg-list" id="sl-messages">
          <div class="field-label">消息</div>
          <div id="sl-msg-list"><div style="color:var(--text-muted);font-size:12px">加载中...</div></div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <input id="sl-msg-input" placeholder="添加消息..." style="flex:1" />
            <button class="btn btn-primary btn-sm" onclick="sendTaskMessage('${t.id}')">发送</button>
          </div>
        </div>
        <div style="margin-top:12px;font-size:11px;color:var(--text-muted)">ID: ${t.id}</div>`;

        bindCopyDropdowns(content);

        // Load messages
        api('GET', `/api/v1/tasks/${taskId}/messages`).then(msgData => {
            const list = document.getElementById('sl-msg-list');
            if (!list) return;
            const msgs = msgData.messages || [];
            if (msgs.length === 0) {
                list.innerHTML = '<div style="color:var(--text-muted);font-size:12px">暂无消息</div>';
            } else {
                list.innerHTML = msgs.map(m => `<div class="msg-item"><div>${esc(m.content)}</div><div class="msg-meta">${esc(m.author || m.sender)} · ${_fmtDate(m.created_at)}</div></div>`).join('');
            }
        }).catch(() => {});

        // Load workflow steps (task-execution-workflow)
        const workflowStepsEl = document.getElementById('sl-workflow-steps');
        if (workflowStepsEl) {
            api('GET', `/api/v1/tasks/${taskId}/workflow-steps?workflow_id=task-execution-workflow`)
                .then(data => {
                    const steps = data.steps || [];
                    if (steps.length === 0) {
                        workflowStepsEl.innerHTML = '<div class="workflow-steps-empty">暂无工作流进度</div><div class="workflow-steps-empty-hint">按任务工作流执行后将在此展示</div>';
                    } else {
                        workflowStepsEl.innerHTML = renderWorkflowSteps(steps);
                    }
                })
                .catch(() => {
                    workflowStepsEl.innerHTML = '<div class="workflow-steps-empty">加载失败</div>';
                });
        }
    }).catch(e => {
        content.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    });
}

export function closeTaskSlideout() {
    document.getElementById('task-slideout-overlay')?.classList.remove('open');
    document.getElementById('task-slideout')?.classList.remove('open');
}

export async function saveTaskFromSlideout(taskId) {
    const groupVal = document.getElementById('sl-group')?.value || '';
    const acceptanceMetVal = document.getElementById('sl-acceptance-met')?.value;
    const body = {
        title: document.getElementById('sl-title')?.value,
        description: document.getElementById('sl-desc')?.value,
        status: document.getElementById('sl-status')?.value,
        priority: document.getElementById('sl-priority')?.value,
        importance: parseInt(document.getElementById('sl-importance')?.value, 10) || 3,
        due_date: document.getElementById('sl-due')?.value || null,
        group_id: groupVal || null,
        acceptance_criteria: document.getElementById('sl-acceptance-criteria')?.value || null,
        acceptance_met: acceptanceMetVal === '' ? null : acceptanceMetVal === 'true',
    };
    try {
        await api('PUT', `/api/v1/tasks/${taskId}`, body);
        toast('任务已保存', 'success');
        loadTasks();
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function createTaskGroupFromSlideout(taskId) {
    const name = prompt('请输入新任务组名称:');
    if (!name || !name.trim()) return;
    try {
        const project = state.activeProject || state.defaultProject || 'default';
        const res = await api('POST', '/api/v1/task-groups', {
            title: name.trim(),
            project,
        });
        const newGroupId = res.id;
        await api('PUT', `/api/v1/tasks/${taskId}`, { group_id: newGroupId });
        toast('任务组已创建并关联', 'success');
        loadTasks();
        showTaskDetail(taskId);
    } catch (e) {
        toast('创建任务组失败: ' + e.message, 'error');
    }
}

export async function deleteTaskFromSlideout(taskId) {
    if (!confirm('确定删除此任务？')) return;
    try {
        await api('DELETE', `/api/v1/tasks/${taskId}`);
        toast('任务已删除', 'success');
        closeTaskSlideout();
        loadTasks();
    } catch (e) {
        toast('删除失败: ' + e.message, 'error');
    }
}

export async function sendTaskMessage(taskId) {
    const input = document.getElementById('sl-msg-input');
    const content = input?.value?.trim();
    if (!content) return;
    try {
        await api('POST', `/api/v1/tasks/${taskId}/messages`, { content });
        input.value = '';
        showTaskDetail(taskId); // reload
    } catch (e) {
        toast('发送失败: ' + e.message, 'error');
    }
}

export async function sedimentTaskGroup(groupId) {
    try {
        const res = await api('POST', `/api/v1/task-groups/${groupId}/sediment`);
        if (res.already_exists) {
            toast(res.message || '该任务组已有组级经验', 'info');
            if (res.experience_id) {
                closeTaskSlideout();
                showDetail(res.experience_id);
            }
        } else {
            toast('组经验已沉淀', 'success');
            const expId = res.id || res.experience_id;
            if (expId) {
                closeTaskSlideout();
                showDetail(expId);
            }
            loadTasks();
        }
    } catch (e) {
        toast('沉淀失败: ' + (e.message || e.detail || '未知错误'), 'error');
    }
}

async function reorderTaskGroups(newOrder) {
    await api('PUT', '/api/v1/task-groups/reorder', { order: newOrder });
    loadTasks();
}

export async function pinTaskGroupToTop(groupId) {
    const currentGroups = window.__tasksCurrentGroups || [];
    const realGroups = currentGroups.filter((g) => !g.isVirtual);
    const idx = realGroups.findIndex((g) => g.id === groupId);
    if (idx === -1) return;
    const newOrder = [groupId, ...realGroups.filter((g) => g.id !== groupId).map((g) => g.id)];
    await reorderTaskGroups(newOrder);
    toast('已置顶', 'success');
}

export async function archiveGroup(groupId) {
    if (!confirm('确定归档此任务组？归档后将不在任务列表显示。')) return;
    try {
        await api('PUT', `/api/v1/task-groups/${groupId}`, { archived: true });
        toast('任务组已归档', 'success');
        loadTasks();
    } catch (e) {
        toast('归档失败: ' + e.message, 'error');
    }
}

export async function unarchiveGroup(groupId) {
    try {
        await api('PUT', `/api/v1/task-groups/${groupId}`, { archived: false });
        toast('任务组已恢复', 'success');
        loadTasks();
    } catch (e) {
        toast('恢复失败: ' + e.message, 'error');
    }
}

export function toggleTaskGroups() {
    const cards = document.querySelectorAll('.task-group-collapsible');
    const btn = document.getElementById('toggle-task-groups-btn');
    if (!btn) return;
    const expanded = btn.dataset.expanded === '1';
    cards.forEach((c, i) => {
        if (i >= 4) c.style.display = expanded ? 'none' : '';
    });
    const hiddenCount = cards.length - 4;
    if (expanded) {
        btn.textContent = `展开全部 (${hiddenCount} 个隐藏)`;
        btn.dataset.expanded = '0';
    } else {
        btn.textContent = '收起';
        btn.dataset.expanded = '1';
    }
}

export function toggleGroupVisibility(groupId) {
    _kanbanInitialized = true;
    if (_kanbanVisibleGroups.has(groupId)) {
        _kanbanVisibleGroups.delete(groupId);
    } else {
        _kanbanVisibleGroups.add(groupId);
    }
    _saveVisibleGroupsToStorage();
    loadTasks();
}

export function showAllTaskGroups() {
    _kanbanInitialized = true;
    _kanbanVisibleGroups.clear();
    (window.__tasksCurrentGroups || []).forEach(g => _kanbanVisibleGroups.add(g.id));
    _saveVisibleGroupsToStorage();
    loadTasks();
}

export function hideAllTaskGroups() {
    _kanbanInitialized = true;
    _kanbanVisibleGroups.clear();
    _saveVisibleGroupsToStorage();
    loadTasks();
}

async function generateTaskPrompt(taskId) {
    try {
        const data = await api('GET', `/api/v1/tasks/${taskId}?with_context=true`);
        const lines = [
            `## 任务: ${data.title}`,
            data.description ? `\n${data.description}` : '',
            `\n优先级: ${data.priority} | 重要度: ${data.importance}/5`,
            data.due_date ? `截止: ${data.due_date}` : '',
        ];
        if (data.experience_context) {
            lines.push(`\n### 关联经验\n${data.experience_context.title}\n${data.experience_context.solution || data.experience_context.description}`);
        }
        lines.push(`\n### 执行后请调用\ntm_task action=update task_id=${taskId} status=completed summary="<执行摘要>"`);
        const prompt = lines.filter(Boolean).join('\n');
        const ok = await copyTextToClipboard(prompt);
        if (ok) toast('已复制到剪贴板，请在 Cursor 中粘贴执行', 'success');
        else toast('复制失败', 'error');
    } catch (e) {
        toast('生成 prompt 失败: ' + e.message, 'error');
    }
}

export { generateTaskPrompt };

export function toggleDupDiff(idx) {
    const el = document.getElementById('dup-diff-' + idx);
    if (el) el.classList.toggle('hidden');
}

export function populateTagSuggestions() {
    const container = document.getElementById('create-tag-suggestions');
    if (!container) return;
    const tags = state.allTags || {};
    const topTags = Object.entries(tags).sort((a, b) => b[1] - a[1]).slice(0, 12);
    if (topTags.length === 0) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = topTags.map(([tag]) =>
        `<span class="tag-suggest-btn" onclick="appendTag('${esc(tag)}')">${esc(tag)}</span>`
    ).join('');
}

export function appendTag(tag) {
    const input = document.getElementById('create-tags');
    if (!input) return;
    const existing = input.value.split(',').map(t => t.trim()).filter(Boolean);
    if (!existing.includes(tag)) {
        existing.push(tag);
        input.value = existing.join(', ');
    }
}

export function populateSettingsProjectDropdown() {
    const sel = document.getElementById('cfg-default-project');
    if (!sel) return;
    const projects = state.availableProjects || [];
    const current = state.defaultProject || 'default';
    let html = '';
    const allProjects = new Set([current, ...projects]);
    allProjects.forEach(p => {
        html += `<option value="${p}"${p === current ? ' selected' : ''}>${p}</option>`;
    });
    sel.innerHTML = html;
}

function _esc(s) { return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function _fmtDate(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

// ===== Quality Scoring Management =====

export async function loadScoringConfig() {
    try {
        const cfg = await api('GET', '/api/v1/config/scoring');
        const fields = {
            'scoring-initial': cfg.initial_score,
            'scoring-max': cfg.max_score,
            'scoring-protection': cfg.protection_days,
            'scoring-decay-rate': cfg.decay_rate,
            'scoring-slow-threshold': cfg.slow_decay_threshold,
            'scoring-slow-rate': cfg.slow_decay_rate,
            'scoring-ref-boost': cfg.reference_boost,
            'scoring-rating-boost': cfg.high_rating_boost,
            'scoring-rating-threshold': cfg.high_rating_threshold,
            'scoring-tier-gold': (cfg.tiers || {}).gold,
            'scoring-tier-silver': (cfg.tiers || {}).silver,
            'scoring-tier-bronze': (cfg.tiers || {}).bronze,
        };
        for (const [id, val] of Object.entries(fields)) {
            const el = document.getElementById(id);
            if (el && val !== undefined) el.value = val;
        }
    } catch (_) { /* non-blocking */ }
}

export async function saveScoringConfig() {
    const g = (id) => { const el = document.getElementById(id); return el ? Number(el.value) : undefined; };
    const body = {
        initial_score: g('scoring-initial'),
        max_score: g('scoring-max'),
        protection_days: g('scoring-protection'),
        decay_rate: g('scoring-decay-rate'),
        slow_decay_threshold: g('scoring-slow-threshold'),
        slow_decay_rate: g('scoring-slow-rate'),
        reference_boost: g('scoring-ref-boost'),
        high_rating_boost: g('scoring-rating-boost'),
        high_rating_threshold: g('scoring-rating-threshold'),
        tiers: {
            gold: g('scoring-tier-gold'),
            silver: g('scoring-tier-silver'),
            bronze: g('scoring-tier-bronze'),
        },
    };
    try {
        await api('PUT', '/api/v1/config/scoring', body);
        alert('打分规则已保存');
    } catch (e) {
        alert('保存失败: ' + e.message);
    }
}

export async function toggleOutdatedPanel() {
    const panel = document.getElementById('outdated-panel');
    if (!panel) return;
    const isHidden = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    if (isHidden) await loadOutdatedList();
}

export async function loadOutdatedList() {
    const list = document.getElementById('outdated-list');
    if (!list) return;
    list.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', '/api/v1/lifecycle/outdated');
        const exps = data.experiences || [];
        if (exps.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:20px">没有 Outdated 经验</p>';
            return;
        }
        list.innerHTML = exps.map(exp => `
            <div style="display:flex;align-items:center;gap:12px;padding:10px;border-bottom:1px solid var(--border)">
                <span style="flex:1;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(exp.title)}</span>
                <span style="font-size:12px;color:var(--text-muted)">Score: ${exp.quality_score}</span>
                <button class="btn btn-secondary btn-sm" onclick="scoreAction('${exp.id}','restore')">恢复</button>
                <button class="btn btn-secondary btn-sm" onclick="scoreAction('${exp.id}','pin')">📌 置顶</button>
                <button class="btn btn-secondary btn-sm" style="color:var(--red)" onclick="if(confirm('确认删除？'))scoreAction('${exp.id}','delete')">删除</button>
            </div>
        `).join('');
    } catch (e) {
        list.innerHTML = `<p style="color:var(--red);padding:12px">${esc(e.message)}</p>`;
    }
}

export async function scoreAction(expId, action) {
    try {
        await api('POST', `/api/v1/lifecycle/experiences/${expId}/score-action`, { action });
        await loadOutdatedList();
        await checkOutdatedCount();
    } catch (e) {
        alert('操作失败: ' + e.message);
    }
}

export async function refreshScores() {
    try {
        const r = await api('POST', '/api/v1/lifecycle/refresh-scores');
        alert(r.message || '已刷新');
        await loadOutdatedList();
        await checkOutdatedCount();
    } catch (e) {
        alert('刷新失败: ' + e.message);
    }
}

export async function checkOutdatedCount() {
    try {
        const data = await api('GET', '/api/v1/lifecycle/outdated');
        const count = (data.experiences || []).length;
        const btn = document.getElementById('btn-manage-outdated');
        const dot = document.getElementById('outdated-dot');
        if (btn) btn.style.display = count > 0 ? 'inline-flex' : 'none';
        if (dot) dot.style.display = count > 0 ? 'block' : 'none';
    } catch (_) { /* ignore */ }
}

export async function checkMergeSuggestions() {
    try {
        const data = await api('GET', '/api/v1/lifecycle/merge-suggestions?limit=5');
        const count = (data.suggestions || []).length;
        const dot = document.getElementById('merge-suggestion-dot');
        if (dot) dot.classList.toggle('active', count > 0);
    } catch (_) { /* ignore */ }
}
