/**
 * Page rendering functions: list, detail, dashboard, drafts, reviews, settings.
 */

import { state, defaultTypeIcons } from './store.js';
import { resolveProjectInput } from './schema.js';
import { esc, formatDate, timeAgo, renderMarkdown } from './utils.js';

function api(...args) {
    return window.__api(...args);
}

function toast(msg, type = 'info') {
    return window.__toast(msg, type);
}

function navigate(page) {
    return window.__navigate(page);
}
const typeIcons = {
    general: '📝',
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
                copyExpOption(option, dropdown);
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
    container.innerHTML = experiences
        .map((exp) => {
            const view = exp.parent || exp;
            const cardId = exp.group_id || view.id || exp.id || '';
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
            const useCount = view.use_count || 0;
            const metricsHtml = `<div class="card-metrics"><span>📊 ${useCount}</span></div>`;
            const tagsStr = (view.tags || []).join(', ');
            const copyId = String(cardId ?? '');
            const copyAttrs = `data-exp-id="${esc(copyId)}" data-exp-title="${esc(view.title || '')}" data-exp-tags="${esc(tagsStr)}" data-exp-created="${esc(view.created_at || '')}" data-exp-created-by="${esc(view.created_by || '')}" data-exp-type="${esc(view.experience_type || 'general')}" data-exp-project="${esc(view.project || '')}"`;
            return `
    <div class="exp-card" onclick="showDetail('${cardId}')">
      <div class="exp-card-header">
        <div class="exp-card-title">
          <span class="type-icon">${typeIcon}</span>${esc(view.title)}${view.status === 'draft' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--accent-glow);color:var(--accent);margin-left:6px">草稿</span>' : ''}${view.visibility === 'private' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--green-bg,#e8f5e9);color:var(--green,#2e7d32);margin-left:6px">仅自己</span>' : ''}${view.visibility === 'global' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:#e0f2fe;color:#0369a1;margin-left:6px">全局</span>' : ''}
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
        <div class="exp-card-tags">${view.visibility === 'global' ? '<span class="tag" style="background:#e0f2fe;color:#0369a1;font-weight:600">全局</span>' : ''}${view.visibility === 'private' ? '<span class="tag" style="background:#f3e8ff;color:#7c3aed;font-weight:600">仅自己</span>' : ''}${(view.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}${exp.children_count > 0 || exp.total_children > 0 ? `<span class="children-badge">${exp.children_count || exp.total_children} steps</span>` : ''}</div>
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
let _listSubTab = 'all'; // 'all' | 'draft'

export function switchListSubTab(tab) {
    if (tab === 'review') tab = 'all'; // review tab removed, fall back to all
    _listSubTab = tab;
    document.querySelectorAll('#page-list .mode-tab').forEach((el) => el.classList.remove('active'));
    const tabEl = document.getElementById(`list-tab-${tab}`);
    if (tabEl) tabEl.classList.add('active');
    const statusFilter = document.getElementById('list-status-filter');
    if (statusFilter) {
        if (tab === 'draft') statusFilter.value = 'draft';
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

    try {
        const statusFilter = document.getElementById('list-status-filter')?.value || '';
        const typeFilter = document.getElementById('list-type-filter')?.value || '';
        const visibilityFilter = document.getElementById('list-visibility-filter')?.value || '';
        let url = `/api/v1/experiences?limit=15&offset=${(page - 1) * 15}`;
        if (projectFilter) url += `&project=${encodeURIComponent(projectFilter)}`;
        if (visibilityFilter) url += `&visibility=${encodeURIComponent(visibilityFilter)}`;
        if (statusFilter) url += `&status=${statusFilter}`;
        if (state.selectedTag) url += `&tag=${encodeURIComponent(state.selectedTag)}`;
        if (typeFilter) url += `&experience_type=${encodeURIComponent(typeFilter)}`;

        const data = await api('GET', url);
        renderExpList('list-content', data.items);
        renderPagination(data);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export function filterByTag(tag) {
    state.selectedTag = tag;
    state.listPage = 1;
    navigate('list');
}

function renderPagination(data) {
    const el = document.getElementById('list-pagination');
    const totalPages = Math.ceil(data.total / data.limit) || 1;
    const currentPage = Math.floor(data.offset / data.limit) + 1;
    if (totalPages <= 1) {
        el.innerHTML = '';
        return;
    }
    el.innerHTML = `
    <button class="btn btn-secondary btn-sm" onclick="loadList(${currentPage - 1})" ${currentPage <= 1 ? 'disabled' : ''}>上一页</button>
    <span class="page-info">${currentPage} / ${totalPages}</span>
    <button class="btn btn-secondary btn-sm" onclick="loadList(${currentPage + 1})" ${currentPage >= totalPages ? 'disabled' : ''}>下一页</button>
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
        const exp = await api('GET', `/api/v1/experiences/${id}`);
        state.currentDetail = exp;
        const typeIcon = typeIcons[exp.experience_type] || defaultTypeIcons[exp.experience_type] || '📝';
        const typeBadges = `<span class="type-icon" style="font-size:20px">${typeIcon}</span>`;

        const backPage = state.detailReferrer || 'list';
        const backLabels = { reviews: '经验列表', drafts: '草稿箱', list: '经验列表', search: '语义搜索', dashboard: '仪表盘' };
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
            ${exp.status === 'published' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--green-bg,#e8f5e9);color:var(--green,#2e7d32);margin-left:12px;vertical-align:middle">已发布</span>' : ''}
            ${exp.visibility === 'private' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#f3e8ff;color:#7c3aed;margin-left:6px;vertical-align:middle">仅自己</span>' : ''}
            ${exp.visibility === 'global' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#e0f2fe;color:#0369a1;margin-left:6px;vertical-align:middle">全局</span>' : ''}
            ${exp.visibility === 'project' ? '<span style="font-size:11px;padding:1px 8px;border-radius:3px;background:#fef3c7;color:#92400e;margin-left:6px;vertical-align:middle">项目内</span>' : ''}
          </h1>
          <div class="detail-meta" style="align-items:center">
            <span>👤 ${esc(exp.created_by)}</span>
            <span>📅 ${formatDate(exp.created_at)}</span>
            <span>📊 ${exp.use_count} 次引用</span>
            ${getCopyDropdownDetailHtml()}
          </div>
          <div style="margin-top:12px">${(exp.tags || []).map((t) => `<span class="tag" onclick="filterByTag('${esc(t)}')">${esc(t)}</span>`).join('')}</div>
        </div>
        <div class="detail-body">
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
                  </div>
                </div>
              `
                  )
                  .join('')}
            </div>
          </div>
          ` : ''}
        </div>
        <div class="detail-actions">
          ${exp.status === 'draft' ? `
            <button class="btn btn-sm" style="background:var(--green);color:#fff;margin-left:4px" onclick="changeExpStatus('${exp.id}','published')">直接发布</button>` : ''}
          ${exp.status === 'published' ? `
            <button class="btn btn-sm" style="background:var(--accent-glow);color:var(--accent)" onclick="changeExpStatus('${exp.id}','draft')">撤回到草稿</button>` : ''}
          <button class="btn btn-primary btn-sm" onclick="openEditModal('${exp.id}')">✏️ 编辑</button>
          <button class="btn btn-primary btn-sm" onclick="openFeedbackModal('${exp.id}')">💬 提交反馈</button>
          <button class="btn btn-danger btn-sm" onclick="deleteExp('${exp.id}')">🗑 删除</button>
          <div style="flex:1"></div>
          <span style="font-size:12px;color:var(--text-muted)">ID: ${exp.id}</span>
        </div>
      </div>
    `;
        bindCopyDropdowns(page);
    } catch (e) {
        page.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

// ===== Drafts =====
export async function loadDrafts() {
    const container = document.getElementById('drafts-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const project = state.activeProject || state.defaultProject || 'default';
        const data = await api('GET', `/api/v1/experiences?status=draft&limit=50&offset=0&project=${encodeURIComponent(project)}`);
        if (!data.items || data.items.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="icon">📝</div><h3>暂无草稿</h3><p>创建经验时勾选"保存为草稿"即可</p></div>';
            return;
        }
        container.innerHTML = data.items
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
    const labels = { draft: '草稿', published: '已发布' };
    const label = labels[newStatus] || newStatus;
    if (!confirm(`确定要将状态改为「${label}」吗？`)) return;
    try {
        const body = { status: newStatus };
        if (newVisibility) body.visibility = newVisibility;
        const res = await api('PATCH', `/api/v1/experiences/${id}/status`, body);
        toast(res.message || '操作成功', 'success');
        showDetail(id);
    } catch (e) {
        toast('状态变更失败: ' + e.message, 'error');
    }
}

export async function publishDraft(id, target = 'personal') {
    const newStatus = target === 'team' ? 'published' : 'published';
    const newVis = target === 'team' ? 'project' : 'private';
    await changeExpStatus(id, newStatus, newVis);
}

// ===== Reviews =====
export async function loadReviews() {
    const container = document.getElementById('reviews-content');
    if (!container) return;
    container.innerHTML = '<div class="empty-state"><div class="icon">📋</div><h3>审核功能已移除</h3><p>经验现在通过状态流转管理（草稿 → 已发布）</p></div>';
}

export async function reviewExperience(id, status) {
    const newStatus = status === 'approved' ? 'published' : 'rejected';
    await changeExpStatus(id, newStatus);
}

// ===== Architecture (removed: architecture page and GitNexus viewer) =====
export async function loadArchitecture() { /* no-op */ }
export async function switchArchitectureTab() { /* no-op */ }

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
    const listEl = document.getElementById('installables-list');
    if (!listEl) return;
    const source = document.getElementById('installables-source-filter')?.value || '';
    const type = document.getElementById('installables-type-filter')?.value || '';
    listEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
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
    const targetProject = document.getElementById('installables-target-project')?.value || '';
    const targetPath = document.getElementById('installables-target-path')?.value?.trim() || '';
    if (!targetProject && !targetPath) {
        toast('请选择安装目标（项目或路径）', 'error');
        return;
    }
    if (!confirm(`确认安装 ${id} 到 ${targetProject || targetPath}？`)) return;
    try {
        const body = { id, source };
        if (targetProject) body.target_project = targetProject;
        if (targetPath) body.target_path = targetPath;
        const result = await api('POST', '/api/v1/installables/install', body);
        toast('安装成功: ' + (result.target_path || ''), 'success');
        const proj = targetProject || (targetPath ? 'current' : null);
        if (proj && document.getElementById('installables-installed-project')?.value === proj) {
            loadInstalledInstallables();
        }
    } catch (e) {
        toast('安装失败: ' + e.message, 'error');
    }
}

export async function loadInstalledInstallables() {
    const project = document.getElementById('installables-installed-project')?.value || '';
    const listEl = document.getElementById('installables-installed-list');
    if (!listEl) return;
    if (!project) {
        listEl.innerHTML = '<div class="empty-state"><p>选择项目后加载已安装项</p></div>';
        return;
    }
    listEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const data = await api('GET', `/api/v1/installables/installed?project=${encodeURIComponent(project)}`);
        const items = data.items || [];
        if (items.length === 0) {
            listEl.innerHTML = '<div class="empty-state"><p>该项目暂无已安装项</p></div>';
            return;
        }
        listEl.innerHTML = items.map((it) => `
            <div class="exp-card" style="cursor:default">
              <div class="exp-card-header">
                <div class="exp-card-title">
                  ${it.type === 'rule' ? '📐' : '🧠'} ${esc(it.name)}
                  <span class="tag">${esc(it.type)}</span>
                </div>
              </div>
              <div class="installable-item-actions" style="margin-top:8px">
                <button class="btn btn-secondary btn-sm" data-project="${esc(project)}" data-item-id="${esc(it.item_id)}" data-type="${esc(it.type)}" data-name="${esc(it.name)}" onclick="openEditInstallableModalFromBtn(this)">编辑</button>
              </div>
            </div>
        `).join('');
    } catch (e) {
        listEl.innerHTML = `<div class="empty-state"><p>加载失败: ${esc(e.message)}</p></div>`;
    }
}

let _editInstallableCtx = { project: '', item_id: '', item_type: '', name: '' };

export function openEditInstallableModalFromBtn(btn) {
    const d = btn.dataset;
    openEditInstallableModal(d.project || '', d.itemId || '', d.type || '', d.name || '');
}

export async function openEditInstallableModal(project, itemId, itemType, name) {
    _editInstallableCtx = { project, item_id: itemId, item_type: itemType, name };
    document.getElementById('edit-installable-title').textContent = `编辑 ${itemType}: ${name}`;
    document.getElementById('edit-installable-content').value = '';
    switchEditInstallableTab('edit');
    document.getElementById('edit-installable-modal').classList.remove('hidden');
    try {
        const data = await api('GET', `/api/v1/installables/custom?project=${encodeURIComponent(project)}&item_id=${encodeURIComponent(itemId)}`);
        document.getElementById('edit-installable-content').value = data.content || '';
    } catch (e) {
        toast('加载内容失败: ' + e.message, 'error');
    }
}

export function switchEditInstallableTab(tab) {
    document.querySelectorAll('.edit-installable-tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
    const editPanel = document.getElementById('edit-installable-edit-panel');
    const previewPanel = document.getElementById('edit-installable-preview-panel');
    if (tab === 'edit') {
        if (editPanel) editPanel.classList.remove('hidden');
        if (previewPanel) previewPanel.classList.add('hidden');
    } else {
        if (editPanel) editPanel.classList.add('hidden');
        if (previewPanel) previewPanel.classList.remove('hidden');
        renderEditInstallablePreview();
    }
}

function renderEditInstallablePreview() {
    const textarea = document.getElementById('edit-installable-content');
    const previewEl = document.getElementById('edit-installable-preview-content');
    if (!textarea || !previewEl) return;
    const raw = textarea.value || '';
    if (typeof marked !== 'undefined') {
        previewEl.innerHTML = marked.parse(raw, { gfm: true, breaks: true });
    } else {
        previewEl.textContent = raw || '(无内容)';
    }
}

export function closeEditInstallableModal() {
    document.getElementById('edit-installable-modal').classList.add('hidden');
}

export async function saveEditInstallableContent() {
    const { project, item_id, item_type } = _editInstallableCtx;
    const content = document.getElementById('edit-installable-content')?.value || '';
    if (!content.trim()) {
        toast('内容不能为空', 'error');
        return;
    }
    try {
        await api('PUT', '/api/v1/installables/custom', {
            project,
            item_id,
            item_type,
            content,
            sync_to_file: true,
        });
        toast('保存成功', 'success');
        closeEditInstallableModal();
        loadInstalledInstallables();
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

// ===== Settings: Health Status =====

/** Load and render /health for ops; no auth required. */
export async function loadHealthStatus() {
    const loadingEl = document.getElementById('health-status-loading');
    const bodyEl = document.getElementById('health-status-body');
    const errorEl = document.getElementById('health-status-error');
    const dotEl = document.getElementById('health-status-dot');
    if (!loadingEl || !bodyEl || !errorEl) return;

    loadingEl.classList.remove('hidden');
    loadingEl.textContent = '正在检测…';
    bodyEl.classList.add('hidden');
    errorEl.classList.add('hidden');
    errorEl.textContent = '';
    if (dotEl) {
        dotEl.className = 'health-dot loading';
        dotEl.title = '加载中…';
    }

    try {
        const base = window.__apiBaseUrl || '';
        const r = await fetch(base ? `${base}/health` : '/health', { method: 'GET', credentials: 'same-origin' });
        const data = await r.json().catch(() => ({}));

        loadingEl.classList.add('hidden');

        const status = data.status || 'unknown';
        const checks = data.checks || {};
        const version = data.version || '';
        const timestamp = data.timestamp || '';

        const dotEl = document.getElementById('health-status-dot');
        if (dotEl) {
            dotEl.className = 'health-dot ' + (status === 'healthy' ? 'healthy' : 'unhealthy');
            dotEl.title = status === 'healthy' ? '正常' : status === 'degraded' ? '降级' : '异常';
        }

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
        const dotEl = document.getElementById('health-status-dot');
        if (dotEl) {
            dotEl.className = 'health-dot unhealthy';
            dotEl.title = '获取失败';
        }
    }
}

// ===== Runtime config (retrieval / search) =====

function setSettingsSaveStatus(msg, isError = false) {
    const el = document.getElementById('settings-save-status');
    if (!el) return;
    el.textContent = msg || '';
    el.classList.toggle('save-status-error', Boolean(isError));
    if (!isError && msg) el.classList.remove('save-status-error');
}

/** Load GET /config/* into settings form and cache. */
export async function loadRuntimeConfigForms() {
    setSettingsSaveStatus('');
    try {
        const [retrieval, search] = await Promise.all([
            api('GET', '/api/v1/config/retrieval'),
            api('GET', '/api/v1/config/search'),
        ]);
        state.cachedRetrievalConfig = retrieval;

        const mt = document.getElementById('cfg-max-tokens');
        if (mt) mt.value = retrieval.max_tokens != null && retrieval.max_tokens !== '' ? String(retrieval.max_tokens) : '';

        const mc = document.getElementById('cfg-max-count');
        if (mc) mc.value = retrieval.max_count != null ? String(retrieval.max_count) : '';

        const ts = document.getElementById('cfg-trim-strategy');
        if (ts && retrieval.trim_strategy) ts.value = retrieval.trim_strategy;

        const tkc = document.getElementById('cfg-top-k-children');
        if (tkc) tkc.value = retrieval.top_k_children != null ? String(retrieval.top_k_children) : '';

        const mar = document.getElementById('cfg-min-avg-rating');
        if (mar) mar.value = retrieval.min_avg_rating != null ? String(retrieval.min_avg_rating) : '';

        const rw = document.getElementById('cfg-rating-weight');
        if (rw) rw.value = retrieval.rating_weight != null ? String(retrieval.rating_weight) : '';

        const sm = document.getElementById('cfg-summary-model');
        if (sm) sm.value = retrieval.summary_model ? String(retrieval.summary_model) : '';

        const smode = document.getElementById('cfg-search-mode');
        if (smode && search.mode) smode.value = search.mode;

        const rrf = document.getElementById('cfg-rrf-k');
        if (rrf) rrf.value = search.rrf_k != null ? String(search.rrf_k) : '';

        const vw = document.getElementById('cfg-vector-weight');
        if (vw) vw.value = search.vector_weight != null ? String(search.vector_weight) : '';

        const fw = document.getElementById('cfg-fts-weight');
        if (fw) fw.value = search.fts_weight != null ? String(search.fts_weight) : '';

        const af = document.getElementById('cfg-adaptive-filter');
        if (af) af.value = search.adaptive_filter ? 'true' : 'false';

        const sg = document.getElementById('cfg-score-gap');
        if (sg) sg.value = search.score_gap_threshold != null ? String(search.score_gap_threshold) : '';

        const mcr = document.getElementById('cfg-min-confidence');
        if (mcr) mcr.value = search.min_confidence_ratio != null ? String(search.min_confidence_ratio) : '';
    } catch (e) {
        setSettingsSaveStatus('加载配置失败: ' + (e.message || String(e)), true);
    }
}

export async function saveRetrievalConfig() {
    setSettingsSaveStatus('');
    const maxTokensRaw = document.getElementById('cfg-max-tokens')?.value?.trim() || '';
    const body = {
        max_tokens: maxTokensRaw === '' ? null : parseInt(maxTokensRaw, 10),
        max_count: parseInt(document.getElementById('cfg-max-count')?.value || '20', 10),
        trim_strategy: document.getElementById('cfg-trim-strategy')?.value || 'top_k',
        top_k_children: parseInt(document.getElementById('cfg-top-k-children')?.value || '3', 10),
        min_avg_rating: parseFloat(document.getElementById('cfg-min-avg-rating')?.value || '0'),
        rating_weight: parseFloat(document.getElementById('cfg-rating-weight')?.value || '0'),
        summary_model: (document.getElementById('cfg-summary-model')?.value || '').trim() || null,
    };
    if (maxTokensRaw !== '' && Number.isNaN(body.max_tokens)) {
        toast('max_tokens 须为数字或留空', 'error');
        return;
    }
    if (Number.isNaN(body.max_count) || body.max_count < 1) {
        toast('max_count 须为 ≥1 的整数', 'error');
        return;
    }
    try {
        const saved = await api('PUT', '/api/v1/config/retrieval', body);
        state.cachedRetrievalConfig = saved;
        setSettingsSaveStatus('检索配置已保存（运行时生效）', false);
        toast('检索配置已保存', 'success');
    } catch (e) {
        setSettingsSaveStatus('保存失败: ' + (e.message || String(e)), true);
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function saveSearchConfig() {
    setSettingsSaveStatus('');
    const body = {
        mode: document.getElementById('cfg-search-mode')?.value || 'hybrid',
        rrf_k: parseInt(document.getElementById('cfg-rrf-k')?.value || '60', 10),
        vector_weight: parseFloat(document.getElementById('cfg-vector-weight')?.value || '0.7'),
        fts_weight: parseFloat(document.getElementById('cfg-fts-weight')?.value || '0.3'),
        adaptive_filter: document.getElementById('cfg-adaptive-filter')?.value === 'true',
        score_gap_threshold: parseFloat(document.getElementById('cfg-score-gap')?.value || '0.15'),
        min_confidence_ratio: parseFloat(document.getElementById('cfg-min-confidence')?.value || '0.6'),
    };
    if (Number.isNaN(body.rrf_k) || body.rrf_k < 1) {
        toast('rrf_k 须为 ≥1 的整数', 'error');
        return;
    }
    try {
        await api('PUT', '/api/v1/config/search', body);
        setSettingsSaveStatus('搜索配置已保存（运行时生效）', false);
        toast('搜索配置已保存', 'success');
    } catch (e) {
        setSettingsSaveStatus('保存失败: ' + (e.message || String(e)), true);
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function loadAllConfig() {
    await loadRuntimeConfigForms();
    toast('已从服务端重新加载配置', 'info');
}

/** Web 去重页：拉取向量相似候选对（须登录）。 */
export async function loadDuplicates() {
    const container = document.getElementById('dedup-results');
    if (!container) return;
    const thrRaw = document.getElementById('dedup-threshold')?.value ?? '0.92';
    let thr = parseFloat(String(thrRaw), 10);
    if (Number.isNaN(thr)) thr = 0.92;
    thr = Math.min(1, Math.max(0.05, thr));
    const project = resolveProjectInput(state.activeProject || state.defaultProject || 'default');
    container.innerHTML = '<div class="loading" style="padding:24px"><div class="spinner"></div><p>正在扫描…</p></div>';
    try {
        const q = new URLSearchParams({ threshold: String(thr), limit: '80' });
        if (project) q.set('project', project);
        const data = await api('GET', `/api/v1/dedup/pairs?${q.toString()}`);
        const pairs = data.pairs || [];
        if (pairs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>未发现达到阈值的重复候选</h3>
                    <p class="hint" style="max-width:520px;margin:12px auto 0;line-height:1.5">
                        项目 <strong>${esc(data.project || project || 'default')}</strong>，
                        阈值 <strong>${data.threshold ?? thr}</strong>。
                        仅统计<strong>已发布</strong>、<strong>父经验</strong>、且<strong>embedding 非空</strong>的记录；
                        子经验标题重叠过低（Jaccard &lt; 0.2）的经验组对会被过滤。
                        可先「刷新经验组向量」再扫描。
                    </p>
                </div>`;
            return;
        }
        container.innerHTML = pairs.map((p) => {
            const a = p.exp_a || {};
            const b = p.exp_b || {};
            const sim = p.similarity ?? 0;
            const prevA = (a.children_preview || []).map((t) => esc(t)).join(' · ') || '—';
            const prevB = (b.children_preview || []).map((t) => esc(t)).join(' · ') || '—';
            return `
            <div class="dedup-pair-card" style="margin-bottom:14px;padding:14px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
                    <span style="font-weight:600;color:var(--accent);">相似度 ${(sim * 100).toFixed(1)}%</span>
                    <div style="display:flex;gap:8px;">
                        <button type="button" class="btn btn-secondary btn-sm" onclick="viewDetail('${a.id}')">打开 A</button>
                        <button type="button" class="btn btn-secondary btn-sm" onclick="viewDetail('${b.id}')">打开 B</button>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px;">
                    <div>
                        <div style="color:var(--text-muted);font-size:11px;margin-bottom:4px;">A · 子经验 ${a.children_count ?? 0}</div>
                        <strong>${esc(a.title || '')}</strong>
                        <div class="hint" style="margin-top:6px;">${prevA}</div>
                    </div>
                    <div>
                        <div style="color:var(--text-muted);font-size:11px;margin-bottom:4px;">B · 子经验 ${b.children_count ?? 0}</div>
                        <strong>${esc(b.title || '')}</strong>
                        <div class="hint" style="margin-top:6px;">${prevB}</div>
                    </div>
                </div>
            </div>`;
        }).join('');
        toast(`发现 ${pairs.length} 对候选`, 'success');
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>扫描失败</h3><p>${esc(e.message || String(e))}</p></div>`;
        toast('扫描失败: ' + e.message, 'error');
    }
}

export async function reembedGroups() {
    try {
        const project = resolveProjectInput(state.activeProject || state.defaultProject || 'default');
        const q = project ? `?project=${encodeURIComponent(project)}` : '';
        const data = await api('POST', `/api/v1/dedup/reembed-group-vectors${q}`);
        const msg = `已更新 ${data.updated}/${data.total_groups} 组父向量` + (data.errors ? `（${data.errors} 组失败）` : '');
        toast(msg, data.errors ? 'info' : 'success');
    } catch (e) {
        toast('刷新失败: ' + e.message, 'error');
    }
}

// ===== Account & Security (change password) =====

export async function loadAccountSecurity() {
    // API key display removed; password change handled by doChangePassword
}

export async function doChangePassword() {
    const oldPwdEl = document.getElementById('settings-old-password');
    const newPwdEl = document.getElementById('settings-new-password');
    const confirmEl = document.getElementById('settings-confirm-password');
    if (!oldPwdEl || !newPwdEl || !confirmEl) return;
    const oldPassword = oldPwdEl.value;
    const newPassword = newPwdEl.value;
    const confirmPassword = confirmEl.value;

    if (!oldPassword) {
        toast('请输入旧密码', 'error');
        return;
    }
    if (!newPassword || newPassword.length < 6) {
        toast('新密码至少 6 个字符', 'error');
        return;
    }
    if (newPassword !== confirmPassword) {
        toast('两次密码输入不一致', 'error');
        return;
    }

    try {
        await api('PUT', '/api/v1/auth/password', { old_password: oldPassword, new_password: newPassword });
        toast('密码修改成功', 'success');
        oldPwdEl.value = '';
        newPwdEl.value = '';
        confirmEl.value = '';
    } catch (e) {
        toast((e && e.message) || '修改失败', 'error');
    }
}

// ===== Key / User Management (admin) =====

export async function loadKeyManagement() {
    const card = document.getElementById('settings-key-mgmt');
    const userMgmtSec = document.getElementById('settings-user-mgmt-section');
    if (!card) return;
    card.style.display = 'block';
    if (userMgmtSec) {
        userMgmtSec.style.display = (state.currentUser && state.currentUser.role === 'admin') ? 'block' : 'none';
    }
    if (!state.currentUser || state.currentUser.role !== 'admin') return;

    try {
        const data = await api('GET', '/api/v1/auth/keys');
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
                    <div style="display:flex; align-items:center; gap:10px;">
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
        await api('PUT', `/api/v1/auth/keys/${id}`, { is_active: true });
        toast('审批成功', 'success');
        loadKeyManagement();
    } catch (e) {
        toast('审批失败: ' + e.message, 'error');
    }
}

export async function rejectUser(id) {
    if (!confirm('确定要拒绝此注册申请？将删除该记录。')) return;
    try {
        await api('DELETE', `/api/v1/auth/keys/${id}`);
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
        await api('POST', '/api/v1/auth/keys', body);
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
        await api('PUT', `/api/v1/auth/keys/${id}`, { role: newRole });
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
        await api('PUT', `/api/v1/auth/keys/${id}`, { is_active: active });
        toast(`用户已${action}`, 'success');
        loadKeyManagement();
    } catch (e) {
        toast(`${action}失败: ` + e.message, 'error');
        loadKeyManagement();
    }
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

function _esc(s) { return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function _fmtDate(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}
// ===== Personal Memory (Task 6) =====
function _pmKind(m) {
    if (m.profile_kind === 'static' || m.profile_kind === 'dynamic') return m.profile_kind;
    return (m.scope === 'context') ? 'dynamic' : 'static';
}

function _renderPmCard(m) {
    const kind = _pmKind(m);
    return `
          <div class="settings-card" style="margin-bottom:12px">
            <div class="settings-card-body" style="padding:12px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
                <div style="flex:1">
                  <span class="tag" style="font-size:11px">${esc(kind)}</span>
                  <span class="tag" style="font-size:11px;opacity:0.85">${esc(m.scope || 'generic')}</span>
                  ${m.context_hint ? `<span class="hint" style="font-size:11px;margin-left:6px">${esc(m.context_hint)}</span>` : ''}
                  <p style="margin:8px 0 0;font-size:14px">${esc(m.content)}</p>
                  <div style="font-size:11px;color:var(--text-muted);margin-top:6px">${m.updated_at ? formatDate(m.updated_at) : ''}</div>
                </div>
                <div style="display:flex;gap:6px">
                  <button class="btn btn-secondary btn-sm" onclick="editPersonalMemory('${m.id}')">编辑</button>
                  <button class="btn btn-secondary btn-sm" onclick="deletePersonalMemory('${m.id}')">删除</button>
                </div>
              </div>
            </div>
          </div>`;
}

export async function loadPersonalMemoryList() {
    const container = document.getElementById('personal-memory-list');
    if (!container) return;
    const scopeEl = document.getElementById('pm-scope-filter');
    const kindEl = document.getElementById('pm-kind-filter');
    const scope = scopeEl ? scopeEl.value || '' : '';
    const profileKind = kindEl ? kindEl.value || '' : '';
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const q = new URLSearchParams();
        if (scope) q.set('scope', scope);
        if (profileKind) q.set('profile_kind', profileKind);
        const qs = q.toString();
        const url = qs ? `/api/v1/personal-memory/list?${qs}` : '/api/v1/personal-memory/list';
        const data = await api('GET', url);
        const items = data.items || [];
        if (items.length === 0) {
            container.innerHTML = `<div class="empty-state"><div class="icon">🧠</div><h3>暂无用户画像条目</h3><p>由 <code>memory_save</code> 解析或在此手动添加</p><button class="btn btn-primary btn-sm" onclick="showAddPersonalMemoryModal()">手动添加</button></div>`;
            return;
        }
        let html = '';
        if (profileKind) {
            html = items.map(_renderPmCard).join('');
        } else {
            const staticItems = items.filter((m) => _pmKind(m) === 'static');
            const dynamicItems = items.filter((m) => _pmKind(m) === 'dynamic');
            if (staticItems.length > 0) {
                html += `<h3 style="font-size:14px;font-weight:600;color:var(--text-secondary);margin:0 0 10px">长期偏好 (static)</h3>`;
                html += staticItems.map(_renderPmCard).join('');
            }
            if (dynamicItems.length > 0) {
                html += `<h3 style="font-size:14px;font-weight:600;color:var(--text-secondary);margin:16px 0 10px">近期语境 (dynamic)</h3>`;
                html += dynamicItems.map(_renderPmCard).join('');
            }
            if (!html) html = items.map(_renderPmCard).join('');
        }
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export function showAddPersonalMemoryModal() {
    const content = prompt('输入偏好或习惯（一句概括）:');
    if (!content || !content.trim()) return;
    const scope = prompt('scope: generic 或 context（直接回车=generic）:') || 'generic';
    const contextHint = scope === 'context' ? prompt('context_hint（适用场景，可选）:') || '' : '';
    savePersonalMemory(null, content.trim(), scope, contextHint || null);
}

async function savePersonalMemory(id, content, scope, contextHint) {
    try {
        if (id) {
            await api('PUT', `/api/v1/personal-memory/${id}`, { content, scope, context_hint: contextHint });
            toast('已更新', 'success');
        } else {
            await api('POST', '/api/v1/personal-memory', { content, scope, context_hint: contextHint });
            toast('已添加', 'success');
        }
        loadPersonalMemoryList();
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function editPersonalMemory(id) {
    try {
        const mem = await api('GET', `/api/v1/personal-memory/${id}`);
        const content = prompt('修改内容:', mem.content || '');
        if (content === null) return;
        const scope = prompt('scope: generic 或 context（直接回车=保持）:', mem.scope || 'generic') || 'generic';
        const contextHint = prompt('context_hint（可选）:', mem.context_hint || '') || null;
        await savePersonalMemory(id, content.trim(), scope, contextHint);
    } catch (e) {
        toast('加载失败: ' + e.message, 'error');
    }
}

export async function deletePersonalMemory(id) {
    if (!confirm('确认删除？')) return;
    try {
        await api('DELETE', `/api/v1/personal-memory/${id}`);
        toast('已删除', 'success');
        loadPersonalMemoryList();
    } catch (e) {
        toast('删除失败: ' + e.message, 'error');
    }
}

// ----- Archives (browse API; semantic search stays on Search / MCP) -----

function _archivesProjectParam() {
    const p = state.activeProject || state.defaultProject || 'default';
    return `project=${encodeURIComponent(p)}`;
}

export function showArchivesListView() {
    document.getElementById('archives-list-view')?.classList.remove('hidden');
    document.getElementById('archives-detail-view')?.classList.add('hidden');
}

/** List archives (keyword + time sort from API). */
export async function loadArchivesList(page = 1) {
    state.archivesListPage = page;
    const container = document.getElementById('archives-list-content');
    const pagEl = document.getElementById('archives-pagination');
    if (!container) return;
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    if (pagEl) pagEl.innerHTML = '';

    const qEl = document.getElementById('archives-q');
    const q = (qEl && qEl.value) ? qEl.value.trim() : '';
    const limit = 20;
    const offset = (page - 1) * limit;
    let qs = `${_archivesProjectParam()}&limit=${limit}&offset=${offset}`;
    if (q) qs += `&q=${encodeURIComponent(q)}`;

    try {
        const data = await api('GET', `/api/v1/archives?${qs}`);
        const items = data.items || [];
        const total = data.total || 0;
        if (!items.length) {
            container.innerHTML = `<div class="empty-state">
                <h3>暂无档案</h3>
                <p>若已通过 MCP 写入 <code>memory_save</code>（<code>scope=archive</code>），请确认当前登录用户与项目筛选一致；团队可见的档案需在关联经验全部发布后才会对他人列出。</p>
            </div>`;
        } else {
            container.innerHTML = items.map((a) => {
                const prev = esc(a.overview_preview || a.solution_preview || '').slice(0, 280);
                const st = esc(a.status || '');
                const idRaw = a.id != null ? String(a.id) : '';
                const hashHref = idRaw
                    ? `#archives/${encodeURIComponent(idRaw)}`
                    : '#archives';
                const meta = [
                    a.created_at ? timeAgo(a.created_at) : '',
                    esc(a.created_by || ''),
                    esc(a.project || ''),
                    st ? `状态 ${st}` : '',
                    typeof a.attachment_count === 'number' ? `${a.attachment_count} 个附件` : '',
                ].filter(Boolean).join(' · ');
                const ctBadge = a.content_type
                    ? `<span class="arch-content-type-badge">${esc(a.content_type)}</span>`
                    : '';
                const valSum = a.value_summary
                    ? `<div class="arch-value-summary">${esc(a.value_summary)}</div>`
                    : '';
                const tagsRow = (a.tags && a.tags.length)
                    ? `<div class="arch-tags">${a.tags.map((t) => `<span class="arch-tag">${esc(t)}</span>`).join('')}</div>`
                    : '';
                return `<a class="arch-card" href="${hashHref}" data-archive-id="${esc(idRaw)}"
                  aria-label="查看档案详情：${esc(a.title || '未命名')}">
                  <div class="arch-card-title">${ctBadge}${esc(a.title || '')}</div>
                  ${valSum}
                  <div class="arch-card-meta">${meta}</div>
                  ${tagsRow}
                  <div class="arch-card-preview">${prev}</div>
                  <div class="arch-card-cta"><span>查看详情</span></div>
                </a>`;
            }).join('');
        }

        const pages = Math.max(1, Math.ceil(total / limit));
        if (pagEl && pages > 1) {
            let phtml = '';
            if (page > 1) phtml += `<button type="button" class="btn btn-secondary btn-sm" onclick="loadArchivesList(${page - 1})">上一页</button>`;
            phtml += `<span style="margin:0 12px;font-size:13px;color:var(--text-muted)">第 ${page} / ${pages} 页（共 ${total} 条）</span>`;
            if (page < pages) phtml += `<button type="button" class="btn btn-secondary btn-sm" onclick="loadArchivesList(${page + 1})">下一页</button>`;
            pagEl.innerHTML = phtml;
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

function _archiveTextBlock(label, text) {
    if (!text || !String(text).trim()) return '';
    return `<section style="margin-bottom:20px"><h3 style="font-size:13px;font-weight:600;color:var(--text-secondary);margin:0 0 8px">${label}</h3><pre class="arch-pre" style="white-space:pre-wrap;word-break:break-word;margin:0;font-size:14px;color:var(--text-primary)">${esc(text)}</pre></section>`;
}

function _archiveScopeLine(a) {
    const parts = [];
    if (a.scope) parts.push(`范围 ${esc(a.scope)}`);
    if (a.scope_ref) parts.push(`引用 ${esc(a.scope_ref)}`);
    if (!parts.length) return '';
    return `<p style="margin:8px 0 0;font-size:12px;color:var(--text-muted)">${parts.join(' · ')}</p>`;
}

/** PageIndex-Lite / long-doc sections stored on archive. */
function _archiveTreeNodesSection(nodes) {
    if (!Array.isArray(nodes) || !nodes.length) return '';
    const blocks = nodes.map((n) => {
        const head = [n.path, n.node_title].filter(Boolean).map(esc).join(' · ');
        const summary = n.content_summary && String(n.content_summary).trim()
            ? `<p style="font-size:12px;color:var(--text-muted);margin:0 0 8px">${esc(n.content_summary)}</p>`
            : '';
        const body = n.content && String(n.content).trim()
            ? `<pre class="arch-pre" style="white-space:pre-wrap;word-break:break-word;margin:0;font-size:13px">${esc(n.content)}</pre>`
            : '';
        if (!summary && !body && !head) return '';
        return `<section style="margin-bottom:16px;padding:12px 14px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius)"><h4 style="margin:0 0 8px;font-size:13px;font-weight:600;color:var(--text-primary)">${head || esc(n.node_title || '')}</h4>${summary}${body}</section>`;
    }).filter(Boolean);
    if (!blocks.length) return '';
    return `<section style="margin-bottom:20px"><h3 style="font-size:13px;font-weight:600;color:var(--text-secondary);margin:0 0 12px">文档分段</h3>${blocks.join('')}</section>`;
}

/** Show attachment path + full content_snapshot / snippet from API. */
function _archiveAttachmentsRich(atts) {
    if (!Array.isArray(atts) || !atts.length) return '';
    const projQs = _archivesProjectParam();
    const items = atts.map((t) => {
        const head = `<strong><code>${esc(t.kind || '')}</code></strong>${t.path ? ` <span style="color:var(--text-muted)">${esc(t.path)}</span>` : ''}${t.git_commit ? ` <code style="font-size:11px">${esc(t.git_commit)}</code>` : ''}`;
        const srcInfo = t.source_path ? `<p style="margin:4px 0 0;font-size:12px;color:var(--text-muted)">本地路径: ${esc(t.source_path)}</p>` : '';
        const dl =
            t.download_api_path
                ? `<p style="margin:8px 0 0;font-size:13px"><a href="${esc(t.download_api_path)}?${projQs}" download target="_blank" rel="noopener">下载附件文件</a>${t.storage === 'local' ? ' <span style="color:var(--text-muted);font-size:12px">(local)</span>' : ''}</p>`
                : '';
        const snap = t.content_snapshot && String(t.content_snapshot).trim();
        const snip = t.snippet && String(t.snippet).trim();
        let body = '';
        if (snap) {
            body += `<h5 style="font-size:11px;margin:10px 0 4px;color:var(--text-secondary)">内容快照</h5><pre class="arch-pre" style="white-space:pre-wrap;word-break:break-word;font-size:13px">${esc(snap)}</pre>`;
        }
        if (snip && snip !== snap) {
            body += `<h5 style="font-size:11px;margin:10px 0 4px;color:var(--text-secondary)">摘录</h5><pre class="arch-pre" style="white-space:pre-wrap;word-break:break-word;font-size:13px">${esc(snip)}</pre>`;
        }
        if (!body) {
            body = `<p style="font-size:12px;color:var(--text-muted);margin:8px 0 0">（无内联快照，仅路径引用）</p>`;
        }
        return `<li style="margin-bottom:16px">${head}${srcInfo}${dl}${body}</li>`;
    }).join('');
    return `<section style="margin-top:20px"><h3 style="font-size:13px;font-weight:600;color:var(--text-secondary);margin:0 0 8px">附件与快照</h3><ul style="margin:0;padding-left:0;list-style:none">${items}</ul></section>`;
}

function _buildUploadFailureCurl(archiveId, filenameHint) {
    const base = typeof location !== 'undefined' ? location.origin : '';
    const qs = _archivesProjectParam();
    const path = `/api/v1/archives/${encodeURIComponent(archiveId)}/attachments/upload?${qs}`;
    const hint = (filenameHint && String(filenameHint).trim()) || 'local.file';
    const safeHint = hint.replace(/'/g, "'\\''");
    return `curl -sS -X POST '${base}${path}' \\\n  -H 'Authorization: Bearer YOUR_API_KEY' \\\n  -H 'X-Upload-Source: web' \\\n  -F 'file=@/path/to/${safeHint}' \\\n  -F 'kind=file'`;
}

function _archiveFailuresHtml(archiveId, failures) {
    if (!Array.isArray(failures) || !failures.length) return { html: '', curlStrings: [] };
    const curlStrings = failures.map((f) =>
        _buildUploadFailureCurl(archiveId, f.client_filename_hint || 'file'),
    );
    const rows = failures
        .map(
            (f, idx) =>
                `<li style="margin-bottom:12px;padding:10px;background:var(--bg-elevated);border-radius:var(--radius);border:1px solid var(--border)">
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">${f.created_at ? esc(f.created_at) : ''} · ${esc(f.error_code || '')}</div>
      <div style="font-size:13px;margin-bottom:8px">${esc(f.error_message || '')}</div>
      <button type="button" class="btn btn-secondary btn-sm" data-archive-copy-curl-idx="${idx}">复制 curl</button>
      <button type="button" class="btn btn-secondary btn-sm" data-archive-resolve-failure="${esc(f.id)}">标记已处理</button>
    </li>`,
        )
        .join('');
    const html = `<section style="margin-bottom:20px;padding:14px;background:rgba(180,80,80,0.08);border:1px solid var(--border);border-radius:var(--radius)">
    <h3 style="font-size:13px;margin:0 0 10px;color:var(--text-primary)">上传曾失败（可用 curl 重试）</h3>
    <ul style="margin:0;padding-left:0;list-style:none">${rows}</ul>
  </section>`;
    return { html, curlStrings };
}

function _archiveUploadFormHtml() {
    return `<section style="margin-bottom:20px;padding:14px;border:1px solid var(--border);border-radius:var(--radius)">
    <h3 style="font-size:13px;margin:0 0 10px">上传附件</h3>
    <p style="font-size:12px;color:var(--text-muted);margin:0 0 10px">与档案 L2 权限一致；白名单后缀见服务端配置 <code>uploads.allowed_extensions</code>。</p>
    <form id="archives-upload-form">
      <div style="margin-bottom:8px">
        <input type="file" name="file" required style="font-size:13px" />
      </div>
      <div style="margin-bottom:8px">
        <label style="font-size:12px;color:var(--text-secondary)">kind</label><br/>
        <input type="text" name="kind" value="file" style="width:100%;max-width:280px" />
      </div>
      <div style="margin-bottom:8px">
        <label style="font-size:12px;color:var(--text-secondary)">备注（snippet）</label><br/>
        <input type="text" name="note" placeholder="可选" style="width:100%;max-width:400px" />
      </div>
      <button type="submit" class="btn btn-primary btn-sm">上传</button>
    </form>
  </section>`;
}

async function _bindArchiveDetailInteractions(archiveId, bodyEl, failureCurlStrings) {
    const form = bodyEl.querySelector('#archives-upload-form');
    if (form) {
        form.addEventListener('submit', async (ev) => {
            ev.preventDefault();
            const fileInput = form.querySelector('input[type=file]');
            const fil = fileInput && fileInput.files && fileInput.files[0];
            if (!fil || !fil.size) {
                toast('请选择非空文件', 'error');
                return;
            }
            const fd = new FormData(form);
            try {
                const qs = _archivesProjectParam();
                await api('POST', `/api/v1/archives/${archiveId}/attachments/upload?${qs}`, fd);
                toast('上传成功', 'info');
                await openArchiveDetail(archiveId);
            } catch (e) {
                toast('上传失败: ' + e.message, 'error');
                await openArchiveDetail(archiveId);
            }
        });
    }
    bodyEl.querySelectorAll('[data-archive-copy-curl-idx]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const i = Number(btn.getAttribute('data-archive-copy-curl-idx'));
            const s = failureCurlStrings[i];
            if (!s) return;
            copyTextToClipboard(s).then((ok) => toast(ok ? '已复制 curl' : '复制失败', ok ? 'info' : 'error'));
        });
    });
    bodyEl.querySelectorAll('[data-archive-resolve-failure]').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const fid = btn.getAttribute('data-archive-resolve-failure');
            if (!fid) return;
            try {
                const qs = _archivesProjectParam();
                await api('PATCH', `/api/v1/archives/${archiveId}/upload-failures/${fid}?${qs}`, { resolved: true });
                toast('已标记处理', 'info');
                await openArchiveDetail(archiveId);
            } catch (e) {
                toast('操作失败: ' + e.message, 'error');
            }
        });
    });
}

export async function openArchiveDetail(id) {
    if (!id) return;
    const listV = document.getElementById('archives-list-view');
    const detailV = document.getElementById('archives-detail-view');
    const body = document.getElementById('archives-detail-body');
    if (!listV || !detailV || !body) return;
    listV.classList.add('hidden');
    detailV.classList.remove('hidden');
    body.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    if (location.hash !== `#archives/${id}`) {
        history.pushState(null, '', `${location.pathname}#archives/${id}`);
    }

    try {
        const qs = _archivesProjectParam();
        const failP = api('GET', `/api/v1/archives/${id}/upload-failures?${qs}`).catch(() => ({ items: [] }));
        const a = await api('GET', `/api/v1/archives/${id}?${qs}`);
        const failRes = await failP;
        const failures = (failRes && failRes.items) || [];
        const { html: failHtml, curlStrings } = _archiveFailuresHtml(id, failures);

        // --- L0: Header ---
        const meta = [
            a.created_by ? esc(a.created_by) : '',
            a.created_at ? `保存于 ${formatDate(a.created_at)}` : '',
            a.project ? esc(a.project) : '',
            a.status ? `状态 ${esc(a.status)}` : '',
        ].filter(Boolean).join(' · ');
        const ctBadge = a.content_type
            ? `<span class="arch-content-type-badge">${esc(a.content_type)}</span>`
            : '';
        const valSum = a.value_summary
            ? `<div class="arch-value-summary">${esc(a.value_summary)}</div>`
            : '';
        const tagsRow = (a.tags && a.tags.length)
            ? `<div class="arch-tags">${a.tags.map((t) => `<span class="arch-tag">${esc(t)}</span>`).join('')}</div>`
            : '';
        const links = (a.linked_experience_ids || []).length
            ? `<p style="font-size:13px;color:var(--text-muted);margin:12px 0">关联经验 ID：${(a.linked_experience_ids || []).map((x) => `<code>${esc(x)}</code>`).join(', ')}</p>`
            : '';

        const l0Header = `
          <header style="margin-bottom:20px;border-bottom:1px solid var(--border);padding-bottom:12px">
            <h2 style="font-size:20px;font-weight:600;margin:0 0 8px;color:var(--text-primary)">${ctBadge}${esc(a.title || '')}</h2>
            ${valSum}
            ${tagsRow}
            <p style="margin:4px 0 0;font-size:12px;color:var(--text-muted)">${meta}</p>
            ${_archiveScopeLine(a)}
          </header>`;

        // --- L1: Overview (expanded by default) ---
        const overviewContent = a.overview
            ? renderMarkdown(a.overview)
            : '<p style="color:var(--text-muted);font-style:italic">无概览内容</p>';
        const l1Section = `
          <div class="archive-section">
            <div class="archive-section-header" onclick="this.nextElementSibling.classList.toggle('collapsed');this.querySelector('.archive-toggle-icon').textContent=this.nextElementSibling.classList.contains('collapsed')?'&#9654;':'&#9660;'">
              <span>概览 / Overview</span>
              <span class="archive-toggle-icon">&#9660;</span>
            </div>
            <div class="archive-section-body">
              ${overviewContent}
            </div>
          </div>`;

        // --- L2: Full Content (collapsed by default) ---
        const solutionHtml = a.solution_doc
            ? `<section style="margin-bottom:16px"><h4 style="font-size:13px;font-weight:600;color:var(--text-secondary);margin:0 0 8px">方案与计划全文</h4><div>${renderMarkdown(a.solution_doc)}</div></section>`
            : '';
        const convHtml = a.conversation_summary
            ? `<section style="margin-bottom:16px"><h4 style="font-size:13px;font-weight:600;color:var(--text-secondary);margin:0 0 8px">对话摘要</h4><div>${renderMarkdown(a.conversation_summary)}</div></section>`
            : '';
        const treeHtml = _archiveTreeNodesSection(a.document_tree_nodes);
        const attHtml = _archiveAttachmentsRich(a.attachments || []);
        const l2Inner = solutionHtml + convHtml + treeHtml + links + attHtml;
        const l2Section = l2Inner.trim()
            ? `<div class="archive-section">
            <div class="archive-section-header" onclick="this.nextElementSibling.classList.toggle('collapsed');this.querySelector('.archive-toggle-icon').textContent=this.nextElementSibling.classList.contains('collapsed')?'&#9654;':'&#9660;'">
              <span>完整内容 / Full Content</span>
              <span class="archive-toggle-icon">&#9654;</span>
            </div>
            <div class="archive-section-body collapsed">
              ${l2Inner}
            </div>
          </div>`
            : '';

        body.innerHTML = `
          ${l0Header}
          ${failHtml}
          ${_archiveUploadFormHtml()}
          ${l1Section}
          ${l2Section}`;
        await _bindArchiveDetailInteractions(id, body, curlStrings);
    } catch (e) {
        body.innerHTML = `<div class="empty-state"><h3>无法打开</h3><p>${esc(e.message)}</p><p style="font-size:13px">若没有权限或档案为草稿且非您创建，服务器会返回未找到。</p></div>`;
    }
}

export function backToArchivesList() {
    showArchivesListView();
    history.pushState(null, '', `${location.pathname}#archives`);
    loadArchivesList(state.archivesListPage || 1);
}
