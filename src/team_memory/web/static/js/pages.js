/**
 * Page rendering functions: list, detail, dashboard, drafts, reviews, settings.
 */

import { state, defaultTypeIcons } from './store.js';
import { esc, formatDate, timeAgo } from './utils.js';
import { resolveProjectInput, loadSchemaAndPopulateFilters } from './schema.js';

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

// ===== Render Experience Cards =====
export function renderExpList(containerId, experiences) {
    const container = document.getElementById(containerId);
    if (!experiences || experiences.length === 0) {
        container.innerHTML = `<div class="empty-state"><div class="icon">📚</div><h3>暂无经验记录</h3><p>点击右上角"新建经验"添加第一条</p></div>`;
        return;
    }
    const severityClass = (s) => (s ? 'severity-badge severity-' + String(s).toLowerCase() : '');
    container.innerHTML = experiences
        .map((exp) => {
            const view = exp.parent || exp;
            const cardId = exp.group_id || view.id;
            const isStale = view.last_used_at && isStaleDate(view.last_used_at);
            const typeIcon = typeIcons[view.experience_type] || defaultTypeIcons[view.experience_type] || '📝';
            const sevBadge = view.severity ? `<span class="${severityClass(view.severity)}">${view.severity}</span>` : '';
            const comp = view.completeness_score != null ? view.completeness_score : null;
            const compBar =
                comp != null
                    ? `<div class="completeness-bar" title="完整度 ${comp}%"><div class="completeness-bar-fill" style="width:${comp}%"></div></div>`
                    : '';
            const matchedNodes = (exp.matched_nodes || [])
                .slice(0, 2)
                .map((n) => `<span class="tag" style="background:var(--accent-glow);color:var(--accent)">#${esc(n.path || '')} ${esc(n.node_title || '')}</span>`)
                .join('');
            const treeScore =
                exp.tree_score !== undefined
                    ? `<span class="tag" style="background:var(--accent-glow);color:var(--accent)">tree ${(Number(exp.tree_score) * 100).toFixed(0)}%</span>`
                    : '';
            return `
    <div class="exp-card" onclick="showDetail('${cardId}')">
      <div class="exp-card-header">
        <div class="exp-card-title">
          <span class="type-icon">${typeIcon}</span>${esc(view.title)}${isStale ? '<span class="stale-badge">疑似过时</span>' : ''}${view.publish_status === 'draft' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--accent-glow);color:var(--accent);margin-left:6px">草稿</span>' : ''}${view.review_status === 'pending' ? '<span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--yellow-bg);color:var(--yellow);margin-left:6px">待审核</span>' : ''} ${sevBadge}
        </div>
        <div class="exp-card-meta">
          ${exp.similarity !== undefined ? `<span class="similarity-badge">${(exp.similarity * 100).toFixed(0)}%</span>` : ''}
          ${view.avg_rating > 0 ? `<span class="rating-badge">★ ${view.avg_rating.toFixed(1)}</span>` : ''}
          <span>${timeAgo(view.created_at)}</span>
        </div>
      </div>
      <div class="exp-card-desc">${esc(view.description || '')}</div>
      ${matchedNodes || treeScore ? `<div style="margin-bottom:8px;display:flex;gap:6px;flex-wrap:wrap">${treeScore}${matchedNodes}</div>` : ''}
      ${compBar ? `<div style="margin-bottom:8px">${compBar}</div>` : ''}
      <div class="exp-card-footer">
        <div class="exp-card-tags">${(view.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}${view.category ? `<span class="tag" style="background:var(--bg-input);color:var(--text-muted)">${esc(view.category)}</span>` : ''}${view.project ? `<span class="tag" style="background:var(--bg-input);color:var(--text-muted)">project:${esc(view.project)}</span>` : ''}${exp.children_count > 0 || exp.total_children > 0 ? `<span class="children-badge">${exp.children_count || exp.total_children} steps</span>` : ''}</div>
        <span style="font-size:12px;color:var(--text-muted)">${esc(view.created_by || '')}</span>
      </div>
    </div>
  `;
        })
        .join('');
}

// ===== Dashboard =====
export async function loadDashboard() {
    try {
        const project = resolveProjectInput(document.getElementById('list-project-filter')?.value);
        const [stats, listData] = await Promise.all([
            api('GET', '/api/v1/stats'),
            api('GET', `/api/v1/experiences?page=1&page_size=5&project=${encodeURIComponent(project)}`),
        ]);

        document.getElementById('stat-total').textContent = stats.total_experiences || 0;
        document.getElementById('stat-recent').textContent = stats.recent_7days || 0;
        document.getElementById('stat-stale').textContent = stats.stale_count || 0;
        document.getElementById('stat-pending').textContent = stats.pending_reviews || 0;
        try {
            const draftsData = await api('GET', '/api/v1/experiences/drafts?page=1&page_size=1');
            document.getElementById('stat-drafts').textContent = draftsData.total || 0;
        } catch (_) {
            document.getElementById('stat-drafts').textContent = '—';
        }

        const tags = stats.tag_distribution || {};
        state.allTags = tags;
        const tagCount = Object.keys(tags).length;
        document.getElementById('stat-tags').textContent = tagCount;

        const tagsBar = document.getElementById('dashboard-tags');
        if (tagCount === 0) {
            tagsBar.innerHTML = '<span class="tag-label">暂无标签</span>';
        } else {
            tagsBar.innerHTML = Object.entries(tags)
                .sort((a, b) => b[1] - a[1])
                .map(([tag, cnt]) => `<span class="tag" onclick="filterByTag('${tag}')">${tag} (${cnt})</span>`)
                .join('');
        }

        renderExpList('dashboard-recent', listData.experiences);
    } catch (e) {
        toast('加载仪表盘失败: ' + e.message, 'error');
    }
}

// ===== Experience List =====
export async function loadList(page = 1) {
    state.listPage = page;
    const container = document.getElementById('list-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const statusFilter = document.getElementById('list-status-filter')?.value || '';
        const typeFilter = document.getElementById('list-type-filter')?.value || '';
        const severityFilter = document.getElementById('list-severity-filter')?.value || '';
        const categoryFilter = document.getElementById('list-category-filter')?.value || '';
        const progressFilter = document.getElementById('list-progress-filter')?.value || '';
        const projectFilterRaw = document.getElementById('list-project-filter')?.value || '';
        const projectFilter = resolveProjectInput(projectFilterRaw);
        let url = `/api/v1/experiences?page=${page}&page_size=15`;
        if (projectFilter) url += `&project=${encodeURIComponent(projectFilter)}`;
        if (statusFilter) url += `&status=${statusFilter}`;
        if (state.selectedTag) url += `&tag=${encodeURIComponent(state.selectedTag)}`;
        if (typeFilter) url += `&experience_type=${encodeURIComponent(typeFilter)}`;
        if (severityFilter) url += `&severity=${encodeURIComponent(severityFilter)}`;
        if (categoryFilter) url += `&category=${encodeURIComponent(categoryFilter)}`;
        if (progressFilter) url += `&progress_status=${encodeURIComponent(progressFilter)}`;

        const data = await api('GET', url);
        renderExpList('list-content', data.experiences);
        renderPagination(data);

        if (Object.keys(state.allTags).length > 0) {
            const bar = document.getElementById('list-tags-bar');
            const tagEntries = Object.entries(state.allTags).sort((a, b) => b[1] - a[1]);
            bar.innerHTML =
                '<span class="tag-label">标签筛选:</span>' +
                `<span class="tag" onclick="filterByTag(null)" style="${!state.selectedTag ? 'background:var(--accent);color:#fff' : ''}">全部</span>` +
                tagEntries
                    .map(
                        ([tag, cnt]) =>
                            `<span class="tag" onclick="filterByTag('${tag}')" style="${state.selectedTag === tag ? 'background:var(--accent);color:#fff' : ''}">${tag} (${cnt})</span>`
                    )
                    .join('');
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${e.message}</p></div>`;
    }
}

export function filterByTag(tag) {
    state.selectedTag = tag;
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

export async function showDetail(id) {
    state.currentPage = 'detail';
    document.querySelectorAll('.page').forEach((p) => p.classList.add('hidden'));
    document.querySelectorAll('.topbar-nav a').forEach((a) => a.classList.remove('active'));
    const page = document.getElementById('page-detail');
    page.classList.remove('hidden');
    page.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const exp = await api('GET', `/api/v1/experiences/${id}`);
        const typeIcon = typeIcons[exp.experience_type] || defaultTypeIcons[exp.experience_type] || '📝';
        const sevClass = exp.severity ? 'severity-badge severity-' + String(exp.severity).toLowerCase() : '';
        const typeBadges = `<span class="type-icon" style="font-size:20px">${typeIcon}</span>${exp.severity ? `<span class="${sevClass}" style="margin-left:8px">${exp.severity}</span>` : ''}${exp.category ? `<span class="tag" style="margin-left:6px;background:var(--bg-input);color:var(--text-muted)">${esc(exp.category)}</span>` : ''}${exp.progress_status ? `<span class="tag" style="margin-left:6px;background:var(--bg-input);color:var(--text-secondary)">${esc(exp.progress_status)}</span>` : ''}`;
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

        page.innerHTML = `
      <button class="back-btn" onclick="navigate('list')">← 返回列表</button>
      <div class="detail-view">
        <div class="detail-header">
          <h1>${typeBadges} ${esc(exp.title)}
            ${exp.publish_status === 'draft' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--accent-glow);color:var(--accent);margin-left:12px;vertical-align:middle">草稿</span>' : ''}
            ${exp.review_status === 'pending' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--yellow-bg);color:var(--yellow);margin-left:12px;vertical-align:middle">待审核</span>' : ''}
            ${exp.publish_status === 'rejected' ? '<span style="font-size:13px;padding:2px 10px;border-radius:4px;background:var(--red-bg);color:var(--red);margin-left:12px;vertical-align:middle">已退回</span>' : ''}
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
        </div>
        <div class="detail-section">
          <h3 style="cursor:pointer" onclick="toggleVersionHistory('${exp.id}')">版本历史 <span id="version-toggle-arrow" style="font-size:11px">▸</span></h3>
          <div id="version-history-panel" class="hidden">
            <div id="version-list" class="version-list">
              <div class="loading"><div class="spinner"></div></div>
            </div>
          </div>
        </div>
        <div class="detail-actions">
          ${exp.publish_status === 'draft' ? `<button class="btn btn-sm" style="background:var(--green);color:#fff" onclick="publishDraft('${exp.id}')">发布</button>` : ''}
          ${exp.review_status === 'pending' ? `
            <button class="btn btn-sm" style="background:var(--green);color:#fff" onclick="reviewExperience('${exp.id}', 'approved')">批准</button>
            <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red)" onclick="reviewExperience('${exp.id}', 'rejected')">退回</button>
          ` : ''}
          <button class="btn btn-primary btn-sm" onclick="openEditModal('${exp.id}')">✏️ 编辑</button>
          <button class="btn btn-primary btn-sm" onclick="openFeedbackModal('${exp.id}')">💬 提交反馈</button>
          ${!exp.summary ? `<button class="btn btn-sm" style="background:var(--accent-glow);color:var(--accent)" onclick="generateSummary('${exp.id}')">📝 生成摘要</button>` : ''}
          <button class="btn btn-danger btn-sm" onclick="deleteExp('${exp.id}')">🗑 删除</button>
          <div style="flex:1"></div>
          <span style="font-size:12px;color:var(--text-muted)">ID: ${exp.id}</span>
        </div>
      </div>
    `;
    } catch (e) {
        page.innerHTML = `<div class="empty-state"><h3>加载失败</h3><p>${e.message}</p></div>`;
    }
}

// ===== Drafts =====
export async function loadDrafts() {
    const container = document.getElementById('drafts-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    try {
        const project = resolveProjectInput(document.getElementById('list-project-filter')?.value);
        const data = await api('GET', `/api/v1/experiences/drafts?page=1&page_size=50&project=${encodeURIComponent(project)}`);
        if (!data.experiences || data.experiences.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="icon">📝</div><h3>暂无草稿</h3><p>创建经验时勾选"保存为草稿"即可</p></div>';
            return;
        }
        container.innerHTML = data.experiences
            .map(
                (exp) => `
      <div class="exp-card" onclick="viewDetail('${exp.id}')">
        <div class="exp-card-header">
          <div class="exp-card-title">${esc(exp.title)} <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--accent-glow);color:var(--accent)">草稿</span></div>
          <div class="exp-card-meta"><span>${timeAgo(exp.created_at)}</span></div>
        </div>
        <div class="exp-card-desc">${esc((exp.description || '').substring(0, 120))}${(exp.description || '').length > 120 ? '...' : ''}</div>
        <div class="exp-card-footer">
          <div class="exp-card-tags">${(exp.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="display:flex;gap:6px">
            <button class="btn btn-sm" style="background:var(--green);color:#fff;font-size:11px;padding:2px 10px" onclick="event.stopPropagation();publishDraft('${exp.id}')">发布</button>
            <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red);font-size:11px;padding:2px 10px" onclick="event.stopPropagation();deleteExp('${exp.id}')">删除</button>
          </div>
        </div>
      </div>
    `
            )
            .join('');
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载草稿失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export async function publishDraft(id) {
    if (!confirm('确定要发布这条经验吗？发布后将出现在搜索结果中。')) return;
    try {
        await api('POST', `/api/v1/experiences/${id}/publish`);
        toast('经验已发布', 'success');
        if (state.currentPage === 'drafts') loadDrafts();
        else if (state.currentPage === 'dashboard') loadDashboard();
    } catch (e) {
        toast('发布失败: ' + e.message, 'error');
    }
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
                (exp) => `
      <div class="exp-card">
        <div class="exp-card-header">
          <div class="exp-card-title" onclick="viewDetail('${exp.id}')" style="cursor:pointer">${esc(exp.title)} <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--yellow-bg);color:var(--yellow)">待审核</span></div>
          <div class="exp-card-meta"><span>来源: ${exp.source || 'unknown'}</span><span>${timeAgo(exp.created_at)}</span></div>
        </div>
        <div class="exp-card-desc">${esc((exp.description || '').substring(0, 200))}</div>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn btn-sm" style="background:var(--green);color:#fff;font-size:12px;padding:4px 16px"
            onclick="reviewExperience('${exp.id}', 'approved')">批准并发布</button>
          <button class="btn btn-sm" style="background:var(--red-bg);color:var(--red);font-size:12px;padding:4px 16px"
            onclick="reviewExperience('${exp.id}', 'rejected')">退回</button>
          <button class="btn btn-sm" style="background:var(--bg-input);color:var(--text-secondary);font-size:12px;padding:4px 16px"
            onclick="viewDetail('${exp.id}')">查看详情</button>
        </div>
        <div class="exp-card-footer" style="margin-top:8px">
          <div class="exp-card-tags">${(exp.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <span style="font-size:12px;color:var(--text-muted)">${esc(exp.created_by || '')}</span>
        </div>
      </div>
    `
            )
            .join('');
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>加载审核队列失败</h3><p>${esc(e.message)}</p></div>`;
    }
}

export async function reviewExperience(id, status) {
    const action = status === 'approved' ? '批准' : '退回';
    let note = null;
    if (status === 'rejected') {
        note = prompt('请输入退回原因（可选）:');
        if (note === null) return;
    } else {
        if (!confirm(`确定要${action}这条经验吗？`)) return;
    }
    try {
        await api('POST', `/api/v1/experiences/${id}/review`, { review_status: status, review_note: note });
        toast(`经验已${action}`, 'success');
        loadReviews();
    } catch (e) {
        toast(`${action}失败: ` + e.message, 'error');
    }
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

// ===== Dedup =====
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
                (pair) => `
      <div class="dup-pair">
        <div class="dup-card">
          <h4>${esc(pair.exp_a.title)}</h4>
          <p>${esc((pair.exp_a.description || '').substring(0, 120))}...</p>
          <div style="margin-top:8px">${(pair.exp_a.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">评分: ${(pair.exp_a.avg_rating || 0).toFixed(1)} · 引用: ${pair.exp_a.use_count || 0}</div>
          <button class="btn btn-primary btn-sm" style="margin-top:12px" onclick="doMerge('${pair.exp_a.id}','${pair.exp_b.id}')">保留此经验</button>
        </div>
        <div class="dup-vs">
          <div class="sim-score">${(pair.similarity * 100).toFixed(1)}%</div>
          <div style="font-size:11px;color:var(--text-muted)">相似度</div>
        </div>
        <div class="dup-card">
          <h4>${esc(pair.exp_b.title)}</h4>
          <p>${esc((pair.exp_b.description || '').substring(0, 120))}...</p>
          <div style="margin-top:8px">${(pair.exp_b.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
          <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">评分: ${(pair.exp_b.avg_rating || 0).toFixed(1)} · 引用: ${pair.exp_b.use_count || 0}</div>
          <button class="btn btn-primary btn-sm" style="margin-top:12px" onclick="doMerge('${pair.exp_b.id}','${pair.exp_a.id}')">保留此经验</button>
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
            (item) => `
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
      <div class="settings-actions" style="margin-top:8px">
        <button class="btn btn-secondary btn-sm" onclick="previewInstallable('${encodeURIComponent(item.id || '')}','${encodeURIComponent(item.source || '')}')">预览</button>
        ${canInstall ? `<button class="btn btn-primary btn-sm" onclick="installInstallable('${encodeURIComponent(item.id || '')}','${encodeURIComponent(item.source || '')}')">安装</button>` : '<span class="hint">仅 admin 可安装</span>'}
      </div>
    </div>
  `
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
    const id = decodeURIComponent(itemIdEncoded || '');
    const source = decodeURIComponent(sourceEncoded || '');
    try {
        const data = await api('GET', `/api/v1/installables/preview?id=${encodeURIComponent(id)}&source=${encodeURIComponent(source)}`);
        const pre = document.getElementById('installables-preview');
        if (pre) {
            const hint = data.truncated ? '\n\n[内容过长，已截断预览]' : '';
            pre.textContent = data.content + hint;
        }
    } catch (e) {
        toast('预览失败: ' + e.message, 'error');
    }
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
        const projectInput = document.getElementById('cfg-default-project');
        if (projectInput) projectInput.value = state.defaultProject;
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
        await loadInstallables();
    } catch (e) {
        toast('加载配置失败: ' + e.message, 'error');
    }
}

export async function loadRetrievalConfig() {
    await loadAllConfig();
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
        toast('正在生成摘要...', 'info');
        await api('POST', `/api/v1/experiences/${id}/summarize`);
        toast('摘要已生成', 'success');
        viewDetail(id);
    } catch (e) {
        toast('摘要生成失败: ' + e.message, 'error');
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
