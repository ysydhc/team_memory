/**
 * Reusable UI components: search bar, modals, cards.
 */

import { state } from './store.js';
import { esc, getSearchHistory, addSearchHistory, clearSearchHistory, renderMarkdown } from './utils.js';
import { resolveProjectInput } from './schema.js';
import { populateTagSuggestions } from './pages.js';

// Re-export renderMarkdown so consumers of components.js can use it.
export { renderMarkdown };

function api(...args) {
    return window.__api(...args);
}

function toast(msg, type = 'info') {
    return window.__toast(msg, type);
}

function navigate(page) {
    return window.__navigate(page);
}

function showDetail(id) {
    return window.__showDetail(id);
}

function loadDashboard() {
    return window.__loadDashboard();
}

function loadList(page) {
    return window.__loadList(page);
}

function loadDrafts() {
    return window.__loadDrafts();
}

// ===== Search =====
export function toggleSearchAdvanced() {
    const panel = document.getElementById('search-advanced');
    const toggle = document.querySelector('.search-advanced-toggle');
    panel.classList.toggle('hidden');
    toggle.classList.toggle('expanded');
    if (!panel.classList.contains('hidden') && state.cachedRetrievalConfig) {
        const cfg = state.cachedRetrievalConfig;
        document.getElementById('search-max-results').placeholder = `默认: ${cfg.max_count}`;
        document.getElementById('search-top-k-children').placeholder = `默认: ${cfg.top_k_children}`;
    }
}

export async function doSearch() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    const container = document.getElementById('search-results');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const body = { query };
        const maxRaw = document.getElementById('search-max-results')?.value ?? '';
        if (maxRaw !== '') {
            const n = parseInt(maxRaw, 10);
            if (!Number.isNaN(n) && n >= 1) body.max_results = n;
        }
        const topKRaw = document.getElementById('search-top-k-children')?.value ?? '';
        if (topKRaw !== '') {
            const n = parseInt(topKRaw, 10);
            if (!Number.isNaN(n) && n >= 0) body.top_k_children = n;
        }
        const minSimRaw = document.getElementById('search-min-similarity')?.value?.trim() ?? '';
        if (minSimRaw !== '') {
            const v = parseFloat(minSimRaw);
            if (!Number.isNaN(v)) body.min_similarity = Math.min(1, Math.max(0, v));
        }
        const project = resolveProjectInput(document.getElementById('search-project')?.value);
        if (project) body.project = project;
        const data = await api('POST', '/api/v1/search', body);
        window.__renderExpList('search-results', data.results);
        if (data.results.length > 0) {
            toast(`找到 ${data.results.length} 条相关经验`, 'success');
            addSearchHistory(query);
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>搜索失败</h3><p>${e.message}</p></div>`;
    }
}

let _searchHistoryHideTimer = null;

function hideSearchHistoryDropdown() {
    const el = document.getElementById('search-history-dropdown');
    if (el) {
        el.classList.remove('open');
        el.setAttribute('aria-hidden', 'true');
    }
}

export function showSearchHistoryDropdown() {
    if (_searchHistoryHideTimer) {
        clearTimeout(_searchHistoryHideTimer);
        _searchHistoryHideTimer = null;
    }
    const list = getSearchHistory();
    const el = document.getElementById('search-history-dropdown');
    if (!el) return;
    el.innerHTML = '';
    if (list.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'search-history-item search-history-empty';
        empty.textContent = '暂无搜索历史';
        el.appendChild(empty);
    } else {
        list.forEach((q) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'search-history-item';
            btn.textContent = q;
            btn.dataset.query = q;
            btn.addEventListener('click', () => {
                const input = document.getElementById('search-input');
                if (input) input.value = q;
                hideSearchHistoryDropdown();
                doSearch();
            });
            el.appendChild(btn);
        });
        const clearBtn = document.createElement('button');
        clearBtn.type = 'button';
        clearBtn.className = 'search-history-clear';
        clearBtn.textContent = '清空历史';
        clearBtn.addEventListener('click', () => {
            clearSearchHistory();
            showSearchHistoryDropdown();
        });
        el.appendChild(clearBtn);
    }
    el.classList.add('open');
    el.setAttribute('aria-hidden', 'false');
}

export function hideSearchHistoryDropdownSoon() {
    if (_searchHistoryHideTimer) clearTimeout(_searchHistoryHideTimer);
    _searchHistoryHideTimer = setTimeout(() => {
        _searchHistoryHideTimer = null;
        hideSearchHistoryDropdown();
    }, 150);
}

// ===== Create Modal =====
export function openCreateModal() {
    document.getElementById('create-modal').classList.remove('hidden');
    state.createMode = 'manual';
    document.getElementById('create-quick-mode').checked = false;
    toggleCreateQuickMode();
    if (!document.getElementById('create-project').value) {
        document.getElementById('create-project').value = state.activeProject || state.defaultProject || 'default';
    }
    if (!document.getElementById('group-parent-project').value) {
        document.getElementById('group-parent-project').value = state.activeProject || state.defaultProject || 'default';
    }
    populateTagSuggestions();
    document.getElementById('create-title').focus();
}

export function switchCreateMode(mode) {
    if (mode !== 'manual' && mode !== 'group') mode = 'manual';
    state.createMode = mode;
    document.getElementById('tab-manual').classList.toggle('active', mode === 'manual');
    document.getElementById('tab-group').classList.toggle('active', mode === 'group');
    document.getElementById('create-mode-manual').classList.toggle('hidden', mode !== 'manual');
    document.getElementById('create-mode-group').classList.toggle('hidden', mode !== 'group');
    document.getElementById('btn-save-exp').style.display = '';
    document.getElementById('btn-save-exp').textContent = mode === 'group' ? '保存经验组' : '保存经验';
}

export function toggleCreateQuickMode() {
    const quick = document.getElementById('create-quick-mode')?.checked || false;
    const ext = document.getElementById('create-extended-fields');
    if (ext) ext.classList.toggle('hidden', quick);
}

export function closeCreateModal(force = false) {
    if (!force) {
        const title = document.getElementById('create-title')?.value?.trim();
        const problem = document.getElementById('create-problem')?.value?.trim();
        const solution = document.getElementById('create-solution')?.value?.trim();
        const gTitle = document.getElementById('group-parent-title')?.value?.trim();
        const gProblem = document.getElementById('group-parent-problem')?.value?.trim();
        const gSolution = document.getElementById('group-parent-solution')?.value?.trim();
        if (title || problem || solution || gTitle || gProblem || gSolution) {
            if (!confirm('当前内容尚未保存，确定关闭？')) return;
        }
    }
    document.getElementById('create-modal').classList.add('hidden');
    ['create-title', 'create-problem', 'create-solution', 'create-tags', 'create-project'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    document.getElementById('create-quick-mode').checked = false;
    toggleCreateQuickMode();
    state.groupChildrenIds = [];
    document.getElementById('group-children-container').innerHTML = '';
    ['group-parent-title', 'group-parent-problem', 'group-parent-solution', 'group-parent-tags', 'group-parent-project'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    switchCreateMode('manual');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const modal = document.getElementById('create-modal');
        if (modal && !modal.classList.contains('hidden')) {
            closeCreateModal();
        }
    }
});

export function addGroupChild() {
    const id = 'gc-' + Date.now();
    state.groupChildrenIds.push(id);
    const idx = state.groupChildrenIds.length;
    const div = document.createElement('div');
    div.className = 'group-child-block expanded';
    div.dataset.id = id;
    div.innerHTML = `
    <div class="child-block-header" onclick="this.parentElement.classList.toggle('collapsed');this.parentElement.classList.toggle('expanded')">
      <span class="child-num">${idx}</span>
      <input type="text" class="child-block-title-input" placeholder="子经验标题..." data-child-title onclick="event.stopPropagation()">
      <span class="child-block-toggle">▸</span>
      <button type="button" class="btn btn-danger btn-sm" onclick="removeGroupChild('${id}');event.stopPropagation()">移除</button>
    </div>
    <div class="child-block-body">
      <div class="form-group">
        <label>问题</label>
        <textarea placeholder="问题描述..." data-child-problem style="min-height:60px"></textarea>
      </div>
      <div class="form-group">
        <label>方案</label>
        <textarea placeholder="解决方案..." data-child-solution style="min-height:60px"></textarea>
      </div>
    </div>
  `;
    document.getElementById('group-children-container').appendChild(div);
    document.querySelectorAll('#group-children-container .child-num').forEach((el, i) => {
        el.textContent = i + 1;
    });
}

export function removeGroupChild(id) {
    const block = document.querySelector(`.group-child-block[data-id="${id}"]`);
    if (block) block.remove();
    state.groupChildrenIds = state.groupChildrenIds.filter((x) => x !== id);
    document.querySelectorAll('#group-children-container .child-num').forEach((el, i) => {
        el.textContent = i + 1;
    });
}

export async function doCreate() {
    if (state.createMode === 'group') {
        const title = document.getElementById('group-parent-title').value.trim();
        const problem = document.getElementById('group-parent-problem').value.trim();
        const solution = document.getElementById('group-parent-solution').value.trim();
        if (!title || !problem || !solution) {
            toast('请填写父经验的标题、问题描述和解决方案', 'error');
            return;
        }
        const tags = document.getElementById('group-parent-tags').value.split(',').map((t) => t.trim()).filter(Boolean);
        const project = resolveProjectInput(document.getElementById('group-parent-project')?.value);

        const children = [];
        document.querySelectorAll('#group-children-container .group-child-block').forEach((block) => {
            const cTitle = (block.querySelector('[data-child-title]') || {}).value?.trim();
            const cProblem = (block.querySelector('[data-child-problem]') || {}).value?.trim() || '';
            const cSolution = (block.querySelector('[data-child-solution]') || {}).value?.trim() || '';
            if (cTitle) {
                children.push({ title: cTitle, problem: cProblem, solution: cSolution });
            }
        });
        if (children.length === 0) {
            toast('请至少添加一个有效的子经验（标题必填）', 'error');
            return;
        }

        try {
            await api('POST', '/api/v1/experiences/groups', {
                parent: { title, problem, solution, tags, project },
                children,
            });
            toast('经验组保存成功', 'success');
            closeCreateModal(true);
            if (state.currentPage === 'list') loadList(state.listPage);
        } catch (e) {
            toast('保存失败: ' + e.message, 'error');
        }
        return;
    }

    const title = document.getElementById('create-title').value.trim();
    const problem = document.getElementById('create-problem').value.trim();
    const quickMode = document.getElementById('create-quick-mode')?.checked || false;

    if (!title || !problem) {
        toast('请填写标题和问题描述', 'error');
        return;
    }

    const solution = quickMode ? null : (document.getElementById('create-solution').value.trim() || null);
    const tags = document.getElementById('create-tags').value.split(',').map((t) => t.trim()).filter(Boolean);
    const project = resolveProjectInput(document.getElementById('create-project')?.value);

    const saveAsDraft = document.getElementById('create-draft-checkbox')?.checked || false;
    const expStatus = saveAsDraft ? 'draft' : 'published';

    try {
        const body = {
            title, problem, solution, tags,
            status: expStatus,
            visibility: 'project',
            project,
        };

        const opts = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + state.apiKey },
            body: JSON.stringify(body),
        };
        const resp = await fetch('/api/v1/experiences', opts);
        const data = await resp.json();

        if (resp.status === 409 && data.status === 'duplicate_detected') {
            const candidateList = (data.candidates || [])
                .map((c) => `• ${c.title} (相似度: ${(c.similarity * 100).toFixed(1)}%)`)
                .join('\n');
            const proceed = confirm(
                `检测到 ${data.candidates?.length || 0} 条相似经验:\n\n${candidateList}\n\n是否仍要保存？`
            );
            if (proceed) {
                const retryBody = {
                    title, problem, solution, tags,
                    status: expStatus,
                    visibility: 'project',
                    project, skip_dedup_check: true,
                };
                await api('POST', '/api/v1/experiences', retryBody);
                toast(saveAsDraft ? '经验已保存为草稿' : '经验保存成功', 'success');
                closeCreateModal(true);
                if (state.currentPage === 'list') loadList(state.listPage);
            }
            return;
        }

        if (!resp.ok) throw new Error(data.detail || '保存失败');
        toast(saveAsDraft ? '经验已保存为草稿' : '经验保存成功', 'success');
        closeCreateModal(true);
        if (state.currentPage === 'list') loadList(state.listPage);
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

// ===== Feedback Modal =====
export function openFeedbackModal(expId) {
    state.feedbackExpId = expId;
    state.feedbackRating = null;
    document.getElementById('fb-comment').value = '';
    document.querySelectorAll('#star-rating input').forEach((r) => { r.checked = false; });
    document.getElementById('feedback-modal').classList.remove('hidden');
}

export function closeFeedbackModal() {
    document.getElementById('feedback-modal').classList.add('hidden');
}

function getSelectedRating() {
    const checked = document.querySelector('#star-rating input:checked');
    return checked ? parseInt(checked.value) : null;
}

export async function submitFeedback() {
    const rating = getSelectedRating();
    if (!rating) {
        toast('请选择评分（1-5 星）', 'error');
        return;
    }
    try {
        await api('POST', `/api/v1/experiences/${state.feedbackExpId}/feedback`, {
            rating,
            comment: document.getElementById('fb-comment').value.trim() || null,
        });
        toast('反馈已提交', 'success');
        closeFeedbackModal();
        showDetail(state.feedbackExpId);
    } catch (e) {
        toast('提交失败: ' + e.message, 'error');
    }
}

// ===== Edit Modal =====
export async function openEditModal(expId) {
    document.getElementById('edit-exp-id').value = expId;
    state.editChildrenIds = [];
    state.editOriginalExp = null;
    document.getElementById('edit-children-container').innerHTML = '';
    document.getElementById('edit-children-section').classList.add('hidden');

    try {
        const exp = await api('GET', `/api/v1/experiences/${expId}`);
        state.editOriginalExp = exp;

        document.getElementById('edit-title').value = exp.title || '';
        document.getElementById('edit-problem').value = exp.description || '';
        document.getElementById('edit-solution').value = exp.solution || '';
        document.getElementById('edit-tags').value = (exp.tags || []).join(', ');
        document.getElementById('edit-experience-type').value = exp.experience_type || 'general';
        document.getElementById('edit-visibility').value = exp.visibility || 'project';
        document.getElementById('edit-project').value = exp.project || '';

        if (exp.children && exp.children.length > 0) {
            document.getElementById('edit-children-section').classList.remove('hidden');
            exp.children.forEach((child) => {
                addEditChild({
                    title: child.title || '',
                    problem: child.description || '',
                    solution: child.solution || '',
                });
            });
        }
        document.getElementById('edit-modal').classList.remove('hidden');
    } catch (e) {
        toast('加载失败: ' + e.message, 'error');
    }
}

export function closeEditModal() {
    document.getElementById('edit-modal').classList.add('hidden');
    state.editChildrenIds = [];
}

export function addEditChild(initial) {
    const id = 'ec-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    state.editChildrenIds.push(id);
    const idx = state.editChildrenIds.length;
    const div = document.createElement('div');
    div.className = 'group-child-block expanded';
    div.dataset.id = id;
    const t = (initial && initial.title) || '';
    const p = (initial && initial.problem) || '';
    const s = (initial && initial.solution) || '';
    div.innerHTML = `
    <div class="child-block-header" onclick="this.parentElement.classList.toggle('collapsed');this.parentElement.classList.toggle('expanded')">
      <span class="child-num">${idx}</span>
      <input type="text" class="child-block-title-input" placeholder="子经验标题..." data-child-title value="${esc(t)}" onclick="event.stopPropagation()">
      <span class="child-block-toggle">▸</span>
      <button type="button" class="btn btn-danger btn-sm" onclick="removeEditChild('${id}');event.stopPropagation()">移除</button>
    </div>
    <div class="child-block-body">
      <div class="form-group">
        <label>问题</label>
        <textarea placeholder="问题描述..." data-child-problem style="min-height:60px">${esc(p)}</textarea>
      </div>
      <div class="form-group">
        <label>方案</label>
        <textarea placeholder="解决方案..." data-child-solution style="min-height:60px">${esc(s)}</textarea>
      </div>
    </div>
  `;
    document.getElementById('edit-children-container').appendChild(div);
    document.querySelectorAll('#edit-children-container .child-num').forEach((el, i) => {
        el.textContent = i + 1;
    });
}

export function removeEditChild(id) {
    const block = document.querySelector(`#edit-children-container .group-child-block[data-id="${id}"]`);
    if (block) block.remove();
    state.editChildrenIds = state.editChildrenIds.filter((x) => x !== id);
    document.querySelectorAll('#edit-children-container .child-num').forEach((el, i) => {
        el.textContent = i + 1;
    });
}

export async function submitEdit() {
    const expId = document.getElementById('edit-exp-id').value;
    const orig = state.editOriginalExp;
    const title = document.getElementById('edit-title').value.trim();
    const problem = document.getElementById('edit-problem').value.trim();

    if (!title || !problem) {
        toast('请填写标题和问题描述', 'error');
        return;
    }

    const solution = document.getElementById('edit-solution').value.trim() || null;
    const tags = document.getElementById('edit-tags').value.split(',').map((t) => t.trim()).filter(Boolean);
    const experience_type = document.getElementById('edit-experience-type')?.value || 'general';
    const visibility = document.getElementById('edit-visibility')?.value || 'project';
    const project = document.getElementById('edit-project')?.value?.trim() || null;

    let children = null;
    const container = document.getElementById('edit-children-container');
    if (container && container.querySelectorAll('.group-child-block').length > 0) {
        children = [];
        container.querySelectorAll('.group-child-block').forEach((block) => {
            const cTitle = (block.querySelector('[data-child-title]') || {}).value?.trim();
            const cProblem = (block.querySelector('[data-child-problem]') || {}).value?.trim() || '';
            const cSolution = (block.querySelector('[data-child-solution]') || {}).value?.trim() || '';
            if (cTitle) {
                children.push({ title: cTitle, problem: cProblem, solution: cSolution });
            }
        });
        if (children.length === 0) {
            toast('子经验至少需要填写标题', 'error');
            return;
        }
    }

    const body = {};
    if (title !== (orig?.title || '')) body.title = title;
    if (problem !== (orig?.description || '')) body.problem = problem;
    if (solution !== (orig?.solution || '')) body.solution = solution;
    if (JSON.stringify(tags) !== JSON.stringify(orig?.tags || [])) body.tags = tags;
    if (experience_type !== (orig?.experience_type || 'general')) body.experience_type = experience_type;
    if (visibility !== (orig?.visibility || 'project')) body.visibility = visibility;
    if (project !== (orig?.project || null)) body.project = project;
    if (children) body.children = children;

    if (Object.keys(body).length === 0) {
        toast('暂无修改', 'info');
        return;
    }

    try {
        const result = await api('PUT', `/api/v1/experiences/${expId}`, body);
        toast('修改成功', 'success');
        closeEditModal();
        showDetail(result.id || result.parent_id || expId);
    } catch (e) {
        toast('修改失败: ' + e.message, 'error');
    }
}

// ===== Delete =====
export async function deleteExp(id) {
    if (!confirm('确定要删除这条经验吗？此操作不可撤销。')) return;
    try {
        await api('DELETE', `/api/v1/experiences/${id}`);
        toast('已删除', 'success');
        navigate('list');
    } catch (e) {
        toast('删除失败: ' + e.message, 'error');
    }
}
