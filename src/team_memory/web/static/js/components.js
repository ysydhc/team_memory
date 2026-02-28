/**
 * Reusable UI components: search bar, modals, webhooks, cards.
 */

import { state } from './store.js';
import { esc } from './utils.js';
import {
    resolveProjectInput,
    applyProjectPlaceholders,
    populateCreateTypeSelector,
    onCreateTypeChange,
    populateEditTypeSelector,
    onEditTypeChange,
    parseGitRefsFromTextarea,
    parseRelatedLinksFromTextarea,
    editGitRefsToText,
    editRelatedLinksToText,
} from './schema.js';
import { populateTagSuggestions } from './pages.js';

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
        document.getElementById('search-min-avg-rating').placeholder = `默认: ${cfg.min_avg_rating}`;
    }
}

export async function doSearch() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    const container = document.getElementById('search-results');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const body = { query, min_similarity: 0.3 };
        const maxResults = document.getElementById('search-max-results').value;
        const topKChildren = document.getElementById('search-top-k-children').value;
        const minAvgRating = document.getElementById('search-min-avg-rating').value;
        const usePageIndexLite = document.getElementById('search-use-pageindex').value;
        const project = resolveProjectInput(document.getElementById('search-project')?.value);
        if (maxResults) body.max_results = parseInt(maxResults, 10);
        if (topKChildren) body.top_k_children = parseInt(topKChildren, 10);
        if (minAvgRating !== '') body.min_avg_rating = parseFloat(minAvgRating);
        if (usePageIndexLite === 'true') body.use_pageindex_lite = true;
        if (usePageIndexLite === 'false') body.use_pageindex_lite = false;
        if (project) body.project = project;
        const data = await api('POST', '/api/v1/search', body);
        window.__renderExpList('search-results', data.results);
        if (data.results.length > 0) toast(`找到 ${data.results.length} 条相关经验`, 'success');
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>搜索失败</h3><p>${e.message}</p></div>`;
    }
}

// ===== Create Modal =====
export function openCreateModal() {
    document.getElementById('create-modal').classList.remove('hidden');
    state.createMode = 'manual';
    if (state.cachedTemplates.length === 0) {
        api('GET', '/api/v1/templates').then((r) => {
            state.cachedTemplates = r.templates || [];
            populateCreateTypeSelector();
            onCreateTypeChange();
        }).catch(() => {});
    } else {
        populateCreateTypeSelector();
        onCreateTypeChange();
    }
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
    state.createMode = mode;
    document.getElementById('tab-manual').classList.toggle('active', mode === 'manual');
    document.getElementById('tab-group').classList.toggle('active', mode === 'group');
    document.getElementById('tab-document').classList.toggle('active', mode === 'document');
    document.getElementById('tab-url').classList.toggle('active', mode === 'url');
    document.getElementById('create-mode-manual').classList.toggle('hidden', mode !== 'manual');
    document.getElementById('create-mode-group').classList.toggle('hidden', mode !== 'group');
    document.getElementById('create-mode-document').classList.toggle('hidden', mode !== 'document');
    document.getElementById('create-mode-url').classList.toggle('hidden', mode !== 'url');
    const showSave = mode === 'manual' || mode === 'group';
    document.getElementById('btn-save-exp').style.display = showSave ? '' : 'none';
    document.getElementById('btn-save-exp').textContent = mode === 'group' ? '保存经验组' : '保存经验';
}

export function toggleCreateQuickMode() {
    const quick = document.getElementById('create-quick-mode')?.checked || false;
    const ext = document.getElementById('create-extended-fields');
    const typeSpecific = document.getElementById('create-type-specific-fields');
    if (ext) ext.classList.toggle('hidden', quick);
    if (typeSpecific) typeSpecific.classList.toggle('hidden', quick);
}

export function closeCreateModal(force = false) {
    if (!force) {
        const title = document.getElementById('create-title')?.value?.trim();
        const problem = document.getElementById('create-problem')?.value?.trim();
        const solution = document.getElementById('create-solution')?.value?.trim();
        const gTitle = document.getElementById('group-parent-title')?.value?.trim();
        const gProblem = document.getElementById('group-parent-problem')?.value?.trim();
        const gSolution = document.getElementById('group-parent-solution')?.value?.trim();
        const parseContent = document.getElementById('parse-content')?.value?.trim();
        const parseUrl = document.getElementById('parse-url')?.value?.trim();
        if (title || problem || solution || gTitle || gProblem || gSolution || parseContent || parseUrl) {
            if (!confirm('当前内容尚未保存，确定关闭？')) return;
        }
    }
    document.getElementById('create-modal').classList.add('hidden');
    ['create-title', 'create-problem', 'create-solution', 'create-tags', 'create-language', 'create-framework', 'create-code', 'create-git-refs', 'create-related-links', 'create-project'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    document.getElementById('create-quick-mode').checked = false;
    toggleCreateQuickMode();
    document.getElementById('create-type-specific-fields').innerHTML = '';
    document.getElementById('parse-content').value = '';
    document.getElementById('parse-url').value = '';
    document.getElementById('url-status').textContent = '';
    state.groupChildrenIds = [];
    document.getElementById('group-children-container').innerHTML = '';
    ['group-parent-title', 'group-parent-problem', 'group-parent-solution', 'group-parent-tags', 'group-parent-language', 'group-parent-framework', 'group-parent-code', 'group-parent-project'].forEach((id) => {
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
      <div class="form-group">
        <label>代码片段</label>
        <textarea placeholder="相关代码..." data-child-code style="font-family:var(--font-mono);font-size:13px;min-height:80px"></textarea>
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

export async function doParse() {
    const content = document.getElementById('parse-content').value.trim();
    if (!content) {
        toast('请先粘贴文档内容', 'error');
        return;
    }

    const btn = document.getElementById('btn-parse');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const data = await api('POST', '/api/v1/experiences/parse-document', { content });
        document.getElementById('create-title').value = data.title || '';
        document.getElementById('create-problem').value = data.problem || '';
        document.getElementById('create-solution').value = data.solution || '';
        document.getElementById('create-tags').value = (data.tags || []).join(', ');
        document.getElementById('create-language').value = data.language || '';
        document.getElementById('create-framework').value = data.framework || '';
        document.getElementById('create-code').value = data.code_snippets || '';
        switchCreateMode('manual');
        toast('文档解析完成，请检查并补充信息', 'success');
    } catch (e) {
        toast('文档解析失败: ' + e.message, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

export async function doParseURL() {
    const url = document.getElementById('parse-url').value.trim();
    if (!url) {
        toast('请输入文档链接', 'error');
        return;
    }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        toast('链接必须以 http:// 或 https:// 开头', 'error');
        return;
    }

    const btn = document.getElementById('btn-parse-url');
    const status = document.getElementById('url-status');
    btn.classList.add('loading');
    btn.disabled = true;
    status.textContent = '正在抓取页面内容...';

    try {
        const data = await api('POST', '/api/v1/experiences/parse-url', { url });
        document.getElementById('create-title').value = data.title || '';
        document.getElementById('create-problem').value = data.problem || '';
        document.getElementById('create-solution').value = data.solution || '';
        document.getElementById('create-tags').value = (data.tags || []).join(', ');
        document.getElementById('create-language').value = data.language || '';
        document.getElementById('create-framework').value = data.framework || '';
        document.getElementById('create-code').value = data.code_snippets || '';
        switchCreateMode('manual');
        toast('链接内容解析完成，请检查并补充信息', 'success');
    } catch (e) {
        toast('链接解析失败: ' + e.message, 'error');
        status.textContent = '解析失败，请检查链接是否可访问';
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
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
        const language = document.getElementById('group-parent-language').value.trim() || null;
        const framework = document.getElementById('group-parent-framework').value.trim() || null;
        const code = document.getElementById('group-parent-code').value.trim() || null;

        const children = [];
        document.querySelectorAll('#group-children-container .group-child-block').forEach((block) => {
            const cTitle = (block.querySelector('[data-child-title]') || {}).value?.trim();
            const cProblem = (block.querySelector('[data-child-problem]') || {}).value?.trim() || '';
            const cSolution = (block.querySelector('[data-child-solution]') || {}).value?.trim() || '';
            const cCode = (block.querySelector('[data-child-code]') || {}).value?.trim() || null;
            if (cTitle) {
                children.push({ title: cTitle, problem: cProblem, solution: cSolution, code_snippets: cCode });
            }
        });
        if (children.length === 0) {
            toast('请至少添加一个有效的子经验（标题必填）', 'error');
            return;
        }

        try {
            await api('POST', '/api/v1/experiences/groups', {
                parent: { title, problem, solution, tags, language, framework, code_snippets: code, project },
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
    const language = quickMode ? null : (document.getElementById('create-language').value.trim() || null);
    const framework = quickMode ? null : (document.getElementById('create-framework').value.trim() || null);
    const code = quickMode ? null : (document.getElementById('create-code').value.trim() || null);
    const project = resolveProjectInput(document.getElementById('create-project')?.value);

    const experienceType = document.getElementById('create-experience-type')?.value || 'general';
    let severity = null;
    let category = null;
    let progressStatus = null;
    let structuredData = null;
    let gitRefs = null;
    let relatedLinks = null;

    if (!quickMode) {
        const sevEl = document.getElementById('create-severity');
        const catEl = document.getElementById('create-category');
        const progEl = document.getElementById('create-progress-status');
        severity = sevEl?.value || null;
        category = catEl?.value || null;
        progressStatus = progEl?.value || null;
        const tpl = (state.cachedTemplates || []).find((t) => (t.experience_type || t.id) === experienceType) || {};
        const structFields = tpl.structured_fields || [];
        if (structFields.length > 0) {
            structuredData = {};
            structFields.forEach((sf) => {
                const fid = 'create-sd-' + (sf.field || '').replace(/_/g, '-');
                const val = document.getElementById(fid)?.value?.trim();
                if (val !== undefined && val !== '') {
                    if (sf.type === 'list') {
                        structuredData[sf.field] = val.split('\n').map((l) => l.trim()).filter(Boolean);
                    } else {
                        structuredData[sf.field] = val;
                    }
                }
            });
            if (Object.keys(structuredData).length === 0) structuredData = null;
        }
        gitRefs = parseGitRefsFromTextarea(document.getElementById('create-git-refs')?.value);
        relatedLinks = parseRelatedLinksFromTextarea(document.getElementById('create-related-links')?.value);
    }

    const saveAsDraft = document.getElementById('create-draft-checkbox')?.checked || false;
    const publishStatus = saveAsDraft ? 'draft' : 'published';

    try {
        const body = {
            title, problem, solution, tags,
            language, framework, code_snippets: code,
            publish_status: publishStatus,
            experience_type: experienceType,
            project,
        };
        if (severity) body.severity = severity;
        if (category) body.category = category;
        if (progressStatus) body.progress_status = progressStatus;
        if (structuredData) body.structured_data = structuredData;
        if (gitRefs) body.git_refs = gitRefs;
        if (relatedLinks) body.related_links = relatedLinks;

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
                    title, problem, solution, tags, language, framework,
                    code_snippets: code, publish_status: publishStatus,
                    experience_type: experienceType, project, skip_dedup_check: true,
                };
                if (severity) retryBody.severity = severity;
                if (category) retryBody.category = category;
                if (progressStatus) retryBody.progress_status = progressStatus;
                if (structuredData) retryBody.structured_data = structuredData;
                if (gitRefs) retryBody.git_refs = gitRefs;
                if (relatedLinks) retryBody.related_links = relatedLinks;
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
        if (state.cachedTemplates.length === 0) {
            const r = await api('GET', '/api/v1/templates');
            state.cachedTemplates = r.templates || [];
        }
        populateEditTypeSelector();

        const exp = await api('GET', `/api/v1/experiences/${expId}`);
        state.editOriginalExp = exp;

        document.getElementById('edit-title').value = exp.title || '';
        document.getElementById('edit-problem').value = exp.description || '';
        document.getElementById('edit-root-cause').value = exp.root_cause || '';
        document.getElementById('edit-solution').value = exp.solution || '';
        document.getElementById('edit-tags').value = (exp.tags || []).join(', ');
        document.getElementById('edit-language').value = exp.programming_language || '';
        document.getElementById('edit-framework').value = exp.framework || '';
        document.getElementById('edit-code').value = exp.code_snippets || '';
        document.getElementById('edit-experience-type').value = exp.experience_type || 'general';
        document.getElementById('edit-visibility').value = exp.visibility || 'project';
        document.getElementById('edit-project').value = exp.project || '';
        document.getElementById('edit-git-refs').value = editGitRefsToText(exp.git_refs);
        document.getElementById('edit-related-links').value = editRelatedLinksToText(exp.related_links);

        onEditTypeChange();

        if (exp.children && exp.children.length > 0) {
            document.getElementById('edit-children-section').classList.remove('hidden');
            exp.children.forEach((child) => {
                addEditChild({
                    title: child.title || '',
                    problem: child.description || '',
                    solution: child.solution || '',
                    code_snippets: child.code_snippets || '',
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
    const c = (initial && initial.code_snippets) || '';
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
      <div class="form-group">
        <label>代码片段</label>
        <textarea placeholder="相关代码..." data-child-code style="font-family:var(--font-mono);font-size:13px;min-height:80px">${esc(c)}</textarea>
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
    const root_cause = document.getElementById('edit-root-cause').value.trim() || null;
    const code_snippets = document.getElementById('edit-code').value.trim() || null;
    const language = document.getElementById('edit-language').value.trim() || null;
    const framework = document.getElementById('edit-framework').value.trim() || null;
    const experience_type = document.getElementById('edit-experience-type')?.value || 'general';
    const visibility = document.getElementById('edit-visibility')?.value || 'project';
    const project = document.getElementById('edit-project')?.value?.trim() || null;
    const severity = document.getElementById('edit-severity')?.value || null;
    const category = document.getElementById('edit-category')?.value || null;
    const progress_status = document.getElementById('edit-progress-status')?.value || null;
    const gitRefs = parseGitRefsFromTextarea(document.getElementById('edit-git-refs')?.value);
    const relatedLinks = parseRelatedLinksFromTextarea(document.getElementById('edit-related-links')?.value);

    const tpl = (state.cachedTemplates || []).find((t) => (t.experience_type || t.id) === experience_type) || {};
    let structured_data = null;
    const structFields = tpl.structured_fields || [];
    if (structFields.length > 0) {
        structured_data = {};
        structFields.forEach((sf) => {
            const fid = 'edit-sd-' + (sf.field || '').replace(/_/g, '-');
            const val = document.getElementById(fid)?.value?.trim();
            if (val !== undefined && val !== '') {
                if (sf.type === 'list') {
                    structured_data[sf.field] = val.split('\n').map((l) => l.trim()).filter(Boolean);
                } else {
                    structured_data[sf.field] = val;
                }
            }
        });
        if (Object.keys(structured_data).length === 0) structured_data = null;
    }

    let children = null;
    const container = document.getElementById('edit-children-container');
    if (container && container.querySelectorAll('.group-child-block').length > 0) {
        children = [];
        container.querySelectorAll('.group-child-block').forEach((block) => {
            const cTitle = (block.querySelector('[data-child-title]') || {}).value?.trim();
            const cProblem = (block.querySelector('[data-child-problem]') || {}).value?.trim() || '';
            const cSolution = (block.querySelector('[data-child-solution]') || {}).value?.trim() || '';
            const cCode = (block.querySelector('[data-child-code]') || {}).value?.trim() || null;
            if (cTitle) {
                children.push({ title: cTitle, problem: cProblem, solution: cSolution, code_snippets: cCode });
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
    if (root_cause !== (orig?.root_cause || '')) body.root_cause = root_cause;
    if (JSON.stringify(tags) !== JSON.stringify(orig?.tags || [])) body.tags = tags;
    if (code_snippets !== (orig?.code_snippets || '')) body.code_snippets = code_snippets;
    if (language !== (orig?.programming_language || '')) body.language = language;
    if (framework !== (orig?.framework || '')) body.framework = framework;
    if (experience_type !== (orig?.experience_type || 'general')) body.experience_type = experience_type;
    if (visibility !== (orig?.visibility || 'project')) body.visibility = visibility;
    if (project !== (orig?.project || null)) body.project = project;
    if (severity !== (orig?.severity || '')) body.severity = severity;
    if (category !== (orig?.category || '')) body.category = category;
    if (progress_status !== (orig?.progress_status || '')) body.progress_status = progress_status;
    if (structured_data !== null && JSON.stringify(structured_data) !== JSON.stringify(orig?.structured_data || {})) {
        body.structured_data = structured_data;
    }
    if (gitRefs !== null && JSON.stringify(gitRefs) !== JSON.stringify(orig?.git_refs || [])) body.git_refs = gitRefs;
    if (relatedLinks !== null && JSON.stringify(relatedLinks) !== JSON.stringify(orig?.related_links || [])) {
        body.related_links = relatedLinks;
    }
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

// ===== Webhook Management =====
export function addWebhookRow(data = {}) {
    const idx = state.webhookRows.length;
    state.webhookRows.push(data);
    const container = document.getElementById('webhook-list');
    const div = document.createElement('div');
    div.id = 'webhook-row-' + idx;
    div.style.cssText = 'display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap';
    div.innerHTML = `
    <input type="text" placeholder="URL" value="${esc(data.url || '')}" style="flex:2;min-width:200px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border);border-radius:var(--radius);padding:4px 8px;font-size:13px" data-field="url">
    <input type="text" placeholder="events (逗号分隔)" value="${esc((data.events || []).join(','))}" style="flex:1;min-width:150px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border);border-radius:var(--radius);padding:4px 8px;font-size:13px" data-field="events">
    <input type="text" placeholder="secret" value="${esc(data.secret || '')}" style="flex:1;min-width:100px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border);border-radius:var(--radius);padding:4px 8px;font-size:13px" data-field="secret">
    <button class="btn btn-danger" onclick="removeWebhookRow(${idx})" style="font-size:12px;padding:4px 8px">删除</button>
    <button class="btn" onclick="testWebhook(${idx})" style="font-size:12px;padding:4px 8px">测试</button>
  `;
    container.appendChild(div);
}

export function removeWebhookRow(idx) {
    const el = document.getElementById('webhook-row-' + idx);
    if (el) el.remove();
    state.webhookRows[idx] = null;
}

export async function testWebhook(idx) {
    const row = document.getElementById('webhook-row-' + idx);
    if (!row) return;
    const url = row.querySelector('[data-field="url"]').value.trim();
    if (!url) {
        toast('请输入 URL', 'error');
        return;
    }
    try {
        const r = await api('POST', '/api/v1/config/webhooks/test', { url });
        if (r.success) toast('测试成功 (HTTP ' + r.status + ')', 'success');
        else toast('测试失败: ' + (r.error || 'HTTP ' + r.status), 'error');
    } catch (e) {
        toast('测试失败: ' + e.message, 'error');
    }
}

export async function saveWebhookConfig() {
    const configs = [];
    state.webhookRows.forEach((_, idx) => {
        const row = document.getElementById('webhook-row-' + idx);
        if (!row) return;
        const url = row.querySelector('[data-field="url"]').value.trim();
        const events = row.querySelector('[data-field="events"]').value.split(',').map((s) => s.trim()).filter(Boolean);
        const secret = row.querySelector('[data-field="secret"]').value.trim();
        if (url) configs.push({ url, events, secret, active: true });
    });
    try {
        await api('PUT', '/api/v1/config/webhooks', configs);
        toast('Webhook 配置已保存 (' + configs.length + ' 条)', 'success');
    } catch (e) {
        toast('保存失败: ' + e.message, 'error');
    }
}

export async function loadWebhookConfig() {
    try {
        const hooks = await api('GET', '/api/v1/config/webhooks');
        state.webhookRows = [];
        document.getElementById('webhook-list').innerHTML = '';
        (hooks || []).forEach((h) => addWebhookRow(h));
    } catch (_) {}
}

// ===== Import / Export =====
export function openImportModal() {
    state.importFile = null;
    document.getElementById('import-file').value = '';
    document.getElementById('import-preview').classList.add('hidden');
    document.getElementById('btn-import').disabled = true;
    document.getElementById('import-modal').classList.remove('hidden');
}

export function closeImportModal() {
    document.getElementById('import-modal').classList.add('hidden');
    state.importFile = null;
}

export function handleImportFile(input) {
    if (!input.files || input.files.length === 0) return;
    state.importFile = input.files[0];
    document.getElementById('import-file-info').textContent = `已选择: ${state.importFile.name} (${(state.importFile.size / 1024).toFixed(1)} KB)`;
    document.getElementById('import-preview').classList.remove('hidden');
    document.getElementById('btn-import').disabled = false;
}

export async function doImport() {
    if (!state.importFile) return;
    const btn = document.getElementById('btn-import');
    btn.disabled = true;
    btn.textContent = '导入中...';

    try {
        const formData = new FormData();
        formData.append('file', state.importFile);
        const opts = {
            method: 'POST',
            headers: {},
            credentials: 'include',
            body: formData,
        };
        if (state.apiKey) opts.headers['Authorization'] = `Bearer ${state.apiKey}`;
        const res = await fetch('/api/v1/experiences/import', opts);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Import failed');

        toast(`导入完成: ${data.imported}/${data.total} 条成功${data.errors && data.errors.length > 0 ? ', ' + data.errors.length + ' 条失败' : ''}`, 'success');
        closeImportModal();
        if (state.currentPage === 'list') loadList(state.listPage);
    } catch (e) {
        toast('导入失败: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '开始导入';
    }
}

export function openExportModal() {
    document.getElementById('export-tag').value = '';
    document.getElementById('export-start').value = '';
    document.getElementById('export-end').value = '';
    document.getElementById('export-format').value = 'json';
    document.getElementById('export-modal').classList.remove('hidden');
}

export function closeExportModal() {
    document.getElementById('export-modal').classList.add('hidden');
}

export async function doExport() {
    const format = document.getElementById('export-format').value;
    const tag = document.getElementById('export-tag').value.trim();
    const start = document.getElementById('export-start').value;
    const end = document.getElementById('export-end').value;

    let url = `/api/v1/experiences/export?format=${format}`;
    if (tag) url += `&tag=${encodeURIComponent(tag)}`;
    if (start) url += `&start=${start}`;
    if (end) url += `&end=${end}`;

    try {
        const opts = { method: 'GET', headers: {}, credentials: 'include' };
        if (state.apiKey) opts.headers['Authorization'] = `Bearer ${state.apiKey}`;
        const res = await fetch(url, opts);
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || 'Export failed');
        }
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `experiences.${format}`;
        a.click();
        URL.revokeObjectURL(a.href);
        toast('导出成功', 'success');
        closeExportModal();
    } catch (e) {
        toast('导出失败: ' + e.message, 'error');
    }
}
