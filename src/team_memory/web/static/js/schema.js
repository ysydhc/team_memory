/**
 * Schema-related dynamic UI: experience types, field rendering, filters.
 */

import { state, defaultTypeIcons } from './store.js';
import { esc } from './utils.js';

export function loadSchemaAndPopulateFilters(api) {
    api('GET', '/api/v1/schema')
        .then((schema) => {
            state.cachedSchema = schema;
            const typeSelect = document.getElementById('list-type-filter');
            if (typeSelect) {
                const current = typeSelect.value;
                let html = '<option value="">全部类型</option>';
                (schema.experience_types || []).forEach((t) => {
                    const icon = defaultTypeIcons[t.id] || '📝';
                    html += `<option value="${esc(t.id)}">${icon} ${esc(t.label || t.id)}</option>`;
                });
                typeSelect.innerHTML = html;
                if (current) typeSelect.value = current;
            }
            const sevSelect = document.getElementById('list-severity-filter');
            if (sevSelect) {
                const current = sevSelect.value;
                let sevHtml = '<option value="">全部严重程度</option>';
                (schema.severity_levels || []).forEach((s) => {
                    sevHtml += `<option value="${esc(s)}">${esc(s)}</option>`;
                });
                sevSelect.innerHTML = sevHtml;
                if (current) sevSelect.value = current;
            }
            const catSelect = document.getElementById('list-category-filter');
            if (catSelect) {
                const current = catSelect.value;
                let catHtml = '<option value="">全部分类</option>';
                (schema.categories || []).forEach((c) => {
                    catHtml += `<option value="${esc(c.id)}">${esc(c.label || c.id)}</option>`;
                });
                catSelect.innerHTML = catHtml;
                if (current) catSelect.value = current;
            }
        })
        .catch(() => {});
}

export function resolveProjectInput(raw) {
    const v = (raw || '').trim();
    if (v) return v;
    return state.defaultProject || 'default';
}

export function applyProjectPlaceholders() {
    const hint = `默认: ${state.defaultProject || 'default'}`;
    const ids = ['list-project-filter', 'search-project', 'create-project', 'group-parent-project'];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.placeholder = hint;
    });
}

export function populateCreateTypeSelector() {
    const sel = document.getElementById('create-experience-type');
    if (!sel) return;
    const current = sel.value;
    const templates =
        state.cachedTemplates.length > 0
            ? state.cachedTemplates
            : [{ experience_type: 'general', name: '通用经验', icon: '📝' }];
    sel.innerHTML = templates
        .map(
            (t) =>
                `<option value="${esc(t.experience_type || t.id || 'general')}">${t.icon || '📝'} ${esc(t.name || t.experience_type || t.id)}</option>`
        )
        .join('');
    if (current) sel.value = current;
    else sel.value = 'general';
}

export function onCreateTypeChange() {
    const typeEl = document.getElementById('create-experience-type');
    const type = typeEl ? typeEl.value : 'general';
    const tpl =
        (state.cachedTemplates || []).find((t) => (t.experience_type || t.id) === type) || {};
    const container = document.getElementById('create-type-specific-fields');
    if (!container) return;
    let html = '';
    const severityOpts = tpl.severity_options || [];
    const categoryOpts = tpl.category_options || [];
    const progressStates = tpl.progress_states || [];
    if (severityOpts.length > 0) {
        html += `<div class="form-group"><label>严重程度</label><select id="create-severity" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${severityOpts.map((s) => `<option value="${s}">${s}</option>`).join('')}</select></div>`;
    }
    if (categoryOpts.length > 0) {
        html += `<div class="form-group"><label>分类</label><select id="create-category" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${categoryOpts.map((c) => `<option value="${c}">${c}</option>`).join('')}</select></div>`;
    }
    if (progressStates.length > 0) {
        html += `<div class="form-group"><label>进度</label><select id="create-progress-status" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${progressStates.map((p) => `<option value="${p}">${p}</option>`).join('')}</select></div>`;
    }
    const structFields = tpl.structured_fields || [];
    structFields.forEach((sf) => {
        const fid = 'create-sd-' + (sf.field || '').replace(/_/g, '-');
        const label = sf.label || sf.field || '';
        const hint = sf.hint || '';
        html += `<div class="form-group"><label>${esc(label)}</label><textarea id="${fid}" placeholder="${esc(hint)}" style="min-height:60px;font-size:13px"></textarea></div>`;
    });
    container.innerHTML = html;
}

export function populateEditTypeSelector() {
    const sel = document.getElementById('edit-experience-type');
    if (!sel) return;
    const current = sel.value;
    const templates =
        state.cachedTemplates.length > 0
            ? state.cachedTemplates
            : [{ experience_type: 'general', name: '通用经验', icon: '📝' }];
    sel.innerHTML = templates
        .map(
            (t) =>
                `<option value="${esc(t.experience_type || t.id || 'general')}">${t.icon || '📝'} ${esc(t.name || t.experience_type || t.id)}</option>`
        )
        .join('');
    if (current) sel.value = current;
    else sel.value = 'general';
}

export function onEditTypeChange() {
    const typeEl = document.getElementById('edit-experience-type');
    const type = typeEl ? typeEl.value : 'general';
    const tpl =
        (state.cachedTemplates || []).find((t) => (t.experience_type || t.id) === type) || {};
    const container = document.getElementById('edit-type-specific-fields');
    if (!container) return;
    const exp = state.editOriginalExp;
    const sd = (exp && exp.structured_data) || {};
    let html = '';
    const severityOpts = tpl.severity_options || [];
    const categoryOpts = tpl.category_options || [];
    const progressStates = tpl.progress_states || [];
    if (severityOpts.length > 0) {
        const v = (exp && exp.severity) || '';
        html += `<div class="form-group"><label>严重程度</label><select id="edit-severity" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${severityOpts.map((s) => `<option value="${s}" ${v === s ? 'selected' : ''}>${s}</option>`).join('')}</select></div>`;
    }
    if (categoryOpts.length > 0) {
        const v = (exp && exp.category) || '';
        html += `<div class="form-group"><label>分类</label><select id="edit-category" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${categoryOpts.map((c) => `<option value="${c}" ${v === c ? 'selected' : ''}>${c}</option>`).join('')}</select></div>`;
    }
    if (progressStates.length > 0) {
        const v = (exp && exp.progress_status) || '';
        html += `<div class="form-group"><label>进度</label><select id="edit-progress-status" style="background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border)"><option value="">—</option>${progressStates.map((p) => `<option value="${p}" ${v === p ? 'selected' : ''}>${p}</option>`).join('')}</select></div>`;
    }
    const structFields = tpl.structured_fields || [];
    structFields.forEach((sf) => {
        const fid = 'edit-sd-' + (sf.field || '').replace(/_/g, '-');
        const val = sd[sf.field];
        const str = Array.isArray(val) ? (val || []).join('\n') : val || '';
        const label = sf.label || sf.field || '';
        const h = sf.hint || '';
        html += `<div class="form-group"><label>${esc(label)}</label><textarea id="${fid}" placeholder="${esc(h)}" style="min-height:60px;font-size:13px">${esc(str)}</textarea></div>`;
    });
    container.innerHTML = html;
}

export function parseGitRefsFromTextarea(text) {
    if (!text || !text.trim()) return null;
    const lines = text.trim().split('\n').filter((l) => l.trim());
    if (lines.length === 0) return null;
    const refs = [];
    lines.forEach((line) => {
        const parts = line.split('|').map((p) => p.trim());
        const type = (parts[0] || 'commit').toLowerCase();
        const url = parts[1] || null;
        const hash = parts[2] || null;
        const desc = parts[3] || null;
        if (url || hash) {
            refs.push({
                type: type === 'pr' ? 'pr' : type === 'branch' ? 'branch' : 'commit',
                url: url || undefined,
                hash: hash || undefined,
                description: desc || undefined,
            });
        }
    });
    return refs.length ? refs : null;
}

export function parseRelatedLinksFromTextarea(text) {
    if (!text || !text.trim()) return null;
    const lines = text.trim().split('\n').filter((l) => l.trim());
    if (lines.length === 0) return null;
    const links = [];
    lines.forEach((line) => {
        const idx = line.indexOf('|');
        const url = (idx >= 0 ? line.slice(0, idx) : line).trim();
        const title = idx >= 0 ? line.slice(idx + 1).trim() : null;
        if (url) links.push({ type: 'other', url, title: title || undefined });
    });
    return links.length ? links : null;
}

export function editGitRefsToText(refs) {
    if (!refs || refs.length === 0) return '';
    return refs
        .map((r) =>
            [r.type || 'commit', r.url || '', r.hash || '', r.description || '']
                .filter(Boolean)
                .join('|')
        )
        .join('\n');
}

export function editRelatedLinksToText(links) {
    if (!links || links.length === 0) return '';
    return links.map((l) => (l.title ? l.url + '|' + l.title : l.url)).join('\n');
}
