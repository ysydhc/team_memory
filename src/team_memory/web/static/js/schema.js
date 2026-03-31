/**
 * Schema-related dynamic UI: experience types, filters, project helpers.
 */

import { state } from './store.js';

export function loadSchemaAndPopulateFilters(_api) {
    const typeSelect = document.getElementById('list-type-filter');
    if (typeSelect) {
        typeSelect.innerHTML = '<option value="">全部类型</option><option value="general">📝 general</option>';
    }
}

export function resolveProjectInput(raw) {
    const v = (raw || '').trim();
    if (v) return v;
    return state.activeProject || state.defaultProject || 'default';
}

export function applyProjectPlaceholders() {
    const proj = state.activeProject || state.defaultProject || 'default';
    const hint = `默认: ${proj}`;
    const ids = ['search-project', 'create-project', 'group-parent-project'];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.placeholder = hint;
    });
}
