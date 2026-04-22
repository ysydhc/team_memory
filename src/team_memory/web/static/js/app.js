/**
 * Main application bootstrap: state, API, auth, routing.
 */

import { state } from './store.js';
import {
    loadSchemaAndPopulateFilters,
    applyProjectPlaceholders,
    resolveProjectInput,
} from './schema.js';
import { esc } from './utils.js';
import * as pages from './pages.js';
import * as components from './components.js';

// ===== Toast (uses DOM container) =====
function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    const container = document.getElementById('toast-container');
    if (container) container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// ===== API Helper =====
async function api(method, path, body = null) {
    const opts = {
        method,
        credentials: 'include',
    };
    const key = state.apiKey || (typeof localStorage !== 'undefined' && localStorage.getItem('api_key'));
    if (body instanceof FormData) {
        opts.headers = {};
        if (key) opts.headers['Authorization'] = `Bearer ${key}`;
        opts.body = body;
    } else {
        opts.headers = { 'Content-Type': 'application/json' };
        if (key) opts.headers['Authorization'] = `Bearer ${key}`;
        if (body) opts.body = JSON.stringify(body);
    }

    const res = await fetch(path, opts);
    if (res.status === 401) {
        doLogout();
        throw new Error('Session expired');
    }
    let data;
    try {
        const text = await res.text();
        data = text ? JSON.parse(text) : {};
    } catch (_) {
        throw new Error(res.statusText || `HTTP ${res.status}`);
    }
    if (!res.ok) {
        const msg = data.detail || res.statusText || 'Request failed';
        const hint = data.ops_hint ? ' [' + data.ops_hint + ']' : '';
        throw new Error(msg + hint);
    }
    return data;
}

// ===== Login mode switching =====
let _loginMode = 'password'; // 'password' | 'register'

function switchLoginMode(mode) {
    _loginMode = mode;
    document.getElementById('login-mode-password').style.display = mode === 'password' ? 'block' : 'none';
    document.getElementById('login-mode-register').style.display = mode === 'register' ? 'block' : 'none';
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('login-success').style.display = 'none';
    const sub = document.getElementById('login-subtitle');
    if (mode === 'register') sub.textContent = '注册新账号';
    else sub.textContent = '团队经验数据库';
}

// ===== Auth =====
async function doLogin() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    if (!username || !password) return;
    const body = { username, password };

    try {
        const data = await api('POST', '/api/v1/auth/login', body);
        if (data.success) {
            state.apiKey = '';
            if (typeof localStorage !== 'undefined') localStorage.removeItem('api_key');
            state.currentUser = { name: data.user, role: data.role };
            showApp();
        } else {
            const err = document.getElementById('login-error');
            err.textContent = data.message || '登录失败';
            err.style.display = 'block';
        }
    } catch (e) {
        const err = document.getElementById('login-error');
        err.textContent = '连接服务器失败';
        err.style.display = 'block';
    }
}

async function doRegister() {
    const username = document.getElementById('reg-username').value.trim();
    const pwd = document.getElementById('reg-password').value;
    const pwd2 = document.getElementById('reg-password2').value;
    const errEl = document.getElementById('login-error');
    errEl.style.display = 'none';

    if (!username || username.length < 2) {
        errEl.textContent = '用户名至少 2 个字符';
        errEl.style.display = 'block';
        return;
    }
    if (!pwd || pwd.length < 6) {
        errEl.textContent = '密码至少 6 个字符';
        errEl.style.display = 'block';
        return;
    }
    if (pwd !== pwd2) {
        errEl.textContent = '两次密码输入不一致';
        errEl.style.display = 'block';
        return;
    }

    try {
        const data = await api('POST', '/api/v1/auth/register', { username, password: pwd });
        if (data.success) {
            const suc = document.getElementById('login-success');
            suc.textContent = data.message || '注册成功，请等待管理员审批';
            suc.style.display = 'block';
            document.getElementById('reg-username').value = '';
            document.getElementById('reg-password').value = '';
            document.getElementById('reg-password2').value = '';
            setTimeout(() => switchLoginMode('password'), 2500);
        } else {
            errEl.textContent = data.detail || data.message || '注册失败';
            errEl.style.display = 'block';
        }
    } catch (e) {
        errEl.textContent = e.message || '注册失败';
        errEl.style.display = 'block';
    }
}

function doLogout() {
    state.apiKey = '';
    if (typeof localStorage !== 'undefined') localStorage.removeItem('api_key');
    api('POST', '/api/v1/auth/logout').catch(() => { });
    document.getElementById('login-screen').style.display = 'flex';
    document.getElementById('app-screen').style.display = 'none';
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('login-success').style.display = 'none';
    switchLoginMode('password');
}

async function checkAuth() {
    const stored = typeof localStorage !== 'undefined' && localStorage.getItem('api_key');
    if (stored) state.apiKey = stored;
    try {
        const data = await api('GET', '/api/v1/auth/me');
        state.currentUser = { name: data.user, role: data.role };
        showApp();
        return true;
    } catch {
        return false;
    }
}

function showApp() {
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app-screen').style.display = 'block';
    document.getElementById('user-name').textContent = state.currentUser.name;
    document.getElementById('user-avatar').textContent = state.currentUser.name[0].toUpperCase();
    api('GET', '/api/v1/config/retrieval').then((cfg) => { state.cachedRetrievalConfig = cfg; }).catch(() => { });
    api('GET', '/api/v1/projects').then((data) => {
        const projects = data.projects || [];
        state.availableProjects = projects;
        const preferred = projects.includes('team_memory') ? 'team_memory' : (projects[0] || 'default');
        state.defaultProject = preferred;
        state.activeProject = preferred;
        populateProjectSwitcher();
        applyProjectPlaceholders();
        reloadCurrentPage();
    }).catch(() => {
        state.defaultProject = 'default';
        state.activeProject = 'default';
        applyProjectPlaceholders();
        reloadCurrentPage();
    });
    applyProjectPlaceholders();
    loadSchemaAndPopulateFilters(api);
    navigateFromHash();
    window.addEventListener('hashchange', navigateFromHash);
    window.addEventListener('popstate', navigateFromHash);
}

// ===== Project Multi-Select =====
state.activeProjects = [];

function populateProjectSwitcher() {
    const projects = state.availableProjects || [];
    const containers = [
        'page-list-projects',
        'page-search-projects',
    ];
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (!container) return;
        const dropdown = container.querySelector('.proj-dropdown');
        if (!dropdown) return;
        const pageKey = containerId.replace('page-', '').replace('-projects', '');
        let html = `<div class="proj-opt selected" data-value="__all__" onclick="toggleProjectOpt(this,'${pageKey}')"><span class="proj-check">✓</span><span>全部项目</span></div>`;
        projects.forEach(p => {
            html += `<div class="proj-opt" data-value="${esc(p)}" onclick="toggleProjectOpt(this,'${pageKey}')"><span class="proj-check">✓</span><span>${esc(p)}</span></div>`;
        });
        dropdown.innerHTML = html;
        updateProjectTrigger(containerId, pageKey);
    });
}

function toggleProjectDropdown(btn) {
    const dropdown = btn.nextElementSibling;
    if (!dropdown) return;
    const isOpen = dropdown.classList.contains('open');
    document.querySelectorAll('.proj-dropdown.open').forEach(d => d.classList.remove('open'));
    if (!isOpen) {
        dropdown.classList.add('open');
        const close = (e) => {
            if (!btn.parentElement.contains(e.target)) {
                dropdown.classList.remove('open');
                document.removeEventListener('click', close);
            }
        };
        setTimeout(() => document.addEventListener('click', close), 0);
    }
}

function getSelectedProjects(pageKey) {
    const containerId = 'page-' + pageKey + '-projects';
    const container = document.getElementById(containerId);
    if (!container) return [];
    const allOpt = container.querySelector('.proj-opt[data-value="__all__"]');
    if (allOpt && allOpt.classList.contains('selected')) return [];
    const selected = [];
    container.querySelectorAll('.proj-opt.selected').forEach(o => {
        if (o.dataset.value !== '__all__') selected.push(o.dataset.value);
    });
    return selected;
}

function toggleProjectOpt(el, pageKey) {
    const containerId = 'page-' + pageKey + '-projects';
    const container = document.getElementById(containerId);
    if (!container) return;
    const isAll = el.dataset.value === '__all__';
    const opts = container.querySelectorAll('.proj-opt');
    if (isAll) {
        opts.forEach(o => o.classList.toggle('selected', o.dataset.value === '__all__'));
    } else {
        el.classList.toggle('selected');
        const allOpt = container.querySelector('.proj-opt[data-value="__all__"]');
        const anySelected = Array.from(opts).some(o => o.dataset.value !== '__all__' && o.classList.contains('selected'));
        if (allOpt) allOpt.classList.toggle('selected', !anySelected);
    }
    updateProjectTrigger(containerId, pageKey);
    if (pageKey === 'list') pages.loadList(1);
}

function updateProjectTrigger(containerId, pageKey) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const label = container.querySelector('.proj-label');
    const tagsEl = container.querySelector('.proj-tags');
    const selected = getSelectedProjects(pageKey);
    if (!selected.length) {
        if (label) label.textContent = '全部项目';
        if (tagsEl) tagsEl.innerHTML = '';
    } else if (selected.length <= 2) {
        if (label) label.textContent = '';
        if (tagsEl) tagsEl.innerHTML = selected.map(p => `<span class="proj-tag">${esc(p)}</span>`).join('');
    } else {
        if (label) label.textContent = '';
        if (tagsEl) {
            tagsEl.innerHTML = selected.slice(0, 2).map(p => `<span class="proj-tag">${esc(p)}</span>`).join('')
                + `<span class="proj-tag">+${selected.length - 2}</span>`;
        }
    }
}

function onProjectMultiChange(pageKey) {
    toggleProjectOpt(event.target.closest('.proj-opt'), pageKey);
}

function onProjectSwitch(value) {
    state.activeProject = value || state.defaultProject || 'default';
    applyProjectPlaceholders();
    const page = state.currentPage;
    if (page === 'list') pages.loadList(1);
    if (page === 'archives') pages.loadArchivesList(1);
}

// ===== Navigation =====
// Hash-to-page mapping for URL-driven navigation (enables browser automation / deep links)
const HASH_TO_PAGE = {
    list: 'list',
    search: 'search',
    archives: 'archives',
    settings: 'settings',
    'personal-memory': 'personal-memory',
    dedup: 'dedup',
    // Legacy redirects
    dashboard: 'list',
    drafts: 'list',
    reviews: 'list',
};

function reloadCurrentPage() {
    if (!state.currentPage) return;
    const page = state.currentPage;
    if (page === 'list') {
        pages.loadList(state.listPage || 1);
    } else if (page === 'archives') {
        pages.loadArchivesList(state.archivesListPage || 1);
    } else if (page === 'personal-memory') {
        pages.loadPersonalMemoryList();
    } else if (page === 'settings') {
        const dotEl = document.getElementById('health-status-dot');
        if (dotEl) {
            dotEl.className = 'health-dot loading';
            dotEl.title = '加载中…';
        }
        pages.loadHealthStatus();
        pages.loadRuntimeConfigForms();
        pages.loadAccountSecurity();
        pages.loadKeyManagement();
    }
}

function navigate(page) {
    if (state.currentPage === page && page !== 'archives') return;
    const fromSettings = state.currentPage === 'settings';
    const toSubPage = ['personal-memory', 'dedup', 'janitor'].includes(page);
    const toSettings = page === 'settings';
    const fromSubPage = ['personal-memory', 'dedup', 'janitor'].includes(state.currentPage);

    if (fromSettings && toSubPage) {
        state.settingsScrollTop = window.scrollY || document.documentElement.scrollTop;
    }

    state.currentPage = page;
    document.querySelectorAll('.page').forEach((p) => p.classList.add('hidden'));
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.remove('hidden');
    document.querySelectorAll('.topbar-nav a').forEach((a) => {
        a.classList.toggle('active', a.dataset.page === page);
    });
    const hash = `#${page}`;
    if (location.hash.slice(1) !== page) {
        history.pushState(null, '', location.pathname + hash);
    }

    if (toSettings && fromSubPage && state.settingsScrollTop > 0) {
        requestAnimationFrame(() => {
            window.scrollTo(0, state.settingsScrollTop);
        });
    }

    if (page === 'list') {
        pages.loadList(state.listPage || 1);
    }
    else if (page === 'search') { }
    else if (page === 'archives') {
        pages.showArchivesListView();
        pages.loadArchivesList(1);
    }
    else if (page === 'personal-memory') pages.loadPersonalMemoryList();
    else if (page === 'janitor') pages.loadJanitorPage();
    else if (page === 'settings') {
        const dotEl = document.getElementById('health-status-dot');
        if (dotEl) {
            dotEl.className = 'health-dot loading';
            dotEl.title = '加载中…';
        }
        pages.loadHealthStatus();
        pages.loadRuntimeConfigForms();
        pages.loadAccountSecurity();
        pages.loadKeyManagement();
    }
}

/** Placeholder: experience list 「批量生成摘要」按钮（避免未定义报错） */
function batchSummarize() {
    toast('批量生成摘要功能尚未接入', 'info');
}

function navigateFromHash() {
    const raw = location.hash.slice(1);
    if (raw.startsWith('detail/')) {
        const id = raw.slice(7);
        if (id) {
            pages.showDetail(id);
            return;
        }
    }
    if (raw === 'archives' || raw.startsWith('archives/')) {
        state.currentPage = 'archives';
        document.querySelectorAll('.page').forEach((p) => p.classList.add('hidden'));
        const archPage = document.getElementById('page-archives');
        if (archPage) archPage.classList.remove('hidden');
        document.querySelectorAll('.topbar-nav a').forEach((a) => {
            a.classList.toggle('active', a.dataset.page === 'archives');
        });
        const segs = raw.split('/').filter(Boolean);
        if (segs.length >= 2 && segs[0] === 'archives') {
            pages.openArchiveDetail(segs[1]);
        } else {
            pages.showArchivesListView();
            pages.loadArchivesList(state.archivesListPage || 1);
        }
        return;
    }
    const page = HASH_TO_PAGE[raw] || 'list';
    navigate(page);
}

// ===== Expose to window for onclick handlers =====
window.__api = api;
window.__toast = toast;
window.__navigate = navigate;
window.__showDetail = pages.showDetail;
window.__viewDetail = pages.viewDetail;
window.__loadDashboard = pages.loadDashboard;
window.__loadList = pages.loadList;
window.__loadDrafts = pages.loadDrafts;
window.__renderExpList = pages.renderExpList;
window.__resolveProjectInput = resolveProjectInput;

// Auth
window.doLogin = doLogin;
window.doLogout = doLogout;
window.doRegister = doRegister;
window.switchLoginMode = switchLoginMode;

// Nav & pages
window.navigate = navigate;
window.onProjectSwitch = onProjectSwitch;
window.toggleProjectDropdown = toggleProjectDropdown;
window.onProjectMultiChange = onProjectMultiChange;
window.getSelectedProjects = getSelectedProjects;
window.toggleProjectOpt = toggleProjectOpt;
window.loadList = pages.loadList;
window.switchListSubTab = pages.switchListSubTab;
window.filterByTag = pages.filterByTag;
window.showDetail = pages.showDetail;
window.viewDetail = pages.viewDetail;
window.backToPreviousDetail = pages.backToPreviousDetail;
window.loadDashboard = pages.loadDashboard;
window.loadDrafts = pages.loadDrafts;
window.changeExpStatus = pages.changeExpStatus;
window.publishDraft = pages.publishDraft;
window.loadReviews = pages.loadReviews;
window.reviewExperience = pages.reviewExperience;
window.appendTag = pages.appendTag;
window.loadInstallables = pages.loadInstallables;
window.previewInstallable = pages.previewInstallable;
window.toggleInstallablePreview = pages.toggleInstallablePreview;
window.installInstallable = pages.installInstallable;
window.loadInstalledInstallables = pages.loadInstalledInstallables;
window.openEditInstallableModalFromBtn = pages.openEditInstallableModalFromBtn;
window.closeEditInstallableModal = pages.closeEditInstallableModal;
window.switchEditInstallableTab = pages.switchEditInstallableTab;
window.saveEditInstallableContent = pages.saveEditInstallableContent;
window.loadPersonalMemoryList = pages.loadPersonalMemoryList;
window.loadArchivesList = pages.loadArchivesList;
window.openArchiveDetail = pages.openArchiveDetail;
window.backToArchivesList = pages.backToArchivesList;
window.showAddPersonalMemoryModal = pages.showAddPersonalMemoryModal;
window.editPersonalMemory = pages.editPersonalMemory;
window.deletePersonalMemory = pages.deletePersonalMemory;
window.batchSummarize = batchSummarize;

// Components
window.toggleSearchAdvanced = components.toggleSearchAdvanced;
window.doSearch = components.doSearch;
window.showSearchHistoryDropdown = components.showSearchHistoryDropdown;
window.hideSearchHistoryDropdownSoon = components.hideSearchHistoryDropdownSoon;
window.openCreateModal = components.openCreateModal;
window.closeCreateModal = components.closeCreateModal;
window.switchCreateMode = components.switchCreateMode;
window.toggleCreateQuickMode = components.toggleCreateQuickMode;
window.addGroupChild = components.addGroupChild;
window.removeGroupChild = components.removeGroupChild;
window.doCreate = components.doCreate;
window.openFeedbackModal = components.openFeedbackModal;
window.closeFeedbackModal = components.closeFeedbackModal;
window.submitFeedback = components.submitFeedback;
window.openEditModal = components.openEditModal;
window.closeEditModal = components.closeEditModal;
window.addEditChild = components.addEditChild;
window.removeEditChild = components.removeEditChild;
window.submitEdit = components.submitEdit;
window.deleteExp = components.deleteExp;
window.togglePin = components.togglePin;
window.reviveExp = components.reviveExp;

// Key management exports
window.loadAccountSecurity = pages.loadAccountSecurity;
window.doChangePassword = pages.doChangePassword;
window.loadKeyManagement = pages.loadKeyManagement;
window.loadRuntimeConfigForms = pages.loadRuntimeConfigForms;
window.saveRetrievalConfig = pages.saveRetrievalConfig;
window.saveSearchConfig = pages.saveSearchConfig;
window.loadAllConfig = pages.loadAllConfig;
window.loadDuplicates = pages.loadDuplicates;
window.loadJanitorPage = pages.loadJanitorPage;
window.runJanitorNow = pages.runJanitorNow;
window.reembedGroups = pages.reembedGroups;
window.approveUser = pages.approveUser;
window.rejectUser = pages.rejectUser;
window.createUserAdmin = pages.createUserAdmin;
window.updateUserRole = pages.updateUserRole;
window.toggleUserActive = pages.toggleUserActive;
window.openAdminCreateUser = function () {
    document.getElementById('admin-create-user-form').style.display = 'block';
};

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    // Login / register: bind here so we use module-scoped functions (inline onclick sees global only).
    document.getElementById('login-btn-submit')?.addEventListener('click', () => doLogin());
    document.getElementById('login-btn-register')?.addEventListener('click', () => doRegister());
    document.getElementById('login-link-to-register')?.addEventListener('click', (e) => {
        e.preventDefault();
        switchLoginMode('register');
    });
    document.getElementById('login-link-to-login')?.addEventListener('click', (e) => {
        e.preventDefault();
        switchLoginMode('password');
    });
    document.getElementById('login-password').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doLogin();
    });
    document.getElementById('reg-password2').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doRegister();
    });
});
