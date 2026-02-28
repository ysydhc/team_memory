/**
 * Main application bootstrap: state, API, auth, routing.
 */

import { state } from './store.js';
import {
    loadSchemaAndPopulateFilters,
    applyProjectPlaceholders,
    resolveProjectInput,
    populateCreateTypeSelector,
} from './schema.js';
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
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
    };
    const key = state.apiKey || (typeof localStorage !== 'undefined' && localStorage.getItem('api_key'));
    if (key) opts.headers['Authorization'] = `Bearer ${key}`;
    if (body) opts.body = JSON.stringify(body);

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
let _loginMode = 'password'; // 'password' | 'apikey' | 'register'

function switchLoginMode(mode) {
    _loginMode = mode;
    document.getElementById('login-mode-password').style.display = mode === 'password' ? 'block' : 'none';
    document.getElementById('login-mode-apikey').style.display = mode === 'apikey' ? 'block' : 'none';
    document.getElementById('login-mode-register').style.display = mode === 'register' ? 'block' : 'none';
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('login-success').style.display = 'none';
    const sub = document.getElementById('login-subtitle');
    if (mode === 'register') sub.textContent = '注册新账号';
    else if (mode === 'apikey') sub.textContent = '使用 API Key 登录';
    else sub.textContent = '团队经验数据库';
}

// ===== Auth =====
async function doLogin() {
    let body;
    if (_loginMode === 'apikey') {
        const key = document.getElementById('login-key').value.trim();
        if (!key) return;
        body = { api_key: key };
    } else {
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;
        if (!username || !password) return;
        body = { username, password };
    }

    try {
        const data = await api('POST', '/api/v1/auth/login', body);
        if (data.success) {
            if (body.api_key) {
                state.apiKey = body.api_key;
                if (typeof localStorage !== 'undefined') localStorage.setItem('api_key', body.api_key);
            } else {
                state.apiKey = '';
                if (typeof localStorage !== 'undefined') localStorage.removeItem('api_key');
            }
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
    api('POST', '/api/v1/auth/logout').catch(() => {});
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
    api('GET', '/api/v1/config/retrieval').then((cfg) => { state.cachedRetrievalConfig = cfg; }).catch(() => {});
    api('GET', '/api/v1/config/project')
        .then((cfg) => {
            state.defaultProject = cfg.default_project || 'default';
            state.activeProject = state.defaultProject;
            applyProjectPlaceholders();
            reloadCurrentPage();
        })
        .catch(() => {
            state.defaultProject = 'default';
            state.activeProject = 'default';
            applyProjectPlaceholders();
            reloadCurrentPage();
        });
    api('GET', '/api/v1/templates')
        .then((r) => {
            state.cachedTemplates = r.templates || [];
            populateCreateTypeSelector();
        })
        .catch(() => { state.cachedTemplates = []; });
    api('GET', '/api/v1/projects')
        .then((r) => {
            state.availableProjects = r.projects || [];
            populateProjectSwitcher();
        })
        .catch(() => {});
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
        'page-list-projects', 'page-tasks-projects',
        'page-search-projects', 'page-usage-projects',
    ];
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (!container) return;
        const dropdown = container.querySelector('.proj-dropdown');
        if (!dropdown) return;
        const pageKey = containerId.replace('page-', '').replace('-projects', '');
        let html = `<div class="proj-opt selected" data-value="__all__" onclick="toggleProjectOpt(this,'${pageKey}')"><span class="proj-check">✓</span><span>全部项目</span></div>`;
        projects.forEach(p => {
            html += `<div class="proj-opt" data-value="${p}" onclick="toggleProjectOpt(this,'${pageKey}')"><span class="proj-check">✓</span><span>${p}</span></div>`;
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
    else if (pageKey === 'tasks') pages.loadTasks();
    else if (pageKey === 'usage') pages.loadUsageStats();
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
        if (tagsEl) tagsEl.innerHTML = selected.map(p => `<span class="proj-tag">${p}</span>`).join('');
    } else {
        if (label) label.textContent = '';
        if (tagsEl) {
            tagsEl.innerHTML = selected.slice(0, 2).map(p => `<span class="proj-tag">${p}</span>`).join('')
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
    else if (page === 'tasks') pages.loadTasks();
    else if (page === 'usage') pages.loadUsageStats();
}

// ===== Navigation =====
// Hash-to-page mapping for URL-driven navigation (enables browser automation / deep links)
const HASH_TO_PAGE = {
    list: 'list',
    tasks: 'tasks',
    search: 'search',
    settings: 'settings',
    usage: 'usage',
    dedup: 'dedup',
    // Legacy redirects
    dashboard: 'list',
    drafts: 'list',
    reviews: 'list',
};

function reloadCurrentPage() {
    if (state.currentPage) navigate(state.currentPage);
}

function navigate(page) {
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

    if (page === 'list') {
        pages.loadList(state.listPage || 1);
        pages.checkOutdatedCount();
    }
    else if (page === 'tasks') pages.loadTasks();
    else if (page === 'search') {}
    else if (page === 'usage') pages.loadUsageStats();
    else if (page === 'dedup') {}
    else if (page === 'settings') {
        pages.loadRetrievalConfig();
        components.loadWebhookConfig();
        pages.loadCurrentSchema();
        pages.loadScoringConfig();
        pages.checkMergeSuggestions();
        if (state.currentUser && state.currentUser.role === 'admin') {
            pages.loadKeyManagement();
        }
    }
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
    const page = HASH_TO_PAGE[raw] || 'tasks';
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
window.switchProjCfgTab = function(tab) {
    document.querySelectorAll('.proj-cfg-tab').forEach(t => t.classList.toggle('active', t.textContent.includes(
        tab === 'basic' ? '基础' : tab === 'scan' ? '扫描' : '可安装'
    )));
    document.querySelectorAll('.proj-cfg-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById('proj-cfg-' + tab);
    if (panel) panel.classList.add('active');
};
window.loadList = pages.loadList;
window.switchListSubTab = pages.switchListSubTab;
window.filterByTag = pages.filterByTag;
window.showDetail = pages.showDetail;
window.viewDetail = pages.viewDetail;
window.loadDashboard = pages.loadDashboard;
window.loadDrafts = pages.loadDrafts;
window.loadTasks = pages.loadTasks;
window.showTaskDetail = pages.showTaskDetail;
window.closeTaskSlideout = pages.closeTaskSlideout;
window.saveTaskFromSlideout = pages.saveTaskFromSlideout;
window.deleteTaskFromSlideout = pages.deleteTaskFromSlideout;
window.createTaskGroupFromSlideout = pages.createTaskGroupFromSlideout;
window.sendTaskMessage = pages.sendTaskMessage;
window.archiveGroup = pages.archiveGroup;
window.toggleTaskGroups = pages.toggleTaskGroups;
window.toggleGroupVisibility = pages.toggleGroupVisibility;
window.generateTaskPrompt = pages.generateTaskPrompt;
window.openCreateTaskModal = () => toast('任务创建面板即将上线', 'info');
window.changeExpStatus = pages.changeExpStatus;
window.publishDraft = pages.publishDraft;
window.loadReviews = pages.loadReviews;
window.reviewExperience = pages.reviewExperience;
window.loadDuplicates = pages.loadDuplicates;
window.doMerge = pages.doMerge;
window.toggleDupDiff = pages.toggleDupDiff;
window.scanStale = pages.scanStale;
window.loadUsageStats = pages.loadUsageStats;
window.appendTag = pages.appendTag;
window.toggleVersionHistory = pages.toggleVersionHistory;
window.viewVersionSnapshot = pages.viewVersionSnapshot;
window.toggleVersionSnapshot = pages.toggleVersionSnapshot;
window.rollbackVersion = pages.rollbackVersion;
window.loadInstallables = pages.loadInstallables;
window.previewInstallable = pages.previewInstallable;
window.toggleInstallablePreview = pages.toggleInstallablePreview;
window.installInstallable = pages.installInstallable;
window.loadAllConfig = pages.loadAllConfig;
window.saveRetrievalConfig = pages.saveRetrievalConfig;
window.saveDefaultProjectConfig = pages.saveDefaultProjectConfig;
window.saveScanDirsConfig = pages.saveScanDirsConfig;
window.loadScanDirsConfig = pages.loadScanDirsConfig;
window.saveSearchConfig = pages.saveSearchConfig;
window.saveRerankerConfig = pages.saveRerankerConfig;
window.saveCacheConfig = pages.saveCacheConfig;
window.savePageIndexLiteConfig = pages.savePageIndexLiteConfig;
window.clearCache = pages.clearCache;
window.switchPreset = pages.switchPreset;
window.generateSchemaFromDoc = pages.generateSchemaFromDoc;
window.applyGeneratedSchema = pages.applyGeneratedSchema;
window.loadCurrentSchema = pages.loadCurrentSchema;
window.generateSummary = pages.generateSummary;
window.batchSummarize = pages.batchSummarize;
window.toggleOutdatedPanel = pages.toggleOutdatedPanel;
window.scoreAction = pages.scoreAction;
window.refreshScores = pages.refreshScores;
window.saveScoringConfig = pages.saveScoringConfig;
window.loadMergePreview = pages.loadMergePreview;

window.addCustomScanPath = function() {
    const container = document.getElementById('custom-scan-paths');
    const row = document.createElement('div');
    row.className = 'scan-path-row custom';
    row.innerHTML = `<input class="scan-path-val" type="text" placeholder="/path/to/scan/dir" value="">` +
        `<span class="scan-path-del" onclick="this.parentElement.remove()">✕</span>`;
    container.appendChild(row);
    row.querySelector('input').focus();
};

window.getCustomScanPaths = function() {
    const inputs = document.querySelectorAll('#custom-scan-paths .scan-path-row.custom input');
    return Array.from(inputs).map(i => i.value.trim()).filter(Boolean);
};

// Components
window.toggleSearchAdvanced = components.toggleSearchAdvanced;
window.doSearch = components.doSearch;
window.openCreateModal = components.openCreateModal;
window.closeCreateModal = components.closeCreateModal;
window.switchCreateMode = components.switchCreateMode;
window.toggleCreateQuickMode = components.toggleCreateQuickMode;
window.addGroupChild = components.addGroupChild;
window.removeGroupChild = components.removeGroupChild;
window.doParse = components.doParse;
window.doParseURL = components.doParseURL;
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
window.addWebhookRow = components.addWebhookRow;
window.removeWebhookRow = components.removeWebhookRow;
window.testWebhook = components.testWebhook;
window.saveWebhookConfig = components.saveWebhookConfig;
window.openImportModal = components.openImportModal;
window.closeImportModal = components.closeImportModal;
window.handleImportFile = components.handleImportFile;
window.doImport = components.doImport;
window.openExportModal = components.openExportModal;
window.closeExportModal = components.closeExportModal;
window.doExport = components.doExport;

// Key management exports
window.loadKeyManagement = pages.loadKeyManagement;
window.approveUser = pages.approveUser;
window.rejectUser = pages.rejectUser;
window.createUserAdmin = pages.createUserAdmin;
window.updateUserRole = pages.updateUserRole;
window.toggleUserActive = pages.toggleUserActive;
window.openAdminCreateUser = function() {
    document.getElementById('admin-create-user-form').style.display = 'block';
};

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    document.getElementById('login-key').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doLogin();
    });
    document.getElementById('login-password').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doLogin();
    });
    document.getElementById('reg-password2').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doRegister();
    });

    // Drag and drop for import
    const zone = document.getElementById('import-drop-zone');
    if (zone) {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                const input = document.getElementById('import-file');
                input.files = e.dataTransfer.files;
                components.handleImportFile(input);
            }
        });
    }
});
