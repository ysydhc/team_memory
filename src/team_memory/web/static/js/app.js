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
    if (state.apiKey) opts.headers['Authorization'] = `Bearer ${state.apiKey}`;
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(path, opts);
    if (res.status === 401) {
        doLogout();
        throw new Error('Session expired');
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    return data;
}

// ===== Auth =====
async function doLogin() {
    const key = document.getElementById('login-key').value.trim();
    if (!key) return;

    try {
        const data = await api('POST', '/api/v1/auth/login', { api_key: key });
        if (data.success) {
            state.apiKey = key;
            state.currentUser = { name: data.user, role: data.role };
            showApp();
        } else {
            const err = document.getElementById('login-error');
            err.textContent = data.message || 'API Key 无效';
            err.style.display = 'block';
        }
    } catch (e) {
        const err = document.getElementById('login-error');
        err.textContent = '连接服务器失败';
        err.style.display = 'block';
    }
}

function doLogout() {
    state.apiKey = '';
    api('POST', '/api/v1/auth/logout').catch(() => {});
    document.getElementById('login-screen').style.display = 'flex';
    document.getElementById('app-screen').style.display = 'none';
    document.getElementById('login-key').value = '';
    document.getElementById('login-error').style.display = 'none';
}

async function checkAuth() {
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
            applyProjectPlaceholders();
        })
        .catch(() => {
            state.defaultProject = 'default';
            applyProjectPlaceholders();
        });
    api('GET', '/api/v1/templates')
        .then((r) => {
            state.cachedTemplates = r.templates || [];
            populateCreateTypeSelector();
        })
        .catch(() => { state.cachedTemplates = []; });
    loadSchemaAndPopulateFilters(api);
    navigate('dashboard');
}

// ===== Navigation =====
function navigate(page) {
    state.currentPage = page;
    document.querySelectorAll('.page').forEach((p) => p.classList.add('hidden'));
    document.getElementById(`page-${page}`).classList.remove('hidden');
    document.querySelectorAll('.topbar-nav a').forEach((a) => {
        a.classList.toggle('active', a.dataset.page === page);
    });

    if (page === 'dashboard') pages.loadDashboard();
    else if (page === 'list') pages.loadList();
    else if (page === 'drafts') pages.loadDrafts();
    else if (page === 'reviews') pages.loadReviews();
    else if (page === 'dedup') {}
    else if (page === 'settings') {
        pages.loadRetrievalConfig();
        components.loadWebhookConfig();
        pages.loadCurrentSchema();
    }
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

// Nav & pages
window.navigate = navigate;
window.loadList = pages.loadList;
window.filterByTag = pages.filterByTag;
window.showDetail = pages.showDetail;
window.viewDetail = pages.viewDetail;
window.loadDashboard = pages.loadDashboard;
window.loadDrafts = pages.loadDrafts;
window.publishDraft = pages.publishDraft;
window.loadReviews = pages.loadReviews;
window.reviewExperience = pages.reviewExperience;
window.loadDuplicates = pages.loadDuplicates;
window.doMerge = pages.doMerge;
window.scanStale = pages.scanStale;
window.toggleVersionHistory = pages.toggleVersionHistory;
window.viewVersionSnapshot = pages.viewVersionSnapshot;
window.toggleVersionSnapshot = pages.toggleVersionSnapshot;
window.rollbackVersion = pages.rollbackVersion;
window.loadInstallables = pages.loadInstallables;
window.previewInstallable = pages.previewInstallable;
window.installInstallable = pages.installInstallable;
window.loadAllConfig = pages.loadAllConfig;
window.saveRetrievalConfig = pages.saveRetrievalConfig;
window.saveDefaultProjectConfig = pages.saveDefaultProjectConfig;
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

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    document.getElementById('login-key').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doLogin();
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
