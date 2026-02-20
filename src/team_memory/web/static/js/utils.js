/**
 * Shared utility functions for team_memory UI (P4-4 modularization)
 */

/** Escape HTML entities to prevent XSS */
export function esc(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** Format a date string for display */
export function formatDate(dateStr) {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleDateString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
    });
}

/** Human-readable relative time (e.g. "5 分钟前") */
export function timeAgo(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前';
    if (diff < 604800) return Math.floor(diff / 86400) + ' 天前';
    return d.toLocaleDateString('zh-CN');
}

/** Truncate a string to maxLen with ellipsis */
export function truncate(str, maxLen = 100) {
    if (!str || str.length <= maxLen) return str;
    return str.slice(0, maxLen) + '...';
}

/** Show a toast notification */
export function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    el.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 10000;
        padding: 12px 24px; border-radius: 8px; color: white;
        font-size: 14px; max-width: 400px; opacity: 0;
        transition: opacity 0.3s;
        background: ${type === 'error' ? '#e53e3e' : type === 'success' ? '#38a169' : '#3182ce'};
    `;
    document.body.appendChild(el);
    requestAnimationFrame(() => el.style.opacity = '1');
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

/** Dark mode toggle */
export function initDarkMode() {
    const saved = localStorage.getItem('darkMode');
    if (saved === 'true' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    }
}

export function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark);
    return isDark;
}

/** Search history management */
const MAX_SEARCH_HISTORY = 20;

export function getSearchHistory() {
    try {
        return JSON.parse(localStorage.getItem('searchHistory') || '[]');
    } catch { return []; }
}

export function addSearchHistory(query) {
    if (!query?.trim()) return;
    let history = getSearchHistory();
    history = history.filter(h => h !== query);
    history.unshift(query);
    if (history.length > MAX_SEARCH_HISTORY) history.pop();
    localStorage.setItem('searchHistory', JSON.stringify(history));
}

export function clearSearchHistory() {
    localStorage.removeItem('searchHistory');
}
