/**
 * API client module for team_memory (P4-4 modularization)
 *
 * Centralizes all API calls with auth handling.
 */

const API_BASE = '/api/v1';

function getHeaders() {
    const h = { 'Content-Type': 'application/json' };
    const key = localStorage.getItem('api_key');
    if (key) h['Authorization'] = `Bearer ${key}`;
    return h;
}

export async function apiGet(path) {
    const resp = await fetch(`${API_BASE}${path}`, { headers: getHeaders() });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
}

export async function apiPost(path, body = {}) {
    const resp = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(body),
    });
    if (!resp.ok && resp.status !== 409) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return { status: resp.status, data: await resp.json() };
}

export async function apiPut(path, body = {}) {
    const resp = await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: getHeaders(),
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
}

export async function apiDelete(path) {
    const resp = await fetch(`${API_BASE}${path}`, {
        method: 'DELETE',
        headers: getHeaders(),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
}

export async function apiLogin(apiKey) {
    const resp = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey }),
    });
    return resp.json();
}
