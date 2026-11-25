/**
 * Typed HTTP client for TeamMemory FastAPI backend (/api/v1/*).
 * Automatically attaches the active tenant via `X-Tenant-Id` header.
 */

const TENANT_STORAGE_KEY = "tm_active_tenant_id";

export function getStoredTenantId(): string | null {
  try {
    return localStorage.getItem(TENANT_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setStoredTenantId(id: string | null): void {
  try {
    if (id) localStorage.setItem(TENANT_STORAGE_KEY, id);
    else localStorage.removeItem(TENANT_STORAGE_KEY);
  } catch {
    // ignore storage errors
  }
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit & { skipTenantHeader?: boolean } = {},
): Promise<T> {
  const { skipTenantHeader, headers: rawHeaders, ...rest } = options;
  const headers = new Headers(rawHeaders);

  if (!headers.has("Content-Type") && rest.body && !(rest.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  if (!skipTenantHeader) {
    const tid = getStoredTenantId();
    if (tid) headers.set("X-Tenant-Id", tid);
  }

  const res = await fetch(path, {
    credentials: "include",
    ...rest,
    headers,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail =
        body?.detail ||
        body?.message ||
        body?.error?.message ||
        (typeof body === "string" ? body : detail);
    } catch {
      // keep default
    }
    throw new ApiError(res.status, detail);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}

// ============================================================
// Types
// ============================================================

export interface TenantSummary {
  id: string;
  slug: string;
  name: string;
  role: "admin" | "member" | "viewer" | string;
  settings_json?: Record<string, unknown>;
  created_at?: string | null;
}

export interface AuthMeResponse {
  user: string;
  role: string;
  tenant_id: string;
  permissions: string[];
  api_key_masked?: string | null;
  current_tenant?: TenantSummary | null;
  tenants: TenantSummary[];
}

export interface ExperienceItem {
  id: string;
  parent_id?: string | null;
  title: string;
  description: string;
  solution?: string | null;
  tags: string[];
  experience_type?: string;
  source?: string;
  created_by: string;
  visibility: string;
  status?: string;
  exp_status?: string;
  project: string;
  quality_score?: number;
  quality_tier?: string;
  is_pinned?: boolean;
  is_deleted?: boolean;
  recall_count?: number;
  used_count?: number;
  negative_feedback_count?: number;
  valid_from?: string | null;
  valid_until?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  // Search debug fields
  similarity?: number;
  score?: number;
  rrf_score?: number;
  fts_rank?: number;
  vector_score?: number;
  fts_score?: number;
  tag_score?: number;
  graph_score?: number;
  rerank_score?: number;
  source_layer?: string;
}

export interface ArchiveItem {
  id: string;
  title: string;
  scope?: string;
  scope_ref?: string | null;
  overview?: string | null;
  solution_doc?: string;
  conversation_summary?: string | null;
  value_summary?: string | null;
  project: string;
  created_by: string;
  status?: string;
  tags?: string[];
  attachments?: Array<{ id: string; kind: string; path?: string; source_path?: string }>;
  linked_experiences?: Array<{ id: string; title: string }>;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface PersonalMemoryItem {
  id: string;
  user_id: string;
  content: string;
  scope: string;
  profile_kind: "static" | "dynamic" | string;
  persona?: string;
  priority?: number;
  context_hint?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface TenantMemberItem {
  id: string;
  tenant_id: string;
  user_id: string;
  role: "admin" | "member" | "viewer";
  active_key_count?: number;
  created_at?: string | null;
}

export interface TenantKeyItem {
  id: number;
  label: string;
  role: string;
  masked: string;
  tenant_id: string;
  created_at?: string | null;
}

// ============================================================
// API Endpoints
// ============================================================

export const api = {
  // Auth
  me: () => apiFetch<AuthMeResponse>("/api/v1/auth/me"),
  login: (body: { username?: string; password?: string; api_key?: string }) =>
    apiFetch<{ success: boolean; user: string; role: string; message: string }>(
      "/api/v1/auth/login",
      { method: "POST", body: JSON.stringify(body), skipTenantHeader: true },
    ),
  register: (body: { username: string; password: string }) =>
    apiFetch<{ message: string }>("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify(body),
      skipTenantHeader: true,
    }),
  logout: () => apiFetch("/api/v1/auth/logout", { method: "POST" }),
  health: () => apiFetch<{ status: string; version?: string; checks?: Record<string, unknown> }>("/health"),

  // Tenants
  listTenants: () =>
    apiFetch<{ tenants: TenantSummary[]; current_tenant_id: string }>("/api/v1/tenants"),
  createTenant: (body: { name: string; slug?: string }) =>
    apiFetch<{ tenant: TenantSummary }>("/api/v1/tenants", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateCurrentTenant: (body: { name?: string; settings_json?: Record<string, unknown> }) =>
    apiFetch<{ tenant: TenantSummary }>("/api/v1/tenants/current", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  listMembers: () => apiFetch<{ members: TenantMemberItem[] }>("/api/v1/tenants/current/members"),
  addMember: (body: {
    user_id: string;
    role: string;
    password?: string;
    generate_api_key?: boolean;
  }) =>
    apiFetch<{ member: TenantMemberItem; api_key?: string }>("/api/v1/tenants/current/members", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateMember: (userId: string, role: string) =>
    apiFetch<{ member: TenantMemberItem }>(
      `/api/v1/tenants/current/members/${encodeURIComponent(userId)}`,
      { method: "PUT", body: JSON.stringify({ role }) },
    ),
  removeMember: (userId: string) =>
    apiFetch<{ message: string }>(
      `/api/v1/tenants/current/members/${encodeURIComponent(userId)}`,
      { method: "DELETE" },
    ),
  listTenantKeys: () => apiFetch<{ keys: TenantKeyItem[] }>("/api/v1/tenants/current/keys"),
  createTenantKey: (body: { label?: string; role?: string }) =>
    apiFetch<{ key: TenantKeyItem; api_key: string }>("/api/v1/tenants/current/keys", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  revokeTenantKey: (keyId: number) =>
    apiFetch<{ message: string }>(`/api/v1/tenants/current/keys/${keyId}`, {
      method: "DELETE",
    }),
  rotateOwnKey: () =>
    apiFetch<{ success: boolean; api_key: string }>("/api/v1/auth/keys/rotate", {
      method: "POST",
    }),
  adminResetPassword: (username: string, new_password: string) =>
    apiFetch<{ message: string }>("/api/v1/auth/admin/reset-password", {
      method: "POST",
      body: JSON.stringify({ username, new_password }),
    }),

  // Projects & Experiences
  listProjects: () => apiFetch<{ projects: string[] }>("/api/v1/projects"),
  listExperiences: (params: {
    limit?: number;
    offset?: number;
    project?: string;
    status?: string;
    tag?: string;
  }) => {
    const qs = new URLSearchParams();
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.offset) qs.set("offset", String(params.offset));
    if (params.project) qs.set("project", params.project);
    if (params.status) qs.set("status", params.status);
    if (params.tag) qs.set("tag", params.tag);
    return apiFetch<{ items: ExperienceItem[]; total: number }>(
      `/api/v1/experiences?${qs.toString()}`,
    );
  },
  getExperience: (id: string) => apiFetch<ExperienceItem>(`/api/v1/experiences/${id}`),
  createExperience: (body: {
    title: string;
    problem: string;
    solution?: string;
    tags?: string[];
    project?: string;
    status?: string;
    visibility?: string;
  }) =>
    apiFetch<{ item: ExperienceItem; message: string }>("/api/v1/experiences", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateExperience: (
    id: string,
    body: {
      title?: string;
      problem?: string;
      solution?: string;
      tags?: string[];
      exp_status?: string;
      visibility?: string;
    },
  ) =>
    apiFetch<ExperienceItem>(`/api/v1/experiences/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  deleteExperience: (id: string, hard = false) =>
    apiFetch<{ message: string }>(`/api/v1/experiences/${id}${hard ? "?hard=true" : ""}`, {
      method: "DELETE",
    }),
  restoreExperience: (id: string) =>
    apiFetch<{ message: string }>(`/api/v1/experiences/${id}/restore`, { method: "POST" }),
  togglePinExperience: (id: string) =>
    apiFetch<{ is_pinned: boolean }>(`/api/v1/experiences/${id}/pin`, { method: "PATCH" }),
  changeExperienceStatus: (id: string, body: { status?: string; visibility?: string }) =>
    apiFetch<{ message: string; experience: ExperienceItem }>(
      `/api/v1/experiences/${id}/status`,
      { method: "PATCH", body: JSON.stringify(body) },
    ),
  supersedeExperience: (id: string) =>
    apiFetch<{ message: string; experience: ExperienceItem }>(
      `/api/v1/experiences/${id}/supersede`,
      { method: "POST" },
    ),

  // Search (normal + debug)
  search: (body: { query: string; project?: string; max_results?: number; include_archives?: boolean }) =>
    apiFetch<{ results: ExperienceItem[]; total: number; reranked?: boolean; cached?: boolean }>(
      "/api/v1/search",
      { method: "POST", body: JSON.stringify(body) },
    ),
  searchDebug: (body: { query: string; project?: string; max_results?: number }) =>
    apiFetch<{
      results: ExperienceItem[];
      total: number;
      duration_ms?: number;
      search_type?: string;
      stage_metrics?: Record<string, unknown>;
    }>("/api/v1/search/debug", { method: "POST", body: JSON.stringify(body) }),

  // Archives
  listArchives: (params: { project?: string; q?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    if (params.project) qs.set("project", params.project);
    if (params.q) qs.set("q", params.q);
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.offset) qs.set("offset", String(params.offset));
    return apiFetch<{ items: ArchiveItem[]; total: number }>(`/api/v1/archives?${qs.toString()}`);
  },
  getArchive: (id: string, project?: string) => {
    const qs = project ? `?project=${encodeURIComponent(project)}` : "";
    return apiFetch<ArchiveItem>(`/api/v1/archives/${id}${qs}`);
  },

  // Personal Memory
  listPersonalMemory: (params?: { profile_kind?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.profile_kind) qs.set("profile_kind", params.profile_kind);
    if (params?.limit) qs.set("limit", String(params.limit));
    return apiFetch<{ items: PersonalMemoryItem[]; total: number }>(
      `/api/v1/personal-memory/list?${qs.toString()}`,
    );
  },
  createPersonalMemory: (body: {
    content: string;
    profile_kind: "static" | "dynamic";
    scope?: string;
    context_hint?: string;
  }) =>
    apiFetch<{ item: PersonalMemoryItem }>("/api/v1/personal-memory", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updatePersonalMemory: (
    id: string,
    body: { content?: string; profile_kind?: string; context_hint?: string },
  ) =>
    apiFetch<PersonalMemoryItem>(`/api/v1/personal-memory/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  deletePersonalMemory: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/v1/personal-memory/${id}`, { method: "DELETE" }),

  // Observability & Ops
  getObservabilityOverview: (days = 30) =>
    apiFetch<Record<string, unknown>>(`/api/v1/observability/overview?days=${days}`),
  runGoldenEval: (limit?: number) =>
    apiFetch<{ success: boolean; eval: Record<string, unknown> }>(
      `/api/v1/observability/eval/run${limit ? `?limit=${limit}` : ""}`,
      { method: "POST" },
    ),
  getJanitorStatus: () => apiFetch<Record<string, unknown>>("/api/v1/janitor/status"),
  runJanitor: (project?: string) =>
    apiFetch<Record<string, unknown>>(
      `/api/v1/janitor/run${project ? `?project=${encodeURIComponent(project)}` : ""}`,
      { method: "POST" },
    ),
  getRetrievalConfig: () => apiFetch<Record<string, unknown>>("/api/v1/config/retrieval"),
  putRetrievalConfig: (body: Record<string, unknown>) =>
    apiFetch<Record<string, unknown>>("/api/v1/config/retrieval", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  getSearchConfig: () => apiFetch<Record<string, unknown>>("/api/v1/config/search"),
  putSearchConfig: (body: Record<string, unknown>) =>
    apiFetch<Record<string, unknown>>("/api/v1/config/search", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
};
