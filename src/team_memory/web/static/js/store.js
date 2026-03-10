/**
 * Shared state and API for team_memory SPA.
 * Provides mutable state and api() used by other modules.
 */

export const state = {
    apiKey: '',
    currentUser: { name: 'anonymous', role: 'member' },
    currentPage: 'dashboard',
    listPage: 1,
    allTags: {},
    selectedTag: null,
    feedbackRating: null,
    feedbackExpId: null,
    groupChildrenIds: [],
    editChildrenIds: [],
    cachedTemplates: [],
    cachedSchema: null,
    editOriginalExp: null,
    defaultProject: 'default',
    activeProject: 'default',
    cachedRetrievalConfig: null,
    cachedInstallables: [],
    webhookRows: [],
    createMode: 'manual',
    importFile: null,
    generatedSchemaData: null,
    detailReferrer: 'list',
    /** Stack of experience ids when navigating via related links; back button pops and shows that detail. */
    detailBackStack: [],
    /** Scroll position when returning to settings from personal-memory/user-expansion/dedup. */
    settingsScrollTop: 0,
    availableProjects: [],
    /** T8: Pre-filled node_key when opening create modal from architecture page. */
    architectureMountNode: null,
    /** Selected clusters for graph (null = 全部). */
    architectureCurrentClusters: null,
    /** Default clusters from config (project entry points). */
    architectureDefaultClusters: [],
    /** Node filter pattern: hide nodes matching (prefix). Empty = show all. */
    architectureFilterPattern: '',
    /** Back stack for architecture view: [{ clusters, filePath, nodeId, path }]. Pop to restore. */
    architectureBackStack: [],
    /** Current file_path used for graph (null when viewing by cluster/full). */
    architectureCurrentFilePath: null,
    /** When true, main sidebar stays open (won't close on click-outside). */
    architectureSidebarPinned: false,
    /** When true, we've already shown "provider not configured" toast this session; avoid repeat. */
    architectureProviderUnavailableShown: false,
};

export const defaultTypeIcons = {
    general: '📝', feature: '🚀', bugfix: '🐛', tech_design: '📐',
    incident: '🔥', best_practice: '✨', learning: '📚', data_quality: '📊',
    pipeline_failure: '🔧', schema_change: '🗂️', deployment: '🚢',
    capacity_planning: '📈', runbook: '📋', postmortem: '📄', note: '📝',
    decision: '⚖️', action_item: '✅',
};
