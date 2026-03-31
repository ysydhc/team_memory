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
    cachedSchema: null,
    editOriginalExp: null,
    defaultProject: 'default',
    activeProject: 'default',
    cachedRetrievalConfig: null,
    cachedInstallables: [],
    createMode: 'manual',
    detailReferrer: 'list',
    /** Stack of experience ids when navigating via related links; back button pops and shows that detail. */
    detailBackStack: [],
    /** Scroll position when returning to settings from personal-memory/user-expansion/dedup. */
    settingsScrollTop: 0,
    availableProjects: [],
    /** Archives browse page (API `/api/v1/archives`). */
    archivesListPage: 1,
};

export const defaultTypeIcons = {
    general: '📝', data_quality: '📊', pipeline_failure: '🔧',
    schema_change: '🗂️', deployment: '🚢', capacity_planning: '📈',
    runbook: '📋', postmortem: '📄', note: '📝', decision: '⚖️',
    action_item: '✅',
};
