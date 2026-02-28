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
    availableProjects: [],
};

export const defaultTypeIcons = {
    general: '📝', feature: '🚀', bugfix: '🐛', tech_design: '📐',
    incident: '🔥', best_practice: '✨', learning: '📚', data_quality: '📊',
    pipeline_failure: '🔧', schema_change: '🗂️', deployment: '🚢',
    capacity_planning: '📈', runbook: '📋', postmortem: '📄', note: '📝',
    decision: '⚖️', action_item: '✅',
};
