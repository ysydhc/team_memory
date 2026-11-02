// Chart labels for RefinementLifecycle, for statechart.mjs.
export const readerActions = [];

export function state(v) {
	if (v.worker_step === "step1_draft_saved") {
		return `Retrying from queue (${v.pg_drafts} draft saved, ${v.pg_published} published)`;
	}
	if (v.worker_step === "step2_draft_saved") {
		return `Refining session (${v.pg_drafts} draft saved, waiting for LLM)`;
	}
	if (v.worker_step === "step2_publishing") {
		return `Publishing refined output (${v.pg_drafts} draft saved)`;
	}
	if (v.session_status === "published") {
		if (v.quality_rejected) {
			return `Session closed after quality gate rejected draft (${v.pg_published} published)`;
		}
		if (v.queue_status === "pending" || v.queue_status === "backoff") {
			return `Session published (${v.pg_published} published), retry still in queue`;
		}
		return `Session published (${v.pg_published} published)`;
	}
	if (v.session_status === "needs_refinement") {
		if (v.reimported_while_publishing) {
			return "Waiting for refinement (re-queued by importer while publishing)";
		}
		if (v.queue_status === "pending") {
			return "Waiting for refinement (retry ready in queue)";
		}
		if (v.queue_status === "backoff") {
			return "Waiting for refinement (retry backing off)";
		}
		if (v.refined_payload) {
			return "Waiting for refinement (saved LLM output ready to replay)";
		}
		return "Waiting for refinement";
	}
	if (v.session_status === "refining") {
		if (v.queue_status === "pending") {
			return "Marked refining (retry ready in queue)";
		}
		if (v.queue_status === "backoff") {
			return "Marked refining (retry backing off)";
		}
		return `Stranded in refining (no retry in queue, ${v.pg_drafts} orphan draft)`;
	}
	if (v.session_status === "publishing") {
		if (v.queue_status === "pending") {
			return "Marked publishing (retry ready in queue)";
		}
		if (v.queue_status === "backoff") {
			return "Marked publishing (retry backing off)";
		}
		return "Marked publishing";
	}
	return "Waiting for refinement";
}

export function group(v) {
	if (v.session_status === "published" && !v.quality_rejected && v.pg_published === 1 && v.queue_status === "none") {
		return "Published cleanly";
	}
	if (v.worker_step !== "idle") {
		return "Worker in flight";
	}
	return "Pending or recovering";
}

const STEPS = {
	BackoffTimerDue: "the retry backoff timer expires",
	DaemonRestartRecovery: "the daemon restarts and resets in-flight sessions",
	ReimportPublishingSession: "an importer re-runs and resets the publishing session to waiting",
	Step1RetryDeferredToBackoff: "the queue retry hits an error and backs off",
	Step1RetryPublishSucceed: "the queue retry finishes LLM extraction and publishes",
	Step1RetrySaveDraft: "the worker starts a queued retry and saves a draft",
	Step2LLMFailEnqueue: "the LLM call fails and the worker queues a retry",
	Step2LLMSucceedMarkPublishing: "the LLM call succeeds and the worker checkpoints the output",
	Step2LLMUnavailableSkip: "the LLM health check fails and the worker skips the session",
	Step2MarkRefiningAndSaveDraft: "the worker marks the session refining and saves a draft",
	Step2PublishFailEnqueue: "publishing fails and the worker queues a retry",
	Step2PublishSucceed: "the quality gate passes and the worker publishes the experience",
	Step2QualityGateDiscard: "the quality gate rejects the draft but calls publish",
	Step2SkipLLMReplayPublish: "the worker replays the saved LLM output without calling the LLM",
};

export function step(action) {
	return STEPS[action] ?? action;
}

export function cause(v, _facts, invariant) {
	if (invariant === "NoStrandedRefiningSession") {
		return "when the LLM endpoint is unreachable, the worker catches the error after marking the session refining and saving an unrefined draft, and exits without resetting the session or queueing a retry";
	}
	if (invariant === "RejectedDraftNeverPublished") {
		return "when the quality gate rejects a draft on structure or score, the rejection branch still calls publish on the saved draft";
	}
	if (invariant === "AtMostOnePublishedExperience") {
		return "after a failed refinement queues a retry and the session is reset to waiting by a restart or importer re-run while its retry stays in the queue, both the normal loop and the retry queue publish it";
	}
	if (invariant === "ImporterPreservesPublishingSession") {
		return "the importer skip guard checks only published, waiting, and refining statuses, so re-running the importer while a session is marked publishing overwrites its status back to waiting";
	}
	return null;
}

export function detail(v) {
	return `status=${v.session_status}, step=${v.worker_step}, queue=${v.queue_status}(${v.queue_count}), drafts=${v.pg_drafts}, pub=${v.pg_published}, q_rej=${v.quality_rejected}, reimp=${v.reimported_while_publishing}`;
}
