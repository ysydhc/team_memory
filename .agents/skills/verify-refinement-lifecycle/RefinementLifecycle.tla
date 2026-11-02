---- MODULE RefinementLifecycle ----
EXTENDS Naturals

VARIABLES
  session_status,
  refined_payload,
  queue_status,
  queue_count,
  worker_step,
  pg_drafts,
  pg_published,
  quality_rejected,
  reimported_while_publishing

vars == <<session_status, refined_payload, queue_status, queue_count, worker_step, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

Init ==
  /\ session_status = "needs_refinement"  \* src/team_memory/daemon/buffers/conversation_buffer.py:500
  /\ refined_payload = FALSE              \* src/team_memory/daemon/buffers/conversation_buffer.py:559
  /\ queue_status = "none"                \* src/team_memory/daemon/buffers/pending_queue.py:23
  /\ queue_count = 0                      \* src/team_memory/daemon/buffers/pending_queue.py:58
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/refinement_worker.py:377
  /\ pg_drafts = 0                        \* src/team_memory/daemon/refinement/refinement_worker.py:584
  /\ pg_published = 0                     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ quality_rejected = FALSE             \* src/team_memory/daemon/refinement/refinement_worker.py:657
  /\ reimported_while_publishing = FALSE  \* src/team_memory/daemon/refinement/chatgpt_importer.py:354

\* Step 1: Process due PendingQueue retry ops first (refinement_worker.py:391-428)
Step1RetrySaveDraft ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/refinement_worker.py:391
  /\ queue_status = "pending"             \* src/team_memory/daemon/buffers/pending_queue.py:88
  /\ session_status /= "published"        \* src/team_memory/daemon/refinement/refinement_worker.py:406
  /\ pg_drafts < 3                        \* src/team_memory/daemon/refinement/refinement_worker.py:584
  /\ worker_step' = "step1_draft_saved"   \* src/team_memory/daemon/refinement/refinement_worker.py:410
  /\ pg_drafts' = pg_drafts + 1           \* src/team_memory/daemon/refinement/refinement_worker.py:584
  /\ UNCHANGED <<session_status, refined_payload, queue_status, queue_count, pg_published, quality_rejected, reimported_while_publishing>>

Step1RetryPublishSucceed ==
  /\ worker_step = "step1_draft_saved"    \* src/team_memory/daemon/refinement/refinement_worker.py:643
  /\ pg_published < 2                     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ refined_payload' = TRUE              \* src/team_memory/daemon/refinement/refinement_worker.py:646
  /\ pg_published' = pg_published + 1     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ session_status' = "published"        \* src/team_memory/daemon/refinement/refinement_worker.py:747
  /\ queue_status' = "none"               \* src/team_memory/daemon/refinement/refinement_worker.py:411
  /\ queue_count' = 0                     \* src/team_memory/daemon/buffers/pending_queue.py:114
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:412
  /\ UNCHANGED <<pg_drafts, quality_rejected, reimported_while_publishing>>

Step1RetryDeferredToBackoff ==
  /\ worker_step = "step1_draft_saved"    \* src/team_memory/daemon/refinement/refinement_worker.py:418
  /\ queue_status = "pending"             \* src/team_memory/daemon/buffers/pending_queue.py:131
  /\ queue_status' = "backoff"            \* src/team_memory/daemon/refinement/refinement_worker.py:428
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:442
  /\ UNCHANGED <<session_status, refined_payload, queue_count, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

BackoffTimerDue ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/refinement_worker.py:392
  /\ queue_status = "backoff"             \* src/team_memory/daemon/buffers/pending_queue.py:88
  /\ queue_status' = "pending"            \* src/team_memory/daemon/buffers/pending_queue.py:88
  /\ UNCHANGED <<session_status, refined_payload, queue_count, worker_step, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

\* Step 2: Process needs_refinement sessions (refinement_worker.py:442-519)
Step2SkipLLMReplayPublish ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/refinement_worker.py:442
  /\ queue_status /= "pending"            \* src/team_memory/daemon/refinement/refinement_worker.py:392
  /\ session_status = "needs_refinement"  \* src/team_memory/daemon/buffers/conversation_buffer.py:549
  /\ refined_payload = TRUE               \* src/team_memory/daemon/refinement/refinement_worker.py:471
  /\ pg_published < 2                     \* src/team_memory/daemon/refinement/refinement_worker.py:474
  /\ pg_published' = pg_published + 1     \* src/team_memory/daemon/refinement/refinement_worker.py:474
  /\ session_status' = "published"        \* src/team_memory/daemon/refinement/refinement_worker.py:483
  /\ queue_status' = "none"               \* src/team_memory/daemon/refinement/refinement_worker.py:485
  /\ queue_count' = 0                     \* src/team_memory/daemon/buffers/pending_queue.py:119
  /\ UNCHANGED <<refined_payload, worker_step, pg_drafts, quality_rejected, reimported_while_publishing>>

Step2MarkRefiningAndSaveDraft ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/refinement_worker.py:442
  /\ queue_status /= "pending"            \* src/team_memory/daemon/refinement/refinement_worker.py:392
  /\ session_status = "needs_refinement"  \* src/team_memory/daemon/buffers/conversation_buffer.py:549
  /\ refined_payload = FALSE              \* src/team_memory/daemon/refinement/refinement_worker.py:471
  /\ pg_drafts < 2                        \* src/team_memory/daemon/refinement/refinement_worker.py:584
  /\ session_status' = "refining"         \* src/team_memory/daemon/refinement/refinement_worker.py:489
  /\ pg_drafts' = pg_drafts + 1           \* src/team_memory/daemon/refinement/refinement_worker.py:584
  /\ worker_step' = "step2_draft_saved"   \* src/team_memory/daemon/refinement/refinement_worker.py:591
  /\ UNCHANGED <<refined_payload, queue_status, queue_count, pg_published, quality_rejected, reimported_while_publishing>>

Step2LLMUnavailableSkip ==
  /\ worker_step = "step2_draft_saved"    \* src/team_memory/daemon/refinement/refinement_worker.py:887
  /\ session_status' = "needs_refinement" \* src/team_memory/daemon/refinement/refinement_worker.py:494
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:493
  /\ UNCHANGED <<refined_payload, queue_status, queue_count, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

Step2LLMFailEnqueue ==
  /\ worker_step = "step2_draft_saved"    \* src/team_memory/daemon/refinement/refinement_worker.py:499
  /\ queue_status' = IF queue_count = 0 THEN "pending" ELSE queue_status \* src/team_memory/daemon/buffers/pending_queue.py:58
  /\ queue_count' = IF queue_count = 0 THEN 1 ELSE queue_count           \* src/team_memory/daemon/buffers/pending_queue.py:61
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:506
  /\ UNCHANGED <<session_status, refined_payload, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

Step2LLMSucceedMarkPublishing ==
  /\ worker_step = "step2_draft_saved"    \* src/team_memory/daemon/refinement/refinement_worker.py:643
  /\ session_status' = "publishing"       \* src/team_memory/daemon/refinement/refinement_worker.py:646
  /\ refined_payload' = TRUE              \* src/team_memory/daemon/buffers/conversation_buffer.py:248
  /\ worker_step' = "step2_publishing"    \* src/team_memory/daemon/refinement/refinement_worker.py:655
  /\ UNCHANGED <<queue_status, queue_count, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

Step2QualityGateDiscard ==
  /\ worker_step = "step2_publishing"     \* src/team_memory/daemon/refinement/refinement_worker.py:656
  /\ quality_rejected' = TRUE             \* src/team_memory/daemon/refinement/refinement_worker.py:658
  /\ session_status' = "published"        \* src/team_memory/daemon/refinement/refinement_worker.py:671
  /\ queue_status' = "none"               \* src/team_memory/daemon/refinement/refinement_worker.py:673
  /\ queue_count' = 0                     \* src/team_memory/daemon/buffers/pending_queue.py:119
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:674
  /\ UNCHANGED <<refined_payload, pg_drafts, pg_published, reimported_while_publishing>>

Step2PublishSucceed ==
  /\ worker_step = "step2_publishing"     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ pg_published < 2                     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ pg_published' = pg_published + 1     \* src/team_memory/daemon/refinement/refinement_worker.py:710
  /\ session_status' = "published"        \* src/team_memory/daemon/refinement/refinement_worker.py:747
  /\ queue_status' = "none"               \* src/team_memory/daemon/refinement/refinement_worker.py:749
  /\ queue_count' = 0                     \* src/team_memory/daemon/buffers/pending_queue.py:119
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:759
  /\ UNCHANGED <<refined_payload, pg_drafts, quality_rejected, reimported_while_publishing>>

Step2PublishFailEnqueue ==
  /\ worker_step = "step2_publishing"     \* src/team_memory/daemon/refinement/refinement_worker.py:719
  /\ queue_status' = IF queue_count = 0 THEN "pending" ELSE queue_status \* src/team_memory/daemon/buffers/pending_queue.py:58
  /\ queue_count' = IF queue_count = 0 THEN 1 ELSE queue_count           \* src/team_memory/daemon/buffers/pending_queue.py:61
  /\ worker_step' = "idle"                \* src/team_memory/daemon/refinement/refinement_worker.py:506
  /\ UNCHANGED <<session_status, refined_payload, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

\* Startup crash recovery (app.py:460-476 -> conversation_buffer.py:224, 260)
DaemonRestartRecovery ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/runtime/app.py:460
  /\ session_status \in {"refining", "publishing"} \* src/team_memory/daemon/buffers/conversation_buffer.py:224
  /\ session_status' = "needs_refinement" \* src/team_memory/daemon/buffers/conversation_buffer.py:260
  /\ UNCHANGED <<refined_payload, queue_status, queue_count, worker_step, pg_drafts, pg_published, quality_rejected, reimported_while_publishing>>

\* Importer re-scan while a session is in 'publishing' state (chatgpt_importer.py:354, agy_transcript_watcher.py:184)
ReimportPublishingSession ==
  /\ worker_step = "idle"                 \* src/team_memory/daemon/refinement/chatgpt_importer.py:353
  /\ session_status \notin {"published", "needs_refinement", "refining", "publishing"} \* src/team_memory/daemon/refinement/chatgpt_importer.py:354
  /\ session_status' = "needs_refinement" \* src/team_memory/daemon/refinement/chatgpt_importer.py:377
  /\ reimported_while_publishing' = TRUE  \* src/team_memory/daemon/observe/agy_transcript_watcher.py:218
  /\ UNCHANGED <<refined_payload, queue_status, queue_count, worker_step, pg_drafts, pg_published, quality_rejected>>

Next ==
  \/ Step1RetrySaveDraft
  \/ Step1RetryPublishSucceed
  \/ Step1RetryDeferredToBackoff
  \/ BackoffTimerDue
  \/ Step2SkipLLMReplayPublish
  \/ Step2MarkRefiningAndSaveDraft
  \/ Step2LLMUnavailableSkip
  \/ Step2LLMFailEnqueue
  \/ Step2LLMSucceedMarkPublishing
  \/ Step2QualityGateDiscard
  \/ Step2PublishSucceed
  \/ Step2PublishFailEnqueue
  \/ DaemonRestartRecovery
  \/ ReimportPublishingSession

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ session_status \in {"needs_refinement", "refining", "publishing", "published"}
  /\ refined_payload \in BOOLEAN
  /\ queue_status \in {"none", "pending", "backoff", "failed"}
  /\ queue_count \in 0..2
  /\ worker_step \in {"idle", "step1_draft_saved", "step2_draft_saved", "step2_publishing"}
  /\ pg_drafts \in 0..3
  /\ pg_published \in 0..2
  /\ quality_rejected \in BOOLEAN
  /\ reimported_while_publishing \in BOOLEAN

AtMostOnePendingQueueEntry ==
  queue_count <= 1

NoStrandedRefiningSession ==
  (worker_step = "idle" /\ session_status /= "published") =>
    (session_status = "needs_refinement" \/ queue_status \in {"pending", "backoff"})

RejectedDraftNeverPublished ==
  quality_rejected => pg_published = 0

AtMostOnePublishedExperience ==
  pg_published <= 1

ImporterPreservesPublishingSession ==
  ~reimported_while_publishing

====
