# Tasks: Instant Feedback

**Total**: 4

## Task 1: Add context field to process_before_prompt
- File: pipeline.py
- Action: format context string from results + persona_context
- Test: 1 new test
- Status: [x] done

## Task 2: Update hook scripts to inject context
- Files: cursor_before_prompt.py, hermes_user_prompt_submit.py
- Action: read response["context"], inject into output
- Test: manual
- Status: [x] done

## Task 3: Verify + commit
- Action: ruff, pytest, e2e-verify, commit, update roadmap
- Status: [x] done
