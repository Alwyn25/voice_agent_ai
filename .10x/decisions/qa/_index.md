# QA Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo adapter | Pending implementation |

## Test Infrastructure [DISCOVERED]
- `core/scripts/` — integration scripts (verify_stage*.ps1, smoke_*.py, eval_*.py, profile_*.py)
- `core/scripts/run_conversation_tests.py` — conversation-level test harness
- `core/scripts/smoke_p1_p4.py` — smoke test covering Stages 1-4
- Unit test directory: not yet established (create `core/tests/telephony/` for Stage 10)
