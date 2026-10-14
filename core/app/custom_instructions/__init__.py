"""Custom Instructions & Dynamic Knowledge — a self-contained feature module.

Kept separate from the engine internals so the whole feature is easy to find, test and
debug in one place. Phase 1 covers the ADVISORY instruction kind and DYNAMIC KNOWLEDGE:

  compiler.py  — author-time: plain English -> classify + compile (one LLM call).
  scope.py     — turn-time scope matching (direction/state/intent/date).
  loader.py    — turn-time DB loaders: active advisory rows, active dynamic facts.
  render.py    — the subordinate "## TENANT PREFERENCES" block for the responder prompt.
  caps.py      — per-tenant caps (row count + injected length).
  service.py   — orchestration used by the admin API (compile-preview, CRUD, audit).

Routing (Phase 2) reuses app/engine/rules and is intentionally not implemented here yet;
the data model (custom_instructions.rule_set_id) already accommodates it.
"""
