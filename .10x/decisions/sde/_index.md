# SDE Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo adapter | Ready to implement |

## Cross-cutting SDE Notes [DISCOVERED]
- Python 3.11 (strict — all type hints, `from __future__ import annotations`)
- Pattern: modules start with a docstring explaining why they exist and what they DON'T do
- Test scripts live in `core/scripts/` (integration) — unit tests dir TBD (create `core/tests/`)
