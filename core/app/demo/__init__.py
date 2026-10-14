"""GTM demo links: private per-prospect demo pages for cold email.

Design: docs/gtm-demo-links.md. Each prospect gets its own tenant (tenants.kind='demo'),
so every tenant-scoped part of the engine (retrieval, caches, synonyms, custom
instructions) isolates one prospect's demo from another's without extra filters.
"""
