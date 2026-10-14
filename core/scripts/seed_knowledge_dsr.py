#!/usr/bin/env python3
"""Replace the knowledge base with what the real documents actually say.

    docker compose exec core python scripts/seed_knowledge_dsr.py            # replace all
    docker compose exec core python scripts/seed_knowledge_dsr.py --dry-run  # just list

A thin entry point. The ONE copy of the DSR knowledge base is
app/packs/real_estate_india/seeds/seed_knowledge_dsr.py — this file only runs it.

It used to be a full second copy, and the two drifted: the fix that stopped the agent telling
callers "the brochure and cost sheet disagree" (8587bac) went into the pack copy only, while
the infrastructure-charge note (bf5b4c1) went into this one only. docs/SETUP.md points here,
so re-seeding by the book put the leaking note straight back. One copy cannot drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.packs.real_estate_india.seeds import seed_knowledge_dsr as _canonical  # noqa: E402

# Re-exported for scripts/apply_inventory_update.py, which reads the pricing document's body.
DOCUMENTS = _canonical.DOCUMENTS

if __name__ == "__main__":
    sys.exit(_canonical.main())
