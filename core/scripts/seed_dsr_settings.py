"""Seed DSR's retrieval synonyms and curated project facts into ``tenant.settings``.

Usage (inside core/, DB up):
    python scripts/seed_dsr_settings.py                       # tenant of rahul@advora.ai
    python scripts/seed_dsr_settings.py --tenant advora       # by tenant slug
    python scripts/seed_dsr_settings.py --email rahul@advora.ai

A thin entry point. The ONE copy is app/packs/real_estate_india/seeds/seed_dsr_settings.py —
this file runs it unchanged, same arguments. It used to be a full second copy that drifted
(its "Configurations" fact said 619-1,422 sq ft, which the cost sheet's 1,560 sq ft unit
contradicts); see scripts/seed_knowledge_dsr.py for why a duplicated seed is how a fixed bug
comes back.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent.parent
                       / "app/packs/real_estate_india/seeds/seed_dsr_settings.py"),
                   run_name="__main__")
