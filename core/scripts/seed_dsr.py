#!/usr/bin/env python3
"""Seed the DSR Highland Greenz real estate demo.

    docker compose exec core python scripts/seed_dsr.py
    docker compose exec core python scripts/seed_dsr.py --only knowledge --wipe-knowledge
    docker compose exec core python scripts/seed_dsr.py --skip conversations

A thin entry point. The ONE copy is app/packs/real_estate_india/seeds/seed_dsr.py — this file
runs it unchanged, same arguments. It used to be a full second copy that drifted (old
/scheduling/centres endpoints, the 1,239-1,422 3 BHK range the cost sheet contradicts); see
scripts/seed_knowledge_dsr.py for why a duplicated seed is how a fixed bug comes back.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent.parent
                       / "app/packs/real_estate_india/seeds/seed_dsr.py"), run_name="__main__")
