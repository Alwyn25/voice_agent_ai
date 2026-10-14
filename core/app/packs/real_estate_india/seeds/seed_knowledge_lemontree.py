#!/usr/bin/env python3
"""Add the Lemon Tree (Legend Properties) project into the SAME tenant as DSR.

    docker compose exec core python app/packs/real_estate_india/seeds/seed_knowledge_lemontree.py            # add
    docker compose exec core python app/packs/real_estate_india/seeds/seed_knowledge_lemontree.py --dry-run  # list only
    docker compose exec core python app/packs/real_estate_india/seeds/seed_knowledge_lemontree.py --purge    # remove them again

WHY THIS EXISTS. This is a DELIBERATE multi-project-in-one-tenant test. The engine assumes
one project per tenant (retrieval is scoped by tenant_id only, and tenant.project_facts holds
a single project), so loading a SECOND project's documents beside DSR lets us measure what
retrieval does when two projects' chunks live in one index — how often a Lemon Tree question
pulls a DSR chunk and vice-versa, and what that does to latency and answer quality.

Unlike seed_knowledge_dsr.py this NEVER deletes existing documents — it only adds. Use
--purge to remove exactly the documents this script created (matched by their title prefix),
leaving DSR untouched.

Every fact below is traceable to a line in the Lemon Tree brochure (Lemon-tree-brochure-MAy.pdf).
Where the brochure gives only marketing copy or an unreadable figure, that is written down as
NOT AVAILABLE rather than guessed — the agent must not quote a price, area, RERA number or
possession date this brochure does not state. All Lemon Tree titles start with the same prefix
so they are easy to spot in /knowledge/search output and easy to purge.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

BASE = "http://127.0.0.1:8000"
EMAIL, PASSWORD = "rahul@advora.ai", "rahul12345"   # SAME tenant as the DSR fixture — intentional
TOKEN = None

# Every Lemon Tree document title starts with this, so the multi-project chunks are obvious in
# search output and --purge can find exactly what to remove.
PREFIX = "Lemon Tree — "

DOCUMENTS = [
    {
        "title": PREFIX + "What the Project Is",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# Lemon Tree by Legend — what the project is

Lemon Tree is a residential apartment development by Legend Properties, marketed as
"A Landmark Development". It offers 2 BHK, 2.5 BHK and 3 BHK apartments, with commercial
retail space on the lower floors of the building.

This is a DIFFERENT project from DSR Highland Greenz. Nothing about DSR — its prices, sizes,
RERA number, towers or location — applies to Lemon Tree, and nothing here applies to DSR. If a
caller is asking about Lemon Tree, answer only from Lemon Tree material.

## What flats are available to buy
2 BHK, 2.5 BHK and 3 BHK. These are the only configurations the brochure shows. It does not
mention 1 BHK, 4 BHK, villas, plots or penthouses, so do not claim any of those exist.

## What the brochure does NOT state
The brochure gives no prices, no cost sheet, no exact unit areas in square feet, no RERA
registration number, no tower or flat counts, no floor count and no possession date. Do not
invent any of these. When asked, say the sales team confirms them and offer a callback.
""",
    },
    {
        "title": PREFIX + "Pricing (Not Published in This Brochure)",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Lemon Tree pricing — nothing is published here

The Lemon Tree brochure carries NO price list, NO cost sheet and NO per-square-foot rate for
any configuration. We cannot quote a price for a 2 BHK, 2.5 BHK or 3 BHK, and must not
estimate one.

Never borrow a figure from another project. In particular, DSR Highland Greenz rates
(11,000 and 11,499 rupees per square foot) have NOTHING to do with Lemon Tree — using them
here would be wrong.

When someone asks the price, say the cost sheet is not published in the brochure and offer a
callback from the sales team or a site visit where current pricing is given.
""",
    },
    {
        "title": PREFIX + "Configurations and Floor Plans",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Configurations and floor plans

Lemon Tree offers three configurations: 2 BHK, 2.5 BHK and 3 BHK.

The brochure includes a ground floor plan and a 5th-to-9th floor plan, which place 2 BHK,
2.5 BHK and 3 BHK units on the typical residential floors. The ground floor holds the
entrance lobby and commercial space.

Exact unit sizes (carpet or super built-up area in square feet), the number of units of each
type, and which floors each type sits on are NOT given as readable figures in the brochure.
Do not state a square footage for any Lemon Tree unit. The sales team confirms exact areas.
""",
    },
    {
        "title": PREFIX + "Amenities",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Amenities at Lemon Tree

The brochure lists fifteen amenities:

1. Entry and exit.
2. Swimming pool and a separate toddler pool.
3. Kids play area.
4. Cricket net (cricket practice net).
5. Seating area.
6. Multi purpose court.
7. Jogging track.
8. Car parking.
9. Boom barrier at the entrance.
10. Entrance lobby.
11. Gym and indoor games.
12. Multi purpose hall.
13. Landscape / landscaped areas.
14. Movie theater.
15. Commercial space.

That is the full amenity list in the brochure. Do not add amenities it does not mention
(for example there is no clubhouse, tennis court or squash court listed here).
""",
    },
    {
        "title": PREFIX + "Location and Connectivity",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Location and connectivity

Lemon Tree is at Channasandra Junction, Whitefield, on the Whitefield - Channasandra Main
Road, Bengaluru.

## Nearby workplaces
Close to ITPL, the EPIP Zone and SAP Labs. The brochure notes an upcoming Whitefield Metro
Station.

## Education nearby
MVJ College, Delhi Public School, Vydehi Institute of Medical Sciences, Gopalan
International School.

## Healthcare nearby
Columbia Asia Hospital, Manipal Hospital, Narayana Multispeciality Hospital.

## Shopping and leisure nearby
Nexus Whitefield, Inorbit Mall, Phoenix Market City.
""",
    },
    {
        "title": PREFIX + "Commercial Space",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Commercial space

Lemon Tree includes commercial retail space on the lower floors of the building, branded
"Lemon Tree by Legend". The ground floor plan marks a commercial space of 13,058.56 square
feet.

The brochure does not give commercial unit sizes, rents or prices individually. Refer any
commercial-space enquiry to the sales team.
""",
    },
    {
        "title": PREFIX + "Contact",
        "doc_type": "sop",
        "effective_from": None, "effective_to": None,
        "content": """# Contact for Lemon Tree

Phone: +91 8884441863 and +91 8884441836.
Website: lemontree.legendproperties.co.in
Developer: Legend Properties.

Prefer booking a callback over reading a number out, so the request is logged and someone is
accountable for it.
""",
    },
    {
        "title": PREFIX + "Marketing Claims — Do Not Assert",
        "doc_type": "compliance",
        "effective_from": None, "effective_to": None,
        "content": """# Marketing claims in the brochure that the agent must NOT repeat as promises

The Lemon Tree brochure contains promotional investment language. The agent must not present
any of it as a guarantee, a factual return, or financial advice:

- "High ROI Guaranteed" — do NOT tell a caller their return is guaranteed.
- "capital appreciation has shown a consistent upward trend of 8-12% annually" — this is a
  market claim about Whitefield, not a promise about this flat. Do not quote it as a
  guaranteed or personal return.
- "Excellent rental yield" and "High resale value" — do not assert these as facts.

If a caller asks about investment returns, rental yield or resale value, do not quote these
numbers as promises. Say the sales team can share market information, and that returns are not
guaranteed. This is not financial advice.

The brochure's own factual, repeatable location claim is fine to use: Whitefield is a
well-connected Bengaluru micro-market near major tech parks. Keep to that.
""",
    },
]


def call(method, path, payload=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=180) as res:
        body = res.read().decode()
        return json.loads(body) if body else None


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would be written")
    ap.add_argument("--purge", action="store_true",
                    help="remove the Lemon Tree documents this script added (DSR untouched)")
    args = ap.parse_args()

    if args.dry_run:
        print(f"\n  {len(DOCUMENTS)} documents (added to the SAME tenant as DSR)\n")
        for d in DOCUMENTS:
            print(f"  {d['title']:52} {d['doc_type']:11} {len(d['content']):>6} chars")
        print()
        return 0

    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    # Tag every Lemon Tree document to the Lemon Tree PROJECT, so retrieval scoped to another
    # project (DSR) can never surface these chunks. Without this the docs land project_id=NULL
    # — treated as tenant-wide and shared into every project's pool — which is exactly how a
    # DSR price question got answered from Lemon's "nothing is published here" sheet. The
    # project must exist first (seed_projects.py creates it); we look it up by slug and refuse
    # to seed untagged rather than silently reintroduce the leak.
    lemon_project_id = None
    if not args.purge:
        projects = call("GET", "/projects") or []
        match = next((p for p in projects
                      if p.get("slug") == "lemon-tree" or p.get("name") == "Lemon Tree"), None)
        if match is None:
            print("\n  ! No 'Lemon Tree' project found. Run app/packs/real_estate_india/seeds/seed_projects.py first so\n"
                  "    these documents can be tagged to it (otherwise DSR retrieval leaks "
                  "them).\n")
            return 1
        lemon_project_id = match["id"]
        print(f"  tagging all documents to project Lemon Tree ({lemon_project_id})\n")

    if args.purge:
        existing = call("GET", "/knowledge/documents?include_expired=true") or []
        removed = 0
        for doc in existing:
            if doc["title"].startswith(PREFIX):
                call("DELETE", f"/knowledge/documents/{doc['id']}")
                removed += 1
        print(f"\n  purged {removed} Lemon Tree document(s); DSR documents left in place\n")
        return 0

    print()
    total = 0
    failures = []
    for spec in DOCUMENTS:
        try:
            res = call("POST", "/knowledge/documents", {**spec, "project_id": lemon_project_id})
            n = res.get("chunks_created", 0)
            total += n
            # Zero chunks means the document is INVISIBLE to the agent, and the endpoint returns
            # 201 either way — a silent zero is how a whole document goes missing while the seeder
            # prints a tidy list. Flag it loudly.
            flag = ""
            if n == 0:
                flag = "  <-- NOT INGESTED, the agent cannot see this"
                failures.append((spec["title"], res.get("error") or res.get("status") or "unknown"))
            print(f"  {spec['title'][:52]:54} {n:>3} chunks{flag}")
        except urllib.error.HTTPError as e:
            failures.append((spec["title"], f"HTTP {e.code}"))
            print(f"  {spec['title'][:52]:54} FAILED {e.code} {e.read().decode()[:120]}")

    print(f"\n  {len(DOCUMENTS)} documents, {total} chunks embedded (added beside DSR).")
    if failures:
        print(f"\n  {len(failures)} DOCUMENT(S) DID NOT INGEST — the agent cannot answer from them:")
        for title, why in failures:
            print(f"    {title}  ({why})")
        print("\n  Usually an embedding failure. Check credits with scripts/check_llm.py,")
        print("  then re-run — it is safe to run again (this script only adds).\n")
        return 1
    print("  Multi-project test is live. Compare retrieval with, e.g.:")
    print("    scripts/show_knowledge.py --grep 'lemon tree'")
    print("    POST /knowledge/search {\"query\": \"what is the price of a 3 bhk\"}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
