#!/usr/bin/env python3
"""Replace the knowledge base with what the real documents actually say.

    docker compose exec core python app/packs/real_estate_india/seeds/seed_knowledge_dsr.py            # replace all
    docker compose exec core python app/packs/real_estate_india/seeds/seed_knowledge_dsr.py --dry-run  # just list

Built from the two source PDFs — the June brochure and the 11,499 cost sheet — rather than
from summaries written from memory. The difference matters: the old set said the project
offered 3 BHK and left 1 and 2 BHK out entirely, so the agent told a caller "we have 3 BHK
flats available" when asked what the project sells. It was answering correctly from a
knowledge base that was wrong.

Everything below is traceable to a line in one of those two documents. Where the documents
DISAGREED with each other, the conflict is resolved HERE, in favour of the cost sheet (the
same figures as seed_units.py, the inventory the agent's tools quote) — never written into
the knowledge base as a note about the disagreement. Every chunk below can reach the model
that speaks to the caller; a "where our documents disagree" note did, and the agent read the
conflict out on a real call ("the brochure lists 1,239 to 1,422 but the cost sheet shows
1,560"). The brochure's 1,422 sq ft 3 BHK is the cost sheet's 1,560 sq ft layout, so the
knowledge base states 1,560 everywhere.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

BASE = "http://127.0.0.1:8000"
EMAIL, PASSWORD = "rahul@advora.ai", "rahul12345"
TOKEN = None

# The project these documents belong to. Tagging every doc to it at CREATION is what stops a
# DSR policy leaking into a Lemon Tree call: retrieval scopes to the resolved project plus
# tenant-wide (project_id NULL) chunks, so an untagged DSR doc is treated as shared and reaches
# every project. Previously tagging happened only as an after-the-fact batch step in
# seed_projects.py (by title prefix), so re-seeding this file alone recreated the docs as NULL
# and reintroduced the leak. Resolved by slug at runtime; if the project does not exist yet
# (a fresh single-project tenant, before seed_projects.py) we fall back to NULL as before.
PROJECT_SLUG = "dsr-highland-greenz"

DOCUMENTS = [
    {
        "title": "What DSR Highland Greenz Offers",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# What we offer

DSR Highland Greenz sells apartments outright for ownership.

We do NOT offer rentals, leases, paying-guest arrangements or short stays. We do not
handle resale of units already sold, and there is no commercial or office space here.
Someone asking to rent should be told plainly that we only sell, and not walked through a
booking flow that cannot help them.

## Configurations — what flats are available to buy

These are all the flat types, sizes and options we sell. Anyone asking what is available,
what they can buy, what sizes there are, or which configurations exist should be given
this whole list — all five, not only the ones with a published price.

1 BHK — 619, 625, 643 and 646 square feet. Tower D only.
1.5 BHK — 780 square feet. Tower D only.
2 BHK — 875 to 1,211 square feet. Towers A, B and C.
2.5 BHK — 1,325 and 1,330 square feet.
3 BHK — 1,239 to 1,560 square feet. Towers A, B and C.

There is no 4 BHK, no villas, no plots, no penthouses and no row houses at this project.
When asked what is available, mention the range, not only the configuration whose cost
sheet happens to be at hand.

This whole-list instruction is for "what's available" questions ONLY. A caller who has
already named a configuration and is now asking something narrower about it — its price,
its size, its facing — should get an answer about THAT configuration, not this full list
again. Repeating it on every later turn just because this document keeps coming up is not
what "mention the range" means.

## The project
10.10 acres at Chikkanayakanahalli, off Sarjapur Road, Bengaluru.
900 flats across 4 towers and 8 wings. 1 basement plus ground plus 12 floors.
Three clubhouses totalling 23,156 square feet.
RERA registration PRM/KA/RERA/1251/446/PR/110522/004876, Karnataka.
""",
    },
    {
        "title": "Which Prices We Can Quote",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Which prices we hold, and which we do not

Cost sheets exist for 2 BHK, 2.5 BHK and 3 BHK. There is NO cost sheet for 1 BHK or
1.5 BHK — we cannot quote any price for those, and must not estimate one.

## The rate is not the same for every configuration
2 BHK and 2.5 BHK are 11,000 rupees per square foot.
3 BHK is 11,499 rupees per square foot.

Never apply one configuration's rate to another, and never multiply a rate by an area to
produce a total yourself. The published totals also fold in infrastructure, clubhouse and
car parking charges, which differ between configurations — a figure worked out from the
rate alone will be wrong by lakhs.

## The cheapest published unit
The cheapest published unit is a 2 BHK of 875 square feet at 1,03,93,750 rupees all in.
Budget questions — what fits a given figure, what the entry price is, the cheapest of a
configuration — are answered by looking the actual units up against that figure with the
inventory tool, never by quoting a stock line. A caller naming a budget wants the units that
sit near it and an honest word about the gap to the next one, not a blanket statement about
the crore mark.

## What a published total includes and excludes
Included — flat cost, infrastructure charges, clubhouse charges, car parking.
NOT included — GST, common area maintenance, corpus fund, legal fees, floor rise,
location and premium charges, franking, stamp duty and registration.

## Infrastructure charges — what the 250 per square foot is for
Infrastructure is 250 rupees per square foot on every configuration, excluding GST. It
covers external electrification, water supply, the sewage treatment plant (STP), the water
treatment plant (WTP) and 100% power backup. It is already inside the published totals,
so never add it on top of a total. On a per-square-foot question,
state it separately from the base rate rather than folding the two into one number.

Published prices are stated to be for a limited period.

## 1 BHK enquiries
Say the cost sheet is not published for 1 BHK yet, and offer a callback from the sales team
or a site visit where current pricing is given. Do not reach for the 2 BHK rate.
""",
    },
    {
        "title": "2 BHK and 2.5 BHK Cost Sheet — 11,000 per square foot",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# 2 BHK and 2.5 BHK cost sheet

Rate 11,000 rupees per square foot on super built-up area. Clubhouse charges are 2,50,000
for a 2 BHK and 3,25,000 for a 2.5 BHK. Car parking is 3,00,000 for all of these units.

## 2 BHK
875 sq ft, East and West facing. Carpet 592. Flat cost 96,25,000. Total 1,03,93,750.
880 sq ft, North facing. Carpet 595. Flat cost 96,80,000. Total 1,04,50,000.
1,099 sq ft, North facing. Carpet 728. Flat cost 1,20,89,000. Total 1,29,13,750.
1,100 sq ft, North facing. Carpet 728. Flat cost 1,21,00,000. Total 1,29,25,000.
1,103 sq ft, West facing. Carpet 726. Flat cost 1,21,33,000. Total 1,29,58,750.
1,104 sq ft, East facing. Carpet 716. Flat cost 1,21,44,000. Total 1,29,70,000.
1,105 sq ft, East and West facing. Carpet 726. Flat cost 1,21,55,000. Total 1,29,81,250.
1,175 sq ft, East and West facing. Carpet 768. Flat cost 1,29,25,000. Total 1,37,68,750.
1,210 sq ft, East and West facing. Carpet 796. Flat cost 1,33,10,000. Total 1,41,62,500.

## 2.5 BHK cost — what a 2.5 BHK flat costs

The price of a 2.5 BHK. Two layouts, both at 11,000 rupees per square foot, with clubhouse
charges of 3,25,000 and car parking of 3,00,000.

2.5 BHK, 1,325 sq ft, South facing. Carpet 859. Flat cost 1,45,75,000. Total 1,55,31,250.
2.5 BHK, 1,330 sq ft, North facing. Carpet 850. Flat cost 1,46,30,000. Total 1,55,87,500.

The 875 square foot 2 BHK at 1,03,93,750 rupees is the cheapest published unit in the
project.
""",
    },
    {
        "title": "3 BHK Cost Sheet — 11,499 per square foot",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# 3 BHK cost sheet

Rate 11,499 rupees per square foot on super built-up area. Clubhouse charges 3,25,000 on
all three. Car parking is 5,50,000 on the two smaller units and 6,00,000 on the largest.

1,239 sq ft, West facing. Carpet 808. Balcony and utility 99. External walls 61.
Flat cost 1,42,47,261. Infrastructure 3,09,750. Total 1,54,32,011.

1,390 sq ft, East and West facing. Carpet 923. Balcony and utility 100. External walls 63.
Flat cost 1,59,83,610. Infrastructure 3,47,500. Total 1,72,06,110.

1,560 sq ft, East and West facing. Carpet 1,032. Balcony and utility 96. External walls 90.
Flat cost 1,79,38,440. Infrastructure 3,90,000. Total 1,92,53,440.

A 3 BHK starts at 1,54,32,011 rupees all in. Anyone with a budget below that is not
looking at a 3 BHK, and should be pointed at the 2 BHK sheet instead.
""",
    },
    {
        "title": "Additional Charges Over the Cost Sheet",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Charges on top of the published total

Common area maintenance — 3.50 rupees per square foot per month, for two years, excluding
GST.
Corpus fund — 20 rupees per square foot, one time.
Legal fees — 30,000 rupees, excluding GST.
Solar water heater — 95,000 rupees excluding GST, for TOP FLOOR FLATS ONLY. Do not quote
it to anyone buying on another floor.
Floor rise — 30 rupees per square foot for each floor, from the 4th floor onwards,
excluding GST. Nothing applies below the 4th.
Franking and agreement registration — as per actuals.
Registration fee and stamp duty — as per actuals.

## Location and premium charges, by unit
These apply ONLY to the specific units listed. Never tell a customer their flat attracts
one without knowing exactly which unit they mean.

Corner flats, 50 rupees per square foot — Tower A wing 1 sr3, wing 2 sr6; Tower B wing 1
sr2, wing 2 sr2 and sr9; Tower C wing 1 sr2 and sr3, wing 2 sr2 and sr3, wing 3 sr2, sr3,
sr8 and sr9.

Premium flats, 75 rupees per square foot — Tower A wing 1 sr1 and sr8, wing 2 sr1 and sr8;
Tower B wing 1 sr4 and sr5, wing 2 sr4, sr5, sr6 and sr7; Tower C wing 1 sr4 and sr5.
Whether this tier applies to a particular flat is confirmed by the sales team for that unit.

Corner and premium flats, 100 rupees per square foot — Tower A wing 1-2 sr7 and sr6, wing 2
sr2, sr3 and sr7; Tower B wing 1 sr3, sr6, sr7, wing 2 sr3 and sr8; Tower C wing 1 sr6 and
sr7, wing 2 sr6 and sr7; Tower D sr5, sr6, sr11 and sr12.

## Car parking
Included in the published totals. Where quoted separately, covered parking is 5.5 lakh and
two single parkings are 6 lakh.
""",
    },
    # "Where Our Documents Disagree" (doc_type sop) USED to sit here. Removed: every conflict
    # it listed is now resolved in the documents themselves (3 BHK 1,560 in the unit mix; 1.5
    # and 2.5 BHK in the configuration list; 2 BHK up to 1,211; the premium-flat charge stated
    # per unit in the charges document), so it had nothing left to say except how the
    # paperwork disagrees — and that is exactly what the agent repeated to callers.
    {
        "title": "Unit Mix by Tower",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Which configurations sit in which tower

Tower A, wings 1 and 2 — 3 BHK of 1,560 and 1,390 sq ft; 2 BHK of 1,099, 1,103 and 1,104.

Tower B, wing 1 — 3 BHK of 1,560 and 1,390; 2 BHK of 1,099, 1,103 and 1,104.
Tower B, wing 2 — 2 BHK only: 875, 880, 1,100, 1,105, 1,175, 1,210 and 1,211.

Tower C, wing 1 — 3 BHK of 1,560 and 1,239; 2 BHK of 1,099, 1,103 and 1,104.
Tower C, wing 2 — 3 BHK of 1,560 and 1,390; 2 BHK of 1,099, 1,103 and 1,104.
Tower C, wing 3 — 2 BHK only: 875, 880, 1,100, 1,105, 1,175 and 1,210.

Tower D — the small-format tower. 1 BHK of 619, 625, 643 and 646 sq ft, and 1.5 BHK of
780 sq ft. There are no 1 BHK units anywhere except Tower D, and no 3 BHK in Tower D.
""",
    },
    {
        "title": "Amenities and Clubhouses",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Amenities and facilities at DSR Highland Greenz

What amenities are included, what facilities residents get, what the clubhouses offer.
This is the full list of amenities — there are more than forty across three clubhouses
totalling 23,156 square feet.

Three clubhouses totalling 23,156 square feet.

## What is in the clubhouse — indoor facilities

What the clubhouse has, what is inside the clubhouses, indoor facilities for residents.
Three clubhouses share the following between them.

Swimming pool, pool deck, separate kids or toddlers pool. Reception. Multipurpose hall and
banquet hall. Squash court. Badminton court. Indoor games room. Senior citizens cards
room. Yoga and meditation room. Creche. Guest bedroom. Indoor gym. Library.

## Sports facilities and games

Sports facilities, games, and things to play — courts, nets and pitches on site.

Children's play area. Tennis court. Basketball court. Mini soccer ground. Cricket practice
nets. Skating rink. Rock climbing wall. Sand pits. Trampoline park.

## Outdoor
Amphitheatre. Outdoor gym. Party lawn with stage. Barbeque zone. Informal seating. Youth
hangout zone. Picnic lawn with hammocks, tables and camping tents. Play lawn. Lawn mound.
Dog park. Jogging track. Cycle lane. Pedestrian bridge and pathways. Elevated yoga,
meditation and bird watching deck. Seasonal stream. Elders seating plaza. Children's play
area with floor games. Informal seating area at first floor.

## Arrival and circulation
Entry arch, entry roundabout, signage wall and signage pylon. Main driveway and block drop
off points. Informal seating and bus waiting area at the entrance. Water feature with
sculpture court at the entry, and a second at the seniors seating plaza. Entry plaza for
the clubhouse.
""",
    },
    {
        "title": "Specifications",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Specifications

Structure — RCC framed for wind and seismic loads. External walls 6 inch solid block
masonry, internal walls 4 inch solid block masonry. Two coats of plastering, internal in
CM 1:6 and external in CM 1:4.

Painting — external is a textured finish with two coats of exterior emulsion of a reputed
brand. Internal is a smooth putty finish with two coats of premium acrylic emulsion over a
coat of primer.

Flooring — bedrooms, living, dining and kitchen in 80cm by 80cm vitrified tiles. Bathrooms
300x300mm anti-skid acid-resistant ceramic. Balconies 200x1200mm. Corridors vitrified with
inlay. Staircases polished kota stone.

Doors and windows — main and internal doors in moulded Acacia engineered wood with
laminated flush shutters and stainless steel hardware. French doors, where present, UPVC
sliding with mosquito mesh. Windows UPVC sliding with glass, mesh and safety grills.

Bathrooms — wash basin with half pedestal, EWC with concealed flush tank or flush valve,
hot and cold concealed single-lever diverter with head shower, geyser provision in all
bathrooms. CP fittings and sanitaryware Vitra, American Standard, Sirius or equivalent.
Glazed or matt ceramic dado of 300x600mm to door height. Utility and wash areas have the
same 300x600mm dado to 3 feet.

Ventilators — UPVC or powder-coated aluminium with fixed glass and exhaust provision, no
safety grill. French doors, where present, have no safety grill either.

Kitchen and utility — treated water connection, water purifier provision, sink inlet and
outlet, washing machine provision, stainless steel single bowl sink with granite counter.

Electrical — concealed copper wiring, Polycab or Anchor or equivalent. AC point in master
bedroom, geyser points in all bathrooms, kitchen points for chimney, fridge, microwave and
mixer. Schneider or equivalent MCB and ELCB. Modular switches.

Telecom — telephone points in master bedroom and living. Cable and DTH provision in master
bedroom and living. In a 1 BHK, TV and cable provision is in the living room ONLY.
Internet point in living.

Security — intercom to security, clubhouse and common areas. Video door phone per unit.
CCTV at entry, exit and driveways.

Lifts and power — branded automatic elevators with granite or vitrified cladding. 100
percent DG backup with acoustic enclosure and AMF panel for all flats and common areas.
""",
    },
    {
        "title": "Location and Connectivity",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Location

Chikkanayakanahalli, off Sarjapur Road, Bengaluru.

RGA Tech Park 2.2 km. Wipro Corporate Office 2.5 km. Decathlon Sports Arena 2.5 km.
Manipal Hospital 5.5 km. Eco Space 6.7 km. Electronic City 9 km. Marathahalli 9 km.

Also nearby — Sakra Hospital, Embassy Tech Village, New Horizon College, Oxford
Engineering College, Primus International School, Karmelaram Railway Station, Bellandur,
Hosa Road, Dommasandra Junction.

## Offices
Bengaluru — DSR Techno Cube, 4th Floor, Block C, Thubarahalli Village, Varthur Main Road,
Bengaluru 560066. Phone 080 4912 3000, mobile 90191 92000.
Corporate — Plot 221, 1st Floor, Road 17, opposite Jubilee Hills International Centre,
Hyderabad 500033. Phone 040 2999 2222.
""",
    },
    {
        "title": "Other DSR Projects",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Other ongoing DSR projects

If Highland Greenz does not suit, these exist. We hold no cost sheets for any of them, so
never quote a price or claim availability — offer a callback instead.

DSR Parkway — off Sarjapur Road. 1, 2 and 3 BHK premium apartments.
DSR RR Avenues — Yelahanka New Town. 2 and 3 BHK apartments.
DSR Vertex and Apex — Kundanahalli Gate Junction, Thubrahalli. 2 and 3 BHK apartments.
DSR Elixir — plotted development at Whitefield.
""",
    },
    {
        "title": "Who to Contact",
        "doc_type": "sop",
        "effective_from": None, "effective_to": None,
        "content": """# Contacts

Home loan assistance — Sastry, 8792582433. Offer this ONLY when someone asks about home
loans or financing. It is not an answer to a general question.

Sales team — Abishek Reddy 8904575960, Arjun 7022023980, Surendra 9611533850, Srinivas
Moorthy 9035261883. Prefer booking a callback over reading a number out, so the request is
logged and someone is accountable for it.

Email sales@dsrinfra.com. Website www.dsrinfra.com.
RERA details at www.rera.karnataka.gov.in under PRM/KA/RERA/1251/446/PR/110522/004876.
""",
    },
    {
        "title": "Site Visit and Booking Process",
        "doc_type": "sop",
        "effective_from": None, "effective_to": None,
        "content": """# Visiting and booking

Site visits run at the DSR Highland Greenz site office at Chikkanayakanahalli, every day
between 10:00 and 18:30. A visit takes about an hour. Slots are confirmed at booking and a
confirmation goes to the registered mobile number. Bring photo identity. Parking available.

To move or cancel a visit, identity is verified first — the last four digits of the mobile
number registered with us.

Booking a unit requires a token amount. The sales team confirms the current amount and the
payment schedule at the visit. The agent does not quote token amounts, discounts or payment
schedules, and cannot negotiate on price.
""",
    },
    {
        "title": "What the Brochure Says About Its Own Accuracy",
        "doc_type": "compliance",
        "effective_from": None, "effective_to": None,
        "content": """# The brochure's own disclaimer

The brochure states plainly that the developer and its agents cannot be held responsible
for inaccuracies or omissions, and that all statements are believed correct but are not to
be regarded as statements or representations of fact.

Floor areas and other measurements are approximations only and are SUBJECT TO FINAL SURVEY.

All visual representation — models, drawings, illustrations, photographs and art renderings
— are artistic impressions only and are not representations of fact.

All information, plans and specifications are current at the time of print and are subject
to change as required or approved by the developer or the relevant authorities, and CANNOT
FORM PART OF A CONTRACT.

## What this means on a call
Areas, layouts and specifications quoted from the brochure are indicative. Never present
them as contractual or final. If a customer is making a decision on an exact square
footage, a facing, or a particular fitting, say that the sales team confirms the final
figures at the agreement stage — because our own brochure says the same thing.
""",
    },
    {
        "title": "Monsoon Booking Offer 2026",
        "doc_type": "promo",
        "effective_from": "2026-06-01", "effective_to": "2026-07-31",
        "content": """# Monsoon Booking Offer 2026

Ran 1 June 2026 to 31 July 2026 and has ended. Not available. Must not be offered, quoted
or described as current. The offer was a waiver of clubhouse charges on bookings confirmed
inside the window.
""",
    },
    {
        "title": "Festive Booking Offer 2026",
        "doc_type": "promo",
        "effective_from": "2026-09-01", "effective_to": "2026-10-31",
        "content": """# Festive Booking Offer 2026

Runs 1 September 2026 to 31 October 2026. Has NOT started. Must not be offered or quoted
before it opens. The offer will be a waiver of the corpus fund on bookings confirmed inside
the window.
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
    ap.add_argument("--keep", action="store_true", help="do not delete existing documents")
    args = ap.parse_args()

    if args.dry_run:
        print(f"\n  {len(DOCUMENTS)} documents\n")
        for d in DOCUMENTS:
            print(f"  {d['title']:46} {d['doc_type']:9} {len(d['content']):>6} chars")
        print()
        return 0

    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    # Resolve THIS project's id so every document is tagged to it (and so the delete below is
    # scoped to it). None means projects are not set up yet — a single-project tenant — in which
    # case docs stay tenant-wide (project_id NULL), exactly as before multi-project.
    projects = call("GET", "/projects") or []
    project_id = next((p["id"] for p in projects if p.get("slug") == PROJECT_SLUG), None)
    if project_id:
        print(f"\n  tagging documents to project {PROJECT_SLUG} ({project_id})")
    else:
        print(f"\n  project '{PROJECT_SLUG}' not found — documents stay tenant-wide (single-project "
              "tenant). Run seed_projects.py to enable multi-project isolation.")

    if not args.keep:
        existing = call("GET", "/knowledge/documents?include_expired=true") or []
        # Delete ONLY this project's documents (plus legacy untagged ones, which were DSR's before
        # tagging existed) — never another project's. Without this scope, re-seeding DSR wiped
        # Lemon Tree's knowledge too, because the old code deleted every document in the tenant.
        if project_id:
            to_delete = [d for d in existing
                         if d.get("project_id") in (project_id, None)]
        else:
            to_delete = existing
        for doc in to_delete:
            call("DELETE", f"/knowledge/documents/{doc['id']}")
        print(f"  removed {len(to_delete)} old document(s)")

    print()
    total = 0
    failures = []
    for spec in DOCUMENTS:
        try:
            # Tag to the resolved project (omit the key entirely when there is none, so the
            # single-project path is byte-for-byte the old request).
            payload = {**spec, "project_id": project_id} if project_id else spec
            res = call("POST", "/knowledge/documents", payload)
            n = res.get("chunks_created", 0)
            total += n
            window = ("always live" if not (spec["effective_from"] or spec["effective_to"])
                      else f"{spec['effective_from'] or '…'} to {spec['effective_to'] or '…'}")
            # Zero chunks means the document is INVISIBLE to the agent, and the endpoint
            # returns 201 either way — ingestion failing is reported in the body, not the
            # status code. A silent zero here is how the entire pricing policy went missing
            # while the seeder printed a tidy list and looked like it had worked.
            flag = ""
            if n == 0:
                flag = "  <-- NOT INGESTED, the agent cannot see this"
                failures.append((spec["title"], res.get("error") or res.get("status") or "unknown"))
            print(f"  {spec['title'][:46]:48} {n:>3} chunks   {window}{flag}")
        except urllib.error.HTTPError as e:
            failures.append((spec["title"], f"HTTP {e.code}"))
            print(f"  {spec['title'][:46]:48} FAILED {e.code} {e.read().decode()[:120]}")

    print(f"\n  {len(DOCUMENTS)} documents, {total} chunks embedded.")
    if failures:
        print(f"\n  {len(failures)} DOCUMENT(S) DID NOT INGEST — the agent cannot answer from them:")
        for title, why in failures:
            print(f"    {title}  ({why})")
        print("\n  Usually an embedding failure. Check credits with scripts/check_llm.py,")
        print("  then re-run this script — it is safe to run again.\n")
        return 1
    print("  Check with: scripts/show_knowledge.py --grep '1 bhk'\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())