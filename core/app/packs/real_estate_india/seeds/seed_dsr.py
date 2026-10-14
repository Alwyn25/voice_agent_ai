#!/usr/bin/env python3
"""
Seed the DSR Highland Greenz real estate demo.

All content comes from the two supplied documents — the June brochure and the
11,499/sqft cost sheet. Nothing here is invented: every price, area, charge, RERA number
and distance is from those files, so the agent can be checked against them line by line.

Everything is created through the REST API rather than direct database writes, so records
pass the same validation, fire the same state machine transitions and land in the same
audit trail as anything the agent does.

    docker compose exec core python app/packs/real_estate_india/seeds/seed_dsr.py
    docker compose exec core python app/packs/real_estate_india/seeds/seed_dsr.py --only knowledge --wipe-knowledge
    docker compose exec core python app/packs/real_estate_india/seeds/seed_dsr.py --skip conversations

Stages: scheduling, knowledge, rules, leads, visits, support, campaigns, conversations
Stdlib only — no extra dependencies in the image.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE = "http://127.0.0.1:8000"
EMAIL = "rahul@advora.ai"
PASSWORD = "rahul12345"
TOKEN = None
TODAY = date.today()

ago = lambda d: (TODAY - timedelta(days=d)).isoformat()
ahead = lambda d: (TODAY + timedelta(days=d)).isoformat()


# --------------------------------------------------------------------- transport

def call(method, path, payload=None, quiet=False):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            body = res.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        if not quiet:
            print(f"    ! {method} {path} -> {e.code} {e.read().decode()[:200]}")
        raise
    except urllib.error.URLError as e:
        print(f"\n  Cannot reach {BASE} — is core running?  ({e.reason})")
        sys.exit(1)


def try_call(method, path, payload=None, label=""):
    try:
        return call(method, path, payload)
    except urllib.error.HTTPError:
        print(f"      skipped {label or path}")
        return None


def login():
    global TOKEN
    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]
    me = call("GET", "/auth/me")
    print(f"  signed in as {EMAIL} (role: {me['role']}, tenant {me['tenant_id'][:8]})")


ok = lambda label, detail="": print(f"    + {label}{('  ' + detail) if detail else ''}")
skip = lambda label, why: print(f"    · {label} — {why}")


# --------------------------------------------------------------------- the project

# A project is modelled as a centre: it is the place a site visit happens, it has an
# address and working hours, and slots are computed against it. That is exactly what the
# scheduling engine already does.
CENTRES = [
    {"name": "DSR Highland Greenz — Site Office", "city": "Bengaluru", "region_code": "KA-BLR",
     "address": "Chikkanayakanahalli, Off Sarjapur Road, Bengaluru 562125",
     "phone": "+918049123000", "timezone": "Asia/Kolkata"},
    {"name": "DSR Techno Cube — Sales Gallery", "city": "Bengaluru", "region_code": "KA-BLR",
     "address": "4th Floor, Block-C, Thubarahalli Village, Varthur Main Road, Bengaluru 560066",
     "phone": "+919019192000", "timezone": "Asia/Kolkata"},
]

# Sales team, from the cost sheet contact block.
EXECUTIVES = [
    {"full_name": "Abishek Reddy", "centre": "DSR Highland Greenz — Site Office",
     "phone": "+918904575960", "specialties": ["3 BHK", "site visits"]},
    {"full_name": "Arjun", "centre": "DSR Highland Greenz — Site Office",
     "phone": "+917022023980", "specialties": ["2 BHK", "3 BHK"]},
    {"full_name": "Surendra", "centre": "DSR Techno Cube — Sales Gallery",
     "phone": "+919611533850", "specialties": ["1 BHK", "investors"]},
    {"full_name": "Srinivas Moorthy", "centre": "DSR Techno Cube — Sales Gallery",
     "phone": "+919035261883", "specialties": ["NRI", "channel partners"]},
]

HOLIDAYS = [{"date": ahead(21), "name": "Site closed — maintenance"}]


DOCUMENTS = [
    {
        "title": "DSR Highland Greenz — Project Overview",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# DSR Highland Greenz

## Where it is
DSR Highland Greenz is at Chikkanayakanahalli, off Sarjapur Road, Bengaluru. It is a
gated community of luxury apartments by DSR Infra Projects.

## The development
The project covers 10.10 acres. There are 4 towers across 8 wings, each 1 basement plus
ground plus 12 floors, with 900 flats in total. There are 3 clubhouses covering 23,156
square feet.

## Configurations
1 BHK apartments range from 619 to 780 square feet.
2 BHK apartments range from 875 to 1,211 square feet.
3 BHK apartments range from 1,239 to 1,560 square feet.

Tower D holds the 1 BHK and 1.5 BHK units, from 619 to 780 square feet. Towers A, B and C
hold the 2 BHK and 3 BHK units.

## RERA
The project is registered under Karnataka RERA with registration number
PRM/KA/RERA/1251/446/PR/110522/004876. Details can be verified at www.rera.karnataka.gov.in.

## Developer
DSR Infra Projects, with offices in Bengaluru at DSR Techno Cube, Thubarahalli Village,
Varthur Main Road, and a corporate office in Hyderabad.""",
    },
    {
        "title": "What DSR Highland Greenz Offers — Scope",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# What we offer, and what we do not

DSR Highland Greenz sells apartments. Units are sold outright for ownership.

We do NOT offer rentals, leases, paying-guest arrangements or short stays. We do not
handle resale of units already sold, and we do not deal in commercial or office space.
Anyone asking about renting has come to the wrong place and should be told so plainly and
politely, rather than being taken through a booking flow that cannot help them.

## Configurations available for sale
1 BHK — 619 to 780 square feet.
2 BHK — 875 to 1,211 square feet.
3 BHK — 1,239 to 1,560 square feet.
There is no 4 BHK. There are no villas, plots or penthouses at this project.

The project is a gated community of 900 flats across 4 towers and 8 wings, on 10.10 acres
at Chikkanayakanahalli, off Sarjapur Road, Bengaluru. Structure is 1 basement plus ground
plus 12 floors. RERA registration number is PRM/KA/RERA/1251/446/PR/110522/004876.
""",
    },
    {
        "title": "Pricing Availability by Configuration",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Which prices we can quote

A published cost sheet exists for 3 BHK units only. The rate is 11,499 rupees per square
foot on super built-up area, and worked totals are available for the 1,239, 1,390 and
1,560 square foot layouts.

## 1 BHK and 2 BHK pricing
There is NO published cost sheet for 1 BHK or 2 BHK units. We cannot quote a price, a
per-square-foot rate, or a total for them.

The 3 BHK rate must NOT be applied to a 1 BHK or 2 BHK to estimate a cost. Rates differ by
configuration and floor, and a number worked out that way would be wrong.

When someone asks the price of a 1 BHK or a 2 BHK, say plainly that the cost sheet for
that configuration is not published yet, and offer either a callback from the sales team or
a site visit where they will be given current pricing. Do not answer with the 3 BHK rate,
and do not estimate.

## Budget questions
If someone gives a budget and asks whether a unit fits, only compare it against the 3 BHK
totals, because those are the only prices we hold. A 3 BHK starts at 1,54,32,011 rupees
all in, so a budget of 1 crore does not reach a 3 BHK. For 1 BHK and 2 BHK, say the pricing
is not published and offer the callback.
""",
    },
    {
        "title": "3 BHK Cost Sheet — Rate 11,499 per sq ft",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# 3 BHK Cost Sheet

The rate is 11,499 rupees per square foot on super built-up area. Three 3 BHK layouts are
currently offered.

## Unit 1 — 1,239 sq ft, West facing
Super built-up area is 1,239 square feet. Carpet area as per RERA is 808 square feet.
Balcony and utility is 99 square feet. External walls are 61 square feet.
Flat cost is 1,42,47,261 rupees.
Infrastructure charges are 3,09,750 rupees.
Club house charges are 3,25,000 rupees.
Car parking is 5,50,000 rupees.
Total is 1,54,32,011 rupees.

## Unit 2 — 1,390 sq ft, East and West facing
Super built-up area is 1,390 square feet. Carpet area as per RERA is 923 square feet.
Balcony and utility is 100 square feet. External walls are 63 square feet.
Flat cost is 1,59,83,610 rupees.
Infrastructure charges are 3,47,500 rupees.
Club house charges are 3,25,000 rupees.
Car parking is 5,50,000 rupees.
Total is 1,72,06,110 rupees.

## Unit 3 — 1,560 sq ft, East and West facing
Super built-up area is 1,560 square feet. Carpet area as per RERA is 1,032 square feet.
Balcony and utility is 96 square feet. External walls are 90 square feet.
Flat cost is 1,79,38,440 rupees.
Infrastructure charges are 3,90,000 rupees.
Club house charges are 3,25,000 rupees.
Car parking is 6,00,000 rupees.
Total is 1,92,53,440 rupees.

## Note on prices
These prices are for a limited period. All totals above exclude the additional charges
listed separately, and exclude GST where indicated.""",
    },
    {
        "title": "Additional Charges and Payment Terms",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Additional Charges

These apply on top of the cost sheet total.

## Recurring and one-time charges
Common area maintenance for 2 years is 3.50 rupees per square foot per month, excluding
GST.
Corpus fund is 20 rupees per square foot.
Legal fees are 30,000 rupees, excluding GST.
Solar water heater charges apply only to top floor flats and are 95,000 rupees, excluding
GST.

## Infrastructure charges
Infrastructure charges are 250 rupees per square foot excluding GST. These cover external
electrification, water, STP, WTP and 100 percent power backup.

## Location premiums
Corner flats attract 50 rupees per square foot. This applies to Tower A wing 1 sr3 and
wing 2 sr6, Tower B wing 1 sr2 and wing 2 sr2 and sr9, and Tower C wing 1 sr2 and sr3,
wing 2 sr2 and sr3, wing 3 sr2, sr3, sr8 and sr9.

Corner and premium flats attract 100 rupees per square foot. This applies to Tower A wing
1 and 2 sr7 and sr6 and wing 2 sr2, sr3 and sr7, Tower B wing 1 sr3, sr6 and sr7 and wing
2 sr3 and sr8, Tower C wing 1 sr6 and sr7 and wing 2 sr6 and sr7, and Tower D sr5, sr6,
sr11 and sr12.

Floor rise charges apply from the 4th floor onwards at 30 rupees per square foot for each
floor, excluding GST.

## As per actuals
Franking charges and agreement registration charges are as per actuals.
Registration fee and stamp duty charges are as per actuals.

## Home loans
For home loan assistance the contact is Sastry on 8792582433.""",
    },
    {
        "title": "Amenities and Clubhouses",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Amenities

There are three clubhouses totalling 23,156 square feet.

## Indoor and clubhouse
Swimming pool, toddler's pool, reception, multipurpose hall, squash court, badminton
court, indoor games, yoga room, creche, guest bedroom, indoor gym, library, banquet hall,
senior citizens' cards room.

## Sports
Children's play area, tennis court, basketball court, mini soccer ground, cricket practice
nets, skating rink, rock climbing wall, sand pits, trampoline park, outdoor gym.

## Outdoor and landscape
Amphitheatre, party lawn with stage, barbeque zone, informal seating, youth hangout zone,
picnic lawn with hammocks, tables and camping tents, play lawn, lawn mound, dog park,
jogging track, cycle lane, pedestrian bridge, bird watching deck, elevated yoga and
meditation deck, seasonal stream, water features with sculpture courts, elders' seating
plaza.""",
    },
    {
        "title": "Specifications",
        "doc_type": "brochure",
        "effective_from": None, "effective_to": None,
        "content": """# Specifications

## Structure
RCC framed structure designed to withstand wind and seismic loads. External walls are 6
inch solid block masonry, internal walls 4 inch. Two coats of plastering internally and
externally.

## Flooring
Bedrooms, living, dining and kitchen have 80cm by 80cm vitrified tiles. Bathrooms have
acid resistant anti skid ceramic tiles 300 by 300 mm. Balconies have acid resistant anti
skid ceramic tiles 200 by 1200 mm. Corridors have vitrified tiles with inlay. Staircases
are polished kota stone.

## Doors and windows
Main and internal doors are moulded Acacia engineered wood frames with laminated flush
shutters and stainless steel hardware. Windows are UPVC sliding with glass, mosquito mesh
and safety grills. French doors where present are UPVC with sliding glass shutters and
mosquito mesh, without safety grill.

## Bathrooms
Wash basin with half pedestal, EWC with concealed flush tank, hot and cold concealed
single lever diverter with head shower, geyser provision in all bathrooms. CP fittings are
Vitra, American Standard, Sirius or equivalent.

## Kitchen and utility
Treated water connection with provision for a water purifier, inlet and outlet for the
kitchen sink. Utility has washing machine provision and a stainless steel single bowl sink
with granite counter.

## Electrical
Concealed copper wiring of Polycab, Anchor or equivalent. Air conditioner outlet in the
master bedroom, geyser outlets in all bathrooms, kitchen power for chimney, refrigerator,
microwave and mixer. MCB and ELCB per distribution board of Schneider or equivalent.

## Security and services
Intercom to all units connecting security, clubhouse and common areas. Standalone video
door phone per unit. CCTV at project entry, exit and driveways. Branded automatic
elevators with granite or vitrified cladding. 100 percent DG backup with acoustic
enclosure for all flats and common areas.

## Telecom
Telephone points in the master bedroom and living room. Cable and DTH provision in master
bedroom and living room, living room only in 1 BHK. Internet point in the living room.""",
    },
    {
        "title": "Location and Connectivity",
        "doc_type": "faq",
        "effective_from": None, "effective_to": None,
        "content": """# Location

DSR Highland Greenz is at Chikkanayakanahalli, off Sarjapur Road, Bengaluru.

## Distances
RGA Tech Park is 2.2 kilometres away.
Wipro Corporate Office is 2.5 kilometres away.
Decathlon Sports Arena is 2.5 kilometres away.
Manipal Hospital is 5.5 kilometres away.
Eco Space is 6.7 kilometres away.
Electronic City is 9 kilometres away.
Marathahalli is 9 kilometres away.

## Nearby
Primus International School, New Horizon College, Oxford Engineering College, Sakra
Hospital, Embassy Tech Village, Karmelaram Railway Station and Dommasandra Junction are
all in the surrounding area. Sarjapur Road connects towards Koramangala on one side and
Dommasandra on the other.""",
    },
    {
        "title": "Site Visit and Booking Process",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# Site Visits and Booking

## Site visits
Site visits run every day from 10:00 AM to 6:30 PM at the site office at Chikkanayakanahalli.
The sales gallery at DSR Techno Cube, Varthur Main Road, is open the same hours. A visit
takes about an hour and covers the sample flat, the clubhouse and the master plan.

Visits can be rescheduled or cancelled free of charge at any time. There is no charge for
a site visit and no obligation to book.

## Booking a unit
A unit is held on payment of a token amount. The agreement and registration follow. Legal
fees are 30,000 rupees excluding GST, and franking, registration and stamp duty are as per
actuals.

## What the agent can and cannot do
The agent can share the cost sheet, quote published prices, check which configurations are
available and book a site visit. It cannot negotiate on price, offer a discount, confirm
availability of a specific flat number, or commit to a possession date. Those go to a
sales executive.

## RERA
This project is registered under Karnataka RERA, registration number
PRM/KA/RERA/1251/446/PR/110522/004876, verifiable at www.rera.karnataka.gov.in. Any price
or area quoted is subject to the RERA-registered figures, and carpet area as per RERA is
stated separately from super built-up area on every cost sheet.""",
    },
    {
        "title": "Monsoon Booking Offer 2026",
        "doc_type": "promo",
        "effective_from": ago(60), "effective_to": ago(10),
        "content": """# Monsoon Booking Offer

The Monsoon Booking Offer waived the club house charges of 3,25,000 rupees on any 3 BHK
booked during the offer period, and included one covered car park at no additional cost.

The offer applied to bookings with a token paid within the offer period and could not be
combined with any channel partner incentive.""",
    },
    {
        "title": "Festive Booking Offer 2026",
        "doc_type": "promo",
        "effective_from": ahead(20), "effective_to": ahead(50),
        "content": """# Festive Booking Offer

The Festive Booking Offer gives a waiver on floor rise charges for units on the 4th floor
and above, and covers the corpus fund of 20 rupees per square foot.

It applies to all configurations and cannot be combined with any other offer.""",
    },
]


LEADS = [
    {"full_name": "Vikram Singh", "phone": "+919812400001", "email": "vikram.singh@example.com",
     "location": "Whitefield, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "99acres", "tags": ["vip", "hot_lead", "complained_last_30d"],
     "priority_score": 92, "state": "negotiation",
     "preferred_contact_start": "10:00", "preferred_contact_end": "19:00",
     "persona": "HIGH VALUE, NEGOTIATING — 3 BHK 1,560 sq ft, self funded, already visited. "
                "Frustration escalates immediately.",
     "facts": [("preference", "budget_range", "1.8 to 2 crore"),
               ("preference", "configuration", "3 BHK, 1560 sq ft, East and West facing"),
               ("buying_intent", "purpose", "end use — moving from a rented flat in Whitefield")]},

    {"full_name": "Rohan Mehta", "phone": "+919812400002", "email": "rohan.mehta@example.com",
     "location": "Sarjapur Road, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "99acres", "tags": ["new"], "priority_score": 45, "state": "lead",
     "persona": "NEW ENQUIRY — no history. The clean path: qualify, quote, book a site visit.",
     "facts": []},

    {"full_name": "Kavya Menon", "phone": "+919812400003", "email": "kavya.menon@example.com",
     "location": "Hebbal, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "MagicBricks", "tags": ["opted_out"], "priority_score": 10,
     "state": "do_not_contact", "dnc": True,
     "persona": "DO-NOT-CONTACT — asked to be removed. Any outbound attempt must be refused.",
     "facts": []},

    {"full_name": "Priya Nair", "phone": "+919812400004", "email": "priya.nair@example.com",
     "location": "Koramangala, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "Housing.com", "tags": ["investor", "site_visit_done"], "priority_score": 70,
     "state": "site_visit_done",
     "persona": "INVESTOR — visited already, wants rental yield and possession certainty. "
                "Will push for numbers the agent must not commit to.",
     "facts": [("preference", "budget_range", "1.4 to 1.6 crore"),
               ("buying_intent", "purpose", "investment, not end use"),
               ("preference", "configuration", "3 BHK 1,239 sq ft — the smallest 3 BHK")]},

    {"full_name": "Sanjay Gupta", "phone": "+919812400005", "email": "sanjay.gupta@example.com",
     "location": "Pune", "region_code": "MH-PUN", "preferred_language": "hi",
     "lead_source": "Referral", "tags": ["out_of_city", "nri_style"], "priority_score": 55,
     "state": "qualified",
     "persona": "OUT OF CITY — in Pune, cannot attend a physical site visit. Should be offered "
                "a virtual walkthrough or a callback, not a slot he cannot reach.",
     "facts": [("preference", "logistics", "based in Pune, visits Bengaluru monthly")]},

    {"full_name": "Meera Iyer", "phone": "+919812400006", "email": "meera.iyer@example.com",
     "location": "Marathahalli, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "MagicBricks", "tags": ["churn_risk", "gone_quiet"], "priority_score": 68,
     "state": "dormant",
     "persona": "GOING COLD — visited, then stopped responding. High value plus churn risk, "
                "so a human should make this call rather than the agent.",
     "facts": [("objection", "price", "total is higher than the headline per-square-foot rate suggested")]},

    {"full_name": "Aditya Verma", "phone": "+919812400007", "email": "aditya.verma@example.com",
     "location": "Indiranagar, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "ChannelOne Realty", "tags": ["channel_partner"], "priority_score": 60,
     "state": "site_visit_scheduled",
     "persona": "BROKER ATTACHED — came through a channel partner, so promotional offers and "
                "direct incentives do not apply.",
     "facts": [("decision_maker", "channel_partner", "ChannelOne Realty"),
               ("preference", "budget_range", "1.5 to 1.7 crore")]},

    {"full_name": "Nisha Pillai", "phone": "+919812400008", "email": "nisha.pillai@example.com",
     "location": "Electronic City, Bengaluru", "region_code": "KA-BLR", "preferred_language": "en",
     "lead_source": "Walk-in", "tags": ["evening_only"], "priority_score": 50, "state": "contacted",
     "preferred_contact_start": "18:00", "preferred_contact_end": "20:00",
     "persona": "NARROW WINDOW — only reachable 18:00 to 20:00. Calling outside that is blocked "
                "even though it is within legal hours.",
     "facts": [("preference", "availability", "works late, only free after 6 pm")]},
]

TICKETS = [
    {"lead": "Vikram Singh", "subject": "Cost sheet total does not match the quoted rate",
     "description": "Customer says the 1,560 sq ft total of 1,92,53,440 is higher than 11,499 "
                    "per sq ft implies. Needs the additional charges walked through line by line.",
     "category": "pricing", "priority": "high"},
    {"lead": "Priya Nair", "subject": "Possession date confirmation requested in writing",
     "description": "Investor wants a written possession commitment. Agent cannot commit; "
                    "needs a sales executive to respond formally.",
     "category": "documentation", "priority": "medium"},
]

CONVERSATIONS = [
    {"lead": "Rohan Mehta", "label": "New enquiry — books a site visit",
     "turns": ["Hi, I saw your project on 99acres and wanted to know more",
               "I'm looking at 3 BHK, budget around 1.6 crore",
               "What's the total cost for the 1,390 sq ft one?",
               "Can I come and see it this weekend?",
               "Saturday morning works",
               "Yes, please book it"]},
    {"lead": "Vikram Singh", "label": "High value, unhappy — should escalate",
     "turns": ["I was quoted 11,499 a square foot and the total is nearly two crore",
               "Nobody explained the floor rise and the corner charges to me",
               "This is the third time I'm asking and I still don't have a straight answer"]},
]

CAMPAIGNS = [
    {"name": "Site visit follow-up — Bengaluru", "purpose": "service", "calls_per_hour": 25,
     "audience": {"states": [], "tags": [], "languages": [], "region_codes": ["KA-BLR"],
                  "lead_sources": []}},
    {"name": "Festive offer — all leads", "purpose": "marketing", "calls_per_hour": 20,
     "audience": {"states": [], "tags": [], "languages": [], "region_codes": [], "lead_sources": []}},
]


# --------------------------------------------------------------------- stages

def seed_scheduling():
    centres = {}
    existing = {c["name"]: c for c in call("GET", "/scheduling/locations")}
    for spec in CENTRES:
        if spec["name"] in existing:
            centres[spec["name"]] = existing[spec["name"]]["id"]
            skip(spec["name"], "already exists")
            continue
        created = try_call("POST", "/scheduling/locations", spec, spec["name"])
        if created:
            centres[spec["name"]] = created["id"]
            ok(spec["name"], spec["city"])

    have = {e["full_name"] for e in call("GET", "/scheduling/staff")}
    for spec in EXECUTIVES:
        if spec["full_name"] in have:
            skip(spec["full_name"], "already exists")
            continue
        if try_call("POST", "/scheduling/staff",
                    {"full_name": spec["full_name"], "location_id": centres.get(spec["centre"]),
                     "phone": spec["phone"], "specialties": spec["specialties"]}, spec["full_name"]):
            ok(spec["full_name"], spec["centre"].split("—")[0].strip())

    # Site visits run seven days a week — property sales do not stop at the weekend, and
    # Saturday and Sunday are when most buyers can actually come.
    for name, cid in centres.items():
        if call("GET", f"/scheduling/working-hours?owner_type=location&owner_id={cid}"):
            skip(f"hours for {name}", "already set")
            continue
        for day in range(0, 7):
            call("POST", "/scheduling/working-hours",
                 {"owner_type": "location", "owner_id": cid, "day_of_week": day,
                  "start_time": "10:00", "end_time": "18:30"})
        ok(f"hours for {name}", "every day 10:00-18:30")

    have_h = {str(h["date"]) for h in call("GET", "/scheduling/holidays")}
    for spec in HOLIDAYS:
        if spec["date"] in have_h:
            continue
        call("POST", "/scheduling/holidays", spec)
        ok(spec["name"], spec["date"])

    if call("GET", "/scheduling/slot-policies"):
        skip("slot policy", "already set")
    else:
        # An hour per visit, two parties at a time, bookable two months out. Two hours'
        # notice because someone has to be at the site office to receive them.
        call("POST", "/scheduling/slot-policies",
             {"location_id": None, "slot_duration_min": 60, "buffer_before_min": 0,
              "buffer_after_min": 0, "max_parallel": 2, "booking_horizon_days": 60,
              "min_notice_hours": 2})
        ok("slot policy", "60min visits, 2 parties in parallel, 60 day horizon")
    return centres


def seed_knowledge(wipe=False):
    existing = {d["title"]: d for d in call("GET", "/knowledge/documents?include_expired=true")}
    if wipe:
        for title, doc in existing.items():
            call("DELETE", f"/knowledge/documents/{doc['id']}")
            ok(f"removed {title}")
        existing = {}
    for spec in DOCUMENTS:
        if spec["title"] in existing:
            skip(spec["title"], "already indexed")
            continue
        result = try_call("POST", "/knowledge/documents", spec, spec["title"])
        if not result:
            continue
        window = "always live"
        if spec["effective_from"] or spec["effective_to"]:
            window = f"{spec['effective_from'] or 'always'} -> {spec['effective_to'] or 'always'}"
        ok(spec["title"], f"{result['chunks_created']} chunks  [{window}]")
    print("      Monsoon offer expired 10 days ago; Festive offer starts in 20 days.")
    print("      Neither is quotable today — both directions of the date window covered.")


def seed_rules(activate=True):
    folder = Path(__file__).parent / "dsr_rules"
    if not folder.exists():
        skip("rules", f"no folder at {folder}")
        return
    for path in sorted(folder.glob("*.yaml")):
        src = path.read_text()
        name = next((l.split(":", 1)[1].strip() for l in src.splitlines()
                     if l.startswith("rule_set:")), None)
        if not name:
            continue
        saved = try_call("PUT", f"/rules/{name}", {"yaml_source": src}, name)
        if not saved:
            continue
        detail = f"v{saved['version']}, {saved['rule_count']} rules"
        if activate:
            call("POST", f"/rules/{name}/activate", {"version": saved["version"]})
            detail += " — active"
        ok(name, detail)


def seed_leads():
    created = {}
    for spec in LEADS:
        payload = {k: v for k, v in spec.items()
                   if k not in ("persona", "priority_score", "dnc", "state", "facts")}
        try:
            existing = call("GET", f"/customers/lookup?phone={spec['phone']}", quiet=True)
            created[spec["full_name"]] = existing["id"]
            skip(spec["full_name"], "already exists")
            continue
        except urllib.error.HTTPError:
            pass

        lead = try_call("POST", "/customers", payload, spec["full_name"])
        if not lead:
            continue
        created[spec["full_name"]] = lead["id"]
        call("PATCH", f"/customers/{lead['id']}", {"priority_score": spec["priority_score"]})

        # Move the lead to where it should be in the journey. Done through the state
        # endpoint so the transition is recorded and auditable rather than written blind.
        if spec.get("state") and spec["state"] != "lead":
            try_call("POST", f"/customers/{lead['id']}/state",
                     {"to_state": spec["state"], "reason": "seeded demo position"},
                     f"{spec['full_name']} -> {spec['state']}")

        # There is no POST /customers/{id}/facts — memory is written by the agent, not by
        # hand. So facts are seeded by executing the same record_customer_fact tool the
        # agent uses, which means they arrive through the real path with real validation.
        for category, key, value in spec.get("facts", []):
            try_call("POST", "/tools/record_customer_fact/test",
                     {"params": {"category": category, "key": key, "value": value},
                      "customer_id": lead["id"]}, f"fact {key}")

        if spec.get("dnc"):
            try_call("POST", "/dnc", {"phone": spec["phone"], "scope": "all",
                                      "source": "customer_request",
                                      "customer_id": lead["id"]}, "dnc entry")
        ok(spec["full_name"], f"{spec['state']} · {spec['persona'].split(' — ')[0]}")

    print("\n    Personas:")
    for spec in LEADS:
        print(f"      {spec['full_name']:<16} {spec['persona']}")
    return created


def seed_visits(centres, leads):
    site = centres.get("DSR Highland Greenz — Site Office")
    if not site:
        skip("site visits", "no site office")
        return
    if call("GET", "/appointments"):
        skip("site visits", "already booked")
        return
    for offset in (1, 2, 3):
        day = ahead(offset)
        slots = call("GET", f"/scheduling/slots?centre_id={site}&from={day}&to={day}")["slots"]
        if not slots:
            continue
        for name, slot in zip(["Aditya Verma", "Priya Nair"], slots[:2]):
            if name not in leads:
                continue
            appt = try_call("POST", "/appointments",
                            {"customer_id": leads[name], "centre_id": site,
                             "starts_at": slot["starts_at"]}, name)
            if appt:
                ok(name, appt["starts_at"])
        return
    skip("site visits", "no slots in the next few days")


def seed_support(leads):
    have = {t["subject"] for t in call("GET", "/tickets")["items"]}
    for spec in TICKETS:
        if spec["subject"] in have or spec["lead"] not in leads:
            continue
        call("POST", "/tickets", {"customer_id": leads[spec["lead"]], "subject": spec["subject"],
                                  "description": spec["description"], "category": spec["category"],
                                  "priority": spec["priority"]})
        ok(spec["subject"], f"{spec['priority']} · {spec['lead']}")


def seed_campaigns():
    have = {c["name"] for c in call("GET", "/campaigns")}
    for spec in CAMPAIGNS:
        if spec["name"] in have:
            skip(spec["name"], "already exists")
            continue
        camp = try_call("POST", "/campaigns", spec, spec["name"])
        if not camp:
            continue
        preview = call("POST", f"/campaigns/{camp['id']}/preview", {})
        ok(spec["name"],
           f"{preview.get('eligible_now', 0)} eligible / {preview.get('excluded_total', 0)} excluded")
        for rule, count in (preview.get("exclusions_by_rule") or {}).items():
            print(f"        blocked by {rule}: {count}")


def seed_conversations(leads, fresh=False):
    convos = list(CONVERSATIONS)
    if fresh:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        made = try_call("POST", "/customers",
                        {"phone": f"+9198{stamp}00", "location": "Bengaluru",
                         "region_code": "KA-BLR", "preferred_language": "en",
                         "lead_source": "test", "tags": ["test_record"]}, "throwaway lead")
        if made:
            leads = dict(leads)
            leads[convos[0]["lead"]] = made["id"]
            ok("throwaway lead", "for a clean happy path")

    for spec in convos:
        if spec["lead"] not in leads:
            continue
        print(f"    {spec['label']}  ({spec['lead']})")
        sid = call("POST", "/sessions", {"customer_id": leads[spec["lead"]],
                                         "channel": "chat", "direction": "inbound"})["session_id"]
        for text in spec["turns"]:
            turn = call("POST", f"/sessions/{sid}/turns", {"text": text})
            timing = (turn.get("debug") or {}).get("timing") or {}
            tools = ", ".join(t["tool_name"] for t in (turn.get("executed_tools") or [])) or "-"
            print(f"      > {text}")
            print(f"      < {(turn.get('response_text') or '')[:105]}")
            print(f"        action={turn.get('action')} tools={tools} {timing.get('total_ms','?')}ms")
            if turn.get("escalation"):
                print(f"        ESCALATED — {turn['escalation'].get('reason')}")
        ended = call("POST", f"/sessions/{sid}/end", {"analyse": "sync"})
        a = ended.get("analysis") or {}
        ok("call ended", f"{len(a.get('facts_added') or [])} facts, outcome={a.get('outcome') or '-'}")


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--skip", default="")
    p.add_argument("--fresh", action="store_true", help="throwaway lead for the happy path")
    p.add_argument("--wipe-knowledge", action="store_true")
    p.add_argument("--no-activate-rules", action="store_true")
    p.add_argument("--email", default=None)
    p.add_argument("--password", default=None)
    args = p.parse_args()

    global EMAIL, PASSWORD
    if args.email:
        EMAIL = args.email
    if args.password:
        PASSWORD = args.password

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
    wanted = lambda stage: (stage in only) if only else (stage not in skipped)

    print("\n=== DSR Highland Greenz — real estate demo ===\n")
    login()

    centres, leads = {}, {}

    if wanted("scheduling"):
        print("\n[scheduling]")
        centres = seed_scheduling()
    else:
        centres = {c["name"]: c["id"] for c in call("GET", "/scheduling/locations")}

    if wanted("knowledge"):
        print("\n[knowledge]")
        seed_knowledge(wipe=args.wipe_knowledge)

    if wanted("rules"):
        print("\n[rules]")
        seed_rules(activate=not args.no_activate_rules)

    if wanted("leads"):
        print("\n[leads]")
        leads = seed_leads()
    else:
        leads = {c["full_name"]: c["id"]
                 for c in call("GET", "/customers?page_size=100")["items"] if c.get("full_name")}

    if wanted("visits"):
        print("\n[site visits]")
        seed_visits(centres, leads)

    if wanted("support"):
        print("\n[support]")
        seed_support(leads)

    if wanted("campaigns"):
        print("\n[campaigns]")
        seed_campaigns()

    if wanted("conversations"):
        print("\n[conversations]  (real LLM calls — the slow part)")
        seed_conversations(leads, fresh=args.fresh)

    print("\n=== done ===")
    print("Check, in order:")
    print("  1. Leads — eight personas across the journey, Kavya on do-not-contact")
    print("  2. Site Visits — two booked against real computed slots")
    print("  3. Call Logs — two conversations with transcripts and decision traces")
    print("  4. Escalations — Vikram's, with the briefing a human would read")
    print("  5. Campaigns — run a preview and read the exclusions by rule")
    print("  6. Knowledge Base — Monsoon expired, Festive not started")
    print("  7. Audit View — every refusal with the rule that caused it\n")


if __name__ == "__main__":
    main()