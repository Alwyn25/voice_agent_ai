#!/usr/bin/env python3
"""
Seed the "Motorworks Auto Care" demo firm.

Everything is created through the REST API rather than direct database writes, so the
records go through the same validation, state machine and audit trail the dashboard and
the agent use. If a step fails here, it would have failed in the UI too — which is the
point.

Run from inside the core container:

    docker compose exec core python app/packs/real_estate_india/seeds/seed_demo.py
    docker compose exec core python app/packs/real_estate_india/seeds/seed_demo.py --only knowledge,rules
    docker compose exec core python app/packs/real_estate_india/seeds/seed_demo.py --skip conversations
    docker compose exec core python app/packs/real_estate_india/seeds/seed_demo.py --wipe-knowledge

Stages: scheduling, knowledge, rules, customers, appointments, support, campaigns,
conversations.

The conversations stage is last and costs real LLM calls — it drives two short chats
through the engine so Conversations, Insights and the Work Queue have something in them.
Without it the dashboard is correct but empty, which reads as broken.

Stdlib only: no extra dependencies in the image.
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


def ago(days):
    return (TODAY - timedelta(days=days)).isoformat()


def ahead(days):
    return (TODAY + timedelta(days=days)).isoformat()


# --------------------------------------------------------------------- transport


def call(method, path, payload=None, quiet=False):
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            body = res.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        if not quiet:
            print(f"    ! {method} {path} -> {e.code} {detail[:220]}")
        raise
    except urllib.error.URLError as e:
        print(f"\n  Cannot reach {BASE} — is core running?  ({e.reason})")
        sys.exit(1)


def login():
    global TOKEN
    out = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})
    TOKEN = out["access_token"]
    me = call("GET", "/auth/me")
    print(f"  signed in as {EMAIL} (role: {me['role']}, tenant {me['tenant_id'][:8]})")


def try_call(method, path, payload=None, label=""):
    """Non-fatal POST. A single rejected row should not abort the remaining stages."""
    try:
        return call(method, path, payload)
    except urllib.error.HTTPError:
        print(f"      skipped {label or path} — see the error above")
        return None


def ok(label, detail=""):
    print(f"    + {label}{('  ' + detail) if detail else ''}")


def skip(label, why):
    print(f"    · {label} — {why}")


# --------------------------------------------------------------------- data


CENTRES = [
    {"name": "Motorworks Vapi", "city": "Vapi", "region_code": "GJ-VAPI",
     "address": "GIDC Char Rasta, Vapi", "phone": "+912602400100", "timezone": "Asia/Kolkata"},
    {"name": "Motorworks Valsad", "city": "Valsad", "region_code": "GJ-VALSAD",
     "address": "Station Road, Valsad", "phone": "+912632240200", "timezone": "Asia/Kolkata"},
    {"name": "Motorworks Silvassa", "city": "Silvassa", "region_code": "GJ-SILV",
     "address": "Naroli Road, Silvassa", "phone": "+912602630300", "timezone": "Asia/Kolkata"},
]

EXECUTIVES = [
    {"full_name": "Anil Rathod", "centre": "Motorworks Vapi",
     "specialties": ["engine", "diagnostics"], "email": "anil@motorworks.example"},
    {"full_name": "Farida Sheikh", "centre": "Motorworks Vapi",
     "specialties": ["bodywork", "paint"], "email": "farida@motorworks.example"},
    {"full_name": "Jignesh Desai", "centre": "Motorworks Valsad",
     "specialties": ["engine", "electrical"], "email": "jignesh@motorworks.example"},
    {"full_name": "Meera Nair", "centre": "Motorworks Silvassa",
     "specialties": ["routine"], "email": "meera@motorworks.example"},
]

HOLIDAYS = [
    {"date": ahead(14), "name": "Independence Day"},
    {"date": ahead(41), "name": "Founders Day (all centres closed)"},
]

# Two of these are deliberately outside their window: one expired, one not started.
# Both directions matter — content that becomes quotable too early is the same
# compliance failure as content that stays quotable too long, and almost nobody tests
# the second one.
DOCUMENTS = [
    {
        "title": "Service Packages and Pricing 2026",
        "doc_type": "pricing",
        "effective_from": None, "effective_to": None,
        "content": """# Service Packages

Motorworks Auto Care runs multi-brand car service centres in Vapi, Valsad and Silvassa.
All prices include labour and taxes unless stated otherwise.

## Essential Service
Essential Service costs 3,200 rupees. It covers engine oil and oil filter replacement,
air filter cleaning, brake inspection, tyre pressure check, battery health check, top-up
of all fluids and an exterior wash. It takes about three hours.

## Comprehensive Service
Comprehensive Service costs 7,500 rupees. It includes everything in Essential Service
plus air filter replacement, fuel filter replacement, spark plug inspection, brake pad
cleaning, wheel alignment, wheel balancing, an air-conditioning performance check and
interior vacuuming. It takes about six hours and is recommended once a year.

## Premium Care
Premium Care costs 25,000 rupees as an annual package. It covers three Comprehensive
Services in a year, unlimited fluid top-ups, free pickup and drop on every visit,
priority booking, a dedicated service advisor and roadside assistance within 50
kilometres of any centre.

## Annual Maintenance Contract
The Annual Maintenance Contract costs 18,000 rupees for petrol cars and 22,000 rupees
for diesel cars. It covers all scheduled services for twelve months. It excludes
accident repairs and consumables damaged by misuse.

## Individual items
Wheel alignment alone is 800 rupees. Wheel balancing is 600 rupees for four wheels.
Air-conditioning gas refill is 2,400 rupees. Headlight restoration is 1,500 rupees.
Ceramic coating starts at 12,000 rupees depending on vehicle size.""",
    },
    {
        "title": "Booking, Rescheduling and Cancellation Policy",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# Booking and Cancellation

## Booking
Service slots run from 9:00 AM to 6:00 PM, Monday to Saturday. All centres are closed on
Sundays. Same-day slots must be booked before 11:00 AM.

## Rescheduling
An appointment can be rescheduled free of charge up to 12 hours before the slot. Inside
12 hours a rescheduling fee of 300 rupees applies. After two reschedules a service
advisor must confirm the booking.

## Cancellation
Cancellation is free more than 24 hours before the appointment. Cancelling within 24
hours carries a fee of 500 rupees. Not arriving without cancelling carries the same 500
rupee fee, added to the next invoice.

## Late arrival
Arriving more than 45 minutes after the slot may mean the service moves to the next
available slot, because the bay will have been released.

## Refunds
Prepaid packages are refundable pro-rata less 10 percent administration charges. Refunds
reach the original payment method within 14 working days.""",
    },
    {
        "title": "Warranty and Workmanship Guarantee",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# Warranty

## Workmanship guarantee
All labour carries a guarantee of 30 days or 1,000 kilometres, whichever comes first. If
the same fault returns inside that window the rework is free.

## Parts warranty
Genuine parts fitted by Motorworks carry the manufacturer warranty, usually 12 months.
Aftermarket parts carry a 6 month warranty. Parts supplied by the customer carry no
warranty, and the workmanship guarantee does not extend to them.

## Not covered
The warranty does not cover damage from accidents, misuse, water ingress or
modification. It does not cover consumables such as wiper blades and bulbs beyond the
first 15 days. It does not cover faults arising from a repair the customer declined.

## Claims
A claim needs the original invoice number. The vehicle must be inspected at a Motorworks
centre. A decision is given within 48 hours of inspection, and a rejected claim can be
reviewed by the service manager on request.""",
    },
    {
        "title": "Pickup and Drop Service",
        "doc_type": "faq",
        "effective_from": None, "effective_to": None,
        "content": """# Pickup and Drop

## Where it is available
Pickup and drop covers Vapi, Valsad, Daman and Silvassa. It is not available outside
these areas; customers elsewhere are welcome to drive in.

## Cost
Pickup and drop costs 400 rupees each way. It is free for Premium Care members and free
on any service billed above 6,000 rupees.

## How it works
A driver collects the vehicle at an agreed time window. A condition report with
photographs is shared before the vehicle leaves. The vehicle is returned washed once the
service is paid for.

## Requirements
Someone above 18 must hand over the vehicle. Insurance and a valid pollution
certificate must be current. Fuel below one quarter may attract a refuelling charge at
cost.""",
    },
    {
        "title": "Centres and Working Hours",
        "doc_type": "faq",
        "effective_from": None, "effective_to": None,
        "content": """# Centres

## Vapi
GIDC Char Rasta. Monday to Saturday, 9:00 AM to 6:00 PM. Eight bays, handling all work
including bodywork and paint.

## Valsad
Station Road. Monday to Saturday, 9:00 AM to 6:00 PM. Four bays, mechanical work only —
bodywork transfers to Vapi.

## Silvassa
Naroli Road. Monday to Saturday, 9:00 AM to 5:30 PM. Three bays, routine servicing.

## Closures
All centres close on Sundays and on national holidays. Roadside assistance runs on those
days for Premium Care members only.""",
    },
    {
        "title": "Complaints and Escalation",
        "doc_type": "policy",
        "effective_from": None, "effective_to": None,
        "content": """# Complaints

## Raising one
A complaint can be raised on a call, at any centre, or by email. Every complaint gets a
reference number the same day.

## Response times
A first response comes within 24 hours. Resolution is targeted within 5 working days.
Anything involving a safety concern is treated as urgent and reviewed the same day by
the service manager.

## Repeat repairs
The same fault returning within 30 days of a repair is treated as a repeat repair. Repeat
repairs are inspected free of charge and reviewed by the service manager rather than the
booking team.

## Escalation path
An unresolved complaint moves to the centre manager after 5 working days and to the
regional service head after 10. A customer may ask for escalation at any point and it is
never refused.""",
    },
    {
        "title": "Monsoon Service Camp 2026",
        "doc_type": "promo",
        "effective_from": ago(60), "effective_to": ago(10),
        "content": """# Monsoon Service Camp

The Monsoon Service Camp gives 20 percent off Comprehensive Service, bringing it to
6,000 rupees. It includes a free wiper blade replacement, a free underbody inspection and
free battery terminal cleaning.

The camp applies to bookings made during the camp period at the Vapi and Valsad centres
only, and cannot be combined with Premium Care or the Annual Maintenance Contract.""",
    },
    {
        "title": "Diwali Service Offer 2026",
        "doc_type": "promo",
        "effective_from": ahead(20), "effective_to": ahead(50),
        "content": """# Diwali Service Offer

The Diwali Service Offer includes a free interior deep clean worth 2,000 rupees with any
Comprehensive Service, and 15 percent off ceramic coating.

It applies at all three centres and cannot be combined with any other discount.""",
    },
]

# Named personas, so a scenario behaves the same way every time it is run.
CUSTOMERS = [
    {
        "full_name": "Priya Shah", "phone": "+919812300001", "email": "priya.shah@example.com",
        "location": "Vapi", "region_code": "GJ-VAPI", "preferred_language": "en",
        "lead_source": "walk_in", "tags": ["vip", "amc_holder", "complained_last_30d"],
        "persona": "ANGRY VIP — high priority, complained recently. Triggers the "
                   "vip_negative_sentiment and third_complaint_this_month rules.",
        "priority_score": 92,
        "preferred_contact_start": "10:00", "preferred_contact_end": "19:00",
    },
    {
        "full_name": "Rakesh Patel", "phone": "+919812300002", "email": "rakesh.patel@example.com",
        "location": "Valsad", "region_code": "GJ-VALSAD", "preferred_language": "gu",
        "lead_source": "referral", "tags": ["new"],
        "persona": "NEW LEAD — Gujarati speaker, no history. Clean happy-path booking.",
        "priority_score": 30,
    },
    {
        "full_name": "Sunita Mehta", "phone": "+919812300003", "email": "sunita.mehta@example.com",
        "location": "Vapi", "region_code": "GJ-VAPI", "preferred_language": "hi",
        "lead_source": "campaign", "tags": ["opted_out"],
        "persona": "DO-NOT-CONTACT — on the DNC list. Any outbound attempt must be refused.",
        "priority_score": 10,
        "dnc": True,
    },
    {
        "full_name": "Imran Qureshi", "phone": "+919812300004", "email": "imran.q@example.com",
        "location": "Silvassa", "region_code": "GJ-SILV", "preferred_language": "en",
        "lead_source": "google_ads", "tags": ["premium_care", "serviced_last_7d"],
        "persona": "REPEAT CUSTOMER — Premium Care member, serviced recently. "
                   "Upsell calls should be refused by recent_purchase_no_upsell.",
        "priority_score": 78,
    },
    {
        "full_name": "Deepak Joshi", "phone": "+919812300005", "email": "deepak.joshi@example.com",
        "location": "Surat", "region_code": "GJ-SURAT", "preferred_language": "en",
        "lead_source": "website", "tags": ["new"],
        "persona": "OUT OF AREA — Surat is outside the pickup zone. "
                   "pickup_area_check and out_of_area_no_pickup_promo should fire.",
        "priority_score": 40,
    },
    {
        "full_name": "Anita Rao", "phone": "+919812300006", "email": "anita.rao@example.com",
        "location": "Vapi", "region_code": "GJ-VAPI", "preferred_language": "en",
        "lead_source": "referral", "tags": ["churn_risk", "amc_holder"],
        "persona": "CHURN RISK — high priority plus churn_risk tag, so churn_risk_escalates "
                   "schedules a human call instead of letting the agent handle it.",
        "priority_score": 71,
    },
    {
        "full_name": "Mahesh Solanki", "phone": "+919812300007", "email": "mahesh.s@example.com",
        "location": "Valsad", "region_code": "GJ-VALSAD", "preferred_language": "gu",
        "lead_source": "walk_in", "tags": ["fleet_account"],
        "persona": "FLEET ACCOUNT — handled by an account manager, so excluded from "
                   "promotional campaigns by fleet_account_manager_only.",
        "priority_score": 65,
    },
    {
        "full_name": "Leena Bhatt", "phone": "+919812300008", "email": "leena.bhatt@example.com",
        "location": "Vapi", "region_code": "GJ-VAPI", "preferred_language": "en",
        "lead_source": "campaign", "tags": ["evening_only"],
        "persona": "NARROW CONTACT WINDOW — only reachable 18:00-20:00. Calling outside "
                   "that is blocked by customer_preferred_hours even though it is legal.",
        "priority_score": 55,
        "preferred_contact_start": "18:00", "preferred_contact_end": "20:00",
    },
]

TICKETS = [
    {"customer": "Priya Shah", "subject": "Invoice higher than the estimate",
     "description": "Billed 9,100 against an estimate of 7,500. Customer wants the "
                    "difference explained before the next service.",
     "category": "billing", "priority": "high"},
    {"customer": "Imran Qureshi", "subject": "Wiper blades replaced without approval",
     "description": "Wipers were changed during a routine service without being asked. "
                    "Customer accepts the work but wants the process reviewed.",
     "category": "process", "priority": "low"},
]

CONVERSATIONS = [
    {
        "customer": "Rakesh Patel",
        "label": "Happy path — books a service",
        # The last line matters: without an explicit yes the confirmation gate correctly
        # parks the booking and nothing executes. A "happy path" that never confirms is
        # not a happy path.
        "turns": [
            "Hi, I want to get my car serviced",
            "It's a Swift, GJ15 AB 1234",
            "Comprehensive service please, at the Vapi centre",
            "What does that cost?",
            "Fine, the earliest morning slot you have",
            "Yes, please book it",
        ],
    },
    {
        "customer": "Priya Shah",
        "label": "Angry VIP — should escalate",
        "turns": [
            "I got my brakes done at Vapi three weeks ago and the noise is back",
            "Yes, exactly the same problem",
            "This is the third time I'm calling and nobody has called me back. I've had enough.",
        ],
    },
]


# --------------------------------------------------------------------- stages


def seed_scheduling():
    # region_code is String(10) in both the centre and customer tables, and the API does
    # not validate length — an over-long code reaches Postgres and comes back as a 500
    # rather than a 422. Cheaper to catch it here than to read a stack trace.
    for spec in CENTRES:
        code = spec.get("region_code") or ""
        if len(code) > 10:
            print(f"    ! region_code '{code}' is {len(code)} chars; the column holds 10")

    centres = {}
    existing = {c["name"]: c for c in call("GET", "/scheduling/locations")}
    for spec in CENTRES:
        if spec["name"] in existing:
            centres[spec["name"]] = existing[spec["name"]]["id"]
            skip(spec["name"], "already exists")
            continue
        created = try_call("POST", "/scheduling/locations", spec, spec["name"])
        if not created:
            continue
        centres[spec["name"]] = created["id"]
        ok(spec["name"], spec["region_code"])

    existing_execs = {e["full_name"] for e in call("GET", "/scheduling/staff")}
    for spec in EXECUTIVES:
        if spec["full_name"] in existing_execs:
            skip(spec["full_name"], "already exists")
            continue
        if not try_call("POST", "/scheduling/staff", {
            "full_name": spec["full_name"],
            "location_id": centres.get(spec["centre"]),
            "email": spec["email"],
            "specialties": spec["specialties"],
        }, spec["full_name"]):
            continue
        ok(spec["full_name"], spec["centre"])

    # Monday=0 .. Saturday=5. Sunday is left out, which is what makes the
    # sunday_workshop_closed rule observable.
    for name, centre_id in centres.items():
        current = call("GET", f"/scheduling/working-hours?owner_type=location&owner_id={centre_id}")
        if current:
            skip(f"hours for {name}", f"{len(current)} rows already set")
            continue
        closing = "17:30" if "Silvassa" in name else "18:00"
        for day in range(0, 6):
            call("POST", "/scheduling/working-hours", {
                "owner_type": "location", "owner_id": centre_id,
                "day_of_week": day, "start_time": "09:00", "end_time": closing,
            })
        ok(f"hours for {name}", f"Mon-Sat 09:00-{closing}")

    existing_holidays = {str(h["date"]) for h in call("GET", "/scheduling/holidays")}
    for spec in HOLIDAYS:
        if spec["date"] in existing_holidays:
            skip(spec["name"], "already set")
            continue
        call("POST", "/scheduling/holidays", spec)
        ok(spec["name"], spec["date"])

    if call("GET", "/scheduling/slot-policies"):
        skip("slot policy", "already set")
    else:
        call("POST", "/scheduling/slot-policies", {
            "location_id": None, "slot_duration_min": 60, "buffer_before_min": 0,
            "buffer_after_min": 15, "max_parallel": 2,
            "booking_horizon_days": 45, "min_notice_hours": 2,
        })
        ok("slot policy", "60min slots, 15min buffer, 2 in parallel")

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
        result = try_call("POST", "/knowledge/documents", {
            "title": spec["title"], "doc_type": spec["doc_type"],
            "content": spec["content"],
            "effective_from": spec["effective_from"], "effective_to": spec["effective_to"],
        }, spec["title"])
        if not result:
            continue
        window = "always live"
        if spec["effective_from"] or spec["effective_to"]:
            window = f"{spec['effective_from'] or 'always'} -> {spec['effective_to'] or 'always'}"
        ok(spec["title"], f"{result['chunks_created']} chunks  [{window}]")

    print("      Monsoon camp expired 10 days ago — rewind 30 days to reach it.")
    print("      Diwali offer starts in 20 days — not quotable yet. Both directions covered.")


def seed_rules(activate=True):
    folder = Path(__file__).parent / "demo_rules"
    if not folder.exists():
        skip("rules", f"no folder at {folder}")
        return
    for path in sorted(folder.glob("*.yaml")):
        yaml_source = path.read_text()
        name = next((line.split(":", 1)[1].strip() for line in yaml_source.splitlines()
                     if line.startswith("rule_set:")), None)
        if not name:
            skip(path.name, "no rule_set: line")
            continue
        saved = call("PUT", f"/rules/{name}", {"yaml_source": yaml_source})
        detail = f"v{saved['version']}, {saved['rule_count']} rules"
        if activate:
            call("POST", f"/rules/{name}/activate", {"version": saved["version"]})
            detail += " — active"
        ok(name, detail)


def seed_customers():
    created = {}
    for spec in CUSTOMERS:
        payload = {k: v for k, v in spec.items()
                   if k not in ("persona", "priority_score", "dnc")}
        try:
            existing = call("GET", f"/customers/lookup?phone={spec['phone']}", quiet=True)
            created[spec["full_name"]] = existing["id"]
            skip(spec["full_name"], "already exists")
        except urllib.error.HTTPError:
            customer = try_call("POST", "/customers", payload, spec["full_name"])
            if not customer:
                continue
            created[spec["full_name"]] = customer["id"]
            if spec.get("priority_score") is not None:
                call("PATCH", f"/customers/{customer['id']}",
                     {"priority_score": spec["priority_score"]})
            ok(spec["full_name"], spec["persona"].split(" — ")[0])

        if spec.get("dnc"):
            try:
                call("POST", "/dnc", {"phone": spec["phone"], "scope": "all",
                                      "source": "customer_request",
                                      "customer_id": created[spec["full_name"]]}, quiet=True)
                ok("  on the DNC list", "outbound to this number must be refused")
            except urllib.error.HTTPError:
                skip("  DNC entry", "already listed")

    print("\n    Personas:")
    for spec in CUSTOMERS:
        print(f"      {spec['full_name']:<16} {spec['persona']}")
    return created


def seed_appointments(centres, customers):
    vapi = centres.get("Motorworks Vapi")
    if not vapi:
        skip("appointments", "no Vapi centre")
        return

    existing = call("GET", "/appointments")
    if existing:
        skip("appointments", f"{len(existing)} already booked")
        return

    # Ask the engine for real availability rather than inventing a time — a hand-picked
    # timestamp would be rejected the moment it fell outside working hours or the
    # minimum-notice window.
    for offset in (2, 3, 4):
        day = ahead(offset)
        slots = call("GET", f"/scheduling/slots?centre_id={vapi}&from={day}&to={day}")["slots"]
        if not slots:
            continue
        for name, slot in zip(["Rakesh Patel", "Imran Qureshi"], slots[:2]):
            if name not in customers:
                continue
            try:
                appointment = call("POST", "/appointments", {
                    "customer_id": customers[name],
                    "centre_id": vapi,
                    "starts_at": slot["starts_at"],
                })
                ok(f"{name}", f"{appointment['starts_at']}")
            except urllib.error.HTTPError:
                pass
        return
    skip("appointments", "no slots available in the next few days")


def seed_support(customers):
    existing = {t["subject"] for t in call("GET", "/tickets")["items"]}
    for spec in TICKETS:
        if spec["subject"] in existing:
            skip(spec["subject"], "already raised")
            continue
        if spec["customer"] not in customers:
            continue
        call("POST", "/tickets", {
            "customer_id": customers[spec["customer"]],
            "subject": spec["subject"], "description": spec["description"],
            "category": spec["category"], "priority": spec["priority"],
        })
        ok(spec["subject"], f"{spec['priority']} · {spec['customer']}")


def seed_campaigns():
    existing = {c["name"] for c in call("GET", "/campaigns")}
    plan = [
        {"name": "Service due — Vapi and Valsad", "purpose": "service", "calls_per_hour": 30,
         "audience": {"states": [], "tags": [], "languages": [],
                      "region_codes": ["GJ-VAPI", "GJ-VALSAD"], "lead_sources": []}},
        {"name": "Diwali offer — all customers", "purpose": "marketing", "calls_per_hour": 20,
         "audience": {"states": [], "tags": [], "languages": [],
                      "region_codes": [], "lead_sources": []}},
    ]
    for spec in plan:
        if spec["name"] in existing:
            skip(spec["name"], "already exists")
            continue
        campaign = call("POST", "/campaigns", spec)
        preview = call("POST", f"/campaigns/{campaign['id']}/preview", {})
        excluded = preview.get("exclusions_by_rule") or {}
        ok(spec["name"],
           f"{preview.get('eligible_now', 0)} eligible / {preview.get('excluded_total', 0)} excluded")
        for rule, count in excluded.items():
            print(f"        blocked by {rule}: {count}")


def seed_conversations(customers, fresh=False):
    """Drives real chats through the engine so the dashboard has history to show.

    With fresh=True the happy path gets a brand-new customer. Memory is cumulative: facts
    and episodic summaries from earlier runs feed into the next prompt, so re-testing the
    same flow against the same person means each attempt inherits the last one's mistakes.
    A test that cannot be repeated cleanly is not a test.
    """
    if fresh:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        payload = {"full_name": f"Test Customer {stamp}",
                   "phone": f"+9198{stamp}00", "location": "Vapi",
                   "region_code": "GJ-VAPI", "preferred_language": "en",
                   "lead_source": "test", "tags": ["test_record"]}
        created = try_call("POST", "/customers", payload, payload["full_name"])
        if created:
            customers = dict(customers)
            customers[CONVERSATIONS[0]["customer"]] = created["id"]
            ok(payload["full_name"], "throwaway customer for the happy path")

    for spec in CONVERSATIONS:
        name = spec["customer"]
        if name not in customers:
            continue
        print(f"    {spec['label']}  ({name})")
        session = call("POST", "/sessions", {
            "customer_id": customers[name], "channel": "chat", "direction": "inbound",
        })
        sid = session["session_id"]
        escalated = False
        for text in spec["turns"]:
            turn = call("POST", f"/sessions/{sid}/turns", {"text": text})
            timing = (turn.get("debug") or {}).get("timing") or {}
            tools = ", ".join(t["tool_name"] for t in (turn.get("executed_tools") or [])) or "-"
            print(f"      > {text}")
            print(f"      < {(turn.get('response_text') or '')[:110]}")
            print(f"        action={turn.get('action')} tools={tools} "
                  f"{timing.get('total_ms', '?')}ms")
            if turn.get("escalation"):
                escalated = True
                print(f"        ESCALATED — {turn['escalation'].get('reason')}")
        ended = call("POST", f"/sessions/{sid}/end", {"analyse": "sync"})
        analysis = ended.get("analysis") or {}
        ok("call ended",
           f"{len(analysis.get('facts_added') or [])} facts learned, "
           f"outcome={analysis.get('outcome') or '-'}")
        if escalated:
            print("        this customer is now frozen from automated contact "
                  "until the escalation is resolved")


# --------------------------------------------------------------------- main

STAGES = {
    "scheduling": None, "knowledge": None, "rules": None, "customers": None,
    "appointments": None, "support": None, "campaigns": None, "conversations": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated stage list")
    parser.add_argument("--skip", default="", help="comma-separated stage list")
    parser.add_argument("--fresh", action="store_true",
                        help="use a throwaway customer for the happy path instead of Rakesh")
    parser.add_argument("--wipe-knowledge", action="store_true",
                        help="delete existing documents before seeding")
    parser.add_argument("--no-activate-rules", action="store_true",
                        help="save rule sets as inactive versions")
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    global EMAIL, PASSWORD
    if args.email:
        EMAIL = args.email
    if args.password:
        PASSWORD = args.password

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skipped = {s.strip() for s in args.skip.split(",") if s.strip()}

    def wanted(stage):
        if only:
            return stage in only
        return stage not in skipped

    print("\n=== Motorworks Auto Care — demo firm ===\n")
    login()

    centres, customers = {}, {}

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

    if wanted("customers"):
        print("\n[customers]")
        customers = seed_customers()
    else:
        customers = {c["full_name"]: c["id"]
                     for c in call("GET", "/customers?page_size=100")["items"]}

    if wanted("appointments"):
        print("\n[appointments]")
        seed_appointments(centres, customers)

    if wanted("support"):
        print("\n[support]")
        seed_support(customers)

    if wanted("campaigns"):
        print("\n[campaigns]")
        seed_campaigns()

    if wanted("conversations"):
        print("\n[conversations]  (real LLM calls — this is the slow part)")
        seed_conversations(customers, fresh=args.fresh)

    print("\n=== done ===")
    print("Open the dashboard. Things worth checking, in order:")
    print("  1. Customers — eight personas, Sunita on do-not-contact")
    print("  2. Appointments — two booked against real computed slots")
    print("  3. Conversations — two calls with transcripts and per-turn traces")
    print("  4. Work Queue — Priya's escalation, with its briefing, plus two tickets")
    print("  5. Campaigns — open a preview and read exclusions_by_rule")
    print("  6. Configuration > Knowledge — Monsoon expired, Diwali not started")
    print("  7. Insights > Blocked Actions — every refusal with the rule that caused it\n")


if __name__ == "__main__":
    main()
