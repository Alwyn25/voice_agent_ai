# GTM Demo Links — Production Design

**Status:** Proposed · **Date:** 2026-10-05 · **Scope:** backend (`core/`) + public demo UI + internal admin screens
**Inputs:** *Test-link cold email spec*, industry sheets (Real estate, D2C, Clinics & diagnostics, Hospitality)

---

## 1. Goal

Every cold-email prospect gets **one private web link** where they talk, in the browser, to a Phonn agent built for **one use case** in their industry. The agent knows the prospect's public company information (the knowledge base) and works on our dummy data (the sample data). No phone number, no meeting needed.

What we need to build:

| # | Deliverable | Audience |
|---|---|---|
| 1 | **KB upload**: public company info per prospect, with a source for every fact | Internal (admin uploads it) |
| 2 | **Sample data upload**: industry dummy data, checked against a schema, with relative dates | Internal (admin uploads it) |
| 3 | **Demo link engine**: tokens, expiry, call limits, separating leads, tracking | Backend |
| 4 | **Public demo UI**: Test console, Knowledge base, Sample data, Contact us / Schedule a call | Prospect |
| 5 | **Lead capture**: contact form → email to our inbox + **Google Sheet**; we call them manually | Internal (sales) |
| 6 | **Admin screens**: industries, prospects, uploads, QA gate, go live, CSV export, conversations, funnel | Internal (ops) |

### Decisions made (2026-10-05)

| topic | decision |
|---|---|
| Who provides the data | **Our team.** The KB content and the dummy data are prepared by our team and **uploaded by an admin** through the admin screens. No scraper and no LLM extraction in this service. |
| Industries | **Not fixed to a list.** An industry is a definition an admin uploads (§4.8). Real estate, D2C, Clinics and Hospitality ship first; **Distributors** is defined but waits for its dummy data; any other industry can be added later without code changes. |
| Contact us / Schedule a call | **Our own form.** The prospect leaves name, email, phone, company, preferred time and a message. The details are **emailed to our inbox** and **stored in a Google Sheet** that the sales team works from. We call them manually. No calendar tool. |
| Lead email | **Configurable in `.env`**: recipients, sender, and the mail provider and its credentials (§13). |
| Leads storage | **Google Sheet** is where the team sees and works the leads (status, notes). The database keeps a backup copy so a lead is never lost if Google is down (§10.3). |
| Call history | **Text transcripts only.** Each demo lead's conversations can be read as text in the admin screens. No audio is stored. |
| Demo domain | **Configurable in `.env`** (`DEMO_PUBLIC_BASE_URL`, `DEMO_ALLOWED_ORIGINS`). |
| Spend and limits | **Configurable in `.env`** (all `DEMO_*` settings, §13). |

### Non-goals (v1)

- Scraping and LLM extraction. The admin uploads finished content.
- A calendar or scheduling tool for "Schedule a call". It's a form, and sales calls back manually.
- Storing call audio. Transcripts only. The "60-second recording" in email 2 of the spec is made by our team outside this system.
- Sending the cold emails. We **export** links for the sending tool (§9.4). The only mail we send is the lead alert to our own inbox.
- Real telephony. Demo calls are browser WebSocket only.

---

## 2. Main principles

1. **One prospect = one agent = one link = one use case.** The spec says not to mix use cases on one link, so the data model enforces it.
2. **One demo tenant per prospect.** Each prospect gets its own tenant (`tenants.kind = 'demo'`), with its own `Agent` and `KnowledgeBase`. The whole engine already isolates by tenant (retrieval, caches, synonyms, custom instructions, memory), so a separate tenant is the only way to be sure one lead's agent can never use another lead's data. Demo tenants are excluded from billing, analytics and usage limits by their `kind`. See §2a.
3. **Every call starts from the same fresh copy of the sample data.** Tool writes (bookings, orders, returns) go into session state, never into real `appointments` / `units` rows. Two people testing the same link never affect each other.
4. **Dates are always relative.** The sample data stores offsets ("tomorrow 11:00"). They're turned into real dates when the call starts, in the prospect's timezone. Never store absolute dates. Campaigns go out late.
5. **A field with no trusted source stays blank.** The agent says the sales team will confirm. Every KB fact can carry `source_url` and `fetched_at` (optional on upload, shown on the Knowledge base tab when present).
6. **Public surface = no trust.** The public page never holds a staff login token, never starts work on a plain GET, and every limit (duration, call count, concurrency, spend) is enforced on the server.
7. **Reuse the existing engine.** The KB pipeline (chunking, embeddings, reranker, parent-child), the voice pipeline (`voice_ws`), agents and guardrails are reused. We add a thin demo layer around them, not a fork.
8. **Every tunable lives in `.env`.** Domain, limits, budget, email provider and recipients, Google Sheet, expiry and retention are settings in `config.py`, read from `.env`, with safe defaults. Changing them needs a restart, not a code change.
9. **Industries are data, not code.** No `if industry == "d2c"` anywhere. Everything that differs between industries (facts fields, dummy data shape, minimum to go live, agent persona, tools, the goal of the call, suggested prompts, QA questions) lives in an **industry definition** (§4.8). Code only knows generic building blocks.

---

## 2a. Isolation between leads

**Requirement:** nothing from one prospect's demo (KB, dummy data, conversations, bookings, leads) can be seen or used by another prospect's link.

**Why a shared demo tenant is not enough.** It was considered and rejected after checking the current code. Today the live call path searches the **whole tenant's** knowledge base, not the agent's own KB:

- the planner's automatic retrieval (`app/engine/planner/planner.py`, both `retrieve()` calls) passes no `knowledge_base_id`
- the `knowledge_lookup` tool (`app/engine/tools/builtin/knowledge_tools.py`) passes no `knowledge_base_id`
- the KB filter in `retrieval.py` also matches chunks whose `knowledge_base_id` is null
- the knowledge cache is versioned per tenant

With all prospects in one tenant, DSR's agent could answer from another developer's price sheet. Other per-tenant features (synonym groups, custom instructions, the responder template cache) would also be shared.

**Design: one tenant per prospect.**

| layer | how it's isolated |
|---|---|
| Knowledge base | separate tenant, so every existing tenant filter and cache key applies unchanged |
| Shared industry KB (disclosure, policies) | **copied** into each prospect's tenant at creation (a few small documents), never shared across tenants |
| Dummy data | `demo_sample_datasets.prospect_id`; resolved per call into `demo_sessions.resolved_context` |
| Bookings / orders made on a call | `demo_sessions.state`, one row per call; a new call starts from a fresh copy |
| Conversations | existing `calls` / conversation rows carry the prospect's `tenant_id`; the public transcript endpoint also checks `session_id` belongs to the token **and** the visitor's session cookie |
| Leads | `demo_leads.prospect_id`; never exposed on the public API |
| Public API | every query starts from `token → demo_link → prospect → tenant_id` and filters on it; no endpoint takes a tenant, agent or session ID from the browser without checking it against the token |
| Voice token | claims carry `tenant_id` + `agent_id` of that prospect only; `voice_ws` checks them against the link again on handshake |

**Tenant model change** (migration 0038): `tenants.kind` (`customer` default · `demo`). The link from prospect to tenant is `demo_prospects.tenant_id` (unique), so there is no circular foreign key. Every sweep over all tenants (dialer, aggregator) goes through `customer_tenant_ids()` in `app/db/models/platform.py`, and a guard test fails if new code adds an unfiltered `select(Tenant.id)` sweep. Billing and dashboards are already per-tenant (by the logged-in user's tenant), and demo tenants have no users. Admin views of all demos use the `demo_*` tables, which hold `prospect_id` → `tenant_id`.

**Cleanup:** deleting or expiring a prospect deletes or suspends its tenant. Nothing from one prospect is left behind in a shared place.

**Guard tests** (in §15):
- Two prospects with conflicting facts (different prices for "2 BHK"): each agent answers only its own, through both the automatic retrieval and the `knowledge_lookup` tool, with a warm cache.
- Token A cannot fetch session B's transcript, even with B's `session_id`.
- Voice token for A is refused on a handshake for B's link.

**Separate improvement, recommended for real customers too:** make the voice path pass the agent's KB IDs to `retrieve()` and stop matching null-KB chunks when a KB is given. That isolates agents within one tenant. It isn't needed for demos once each prospect has its own tenant, so it should be its own change, not part of this one.

---

## 3. Architecture overview

```
                   ┌──────────────────────────── INTERNAL (staff login) ───────────────────────────┐
  Admin (us) ─────►│ /admin/demo/*                                                                  │
                   │   industries ─► industry definition (fields, data shape, tools, goal, QA)     │
                   │   prospects ─► KB upload ─► (existing ingest pipeline) ─► KnowledgeBase        │
                   │             └► sample-data upload ─► schema check ─► demo_sample_datasets      │
                   │             └► QA checklist ─► go live ─► demo_links (token hash, expiry)      │
                   │             └► batch export.csv ─► email sending tool                          │
                   │   conversations (text)                                                         │
                   └───────────────────────────────────────────────────────────────────────────────┘
                   ┌──────────────────────────── PUBLIC (token only) ──────────────────────────────┐
  Prospect ───────►│ {DEMO_PUBLIC_BASE_URL}/t/{token}   (static SPA, noindex)                       │
                   │   GET  /public/demo/{token}            page data                               │
                   │   POST /public/demo/{token}/session    ─► short-lived demo voice token         │
                   │   WS   /ws/voice?token=…               ─► voice pipeline + demo tools          │
                   │   POST /public/demo/{token}/events     page view / CTA beacons                 │
                   │   POST /public/demo/{token}/contact    contact / schedule a call ─┐            │
                   └───────────────────────────────────────────────────────────────────┼───────────┘
                   ┌──────────────────────────── JOBS ─────────────────────────────────▼───────────┐
                   │ lead_notifier (outbox): email to DEMO_LEAD_NOTIFY_EMAILS + Google Sheet row   │
                   │ sheet_sync: read Status/Notes back from the sheet · write demo activity tab   │
                   │ expire_demo_links (hourly) · follow_up_buckets (hourly) · retention (daily)   │
                   └───────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                   Google Sheet (sales team works here)
                                     tab "Leads"          one row per form submission
                                     tab "Demo activity"  one row per prospect link
```

---

## 4. Data model (migration `0038_demo_links`)

The `demo_*` tables are the admin-side catalogue across all demos, so they are **not** tenant-scoped. Each prospect points to its own demo tenant (§2a), and everything the engine touches (agent, KB, calls, conversations) lives inside that tenant.

### 4.1 `demo_batches`
A batch of prospects sent together (the spec: 50 per batch, 10 per industry). Funnel metrics are reported per batch.

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| name | varchar(120) | e.g. `2026-10 batch 1` |
| created_by | uuid FK users | |
| created_at | timestamptz | |

**Model as decided 2026-10-06 (replaces one link per person).** A sheet row is a **company record**: one prospect company plus the one project/product its demo is about. Each record has **one private link**. The **people** at that company are separate contacts. Each person is sent the company link with **their own recipient code** (`?r=…`), so opens, calls and questions are still attributed to a person, without their email address in the URL. A company's link can also be sent to a person at another company (e.g. a group's head office) by adding them as a recipient.

```
demo_prospects (company record) 1 ──< demo_links (one active) 1 ──< demo_link_recipients >── 1 demo_contacts (people)
                                                                │
                                                                └──< demo_visitors (one per browser) ──< demo_sessions, demo_events
```

### 4.2 `demo_prospects` (company data records)

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| tenant_id | uuid FK tenants, unique | the record's own demo tenant (§2a) |
| industry_key | varchar(40) | any defined industry (§4.8), not a fixed enum; FK with `industry_version` to `demo_industry_versions` |
| industry_version | int | the industry definition version this record was built from; later edits to the definition don't silently change a live demo |
| record_key | varchar(160) | `slug(company)--slug(project)`; unique per industry, so re-importing the sheet updates instead of duplicating |
| company_name | varchar(200) | as the company writes it |
| company_name_pronunciation | varchar(200) | for TTS; QA item 1 |
| city, website_url | | from the sheet |
| trigger_note | text | the "why now" used in email line 1 |
| research_verified | bool | the sheet's "Verified" column |
| source_file, source_row | | where the record was imported from (audit) |
| timezone | varchar(64) | default `Asia/Kolkata`; used to turn relative slots into dates |
| agent_id, knowledge_base_id | uuid FK | created with the record's tenant |
| status | varchar | `draft → ready_for_qa → qa_passed → live → expired · revoked · dropped` |
| dropped_reason | text | e.g. `below_minimum: Sales office is missing` |
| created_by, created_at, updated_at | | |

### 4.2a `demo_contacts` (the people; one link each)

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| prospect_id | uuid FK | their own company record |
| batch_id | uuid FK \| null | the email batch they go out in |
| full_name, first_name, role | | first name is used in "Hi [first name]" |
| email | varchar(254) \| null | a contact without an email can't be sent a link until one is added |
| linkedin_url | text \| null | |
| needs_review | bool | the importer couldn't pair this person's details with confidence (§5.5) |
| review_notes | text | why, e.g. "email paired by similar spelling; please confirm" |
| status | varchar | `active · excluded` |

Unique: one name per company (case-insensitive), one email per company, one email per batch (the same person is never emailed twice in one batch).

### 4.3 `demo_prospect_fields`
The prospect's public facts (project name, prices, policies…), prepared by our team and uploaded by an admin. One row per field so each field can keep its source.

| column | type | notes |
|---|---|---|
| prospect_id | uuid FK | |
| field_key | varchar(60) | e.g. `project_name`, `rera_number`, `price`, `return_policy` |
| value | text \| null | null = blank; the agent says the team will confirm |
| source_url | text \| null | optional; shown as the source link on the Knowledge base tab |
| fetched_at | timestamptz \| null | optional; "checked on" date |
| confidence | varchar(10) | `confirmed · needs_check` |

`UNIQUE (prospect_id, field_key)`.

### 4.4 `demo_sample_datasets`

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| prospect_id | uuid FK | null = industry **template** (shared default) |
| industry_key | varchar(40) FK | |
| industry_version | int | the definition version whose data schema this was checked against (§6.2) |
| data | jsonb | checked against the schema; holds placeholders and relative times |
| version | int | goes up on every upload; older versions kept for audit and rollback |
| is_active | bool | one active per prospect (partial unique index) |
| uploaded_by, created_at | | |

### 4.5 `demo_links`

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| prospect_id | uuid FK | the company record this link is the demo of |
| token_hash | char(64) UNIQUE | SHA-256 of the raw token; every lookup uses this |
| token_ciphertext | text \| null | the token encrypted with `DEMO_LINK_TOKEN_KEYS` (Fernet), so the URLs can be exported again for people added later. Null = a link made before 0039 (can't be exported; make a new one) |
| token_prefix | char(6) | first 6 chars, so support can find a link without the secret |
| status | enum | `active · expired · revoked` |
| activated_at, expires_at | timestamptz | `expires_at = activated_at + DEMO_LINK_TTL_DAYS` |
| max_calls_per_day | int \| null | null = use `DEMO_MAX_CALLS_PER_LINK_PER_DAY` from `.env` |
| max_call_seconds | int \| null | null = use `DEMO_MAX_CALL_SECONDS` |
| max_total_minutes | int \| null | null = use `DEMO_MAX_MINUTES_PER_LINK` (lifetime cost ceiling per link) |
| revoked_at, revoked_reason | | |

The limit columns are **optional per-link overrides**. Normally they stay null and the `.env` value applies, so changing `.env` changes every link at once. An admin can raise one link's limits (e.g. a hot lead who wants a longer call) without touching the others.

Only one `active` link per **company record** (partial unique index `uq_demo_links_one_active_per_record`). Making a new link revokes the old one, and its recipients move to the new link with fresh codes (the old URLs stop working). Re-exporting without "replace" gives the same URLs as last time.

### 4.5a `demo_link_recipients` (who a company link was sent to) — migration 0039

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| demo_link_id | uuid FK | |
| contact_id | uuid FK | usually a person at the link's company; may be anyone (marked "other company" in the export) |
| ref_code | varchar(16) UNIQUE | 10 random URL-safe characters (56 bits), in this person's copy of the URL as `?r=`. Attribution, not a credential: it opens nothing without the token, and a code of another link is ignored |
| last_exported_at | timestamptz \| null | last time this person's URL went into an export; "emailed" in the funnel |

`UNIQUE (demo_link_id, contact_id)`.

**Why a code and not the email in a UTM tag:** URLs are logged by proxies and mail scanners, show in browser history and screenshots, and travel with a forwarded email. An email address there would leak to whoever the link is forwarded to. The code gives the admin the same answer (name, email, role) with none of that.

### 4.5b `demo_visitors` (who opened it) — migration 0039

One row per browser that opened a company link (`visitor_key` = the page's first-party id; `UNIQUE (demo_link_id, visitor_key)`). Created only by the page's own script, never by a plain GET.

| column | notes |
|---|---|
| recipient_id | the person this browser is credited to |
| attribution | `recipient` (first browser on that person's code) · `shared` (another browser on the same code: their phone, or someone they forwarded the email to) · `self_identified` (typed their name and email) · `unknown` (no code: the link was copied from the address bar) |
| identified_name / _email / _role / _at | what the visitor typed in the optional "Who's trying the demo?" card. If the email is a contact of the company, the visitor becomes that person; otherwise the visit stays credited to whoever's email brought them ("via Asha's email") |
| first_seen_at, last_seen_at, first_viewed_at | first_viewed_at = first human signal (page seen ≥ 2 s, a click, a call) |
| landing_ref, source_step, utm | how they arrived; first touch wins |
| ip_hash, user_agent | first touch; never the raw IP |

`demo_sessions` and `demo_events` carry `demo_visitor_id` + `recipient_id`, so every call, tab view and question has a person.

### 4.6 `demo_sessions`
One row per browser call.

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| demo_link_id | uuid FK | |
| visitor_id | uuid | first-party cookie; counts distinct people when a link is forwarded |
| call_id | uuid FK calls | the real call row → transcript, cost, latency come for free |
| started_at, ended_at, duration_seconds | | |
| resolved_context | jsonb | the sample data **after** relative dates were turned into real ones; kept so the transcript can be checked later |
| state | jsonb | what the tools changed: `{bookings:[…], orders:[…], returns:[…]}` |
| outcome | enum | `goal_completed · partial · none`; from the industry's goal rule (§4.8, §6.5) |
| end_reason | enum | `user_hangup · max_duration · idle · error · budget` |
| ip_hash, user_agent | | ip is hashed with a secret salt; never the raw IP |

### 4.7 `demo_events`

`demo_events(id, demo_link_id, visitor_id, session_id?, type, source_step?, occurred_at, meta jsonb)`
`type ∈ page_viewed · kb_tab_viewed · sample_tab_viewed · call_started · call_ended · transcript_viewed · cta_real_calls_clicked · contact_submitted · expired_page_viewed · visitor_identified` (+ `demo_visitor_id`, `recipient_id` since 0039)

Indexes: `demo_events (demo_link_id, type, occurred_at)`, `demo_sessions (demo_link_id, started_at)`.

### 4.7a `demo_leads` (Contact us / Schedule a call submissions)

The sales team works leads in the **Google Sheet** (§10.3). This table is the **safety copy and sync record**: every submission is written here first, in the same request, so a lead is never lost if Google or the mail server is down. The notifier then pushes it to the sheet and the inbox, and `call_status` / `notes` are read back from the sheet.

| column | type | notes |
|---|---|---|
| id | uuid PK | |
| demo_link_id | uuid FK \| null | null if the form was sent from an expired page or without a link |
| prospect_id | uuid FK \| null | filled from the link, so we know which cold-email lead this is |
| kind | enum | `contact · schedule_call · real_calls_pilot · rebuild_request` |
| name | varchar(120) | required |
| email | varchar(254) | required, format checked |
| phone | varchar(20) | required for `schedule_call` and `real_calls_pilot` (we call them back); normalised to E.164 with the existing `services/phone.py` |
| company | varchar(200) | prefilled from the prospect when known |
| role | varchar(120) | optional |
| preferred_time | varchar(120) | free text: "weekdays after 4 pm" (no calendar) |
| message | text | optional, max 2,000 chars |
| last_session_id | uuid FK \| null | their latest demo call, so sales can read the conversation before calling |
| call_status | varchar(30) | `New · Contacted · No answer · Follow up · Converted · Not interested`; **edited in the Google Sheet**, copied back here by `sheet_sync` |
| notes | text | sales notes after the manual call; edited in the sheet, copied back |
| email_notified_at | timestamptz \| null | set by the notifier when the email went out |
| sheet_synced_at | timestamptz \| null | set when the row was added to the sheet |
| sheet_row_key | varchar(40) | the `Lead ID` written in the sheet (short form of `id`); used to find the row again even if the team sorts or moves rows |
| notify_attempts, last_notify_error | int, text | for retries and alerts |
| created_at, updated_at | | |

`UNIQUE (email, kind, date(created_at))` stops a double-click or refresh creating duplicate leads on the same day.

### 4.8 Industries: `demo_industries` (definitions, not hard-coded)

The system is **not limited to the five industries in the spec.** An industry is a versioned definition that an admin uploads (YAML or JSON) or edits in the admin screen. Adding a new industry (say, "Education: admission enquiry" or "Insurance: renewal reminder") is an upload, not a code change, as long as it uses the building blocks in §6.5.

Two tables (as built in 0038):

`demo_industries`: one row per industry.

| column | type | notes |
|---|---|---|
| key | varchar(40) PK | `real_estate`, `d2c`, `clinics`, `distributors`, `hospitality`, …; lowercase snake case (CHECK) |
| label | varchar(80) | shown in the UI and the email ("Clinics & diagnostics") |
| status | varchar | `draft · active · archived`; only `active` industries can have live links |
| current_version | int | the version new prospects are built from |
| updated_by, created_at, updated_at | | |

`demo_industry_versions`: every version ever saved, never edited.

| column | type | notes |
|---|---|---|
| industry_key, version | PK | |
| definition | jsonb | the full definition below, checked against a meta-schema on upload |
| created_by, created_at | | |

Prospects and dummy datasets reference `(industry_key, industry_version)` with a foreign key, so a version in use can't be deleted.

**What one definition contains:**

```yaml
key: distributors
label: Distributors
version: 1
use_case:
  key: retailer_phone_order
  label: Retailer phone order
  prospect_plays: A kirana or restaurant owner
  goal: order_read_back            # a goal rule from §6.5
  goal_label: Order read back with quantities and a total

facts_fields:                      # → facts template + minimum-to-go-live check (§5.4)
  - {key: company_name,   label: Company name,          required: true}
  - {key: brands,         label: Brands distributed,    required: true}
  - {key: service_area,   label: Areas served,          required: true}
  - {key: ordering_terms, label: MOQ / delivery days,   required: true}
  - {key: credit_terms,   label: Credit terms,          required: false}

minimum:                           # extra rules beyond required fields
  - {collection: catalog, min_items: 5}

sample_data:                       # → JSON Schema for the dummy data upload (§6.2)
  caller: {type: sample_caller}
  catalog:
    type: list
    min_items: 5
    item: {sku: string, name: string, pack: string, rate_inr: money, moq: integer}
  delivery_slots: {type: list, item: {type: relative_slot}}

tools:                             # building blocks from §6.5, renamed for the agent
  - {block: lookup,        name: check_item,   collection: catalog, match_on: [name, sku]}
  - {block: create_record, name: place_order,  collection: orders,
     fields: {lines: [{sku: string, qty: integer}]}, compute: order_total, ref_prefix: ORD-}

agent:
  persona: Order desk for {{company_name}}
  languages: [en-IN, hi-IN, hinglish]
  greeting: "Namaste, {{company_name}} order desk. Yeh demo sample data par chal raha hai. Aaj kya order karna hai?"
  instructions: |
    Take the retailer's order line by line, confirm quantities, read back the full order with a total…
  guardrails:
    - Never invent a product, price or discount that is not in the catalog.

prompts:                           # the 3 suggestions on the test console, one in Hinglish
  - "20 carton Maggi aur 10 Parle-G bhej do"
  - "What's the minimum order?"
  - "Can I get it delivered tomorrow?"

qa_questions:                      # automated QA set (§11)
  - {say: "Parle-G ka rate kya hai?", expect: "Reads the catalog rate; no invented discount"}

compliance:                        # extra QA checklist items for this industry
  - Refuses to change prices or give credit beyond the stated terms
```

**What ships at launch:**

| industry | definition | dummy data | can go live |
|---|---|---|---|
| Real estate | yes | from the industry sheet | yes |
| D2C | yes | from the industry sheet | yes |
| Clinics & diagnostics | yes | from the industry sheet | yes |
| Hospitality | yes | from the industry sheet | yes |
| **Distributors** | yes (`status=active`) | **not yet** | **no**: blocked by the minimum-to-go-live check until its dummy data is uploaded. Nothing else to build. |
| any new industry | uploaded later | uploaded later | after its QA passes |

**Why a definition and not code per industry:**
- The five industries share the same pipeline. What differs is data (fields, dummy data, prompts) plus a choice of tools and goal.
- Adding an industry must not need a deploy, because GTM will test new verticals faster than release cycles.
- Versioning keeps a live demo stable: a prospect stays on the definition version it was built with until an admin clicks "Rebuild on latest".

**Limit of this approach:** a new industry that needs behaviour none of the building blocks cover (e.g. a live payment link) needs one new building block in code. After that, every industry can use it.

---

## 5. KB upload

### 5.1 Who uploads what

Our team prepares the content. An **admin uploads it from the admin screen** (§11). There is no scraper in this service.

| what | format | where it goes |
|---|---|---|
| Company documents (brochure, price sheet, FAQ, policies) | PDF, DOCX, TXT, MD, HTML | prospect's KB → chunked and embedded |
| Company facts sheet (project name, prices, timings, policies…) | **XLSX/CSV with columns `field_key, value, source_url, fetched_at`**, or the form in the admin screen | `demo_prospect_fields` + the generated "Company facts" KB document |
| Whole batch at once | ZIP: `prospects.xlsx` (one row per prospect) + one folder per prospect with its facts sheet, documents and sample data | everything above, per prospect, in one upload |

We give the team **one downloadable facts template per industry** (`GET /admin/demo/fields/template/{industry}.xlsx`), **generated from the industry definition's `facts_fields`**, with the field keys already filled in, so they only type values. A new industry gets its template automatically.

### 5.2 Endpoints (staff login, role `admin`)

```
GET    /admin/demo/industries                   list definitions (+ readiness: has template data? active?)
POST   /admin/demo/industries                   upload a new definition (yaml/json), checked against the meta-schema
PUT    /admin/demo/industries/{key}             new version of a definition
POST   /admin/demo/industries/{key}/validate    dry run: errors + generated templates preview
POST   /admin/demo/prospects                    create one prospect (company, industry, buyer…)
POST   /admin/demo/batches/{id}/import          ZIP bulk import (dry_run=1 supported)
POST   /admin/demo/prospects/{id}/kb/files      multipart: files[] (pdf, docx, html, txt, md)
POST   /admin/demo/prospects/{id}/kb/text       {"documents":[{"title","text","source_url?","doc_type"}]}
POST   /admin/demo/prospects/{id}/fields        facts sheet upload (xlsx/csv) or JSON upsert
GET    /admin/demo/fields/template/{industry}.xlsx
GET    /admin/demo/prospects/{id}/kb            documents + ingest status
DELETE /admin/demo/prospects/{id}/kb/{doc_id}
```

**Bulk import is all or nothing per prospect.** A prospect folder with any error is skipped and reported (`row 7, Sunrise Clinics: doctors sheet missing column specialty`). The valid ones are imported. `dry_run=1` shows the full report before anything is saved.

### 5.3 How it works

- **Thin wrapper, no new pipeline.** Each call creates `KnowledgeDocument` rows in the prospect's KB (`knowledge_base_id = prospect.knowledge_base_id`) and hands them to the same ingest path `/knowledge/documents/upload` and `/knowledge-bases/{id}/sources` use. Chunking, embeddings, parent-child, reranker and embedding profiles all apply.
- **Where the source goes:** `KnowledgeDocument.source_uri = source_url` (when given), `meta = {"fetched_at":…, "demo_prospect_id":…, "uploaded_by":…}`. The public Knowledge base tab reads from here (§10.2).
- **Facts sheet → KB document.** The uploaded fields (§4.3) are also turned into one generated "Company facts" document, one line per field: `Project name: DSR Highland Greenz (source: …)`. Blank fields are written as `RERA number: not published — the sales team will confirm.` This is the most reliable way to make the agent decline instead of inventing an answer.
- **Background and idempotent.** Ingest runs in the background. `content_sha256` in `meta` makes re-uploading the same file a no-op. The endpoint returns `202 {document_ids, status:"processing"}`.
- **Limits:** `DEMO_UPLOAD_MAX_FILE_MB` (default 20) per file, `DEMO_UPLOAD_MAX_FILES_PER_PROSPECT` (default 50), MIME sniffed (not trusted from the extension), HTML stripped of scripts.
- **Shared industry KB.** One KB per industry (disclosure line, "sample data" policy, clinics no-medical-advice policy, off-topic handling) is attached to every agent in that industry alongside the prospect's KB. It's written once and reused.
- **Audit.** Every upload records who uploaded what and when, so a wrong price on a call can be traced back to the file it came from.

### 5.4 Minimum-to-go-live check

One generic checker reads the rules from the industry definition (`facts_fields[].required` + `minimum`) and runs them on the prospect's fields and active dataset. There is no per-industry code. It catches a missing or half-filled upload before QA time is spent on it. A prospect that fails cannot move to `ready_for_qa`, and the admin screen lists exactly what is missing ("catalog has 2 items, needs 5"). The same check blocks an industry that has no dummy data yet (Distributors today).

Starting values written into the launch definitions:

| industry | minimum |
|---|---|
| real_estate | project_name, location, ≥ 2 BHK configurations, sales_office address (from the spec) |
| d2c | brand, ≥ 1 product (best-seller), shipping policy, return/exchange policy |
| clinics | clinic name, ≥ 1 branch with address, ≥ 2 doctors with specialty, timings |
| distributors | company, ≥ 5 SKUs with pack size, ordering terms (MOQ / delivery days) |
| hospitality | property name, address, booking type (table/room), timings or room types |

Because they're data, they can be tightened or loosened after the first batch without a deploy.

---

## 6. Sample data upload

### 6.1 Endpoints

```
GET  /admin/demo/sample-data/schema/{industry}          JSON Schema (drives the admin form + the CSV template)
GET  /admin/demo/sample-data/template/{industry}.csv    downloadable template
PUT  /admin/demo/sample-data/templates/{industry}       upload or replace the industry template dataset
POST /admin/demo/prospects/{id}/sample-data?dry_run=1   json | csv | xlsx → errors per field + resolved preview
POST /admin/demo/prospects/{id}/sample-data             same, commits a new version
GET  /admin/demo/prospects/{id}/sample-data/preview?at=2026-10-10T19:00+05:30
POST /admin/demo/prospects/{id}/sample-data/rollback/{version}
```

### 6.2 Schema generated from the industry definition

There are no hand-written per-industry Python classes. The `sample_data` block of the industry definition (§4.8) is compiled into a **JSON Schema** when the definition is uploaded, and every dummy data upload is checked against it (with the `jsonschema` library). The admin form, the XLSX template and the error messages all come from that one schema.

The definition is built from a small set of **field types** that are written once in code and reused by every industry:

| type | meaning | example |
|---|---|---|
| `string`, `integer`, `boolean`, `money` | plain values; `money` is a number in rupees, spoken correctly ("1.04 crore") | `rate_inr: money` |
| `relative_slot` | `{day_offset, time, label?}`; turned into a real date at call start | tomorrow 11:00 |
| `relative_day` | a day offset, may be negative (an order placed 3 days ago) | `placed: -3` |
| `sample_caller` | `{name, phone_masked}`; phone must be masked | Rahul |
| `ref_code` | a booking or order reference with a fixed prefix | `VIS-1187` |
| `enum` | one of a list | `status: [placed, shipped, out_for_delivery, delivered]` |
| `list` / `object` | nesting of the above, with `min_items` | catalog, doctors, orders |
| `text_template` | text that may contain `{{facts_field}}` placeholders | "Facebook lead form for {{project_name}}" |

For example, real estate's block (from the industry sheet):

```yaml
sample_data:
  caller: {type: sample_caller}
  lead_source: {type: text_template}            # "Facebook lead form for {{project_name}}"
  enquiry_minutes_before_call: {type: integer, default: 10}
  free_visit_slots: {type: list, min_items: 2, item: {type: relative_slot}}
  visit_location: {type: text_template}
  sales_contact: {type: sample_caller}
  booking_reference: {type: ref_code, prefix: VIS-}
  qualifying_questions: {type: list, item: {type: string}}
```

A new industry only needs a new block like this. If it needs a genuinely new field type, that one type is added in code and every industry can use it after that.

Rules shared by every schema:

- **Only relative times.** No `datetime` field accepts an absolute date. The validator rejects ISO dates anywhere in the payload.
- **Placeholders** `{{field_key}}` must refer to a `demo_prospect_fields.field_key` that exists and is **not blank** for that prospect. If one doesn't, the upload fails with `unresolved_placeholder: best_seller_price`.
- **Personal data guard:** phone numbers must be masked, so real contact details can't leak into a public page.
- **Errors per field:** each error is returned as `{loc: "orders[0].status", msg, input}`. For CSV/XLSX, `loc` is translated to `row 3, column status`.

### 6.3 CSV / XLSX

Nested data (slots, orders, catalog) uses **one sheet per list** in XLSX (`caller`, `slots`, `orders`, …), or a ZIP of CSVs with the same names. A parser turns them into JSON, and every format goes through the same schema check. **XLSX is what the team will normally use**: one downloadable template per industry, `GET /admin/demo/sample-data/template/{industry}.xlsx`, generated from the definition (one sheet per list, a header per field, example rows if the definition has them). JSON is accepted too.

**The industry template saves work.** The dummy data is mostly the same for every prospect in an industry (Rahul, the slots, VIS-1187). The admin uploads it **once per industry** as the template. Each new prospect gets a copy automatically, with `{{placeholders}}` filled from that prospect's facts. A per-prospect upload is only needed when one prospect's data differs.

### 6.4 Turning the dataset into real values at call start

`resolve_dataset(dataset, prospect_fields, now, tz) -> ResolvedContext`:

1. Fill in `{{placeholders}}` from the prospect fields.
2. Turn every `relative_slot` / `relative_day` into a real date. Optional rules from the industry definition apply (e.g. `skip_weekdays: [sunday]` for a clinic with no Sunday OPD).
3. Render the spoken form ("Thursday, 1 October, 11 am") with the existing `services/spoken.py` and `packs/locale.py`.
4. Freeze the result into `demo_sessions.resolved_context`. Everything the agent says on this call comes from that snapshot.

`/sample-data/preview?at=…` runs the same function, so ops can see exactly what a prospect will hear on any date.

### 6.5 How the agent uses the sample data

A `SampleDataProvider` interface is passed into the tool runtime for demo calls only:

```python
class SampleDataProvider(Protocol):
    def context(self) -> ResolvedContext: ...
    async def record(self, kind: str, payload: dict) -> dict: ...   # writes to demo_sessions.state
```

**Tools are generic building blocks**, set up per industry by the definition's `tools` list. Every use case in the spec (and most others) is a combination of these five:

| block | does | used by |
|---|---|---|
| `lookup` | find items in a dataset list by matching fields (order number, product name, doctor, SKU) | D2C order status, distributors item check, clinics doctor search |
| `list_slots` | list free slots from a dataset list, filtered (by doctor, branch, party size) | real estate visits, clinics, hospitality |
| `book_slot` | reserve a free slot for the caller; returns the reference from the dataset; the slot is then taken for the rest of this call | real estate visit, clinics appointment, hospitality table/room |
| `create_record` | create a new record (order, return, exchange) with the fields given, optional `compute` (e.g. `order_total`), reference with prefix | distributors order, D2C return/exchange |
| `update_record` | change a field on an existing record within allowed values | D2C "cancel my order" |

Each block is registered for the agent under the name and description from the definition (`place_order`, `check_order_status`…), so the LLM sees tools that make sense for that industry, while the code underneath is the same. They're attached only to demo agents through `AgentTool`. All writes go to `demo_sessions.state`; no real domain tables are touched, and no new order or booking tables are needed for any industry.

**The goal check is also a building block.** The definition picks a goal rule, and `outcome` is computed from session state after the call:

| goal rule | `goal_completed` when |
|---|---|
| `slot_booked` | `book_slot` succeeded |
| `record_created` | `create_record` succeeded for the named collection |
| `lookup_answered` | `lookup` returned a record (plus optional: a follow-up `create_record` if the caller asked) |
| `order_read_back` | `create_record` on orders succeeded with ≥ 1 line and a total |

- Booking and order references come from the dataset (`VIS-1187`) so the read-back is predictable and QA can check it.
- The existing real-estate and healthcare scheduling tools are **not** used for demos. Using one generic path for every industry means one thing to test and secure, and no risk of a demo writing into real `appointments`.

---

## 7. Agent per prospect

When a prospect is created:

0. A new tenant: `kind=demo`, `demo_prospect_id`, `name="{company} (demo)"`, timezone from the prospect, recording off.
1. `Agent` in that tenant: `status=draft`, `name="{company} demo"`, `purpose_label=use_case.key`, languages, persona, `system_instructions` and greeting **from the industry definition's `agent` block** with placeholders filled. The disclosure line is always added by code, so no definition can leave it out.
2. `KnowledgeBase` for the prospect, plus the shared industry KB, both attached through `AgentKnowledgeBase`.
3. The tool building blocks listed in the definition, attached through `AgentTool` (only those, nothing else).
4. Guardrails through `AgentGuardrail`:
   - **Always added (from code, every industry):** never invent prices, numbers, names or discounts that aren't in the KB or sample data; call sample data "sample" if asked; stay on the one use case and decline other topics politely; never claim to be human.
   - **From the definition:** the industry's own rules (clinics: no medical advice, emergencies go to a human; real estate: never state a RERA number or bank that isn't given).

Re-running "regenerate agent" rebuilds the prospect's agent from its pinned definition version, or from the latest version with "Rebuild on latest".

---

## 8. Demo links: tokens, sessions and security

### 8.1 Token

- `raw = secrets.token_urlsafe(16)` (128 bits), one per company record. A person's URL is `{DEMO_PUBLIC_BASE_URL}/t/{raw}?r={ref_code}&utm_source=email&utm_medium=cold_email&utm_campaign={batch}&utm_content=e1`. The domain comes from `.env`, so moving from a Railway URL to `demo.phonn.ai` later is a config change only. Links already sent keep working as long as the old domain still points at the demo app.
- Lookups use `sha256(raw)` through the unique index. The token is also kept **encrypted** (`token_ciphertext`, Fernet keys in `DEMO_LINK_TOKEN_KEYS`; first key encrypts, all decrypt, so keys rotate). That's needed because one company link goes into many emails over time: a person added next week must get the same link. A database dump alone does not reveal any token.
- **Neither the company name nor anyone's email is in the URL.** URLs end up in logs, proxies, screenshots and forwarded emails.
- Admins can show a link's URLs again (`GET /admin/demo/links/{id}/urls`); every reveal is written to the audit log.

### 8.2 Public endpoints

```
GET  /public/demo/{token}
     200 → {company, header, industry, use_case, prompts[3], sample_panel, kb_facts[], expires_at, disclosure}
     410 → {expired: true, company} — page shows "request a fresh demo" (still a lead)
     404 → unknown/revoked (same body for both: don't reveal which)

POST /public/demo/{token}/session          body: {visitor_id, turnstile_token}
     201 → {session_id, voice_token, ws_url, max_seconds}
     429 → daily calls / concurrency / budget limit hit (with a friendly message for the UI)

POST /public/demo/{token}/visit            body: {visitor_id, ref?, source_step?, utm?}
     200 → {ask_identity}  — the "a person is looking" beacon: creates/attributes the visitor, writes page_viewed
POST /public/demo/{token}/identify         body: {visitor_id, name, email, role?}  — optional "who's trying this?"
     200 → {ok}  — same answer whether or not the email matched a contact

GET  /public/demo/{token}/sessions/{session_id}/transcript     (needs the session cookie set at /session)
POST /public/demo/{token}/events           body: {visitor_id, type, ref?, utm?, session_id?}  — returns 204; page_viewed is refused here (only /visit records opens)
POST /public/demo/{token}/contact          body: {kind, name, email, phone, company?, role?, preferred_time?, message?, turnstile_token}
     201 → {lead_id}  — the lead is saved first; email + sheet happen after (§10.3)
POST /public/demo/contact                  same, for the expired page or a visit without a link
```

**Page data, as the demo page (`Generic-voice-agent-UI/src/demo/`) consumes it** (built 2026-10-05 against a mock; the backend must return exactly this):

```jsonc
{
  "company_name": "Acme Homes", "record_name": "Acme Green Acres", "industry_label": "Real estate",
  "header": "Acme Homes × Phonn: a demo agent built from your public website",
  "use_case": "New-project enquiry and site-visit booking",
  "prompts": ["…", "…", "…"],
  "disclosure": "Built by Phonn from public information. Sample data only.",
  "expires_at": "2026-10-12T10:00:00+05:30",
  "max_call_seconds": 300,
  "sample": {                              // the record's dummy data, resolved at request time
    "summary":  [{"label": "You are", "value": "Rahul, who just enquired via …"}],
    "sections": [{"title": "Free site-visit slots",
                  "items": [{"label": "Slot 1", "value": "tomorrow, Tuesday 6 October at 11 am"}]}]
  },
  "knowledge": {                           // every fact the agent was given
    "facts": [{"label": "Price", "value": "₹1.04 Cr onwards",
               "source_url": "https://…", "checked_on": "2026-10-01"},
              {"label": "RERA number", "value": null}],          // null = not published
    "documents": [{"title": "Acme Green Acres brochure", "kind": "pdf"}],   // titles only
    "not_published": "Not published. The agent says the sales team will confirm."
  }
}
```

The `sample` sections are built generically from the industry's dummy-data schema (titles and labels), so a new industry needs no UI change. Only titles of knowledge documents are exposed, never their content. Sample phones stay masked ("+91 98XXX XXX01"); the page says the people are sample data.

### 8.3 Demo voice token and `voice_ws`

Today `voice_ws._authenticate` ([core/app/api/voice_ws.py:105](../core/app/api/voice_ws.py)) only accepts a user access token. Add a second token type:

```
typ = "demo_voice"
claims = {tenant_id, agent_id, demo_link_id, session_id, max_seconds, jti}
ttl = 300 s, single-use (jti stored in Redis or a DB table and burned on handshake)
```

In `voice_call()`:

1. Try `decode_token(token, expected_type="demo_voice")` first, then fall back to the existing user path.
2. Check again: link still `active`, not expired, `session_id` belongs to that link, `jti` not already used.
3. Check the `Origin` header against `DEMO_ALLOWED_ORIGINS`.
4. Force `direction="inbound"`. Demo tokens can never reach the outbound dial path.
5. Build the pipeline with the `SampleDataProvider` for `session_id`, and start a **hard timer** at `max_seconds`. When it fires, the agent says the closing line and the socket closes with code `4408`.
6. On close, write `ended_at`, `duration_seconds`, `end_reason`, compute `outcome` from `state` against the use-case goal (§4.8), and emit `call_ended`.

A staff login token never reaches the public page.

### 8.4 Abuse and cost limits

| control | `.env` setting | default | enforced in |
|---|---|---|---|
| Max call length | `DEMO_MAX_CALL_SECONDS` | 300 | `voice_ws` timer, per call, from the agent's first audio |
| Calls per link per day | `DEMO_MAX_CALLS_PER_LINK_PER_DAY` | 10 | `/session` |
| Concurrent calls per link | `DEMO_MAX_CONCURRENT_CALLS_PER_LINK` | 1 | `/session`, counted and inserted under one Postgres advisory lock |
| Lifetime minutes per link | `DEMO_MAX_MINUTES_PER_LINK` | 60 | `/session` |
| Global concurrent demo calls | `DEMO_MAX_CONCURRENT_CALLS_GLOBAL` | 20 | `/session` |
| Daily demo spend budget | `DEMO_DAILY_BUDGET_USD` | 25 (0 = no cap) | `/session`, using P4.06 per-stage cost tracking; above it → 429 + alert |
| Budget alert threshold | `DEMO_BUDGET_ALERT_PCT` | 80 | budget check → email to `DEMO_ALERT_EMAILS` |
| Per-IP rate limit | `DEMO_RATE_LIMIT_PER_MIN`, `DEMO_SESSIONS_PER_IP_PER_HOUR` | 30, 5 | every `/public/demo` route (Redis; in-process counter while Redis is down) |
| Caller IP | `DEMO_CLIENT_IP_HEADER`, `DEMO_TRUSTED_PROXY_HOPS` | x-forwarded-for, 1 | the entry our own proxy appended, never the client-written left part of X-Forwarded-For |
| Bot check | `DEMO_TURNSTILE_ENABLED`, `DEMO_TURNSTILE_SECRET` (+ page `VITE_TURNSTILE_SITE_KEY`) | required outside `ENV=local` (boot check) | `/session` and `/contact`; fails closed; token must come from a demo origin and carry the route's action (`demo_session` / `demo_contact`) |
| Idle timeout | existing `silence_timeout_seconds` | 6 | pipeline |

All of these are read from `.env` (full list in §13). The per-link columns in §4.5 can override the first four for a single link.

**How the budget works:** before issuing a session, `/session` adds up today's demo spend (from the per-stage cost rows of calls in all `kind=demo` tenants, cached for 60 s) plus an estimate for one more full-length call. If that would go over `DEMO_DAILY_BUDGET_USD`, the request gets 429 and the page shows "Demo is busy, leave your details and we'll call you", which turns into a lead instead of a dead end. "Today" uses `DEMO_BUDGET_TIMEZONE` (default `Asia/Kolkata`).

### 8.5 Headers and privacy

- `X-Robots-Tag: noindex, nofollow, noarchive` on the API and the SPA; `<meta name="robots" content="noindex">`; `robots.txt` disallows `/`.
- `Referrer-Policy: no-referrer` so the token doesn't leak to third-party scripts.
- No third-party analytics on the demo page. All tracking is first-party (`/events`).
- Strict CSP; the only allowed origins are the API, the WS endpoint and Turnstile.
- Disclosure footer: *"Built by Phonn from public information. Sample data only. Link expires on {date}."*
- `ip_hash = HMAC(secret, ip)`. Raw IPs are never stored.

---

## 9. Separating leads and tracking

### 9.1 Attribution

`token → demo_link → prospect → batch → industry` for the company, and `?r= code → recipient → contact` for the person. Every session and event carries `demo_link_id`, `demo_visitor_id` and `recipient_id`, so all activity traces back to a company and, where we can tell, a person, without fingerprinting.

- **Person:** each person's URL carries their recipient code. The page reads it (with the UTM tags and the email step), sends it with `/visit`, and removes all of them from the address bar (`history.replaceState`, other params and the `#tab` kept). A link copied from the address bar and passed on therefore arrives without a code and shows as "unknown" instead of crediting the original person.
- **Email step:** `utm_content=e1|e2|e3|e4|li` (or `?s=` when `DEMO_UTM_ENABLED=false`).
- **UTM tags:** the five standard keys are stored on the visitor (first touch) and on each `page_viewed`. Unknown keys and odd values are dropped.
- **Forwarding:** each browser is one visitor (first-party id). A second browser on the same person's code is `shared` (their phone, or a colleague they forwarded the email to). "3 people from this company tried it" is a strong buying signal.
- **Unknown visitors** see an optional, skippable "Who's trying the demo?" card (`DEMO_IDENTIFY_PROMPT=unknown_only`, also shown for `shared`). It never blocks the call. If the email they type is a contact of the company, the visitor becomes that person.
- **Questions asked:** every caller utterance (role `customer`) of a demo call, with who said it and the agent's reply: per link in `GET /admin/demo/links/{id}/activity`, across demos (searchable) in `GET /admin/demo/questions?q=`.

### 9.2 Telling real opens from mail scanners

Corporate mail security (Outlook Safe Links, Mimecast, Proofpoint, Barracuda) fetches every link in an email before a human sees it. **A server GET is not an open.**

- `page_viewed` is sent only by the SPA's JavaScript **after** the page renders **and** (the page has been visible for 2 s **or** the visitor interacts). Scanners don't run this, and headless scanners usually don't meet the visibility condition.
- A plain `GET /public/demo/{token}` changes nothing and is not counted.
- `call_started` needs microphone permission, so it's always a real person.

### 9.3 Follow-up buckets (hourly job)

From the spec's trigger table:

| bucket | rule | action |
|---|---|---|
| `talked_long` | any session with `duration_seconds ≥ 120` | same day: founder writes a personal email quoting what they asked (transcript linked) |
| `talked_short` | sessions exist, all < 120 s | next day: "what felt off?" email |
| `opened_no_talk` | `page_viewed`, no `call_started` | day 3: email 2 with the 60-second recording (made by our team) |
| `never_opened` | no `page_viewed` | day 3: email 2; day 6: LinkedIn note |
| `hot` | `cta_real_calls_clicked` or `contact_submitted` | **immediately**: lead email to `DEMO_LEAD_NOTIFY_EMAILS` + row in the leads sheet (§10.3); sales calls within 1 hour |

Exposed as `GET /admin/demo/batches/{id}/follow-ups` and as an outgoing webhook (`DEMO_WEBHOOK_URL`), so the email tool or CRM can drive the sequence. `hot` is pushed when the event happens, not in the hourly job.

### 9.4 Batch export for the email tool

**As built (2026-10-06):** `POST /admin/demo/links/generate?format=csv` (filters: records, batch, industry; `regenerate` replaces links). One link per company, **one row per person**:

```
first_name,full_name,email,role,company,record_name,industry,link,link_e1,link_e2,link_e3,link_e4,link_li,
company_link,expires_on,needs_review,review_notes,other_company,prompt_1,prompt_2,ref_code,record_id,link_id,contact_id
```

`link` = the first step's URL; one `link_<step>` column per `DEMO_EXPORT_STEPS` entry, each with the person's code and UTM tags. `company_link` has no code (for places where a person can't be named). Re-running gives the same URLs; exporting marks those people as emailed.

Planned columns not built yet: `trigger_note`, `agent_knows_line`.

`agent_knows_line` is generated from the fields ("trained on your store, shipping and return policies"), and the prompts come from the industry template ("mera order kab aayega?"). Email 1 can be mail-merged without manual edits.

### 9.5 Funnel metrics

`GET /admin/demo/metrics?batch_id=&industry=`:

- Link open rate = prospects with `page_viewed` ÷ prospects emailed (target 25%+)
- Talk rate = prospects with `call_started` ÷ prospects with `page_viewed` (target 40%+)
- Pilot rate = prospects that are `hot` ÷ prospects emailed (target 5%+)
- QA failure rate = prospects that failed QA at least once ÷ prospects that reached QA (above 20% → fix the uploaded data or the industry template before the next batch)
- Also: median call length, goal-completed rate per use case, cost per demo call

---

## 10. Public demo UI

### 10.1 Delivery

- A **separate Vite entry or small app** (`voice-agent-ui/demo/` or its own package), deployed as static files on the domain in `DEMO_PUBLIC_BASE_URL`. The UI reads its API and WebSocket URLs from build-time env (`VITE_DEMO_API_URL`), so no domain is written in code. No dashboard bundle, auth context or admin code ships to prospects.
- Reuses [`useVoiceCall.js`](../../voice-agent-ui/src/pages/console/useVoiceCall.js) and the pipecat client. It's given the demo `ws_url` + `voice_token` from `/session`.
- Mobile first: many prospects will open the email on a phone. Check microphone permission on iOS Safari and Android Chrome.

### 10.2 Pages (tabs on one page, URL `/t/{token}#console|knowledge|sample|contact`)

**Test console** (default)
- Header: **"{Company} × Phonn: a demo agent built from your public website"**
- Big **Talk to the agent** button → microphone explainer → `/session` → connect. Shows connecting / listening / speaking and a countdown of the time left.
- 3 suggested things to say (one in Hinglish) as chips, so the prospect knows what to test.
- Live transcript while the call runs.
- After the call: the full transcript, the goal result ("Site visit booked: Thu 1 Oct, 11 am, VIS-1187"), and a primary **Run this on real calls for 14 days** button → contact form with `kind=real_calls_pilot`, prefilled from the prospect.
- Errors: microphone denied (how to enable it), limit hit (429 → "try again later or book a call"), connection drop (retry).

**Knowledge base**
- Read-only list of what the agent knows, grouped (Project, Pricing, Amenities, Location…). Each fact shows a **source link** and "checked {date}".
- Blank fields are shown honestly: "RERA number: not published, the sales team confirms". This shows we don't make things up.

**Sample data**
- The dummy dataset with dates already filled in for *now* (the same `resolve_dataset` as the call), clearly marked **Sample**: the caller, the free slots, the sample order, the catalog.
- Doubles as a script: "Ask about order #PH-20931".

**Contact us / Schedule a call**
- One form, two buttons at the top: **Contact us** / **Schedule a call** (sets `kind`).
- Fields: name\*, work email\*, phone\* (required for Schedule a call; we call back manually), company (prefilled from the link), role, **preferred time to call** (free text, e.g. "Tomorrow after 3 pm"), message.
- Checked in the browser and again on the server. Turnstile when enabled. Submit button disabled while sending (no double leads).
- Confirmation screen: "Thanks {name}, our team will call you on {phone} {preferred_time}." No calendar tool.

**Expired state**: company header, "This demo has expired", and a one-click **Rebuild my demo** (`kind=rebuild_request`) that opens the same form.

**Footer** on every tab: the disclosure line and the expiry date.

### 10.3 Leads: email + Google Sheet

When a prospect submits the form:

```
POST /contact ─► save demo_leads row (commit) ─► 201 to the browser
                         │
                         └─► lead_notifier (background, retried)
                               ├─► email to DEMO_LEAD_NOTIFY_EMAILS
                               └─► append row to Google Sheet tab "Leads"

sheet_sync (every DEMO_SHEET_SYNC_MINUTES)
   ├─► read "Status" and "Notes" columns from the sheet ─► demo_leads.call_status / notes
   └─► rewrite the "Demo activity" tab (one row per prospect link)
```

**1. Save first, notify after (outbox pattern).** The browser gets success as soon as the lead is in the database. The email and sheet run in the background. If the mail provider or Google is down, nothing is lost: a sweep every `DEMO_NOTIFY_RETRY_MINUTES` (default 5) retries rows whose `email_notified_at` / `sheet_synced_at` is still empty. After `DEMO_NOTIFY_MAX_ATTEMPTS` (default 5) failures, an alert goes to `DEMO_ALERT_EMAILS` and the lead shows a red "not synced" badge in the admin screen.

Why keep a database copy when the team works in the sheet: a Google Sheet can be edited by anyone with access, rows can be deleted by accident, and Google's API has rate limits and outages. The database copy means the sheet can always be rebuilt (`POST /admin/demo/leads/resync-sheet`), and the follow-up logic (§9.3) never depends on someone not breaking the sheet.

**2. The email.** Everything about it is set in `.env` (§13):

| setting | what |
|---|---|
| `DEMO_EMAIL_PROVIDER` | `smtp` (default), `sendgrid`, `resend`, or `none` (email off, sheet only) |
| `DEMO_LEAD_NOTIFY_EMAILS` | who receives leads, comma-separated |
| `DEMO_LEAD_FROM_EMAIL`, `DEMO_LEAD_FROM_NAME` | the sender |
| `SMTP_*` or `DEMO_EMAIL_API_KEY` | credentials for the chosen provider |

`smtp` covers Google Workspace (app password), Zoho, Amazon SES, Outlook/Microsoft 365, or any mail server, so switching mailbox or provider is a `.env` change. The API providers are behind the same small `LeadMailer` interface, so adding another one later is a single class.

The email looks like this:

```
To:       DEMO_LEAD_NOTIFY_EMAILS (comma-separated, e.g. founder + sales)
From:     DEMO_LEAD_FROM_EMAIL
Reply-To: the lead's email            ← hitting Reply answers the lead directly
Subject:  [Demo lead] Schedule a call · Sunrise Clinics (Clinics) · Priya Sharma

Name:            Priya Sharma
Phone:           +91 98xxx xxxxx  (tap to call)
Email:           priya@sunriseclinics.in
Company / role:  Sunrise Clinics · Operations Head
Preferred time:  Tomorrow after 3 pm
Message:         …

Demo activity:   3 calls, longest 4m 12s, goal completed (appointment booked)
Last call:       first 6 lines of the transcript …
Open in admin:   {DASHBOARD_URL}/demo/leads/{lead_id}   (full conversations as text)
Google Sheet:    https://docs.google.com/spreadsheets/d/{DEMO_LEADS_GSHEET_ID}
```

- Everything the prospect typed is HTML-escaped and length-limited, so the form can't inject content into our email.
- The demo link token is **never** included in the email.
- Plain-text and HTML versions are both sent.

**3. The Google Sheet.** One spreadsheet (ID in `.env`), two tabs (names in `.env`):

*Tab "Leads"*: one row per form submission. The sales team works from here.

```
Lead ID | Received at | Kind | Name | Phone | Email | Company | Role | Industry | Preferred time |
Message | Demo calls | Longest call | Goal completed | Batch | Admin link | Status | Called by | Notes
```

*Tab "Demo activity"*: one row per prospect link, rewritten by `sheet_sync`, so the team can see the whole funnel without opening the admin screen.

```
Prospect | Company | Industry | Batch | Buyer email | Link status | Expires on | First opened |
Visitors | Calls | Longest call | Goal completed | Follow-up bucket | Lead submitted
```

How it's set up:

- **Access:** a Google Cloud **service account** (key in `.env` as `GOOGLE_SERVICE_ACCOUNT_JSON`, base64 or a file path). The spreadsheet is shared with the service account's email as **Editor**. Nobody's personal Google login is used, and access can be removed by un-sharing the sheet.
- **Library:** `gspread` (wraps the Google Sheets API v4).
- **Who edits which columns:** columns from `Lead ID` to `Admin link` are written by the system and **protected** (Google sheet range protection: only the service account can edit them). `Status`, `Called by` and `Notes` are for the team. `Status` is a dropdown (data validation) with the values in `DEMO_LEAD_STATUSES`.
- **Two-way sync:** new leads are appended. Every `DEMO_SHEET_SYNC_MINUTES` (default 5), `sheet_sync` reads `Status`, `Called by` and `Notes` back into `demo_leads`, so the admin screen and the follow-up buckets see what sales did. The sheet wins for those three columns; the database wins for everything else.
- **Rows are found by `Lead ID`, not by row number.** The team can sort, filter, colour or add their own columns. Columns are matched by **header name**, so moving a column doesn't break the sync. If a required header is missing, sync stops and alerts instead of writing into the wrong column.
- **Formula injection is blocked.** Values are written with `valueInputOption=RAW`, and any typed value starting with `=`, `+`, `-` or `@` is prefixed with `'`. A prospect can't type `=IMPORTXML(…)` into the form and have it run in our sheet.
- **Rate limits:** Google allows about 60 write requests per minute per service account. Appends are batched (one API call per sweep for all pending leads), the activity tab is written in one `batch_update`, and 429 responses are retried with exponential backoff.
- **Setup check:** on startup (and on `GET /admin/demo/config`) the app confirms it can open the spreadsheet and that both tabs and their headers exist. It creates them if `DEMO_SHEET_AUTO_CREATE_TABS=true`.
- **Turning it off:** leave `DEMO_LEADS_GSHEET_ID` empty and leads go to the email and the admin screen only. A `leads.xlsx` download from the admin screen is always available as a fallback, built from the database.

**4. After the manual call**, the caller sets `Status` (New → Contacted → No answer → Follow up → Converted / Not interested), their name in `Called by`, and notes in the sheet. Filtering the sheet by `Status = New` gives today's call list, so nobody is called twice and nobody is forgotten. The same fields can also be edited in the admin screen; that change is written to the sheet on the next sync.

---

## 11. Admin screens (existing dashboard, `admin` role)

**As built (2026-10-05):** a **Demo links** page in `Generic-voice-agent-UI` (sidebar group "Growth"), shown only when `VITE_DEMO_ADMIN_ENABLED=true` **and** the backend accepts the user as a demo admin (the page probes `GET /admin/demo/industries`). The backend enforces the same rule on every request.
- **Tabs:** Overview (funnel + spend), Import (preview → apply), Records, Links (generate the CSV, revoke, extend), Calls (outcome, length, transcript).
- **Record drawer:** facts (inline edit), people (review, edit, exclude, get a link to any company's demo, add a person), documents (upload or delete, re-index), dummy data (preview at any moment), and a text test chat with the real agent.
- **Backend additions for the page:** `GET /admin/demo/sessions`, `GET /admin/demo/sessions/{id}/transcript`, `GET /admin/demo/metrics`.
- **QA gate:** not built yet (phase 7).


1. **Batches → Prospects table:** status, industry, minimum-to-go-live result, QA status, link status, opens/talks, bucket.
2. **Prospect detail:**
   - *Facts*: upload the facts sheet or edit fields inline; source links; blank-field badges.
   - *KB*: upload documents/text, ingest status, re-ingest.
   - *Sample data*: upload (or use the industry template), schema errors, **preview at a date**, version history and rollback.
   - *Agent*: regenerate from template, open in the existing test console.
   - *QA*: the checklist below + the QA call's transcript.
   - *Link*: go live / extend / revoke; copy link (shown once).
   - *Conversations*: every demo call on this link as **text**, newest first (see 3).
   - *Activity*: timeline of page views, calls, CTA clicks and leads.
3. **Conversations** (text only, no audio):
   - List: time, visitor (#1, #2… when a link was forwarded), duration, goal result, end reason.
   - Open one: the full transcript as a chat (Caller / Agent), with the tool actions shown inline ("booked visit: Thu 1 Oct 11:00, VIS-1187") and the sample data the call used.
   - Search across all demo conversations ("discount", "RERA") to spot what prospects ask most.
   - Built on the existing `calls` / conversation tables through `demo_sessions.call_id`, so no new transcript storage is needed.
4. **Leads** (the team mainly works in the Google Sheet; this is the same data):
   - Table: received at, kind, name, phone, email, company, industry, preferred time, demo calls, status (synced from the sheet).
   - Open one: details + their demo conversations + notes.
   - **Open Google Sheet** button; **Download leads.xlsx** fallback; **Resync sheet** (rebuilds the sheet from the database if rows were deleted by mistake).
   - "Not synced" badge if the email or sheet failed after all retries, with a **Resend** button.
5. **Industries:**
   - List of definitions with status and readiness ("Distributors: definition active, no dummy data yet → can't go live").
   - Upload or edit a definition (YAML/JSON) with validation errors shown inline; version history.
   - Download the generated facts and dummy data templates for that industry.
6. **QA gate** (all must pass, saved with who checked it and when). Five items apply to every industry; the industry definition's `compliance` list adds its own:
   - [ ] Says the company and project/product names correctly, including pronunciation
   - [ ] Gets price, address and timings right against the source page
   - [ ] Completes the use case end to end
   - [ ] Handles one Hinglish question and one off-topic question without inventing facts
   - [ ] Says the disclosure line and calls sample data "sample" if asked
   - [ ] *(from the definition, e.g. Clinics)* refuses medical advice and routes emergencies to a human
7. **Go live** requires: minimum met, KB ingest `ready`, active dataset, all QA items passed, a QA call ≥ 3 minutes on record (its transcript). Then: agent → `live`, link created, `expires_at` set, prospect → `live`.

### Automated QA support (reduces the target of about 10 minutes of human time per link)

- **QA question sets per industry** (the definition's `qa_questions`, taken from each industry sheet's QA table, e.g. "2 BHK ka starting price kya hai?" → "From ₹1.04 Cr, sizes 875–1,104 sq ft") run as a text-mode test against the prospect's agent, scored like the existing RAG eval (answered from the KB, nothing false or invented). Results are pre-filled into the QA screen; a human still makes the call.
- **Pronunciation check**: TTS renders the company and product names into a clip for the reviewer to listen to.

---

## 12. Lifecycle jobs

| job | schedule | does |
|---|---|---|
| `expire_demo_links` | hourly | `active` links past `expires_at` → `expired`; agent → `paused`; prospect → `expired` |
| `follow_up_buckets` | hourly | recompute §9.3; push changes to the webhook if `DEMO_WEBHOOK_URL` is set |
| `lead_notifier` | on submit + sweep every `DEMO_NOTIFY_RETRY_MINUTES` | email + append to the Leads tab for new leads, retries failures (§10.3) |
| `sheet_sync` | every `DEMO_SHEET_SYNC_MINUTES` | read Status / Called by / Notes back from the sheet; rewrite the Demo activity tab |
| `demo_retention_purge` | daily | `DEMO_TRANSCRIPT_RETENTION_DAYS` (default 90) after a link expires, delete its transcripts, events and visitors (including names and emails typed into the identity card), **unless** the prospect became a lead (leads and their conversations are kept until deleted by hand) |
| `prospect_erasure` | on request | `DELETE /admin/demo/prospects/{id}`: link, sessions, events, transcripts, KB docs and chunks, agent (soft delete), leads; recorded in the audit log |

No audio is ever stored for demo calls, so there are no recordings to clean up. The voice pipeline runs with recording off for `kind=demo` tenants.

---

## 13. Configuration (`.env`)

All settings are added to `Settings` in `core/app/config.py` (pydantic-settings, already reads `.env`), with the defaults below. Change `.env` and restart. No code change.

```dotenv
# ── Feature switch ──────────────────────────────────────────────────────────
DEMO_ENABLED=true                         # false = /public/demo/* returns 404, admin screens hidden
DEMO_TENANT_SLUG_PREFIX=demo-             # each prospect's tenant slug: demo-{short id}

# ── Domain / URLs ───────────────────────────────────────────────────────────
DEMO_PUBLIC_BASE_URL=https://demo.phonn.ai            # base of every link: {this}/t/{token}
DEMO_ALLOWED_ORIGINS=https://demo.phonn.ai            # comma-separated; CORS + WebSocket Origin check
DASHBOARD_URL=https://app.phonn.ai                    # used for "Open in admin" links in lead emails

# ── Link lifetime ───────────────────────────────────────────────────────────
DEMO_LINK_TTL_DAYS=7
DEMO_DEFAULT_TIMEZONE=Asia/Kolkata                    # for relative slots when a prospect has none

# ── Company links and tracking (0039) ───────────────────────────────────────
DEMO_LINK_TOKEN_KEYS=                     # Fernet key(s), required; first encrypts, all decrypt
DEMO_UTM_ENABLED=true
DEMO_UTM_SOURCE=email
DEMO_UTM_MEDIUM=cold_email                # LinkedIn URLs (step li) use linkedin / social
DEMO_UTM_CAMPAIGN=                        # empty = batch name (slugged), else demo_<industry>
DEMO_EXPORT_STEPS=e1,e2,e3,e4,li          # one URL column per step in the export
DEMO_IDENTIFY_PROMPT=unknown_only         # off | unknown_only | always

# ── Call limits ─────────────────────────────────────────────────────────────
DEMO_MAX_CALL_SECONDS=300
DEMO_MAX_CALLS_PER_LINK_PER_DAY=10
DEMO_MAX_CONCURRENT_CALLS_PER_LINK=1
DEMO_MAX_MINUTES_PER_LINK=60
DEMO_MAX_CONCURRENT_CALLS_GLOBAL=20

# ── Spend ───────────────────────────────────────────────────────────────────
DEMO_DAILY_BUDGET_USD=25                  # 0 = no cap
DEMO_BUDGET_ALERT_PCT=80
DEMO_BUDGET_TIMEZONE=Asia/Kolkata         # when "today" starts and ends

# ── Abuse protection ────────────────────────────────────────────────────────
DEMO_RATE_LIMIT_PER_MIN=30
DEMO_SESSIONS_PER_IP_PER_HOUR=5
DEMO_TURNSTILE_ENABLED=false              # required outside ENV=local
DEMO_TURNSTILE_SECRET=
DEMO_IP_HASH_SECRET=change-me             # HMAC key for ip_hash
DEMO_CLIENT_IP_HEADER=x-forwarded-for     # where the caller's IP comes from
DEMO_TRUSTED_PROXY_HOPS=1                 # proxies in front of the API (Railway = 1)

# ── Uploads ─────────────────────────────────────────────────────────────────
DEMO_UPLOAD_MAX_FILE_MB=20
DEMO_UPLOAD_MAX_FILES_PER_PROSPECT=50

# ── Lead email ──────────────────────────────────────────────────────────────
DEMO_EMAIL_PROVIDER=smtp                  # smtp | sendgrid | resend | none
DEMO_LEAD_NOTIFY_EMAILS=founder@phonn.ai,sales@phonn.ai
DEMO_ALERT_EMAILS=founder@phonn.ai        # budget / sync-failure alerts
DEMO_LEAD_FROM_EMAIL=demo-leads@phonn.ai
DEMO_LEAD_FROM_NAME=Phonn Demo Leads
DEMO_LEAD_EMAIL_SUBJECT_PREFIX=[Demo lead]
SMTP_HOST=smtp.gmail.com                  # used when DEMO_EMAIL_PROVIDER=smtp
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=                            # app password / SES SMTP credential, never committed
SMTP_STARTTLS=true
DEMO_EMAIL_API_KEY=                       # used when DEMO_EMAIL_PROVIDER=sendgrid|resend
DEMO_NOTIFY_RETRY_MINUTES=5
DEMO_NOTIFY_MAX_ATTEMPTS=5

# ── Leads Google Sheet ──────────────────────────────────────────────────────
DEMO_LEADS_GSHEET_ID=                     # the spreadsheet ID from its URL; empty = sheet off
GOOGLE_SERVICE_ACCOUNT_JSON=              # base64 of the key JSON, or a path to the key file
DEMO_SHEET_LEADS_TAB=Leads
DEMO_SHEET_ACTIVITY_TAB=Demo activity     # empty = don't write the activity tab
DEMO_SHEET_SYNC_MINUTES=5                 # how often Status/Notes are read back + activity rewritten
DEMO_SHEET_AUTO_CREATE_TABS=true          # create missing tabs/headers on startup
DEMO_LEAD_STATUSES=New,Contacted,No answer,Follow up,Converted,Not interested

# ── Industries ──────────────────────────────────────────────────────────────
DEMO_INDUSTRIES_SEED_DIR=app/demo/industries  # launch definitions (yaml) loaded on first start;
                                              # after that, admins manage them in the admin screen
DEMO_IMPORT_NEW_COLUMNS=propose           # a workbook column the industry doesn't map: propose | ignore
DEMO_IMPORT_IGNORE_COLUMNS=#,Minimum met? # bookkeeping columns that never become facts

# ── Tracking / retention ────────────────────────────────────────────────────
DEMO_WEBHOOK_URL=                         # optional: follow-up bucket changes are POSTed here
DEMO_TRANSCRIPT_RETENTION_DAYS=90
```

Rules:

- **Checked at startup.** If `DEMO_ENABLED=true`, the app checks the settings that must go together and refuses to start with a clear message if one is missing: `DEMO_PUBLIC_BASE_URL` must be https in production, `DEMO_ALLOWED_ORIGINS` must include its origin, the chosen `DEMO_EMAIL_PROVIDER` must have its credentials, the Google key is needed when `DEMO_LEADS_GSHEET_ID` is set (and the sheet must be openable with it), and the Turnstile secret is needed when Turnstile is on. `DEMO_EMAIL_PROVIDER=none` together with an empty `DEMO_LEADS_GSHEET_ID` is refused, because leads would reach nobody. A misconfigured deploy fails loudly instead of silently dropping leads.
- **Secrets** (`SMTP_PASSWORD`, `DEMO_TURNSTILE_SECRET`, `DEMO_IP_HASH_SECRET`, the Google key) live only in `.env` / Railway variables and are added to `.env.example` with empty values.
- **Visible to admins.** `GET /admin/demo/config` shows the current values (secrets masked), so anyone can check what's in effect without server access.

---

## 14. Observability

- Structured logs carry `demo_link_id`, `session_id`, `prospect_id` (never the raw token).
- Metrics: demo sessions active, `/session` 429s by reason, WS handshake failures by close code, demo spend today vs budget, time to first audio (existing TTFA) for demo calls.
- Alerts: budget at 80% / 100%, error rate on `/public/*` > 2%, any `hot` lead (business alert), leads not synced to the sheet or email after max attempts, sheet headers missing or sheet unreachable.
- Cost per call comes from the existing per-stage cost tracking on `calls`, joined through `demo_sessions.call_id`.

---

## 15. Testing

| layer | tests |
|---|---|
| Industry definitions | meta-schema rejects bad definitions (unknown field type, unknown tool block, missing goal); each launch definition compiles; **a test-only "fake industry" definition goes end to end (upload → templates → prospect → agent → call → goal) to prove no code is tied to the five known industries**; pinned version stays stable after a definition edit |
| Schemas | each industry: valid sample passes; absolute dates rejected; placeholder that doesn't resolve rejected; unmasked phone rejected; CSV/XLSX `loc` mapping |
| Tool blocks | each block (`lookup`, `list_slots`, `book_slot`, `create_record`, `update_record`) and each goal rule, independent of industry |
| Relative dates | `resolve_dataset` at fixed "now"s (the spec's examples: a call on 30 Sep offers 1 & 2 Oct, a call on 10 Oct offers 11 & 12 Oct), across midnight and timezone edges |
| Minimum check | pass and fail per industry |
| Isolation (§2a) | two prospects with conflicting facts each answer only their own (auto retrieval + `knowledge_lookup`, warm cache); token A can't read session B's transcript; voice token A refused on link B; demo tenants absent from billing and analytics |
| Tokens | hash lookup; revoked = unknown (same response); expired → 410; demo token can't reach the outbound path; `jti` reuse rejected; wrong Origin rejected |
| Limits | daily cap, concurrency, lifetime minutes, global cap, budget → 429; `max_seconds` cuts the call |
| Sessions | two parallel sessions on one link don't share `state`; outcome computed per use case |
| Events | `page_viewed` only counted from the beacon; `?s=` attribution; follow-up bucket rules |
| Leads | lead saved even when email or Google fails; retry sweep sends it later; alert after max attempts; duplicate submit same day → one lead; HTML in message is escaped in the email; token never appears in the email; each email provider through the `LeadMailer` interface (fake server) |
| Google Sheet | against a fake Sheets client: append batches pending leads; Status/Notes read back by `Lead ID` after the team sorts rows; columns matched by header after reordering; missing header stops sync and alerts; `=HYPERLINK(…)` in a form field is written as text, not a formula; 429 retried with backoff; resync rebuilds deleted rows; sheet off when `DEMO_LEADS_GSHEET_ID` is empty |
| Config | app refuses to start on bad combinations (§13); every limit read from settings, not hard-coded (tests override settings and check the new value is enforced) |
| Uploads | facts sheet and sample data XLSX round trip; bulk ZIP with one bad prospect imports the rest and reports the bad one |
| Agent QA | industry QA question sets in CI as a regression gate for template changes |
| E2E | Playwright: open link → microphone (fake media stream) → talk → transcript → CTA → contact |

---

## 16. Build phases

### 16.1 Overview

| phase | name | size | depends on | result you can see at the end |
|---|---|---|---|---|
| 0 | Foundations | S | — | settings load from `.env`, migration applied, feature switched off |
| 1 | Industries and dummy data engine | M | 0 | the 5 industry definitions compile; DSR dummy data previews with correct dates |
| 2 | Admin uploads and per-prospect tenant | L | 1 | DSR prospect created from uploaded files in its own tenant; KB `ready` |
| 3 | Demo agent and tool blocks | L | 2 | text chat with DSR's demo agent books a visit; QA question set passes |
| 4 | Public API, demo voice and limits | L | 3 | browser voice call to DSR's agent through a demo link; limits enforced |
| 5 | Leads: email and Google Sheet | M | 0 (models), 4 (link context) | a test lead lands in the inbox and the sheet; status in the sheet syncs back |
| 6 | Public demo UI | L | 4 (API contract), 5 (contact) | the 4-tab demo page works on phone and desktop |
| 7 | Admin screens, QA gate, jobs and tracking | L | 2–5 | full internal workflow: upload → QA → go live → export → track |
| 8 | Hardening and internal dry run | M | 1–7 | 4 real-looking demos run by the team end to end; checklist in §16.10 green |
| 9 | Pilot batch | — | 8 | first 40 links sent; funnel numbers visible |

Size is relative effort (S < M < L), not a date. Each phase is its own branch and PR, merged behind `DEMO_ENABLED=false` until phase 8, so half-built demo code never reaches the public.

**What can run in parallel:**

```
0 ─► 1 ─► 2 ─► 3 ─► 4 ─────────────► 7 ─► 8 ─► 9
 │                   │                ▲
 └──► 5 (leads) ─────┼────────────────┤
                     └─► 6 (UI) ──────┘
```

- Phase 5 (leads) only needs phase 0's models, so a second developer can build it alongside phases 1–4. It gets the link context from phase 4 at the end.
- Phase 6 (UI) can start as soon as phase 4's API shapes are written down (§8.2), against a mocked API, and switch to the real one when phase 4 merges.

### 16.2 Phase 0: Foundations — **built 2026-10-05**

What was built differs from the plan below in a few small ways:
- `DEMO_ENABLED` defaults to **false** in code and `.env.example` (the §13 block shows the production value).
- `tenants.demo_prospect_id` was dropped in favour of `demo_prospects.tenant_id` (unique), avoiding a circular foreign key.
- Industries use two tables (`demo_industries` + `demo_industry_versions`) so old versions are kept and pinned by foreign key.
- New libraries (`openpyxl`, `gspread`, `google-auth`, `aiosmtplib`) are added in the phases that first use them (2 and 5), not here, so nothing unused is pinned. `demo_links` also got a `max_concurrent_calls` override.
- Migration 0038 sits on 0036. R5.04's 0037 is on another branch; whichever merges second must re-point its `down_revision`.
- Because migration 0001 runs `Base.metadata.create_all()`, a **fresh** database gets the demo tables from the models, not from 0038. So every constraint, partial index and server default is declared in both `app/db/models/demo.py` and 0038 with the same names. `tests/test_demo_schema_parity.py` builds both paths on a scratch database and fails if they differ (run with `TEST_ADMIN_DATABASE_URL`, skipped otherwise).

Files: `app/config.py` (demo settings), `app/demo/settings.py` (startup check, called from `main.py` lifespan), `app/db/models/demo.py`, `app/db/models/platform.py` (`Tenant.kind`, `customer_tenant_ids()`), `migrations/versions/0038_demo_links.py`, `app/services/aggregator.py` + `app/services/telephony/dialer.py` (customer tenants only), `.env.example`, `tests/test_demo_settings.py`, `tests/test_demo_tenant_scope.py`, `tests/test_demo_schema_parity.py`.

Plan as written:

**Goal:** everything later phases need, with no behaviour change for existing users.

Backend:
- `core/app/config.py`: all `DEMO_*`, `SMTP_*`, Google Sheet settings from §13 with defaults; a `validate_demo_settings()` startup check (§13 rules). Add every key to `.env.example`.
- Migration `0038_demo_links.py`:
  - `tenants.kind` (`customer` default) and `tenants.demo_prospect_id`
  - tables `demo_industries`, `demo_batches`, `demo_prospects`, `demo_prospect_fields`, `demo_sample_datasets`, `demo_links`, `demo_sessions`, `demo_events`, `demo_leads`, with the constraints and indexes from §4
  - a downgrade that drops them cleanly (guard index drops, as in 0020)
- `core/app/db/models/demo.py`: the SQLAlchemy models.
- Billing, usage limits, dashboards, analytics aggregator: filter `tenants.kind = 'customer'` so demo tenants never show up there.
- `core/app/demo/` package skeleton: `industries/`, `schema/`, `dataset/`, `tools/`, `public/`, `leads/`, `jobs/`.
- New dependencies: `openpyxl`, `gspread`, `google-auth`, `aiosmtplib`. (`jsonschema` and `PyYAML` are already in `requirements.txt`.)

Tests: migration up and down on a copy of the dev database; settings validation (each bad combination refused); existing analytics and billing tests still pass with a demo tenant present.

**Exit:** app starts with `DEMO_ENABLED=false` and behaves exactly as before; with `DEMO_ENABLED=true` and bad settings it refuses to start with a clear message.

### 16.3 Phase 1: Industries and dummy data engine

**Goal:** the generic, industry-independent core: definitions, schemas and date handling, as pure code with no API yet.

Backend (`core/app/demo/`):
- `industries/meta_schema.py`: the meta-schema every industry definition must pass (§4.8).
- `schema/field_types.py`: `string`, `integer`, `boolean`, `money`, `relative_slot`, `relative_day`, `sample_caller`, `ref_code`, `enum`, `list`, `object`, `text_template` (§6.2).
- `schema/compiler.py`: definition `sample_data` block → JSON Schema; definition `facts_fields` → facts template spec.
- `dataset/validate.py`: check a dataset against the compiled schema; global rules (no absolute dates, masked phones, placeholders that resolve); errors with `loc`.
- `dataset/resolve.py`: `resolve_dataset(dataset, fields, now, tz)` → `ResolvedContext` with real dates and spoken forms (reuse `services/spoken.py`, `packs/locale.py`).
- `industries/minimum.py`: the generic minimum-to-go-live checker.
- `industries/seed/*.yaml`: launch definitions for real estate, D2C, clinics, hospitality and distributors, written from the industry sheets. Loaded into `demo_industries` on first start from `DEMO_INDUSTRIES_SEED_DIR`.

Tests:
- every launch definition passes the meta-schema and compiles
- the DSR Highland Greenz dummy data (from the real-estate sheet) validates
- `resolve_dataset` at the spec's dates: a call on 30 Sep offers 1 and 2 Oct; a call on 10 Oct offers 11 and 12 Oct; midnight and timezone edges
- each global rule rejects a bad dataset with the right `loc`
- a test-only fake industry definition compiles and validates (proves nothing depends on the 5 known industries)

**Exit:** all of the above green; a script prints DSR's resolved dummy data for any given date.

#### Phase 1 as built (2026-10-05), real estate only

Scope was narrowed to **real estate** (the other industries come later, as YAML only), and widened to include the **workbook dry-run importer**. The prospect workbook (`Phonn_list_3.xlsx`) turned out to be company records plus the people to email, so the importer belongs with the engine.

| file | what it does |
|---|---|
| `app/demo/industries/seed/real_estate.yaml` | the real-estate definition: 12 facts mapped to the sheet's columns, the PDF's minimum (project, location, ≥2 BHK types, sales office), the PDF's dummy scenario as the template (Rahul, slots tomorrow 11 am / 4 pm / day after 11 am, Priya, VIS-1187), agent persona and guardrails, tools, prompts, the PDF's QA questions |
| `app/demo/industries/definition.py` | meta-schema + cross-checks (unknown placeholders, tools pointing at real slot lists, goal needs a matching tool, template passes its own schema); compiles to `IndustryDefinition` |
| `app/demo/industries/registry.py` | loads seeds; on startup (when `DEMO_ENABLED`) creates missing industries + version + template dataset. **Never overwrites** an industry that already exists (admin edits are safe) |
| `app/demo/sample/compiler.py` | the 12 field types → JSON Schema |
| `app/demo/sample/validate.py` | schema check + shared rules: no absolute dates, masked phones, placeholders only in templates and only naming filled-in facts |
| `app/demo/sample/resolve.py` | relative slots → real dates in the record's timezone with spoken forms; past slots dropped; skipped weekdays shifted |
| `app/demo/facts.py` | sheet cell → fact (`text`, `list`, `unit_types`, `month`, `url`); BHK parsing; go-live minimum check |
| `app/demo/contacts.py` | splits the people columns into contacts: best-match email pairing (handles `achauhan@`, `gollapellitejesh@`, `sreedhar.g@`), near-spelling pairing flagged for review, shared inboxes kept as company emails, webmail flagged, never guesses on a tie |
| `app/demo/importer.py` | dry-run import of an industry tab → `ImportReport` (per row: ready / below minimum / invalid with reasons, contacts, warnings). Phase 2 persists from this same report |
| `scripts/demo_import_preview.py` | CLI: preview a workbook, optionally resolve the dummy data at any date (`--preview-at`), write JSON |

Run against the real workbook (Real estate tab, 30 companies):

- **17 records ready, 13 below minimum.** The sheet's own "Minimum met?" says Yes for all 30. It only checks for non-empty cells. Ours applies the PDF's rule, so the research team can see exactly what's missing.
  - 5 records have only one BHK type: Adarsh, Gaur, Appaswamy, Siddha, PS Group.
  - 4 records have no configurations at all: Chandak, Mahagun, ATS, Merlin.
  - 7 records have no sales office address: Concorde, Mahagun, ATS, Gaur, Aparna, My Home, Vasavi.
- **153 contacts, 113 with a matched email; 51 flagged for review.** These mostly lack an email in the sheet, and a few have only a personal webmail address. One record (Eldeco) lists five emails belonging to people not in its name list. Ambiguous matches are never guessed.

Tests: `test_demo_sample_engine.py`, `test_demo_industry_definition.py` (includes a made-up "pet grooming" industry going end to end), `test_demo_facts_contacts.py`, `test_demo_importer.py` (synthetic workbook with the real layout; no real people's data in the repo), `test_demo_registry_db.py` (scratch DB; needs `TEST_ADMIN_DATABASE_URL`).

### 16.4 Phase 2: Admin uploads and per-prospect tenant

#### Phase 2 as built (2026-10-05)

**Access.** Roles are per tenant and `admin` holds `*`, so the plain role check would have let **any customer's admin** read every prospect and lead. `/admin/demo/*` is therefore limited to principals from `DEMO_ADMIN_TENANT_IDS` (our own staff tenant, a new required `.env` setting) holding `demo:admin` (`*` includes it). With `DEMO_ENABLED=false` every route answers 404.

**Endpoints** (`core/app/api/demo_admin.py`; logic in `core/app/demo/records.py`):

| area | routes |
|---|---|
| industries | `GET /industries`, `GET /industries/{key}`, `POST /industries/validate`, `POST /industries` (next version; `activate`), `GET /industries/{key}/facts-template.xlsx`, `GET/PUT /industries/{key}/dummy-data`, `POST …/dummy-data/rollback/{v}` |
| imports | `POST /imports` (upload → stored dry-run report, writes nothing else), `GET /imports`, `GET /imports/{id}`, `POST /imports/{id}/apply`, `POST /imports/{id}/discard` |
| batches | `POST /batches`, `GET /batches` |
| records | `GET /records` (search, industry, minimum filter, paging), `GET/PATCH /records/{id}`, `PATCH /records/{id}/facts`, `POST /records/{id}/reindex` |
| contacts | `GET/POST /records/{id}/contacts`, `PATCH /contacts/{id}` (edit, batch, exclude, `reviewed`) |
| documents | `GET/POST /records/{id}/documents` (PDF, HTML, TXT, MD or pasted text), `DELETE …/{doc_id}` |
| dummy data | `GET/PUT/DELETE /records/{id}/dummy-data`, `POST …/rollback/{v}`, `GET …/preview?at=` |

**How it behaves:**
- **Preview, then apply.** An upload is parsed and its report stored in `demo_imports`. The report marks each record as create or update, with the facts that would change. **Apply saves exactly that stored report**, once. It's refused if the industry definition changed since the preview.
- **New record** → its own `kind=demo` tenant and knowledge base, plus a generated **Company facts** document. Blank facts are written as "not published; the sales team will confirm", so the agent finds an answer instead of guessing. Embedding runs after the commit, one document at a time. If a deploy interrupts it, `POST /records/{id}/reindex` re-queues it.
- **Re-import:** facts follow the sheet. Contacts are never overwritten: only blanks are filled and new people added, so an admin's review survives. People missing from the sheet are reported, not deleted. An unchanged re-import re-embeds nothing, because the document's content hash is unchanged.
- **Dummy data:** a record uses the industry template until it gets its own. Every save is a new version; rollback and "back to template" are one call each. The preview shows exactly what a call at a given moment would get.
- **Documents:** the file type is read from the bytes, not the extension. HTML is stripped to visible text; Word/ZIP files and scanned PDFs are refused with a clear message. Identical content is refused as a duplicate. The generated facts document can't be deleted (edit the facts instead).
- **Audit:** every change writes an `audit_events` row on the admin's tenant.

**Schema additions in 0038 (still unreleased):** `demo_imports`.
**New dependencies:** `pypdf`, plus `openpyxl` from Phase 1, so rebuild the core image.

**Verified:** the real workbook imports into a scratch database: 30 records, 153 contacts, 30 facts documents queued, and an identical re-import changes 0 facts. `tests/test_demo_admin_api.py` (15 tests, scratch DB, embedding stubbed) covers:
- **Access control:** a customer's admin is refused, and every route is 404 while disabled.
- **Industries:** publishing a new version, and refusing an identical one.
- **Imports:** preview, apply and re-import.
- **Facts edits:** the generated document is re-rendered.
- **Contacts:** an admin's review survives a re-import, and the contact rules hold.
- **Documents:** HTML stripping, duplicates and Word files refused, and one record can't reach another record's documents.
- **Dummy data:** the full lifecycle.
- **Isolation:** separate tenants and knowledge bases per record.

Plan as written:

**Goal:** an admin can create prospects and upload everything a demo needs.

Backend (`core/app/api/demo_admin.py`, admin role only):
- Industries: list, upload, new version, validate (§5.2).
- Prospects: create one → creates the **demo tenant** (§2a), its KB, and copies the shared industry KB documents into it.
- Facts: upload XLSX/CSV or JSON upsert; generated "Company facts" KB document (blank fields written as "the sales team will confirm").
- KB documents: files and text, through the existing ingest pipeline into the prospect's tenant and KB; source metadata; size and count limits; content-hash dedupe.
- Dummy data: industry template upload; per-prospect upload with `dry_run`, versions, rollback, preview at a date.
- Templates: generated facts and dummy-data XLSX per industry.
- Bulk ZIP import per batch with dry-run report; one bad prospect doesn't block the rest.
- Audit log entries for every upload.

Tests: XLSX round trips; bulk import with one bad folder; uploading to prospect A never creates rows in prospect B's tenant; minimum check result shown per prospect; Distributors reports "no dummy data".

**Exit:** DSR prospect created entirely from uploaded files (no manual DB edits), KB `ready`, minimum check green, preview matches phase 1.

### 16.5 Phase 3: Demo agent and tool blocks

#### Phase 3 as built (2026-10-05), real estate

**Design change from §6.5 for real estate.** The planner references `book_appointment` 57 times across 12 files: identity checks, slot-was-offered, booking-claim and read-back guards. New generic tool names would bypass all of that. So an industry may now run on an existing **engine pack** (definition `engine:` block). The demo tenant uses that pack's own tools and planner, configured from the record. The generic building blocks stay for future industries without a pack.

What `provision_demo_agent()` (`app/demo/agent.py`) writes, **only inside the record's own demo tenant**:

| row | content |
|---|---|
| `tenant_pack_config` | references `real_estate_india` (the pack itself is not changed) |
| `tool_definitions` | the definition's **allowlist** of pack tools. Excluded: `inventory_query` (an empty unit search made the agent deny a listed 2 BHK), `send_document` (WhatsApp), `create_customer`, tickets, do-not-contact. `executive_id` is removed from the demo copies (a live run passed "Priya" as `executive_id` and the booking failed) |
| `locations` | one visit location: record name, sales office (or location) address, `booking_reference` from the dummy data |
| `tenants.settings.project_facts` | every fact (blanks as "not published; the sales team will confirm"), in the planner's context on every turn |
| `prompt_templates` (persona) | industry persona and instructions; what the agent knows before the call (sample lead, lead source); the sales contact for callbacks; the booking reference; qualifying questions; languages; always-on guardrails (demo/sample-data disclosure, no invented facts, stay on topic, never claim to be human) + industry guardrails |
| `agents` | a row describing the agent |

Provisioning is idempotent and runs on import, fact edits, dummy-data changes, and at the start of every call. It refuses any tenant that isn't `kind='demo'`.

**Two small changes to shared engine code.** Both are additive, and inert for customer tenants:
- `services/scheduling.compute_slots`: for `kind='demo'` tenants it returns the record's dummy-data slots, resolved at call time (`app/demo/scheduling.py`). Customer tenants return early from a cached tenant-kind lookup and run exactly as before. Search, availability checks, booking and reschedule are otherwise untouched.
- `book_appointment` output and the P3.04 booking template: they add `booking_reference` / read it back **only when the location carries one**. Only demo locations do. Existing template tests still pass, with new ones added.

**Each call:** `start_demo_conversation()` creates a fresh sample lead ("Rahul", placeholder `+910…` number, `demo_sample` tag) and speaks the definition's greeting with the disclosure. Every call starts from a clean calendar; other callers' demo bookings don't reduce availability.

**Admin endpoints:** `POST /records/{id}/agent` (provision), `POST /records/{id}/chat`, `POST /records/{id}/chat/{session}/turns` (real engine, admin sees the debug envelope), `GET /records/{id}/chat/{session}`.

**Live verification** (real LLM, scratch DB, Sumadhura from the real workbook, 10+ conversations):
- **Correct from the facts:** RERA number and possession date. Price not published, so the agent offers a sales callback. Banks not published, so "the team will confirm". Discount goes to a Priya callback.
- **Disclosure:** the greeting says it's a demo on sample data, and "Is this real data?" is answered correctly.
- **Booking:** "Kal 11 baje visit book karo" → locate centre → search slots (dummy 11 am) → book → "Your visit is confirmed for Tuesday, 6 October at 11:00 AM, at Sumadhura Edition, booking reference VIS-1187."
- **After the allowlist and `executive_id` fixes:** 4 of 5 consecutive runs booked correctly.
- **The fifth hit an engine-wide guard gap.** The reply was "Your visit for the 2 BHK … is confirmed" after only a slot search, with no booking. `booking_claim.py` misses claims with a phrase between "your visit" and "is confirmed". This affects customer tenants too, so it's flagged as its own fix, not changed inside demo work.

**Not changed:** any existing seed script, pack, tenant, `industry_packs` row or global tool definition. The tests assert that no engine row written by the demo lands in a customer tenant.

#### UI plan (decided 2026-10-05)

- **Demo console:** a **separate Vite entry** in `Generic-voice-agent-UI` (`npm run build:demo` → its own bundle on `DEMO_PUBLIC_BASE_URL`).
  - `npm run build` and the dashboard stay exactly as they are.
  - The console reuses the existing Test Console's look and the voice code; `useVoiceCall.js` gets only an optional URL parameter.
  - No customer picker, outbound mode or debug panel. The server also stops sending the per-turn debug data (`turn_debug`: tools, rules, cost) on demo connections.
- **Demo analytics:** a "Demo links" page in the dashboard, shown only when `VITE_DEMO_ADMIN_ENABLED=true` and the user is a demo admin. It reuses the Analytics page components. Demo calls live in demo tenants, so they never appear in the regular Analytics page.
- **Links:** ~~one per person~~ **one per company (2026-10-06)**, for every company, including those below the minimum. Every active person gets their own code on it; contacts flagged for review are included and the export marks the flag.

**Goal:** a working demo agent per prospect, testable by text before any voice or public page exists.

Backend:
- `demo/agent_builder.py`: from the pinned industry definition, create or regenerate the prospect's `Agent` (persona, languages, greeting + disclosure line added by code), attach KBs, tools and guardrails (always-on guardrails from code + industry ones) (§7).
- `demo/tools/`: the 5 building blocks (`lookup`, `list_slots`, `book_slot`, `create_record`, `update_record`) registered in the existing tool registry under the names the definition gives, attached through `AgentTool`.
- `demo/provider.py`: `SampleDataProvider` reading `resolved_context` and writing `demo_sessions.state`.
- `demo/goals.py`: goal rules → `outcome`.
- Text-mode runner: `POST /admin/demo/prospects/{id}/chat` runs a turn through the normal engine with a demo session, for QA and for automated tests.
- Automated QA: run the definition's `qa_questions` against the agent, score like the RAG eval, store the result for the QA screen.

Tests:
- each tool block and goal rule on its own
- DSR: "Kal 11 baje visit book karo" books tomorrow's 11 am slot and reads back VIS-1187; RERA and bank questions are not invented
- two parallel sessions on one prospect don't share bookings
- **isolation:** two prospects with conflicting "2 BHK" prices each answer only their own, via automatic retrieval and the `knowledge_lookup` tool, with a warm cache
- fake industry from phase 1 goes end to end by text

**Exit:** DSR's QA question set passes in text mode; fake-industry and isolation tests green.

### 16.6 Phase 4: Public API, demo voice and limits

**Goal:** a prospect can talk to their agent through their link, safely.

Backend:
- Link creation: token generation, hash storage, expiry, revoke, one active link per prospect (§8.1).
- `core/app/api/demo_public.py`: page data, `/session`, transcript, `/events` (§8.2), 404/410 handling, `noindex` and `Referrer-Policy` headers, CORS from `DEMO_ALLOWED_ORIGINS`.
- `demo_voice` token type in `security/auth.py`: short TTL, single-use `jti` in Redis.
- `voice_ws.py`: demo branch in `_authenticate` (§8.3): re-check link, Origin check, inbound only, `SampleDataProvider` attached, hard call timer, session close writes duration, end reason and outcome.
- Limits (§8.4): per-link daily calls, concurrency, lifetime minutes, global concurrency (Redis counters with TTL), daily budget from cost rows, per-IP rate limits, Turnstile verify when enabled. All values from `.env`, per-link overrides respected.
- Events: beacon-only `page_viewed`, `?s=` step attribution, `visitor_id` cookie.

Tests (§15 Tokens, Limits, Sessions, Events, Isolation rows): expired → 410; revoked = unknown → 404; demo token cannot reach the outbound path; reused `jti` refused; wrong Origin refused; each limit returns 429 and is read from settings; token A can't read session B's transcript; voice token A refused on link B.

**Exit:** a raw WebSocket test client (and the existing console hook pointed at the demo endpoints) holds a full voice call with DSR's agent through a demo link; the call is cut at `DEMO_MAX_CALL_SECONDS`; spend shows up against the budget.

#### Change 2026-10-06: per-company links and who-opened tracking

**Why:** one link per person meant colleagues got different demos' URLs for the same company, and nothing told us who opened a forwarded link. Now each company has one link, and each person's copy carries a code.

- **Migration 0039** (`0039_demo_company_links.py`): `demo_link_recipients`, `demo_visitors`; `demo_links` drops `contact_id`, gains `token_ciphertext` and one-active-per-record; sessions and events gain `demo_visitor_id` / `recipient_id`; event type `visitor_identified`. Existing per-person links become recipients (old calls keep their person) and are revoked, since their tokens were never stored. Guarded for both the fresh (`create_all`) and the upgrade path; the parity test checks fresh, 0038+0039, and 0039 over an existing 0038.
- **Code:** `app/demo/links.py` (company links, encryption, URLs with code + UTM), `app/demo/visitors.py` (attribution, identify), `app/demo/activity.py` (per-link activity, questions, "who" for every row), `app/api/demo_public.py` (`/visit`, `/identify`; events and sessions attributed), `app/api/demo_admin.py` (below).
- **Admin API:** `POST /records/{id}/link` (get or `replace`), `GET /links` (per company, with people opened / other visitors / calls / last activity), `GET /links/{id}/urls` (audited), `POST /links/{id}/recipients`, `GET /links/{id}/activity`, `GET /questions`, `POST /links/generate` (one row per person). `POST /contacts/{id}/link` is gone. Sessions, transcripts and metrics now say who; metrics count per company and per person.
- **UI:** demo page reads and strips `r`/`s`/`utm_*`, records the visit after 2 s visible or a touch, shows the optional identity card. Admin: Links tab per company with an activity view (people, other visitors, questions, calls, timeline) and a "Links to send" view; new Questions tab; Overview split into Companies and People; the record drawer shows the company link and who opened it.
- **Tests:** `test_demo_links_api.py` (26), parity + carry-over migration test, settings tests for the new keys.

### 16.7 Phase 5: Leads, email and Google Sheet

#### As built (2026-10-06)

- **Code:** `app/demo/leads/` — `service.py` (intake, rows, the worker: push pending leads, sheet sync, alerts, advisory lock), `sheets.py` (tabs/headers, append by Lead ID, read-back, activity tab, formula guard; logic on a `SheetBackend` interface, `GSpreadBackend` for real), `mailer.py` (`smtp` · `sendgrid` · `resend` · `none`). Worker started in `main.py` when `DEMO_ENABLED`.
- **Public:** `POST /public/demo/{token}/contact` (`contact` · `schedule_call` · `real_calls_pilot` · `rebuild_request`; phone required for the call kinds; works on an expired link, not a revoked one; same email + kind on one day returns the first lead; an unknown or forwarded visitor becomes identified). The tokenless `/public/demo/contact` was not built.
- **Admin:** `GET /leads`, `GET /leads/status?check=true`, `POST /leads/sync`, `POST /leads/{id}/resend`, `POST /leads/resync-sheet`, `GET /leads.xlsx`.
- **Sheet columns:** Leads = Lead ID … Admin link (ours) + Status, Called by, Notes (team's); we also refresh Demo calls / Longest call / Goal completed / Questions asked on existing rows. Demo activity = one row per non-revoked link incl. "Opened by" names. A number or phone number is never prefixed with `'`.
- **UI:** demo page **Schedule a call** tab (Contact us dropped 2026-10-06: same purpose; the API still accepts `contact`), "Run this on real calls for 14 days" after a call, "Rebuild my demo" form on the expired page; admin Leads tab (status bar with sheet link, check, sync now, put back deleted rows, leads.xlsx; per lead: status from the sheet, delivery, resend, their call).
- **Not built yet:** status dropdown is best effort; protected ranges for our columns. (Turnstile on the page: built 2026-10-06, `src/demo/turnstile.js`.)
- **Tests:** `test_demo_leads_sheet.py` (fake sheet), lead tests in `test_demo_links_api.py`.

**Goal:** no lead is ever lost, and the sales team gets every lead in their inbox and their sheet.

Backend (`core/app/demo/leads/`):
- `POST /public/demo/{token}/contact` and `/public/demo/contact`: validation, phone normalisation, Turnstile, same-day duplicate guard, `demo_leads` row committed before responding (§10.3).
- `mailer.py`: `LeadMailer` interface; `smtp` implementation first (aiosmtplib), then `sendgrid` / `resend`; `none`. HTML-escaped, plain-text + HTML, Reply-To the lead, no token in the email.
- `sheets.py`: gspread client from the service account; create tabs and headers if missing; append in batches; read Status / Called by / Notes back by `Lead ID`; columns matched by header; `RAW` writes + formula-prefix escaping; backoff on 429; protected system columns; Status dropdown.
- `notifier.py`: outbox sweep every `DEMO_NOTIFY_RETRY_MINUTES`, attempts counter, alert after `DEMO_NOTIFY_MAX_ATTEMPTS`.
- `sheet_sync.py`: two-way sync every `DEMO_SHEET_SYNC_MINUTES`; rewrites the Demo activity tab.
- `leads.xlsx` fallback export; `POST /admin/demo/leads/resync-sheet`.
- Background loops follow the existing `asyncio.create_task(...)` pattern in `main.py`, each guarded by a **Postgres advisory lock**, so running more than one server instance never sends a lead twice.

Tests (§15 Leads and Google Sheet rows), against a fake SMTP server and a fake Sheets client; plus one manual check against a real test sheet and inbox.

**Exit:** a test lead arrives in the inbox and the Leads tab within a minute; typing "Contacted" in the sheet shows up in the database on the next sync; with SMTP and Google both down, the lead is kept and delivered once they're back.

#### Change 2026-10-07: every workbook tab, data-only industries, new columns

The import was real-estate only. Now any tab of the prospect workbook imports, and a tab or
column added later needs no code.

- **Every tab has an industry.** Seeds added for `d2c`, `hospitals`, `restaurants`, `hotels` and
  `distributors` (from each tab and the workbook's Field guide). They are created on the next
  start (seeds only create; existing industries are untouched).
- **Data-only industries.** A definition's demo-agent part (`sample_data`, `sample_template`,
  `agent`, `prompts`, plus `engine`/`tools`) is now all-or-nothing and optional. Without it the
  industry is *data only*: its tab imports (records, own tenant and KB, people, facts, Company
  facts document) but it has no dummy data, test chat or links (409 with the reason) until a
  later version adds the agent. The five new industries are data only; real estate is unchanged.
  `GET /industries` reports `has_agent`, `can_import`, `tab_names`, `use_case`.
- **Either/or minimums.** `minimum` accepts `{any_of: [a, b], min_items?, message}` ("doctors or
  services", "menu or price range") besides `{field, min_items}`.
- **Pick the tab, industry detected.** `POST /admin/demo/imports/inspect` lists every tab with the
  industry whose `sheet.tab_names` claims it, header row, rows and example rows, use case; tabs
  that look like prospect lists but no industry claims are `kind: new`. `POST /imports` takes
  `tab` (any tab, e.g. a renamed one) besides `industry_key`.
- **New columns.** A column the definition doesn't map is not dropped. The preview returns it in
  `report.new_columns` with its values already in the records: `action: map` when it feeds an
  existing fact (a fact no column fed yet, matched by name, e.g. "Areas served" → `service_area`;
  or a renamed column, matched by spelling, e.g. "Shipping timeline" → `shipping_timelines`), else
  `action: add` (a new optional text fact keyed from the header). Applying publishes the
  definition + those columns as the industry's **next version** (audit event
  `demo.industry_columns_add`) and saves the records on it; dummy data is untouched. Unticking a
  column re-runs the preview with `skip_columns` (JSON list), so what is previewed is exactly
  what is applied. Bookkeeping columns (`DEMO_IMPORT_IGNORE_COLUMNS`, `sheet.ignore_columns`)
  never become facts; `sheet.extra_columns: ignore` (or `DEMO_IMPORT_NEW_COLUMNS=ignore`) turns
  proposals off, and a missing column then gets a "renamed to …?" hint. A preview made before
  the industry's version changed is refused on apply (409).
- **A brand-new tab.** `POST /admin/demo/industries/draft` (file, tab) drafts a data-only
  definition from the tab: company/people columns are recognised from the existing industries'
  mappings, the first other column is the company, the rest are facts; the Field guide (if any)
  gives descriptions, required facts and the trusted-source rule. The admin edits the YAML and
  publishes it with `POST /industries`; the tab then imports like any other.
- **UI (Import tab).** Upload → tabs table (industry, rows, use case; "Set up industry" for a new
  tab, opening the drafted YAML) → preview with a "columns this industry doesn't have yet" card
  (tick/untick) → apply. Record drawer hides Dummy data / Test chat and the link button for data-only records.
- Code: `app/demo/workbook.py` (inspect, Field guide, draft), `app/demo/importer.py`
  (`propose_columns`, `apply_column_proposals`), `records.add_columns_to_industry`. Tests:
  `tests/test_demo_import_industries.py`, `tests/test_demo_import_industries_db.py`.

### 16.8 Phase 6: Public demo UI

**Goal:** the page the prospect actually sees.

Frontend (separate Vite entry or package in `voice-agent-ui`, §10.1):
- Routing `/t/{token}` with tabs: Test console, Knowledge base, Sample data, Contact us / Schedule a call; expired state; 404 state.
- Test console: Talk button, microphone explainer and permission handling, connecting / listening / speaking states, countdown, live transcript, 3 prompt chips, post-call transcript and goal result, "Run this on real calls" button. Reuses `useVoiceCall.js`.
- Knowledge base and Sample data tabs from the page-data API.
- Contact form with client-side validation, Turnstile, disabled submit while sending, confirmation screen.
- Beacon for `page_viewed` (render + visible 2 s or interaction), CTA events, `?s=` stripped after reading.
- `noindex` meta, `robots.txt`, CSP, no third-party analytics; API URLs from `VITE_DEMO_API_URL`.
- Mobile first: iOS Safari and Android Chrome microphone flows.

Tests: Playwright E2E with a fake microphone stream: open link → talk → transcript → CTA → contact; expired link; limit-hit message; phone and desktop viewports.

**Exit:** E2E green on mobile and desktop viewports; Lighthouse accessibility pass; page loads fast on a phone connection.

### 16.9 Phase 7: Admin screens, QA gate, jobs and tracking

**Goal:** the team can run the whole workflow without developers.

Frontend (existing dashboard, admin role, §11): Industries; Batches → Prospects table; Prospect detail tabs (Facts, KB, Sample data, Agent, QA, Link, Conversations, Activity); Conversations list, transcript view and search; Leads (sheet link, xlsx fallback, resync, resend); config view (`GET /admin/demo/config`, secrets masked).

Backend:
- QA gate: checklist (5 common items + the definition's `compliance` items) saved with who and when; go live blocked until minimum, KB ready, dataset active, QA passed and a ≥ 3-minute QA call transcript exist.
- Batch export CSV for the email tool (§9.4).
- Jobs (§12): `expire_demo_links`, `follow_up_buckets` (+ optional webhook), `demo_retention_purge`, `prospect_erasure` (deletes the prospect's tenant).
- Funnel metrics endpoint (§9.5).
- Alerts (§14): budget, sync failures, public error rate, hot leads.

Tests: go-live gate refuses each missing condition; expiry flips link, agent and prospect; follow-up buckets match the spec's trigger table; erasure leaves nothing behind for that prospect.

**Exit:** a non-developer creates a prospect, uploads data, runs QA, goes live, exports the CSV and reads a test conversation, all from the screens.

### 16.10 Phase 8: Hardening and internal dry run

**Goal:** prove it under real conditions before any prospect sees it.

- Build 4 internal demos (real estate, D2C, clinics, hospitality) from the real industry sheets.
- Team members play the prospect on phones and laptops, including Hinglish, off-topic and "is this real data?" questions; run the full QA checklist.
- Load check: `DEMO_MAX_CONCURRENT_CALLS_GLOBAL` simultaneous calls; confirm latency and that extra callers get the friendly 429.
- Failure drills: stop Redis, block SMTP, revoke the Google key, exhaust the budget. Each fails safe and alerts.
- Security pass: run `/security-review` on the branch; check headers, CORS, Origin, token handling, formula injection, isolation tests.
- Production `.env` filled and checked by `GET /admin/demo/config`; set `DEMO_ENABLED=true` in production.

**Exit checklist:**
- [ ] all §15 test rows green in CI
- [ ] 4 internal demos pass QA
- [ ] test lead reaches inbox and sheet in production
- [ ] failure drills each alert and recover
- [ ] isolation tests green against the production-like database
- [ ] retention and erasure verified on a throwaway prospect

### 16.11 Phase 9: Pilot batch

- Batch 1: 40 prospects (10 each across the 4 industries with data). Distributors joins a later batch once its dummy data is uploaded; no code change needed.
- Export CSV → email tool → send.
- Watch daily: opens, talk rate, hot leads, QA failures, spend vs budget.
- After two batches: reset the targets (§9.5), tune minimums and prompts in the industry definitions, and adjust `.env` limits from real usage.

---

## 17. Open decisions

Nothing blocks the build. These are values to fill in before going live, all in `.env` or the admin screen:

1. **Lead email**: the recipient addresses, the sender address, and which provider sends (SMTP account details or an API key).
2. **Google Sheet**: create the spreadsheet, create a Google Cloud service account, and share the sheet with it as Editor. Put its ID and the key in `.env`. Tabs and headers are created automatically.
3. **Distributors dummy data**: upload it whenever it's ready; the industry turns on by itself once the minimum check passes.
4. **Minimums for D2C, Clinics and Hospitality**: the starting values in §5.4 are our proposal; adjust them in the industry definitions if the team disagrees.
