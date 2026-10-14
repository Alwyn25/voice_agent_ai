"""Dry-run a prospect workbook against an industry definition and print what would be
imported (docs/gtm-demo-links.md §5). Writes nothing.

    cd core
    python scripts/demo_import_preview.py path/to/Phonn_list.xlsx --industry real_estate
    python scripts/demo_import_preview.py path/to/Phonn_list.xlsx --json report.json
    python scripts/demo_import_preview.py path/to/Phonn_list.xlsx --preview-at 2026-10-09T19:00+05:30

--preview-at also resolves each ready record's dummy data at that moment, so you can hear
exactly which visit slots a call at that time would offer.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo.importer import parse_workbook  # noqa: E402
from app.demo.industries.registry import load_seed_definitions  # noqa: E402
from app.demo.sample.resolve import resolve_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workbook")
    ap.add_argument("--industry", default="real_estate")
    ap.add_argument("--json", help="write the full report as JSON to this file")
    ap.add_argument("--preview-at", help="ISO timestamp with offset; resolve dummy data at it")
    args = ap.parse_args()

    defs = {d.key: d for d in load_seed_definitions()}
    if args.industry not in defs:
        print(f"unknown industry {args.industry!r}; seeds: {', '.join(sorted(defs))}")
        return 2
    definition = defs[args.industry]
    report = parse_workbook(args.workbook, definition)

    print(f"{report.source_name} › tab {report.tab!r} › header row {report.header_row}")
    print("summary:", json.dumps(report.summary))
    for e in report.errors:
        print("ERROR:", e)
    for w in report.warnings:
        print("warning:", w)
    for s in report.skipped:
        print(f"skipped row {s['row']} ({s['company']}): {s['reason']}")
    for r in report.records:
        project = r.facts.get("project_name").value if "project_name" in r.facts else ""
        print(f"\nrow {r.row:>3}  [{r.status}]  {r.company_name} — {project}")
        for m in r.minimum_missing:
            print(f"        minimum: {m}")
        for e in r.dataset_errors:
            print(f"        dummy data: {e}")
        n_rev = sum(c["needs_review"] for c in r.contacts)
        print(f"        contacts: {len(r.contacts)} ({n_rev} need review); "
              f"shared inboxes: {', '.join(r.company_emails) or '-'}")
        for w in r.warnings:
            print(f"        note: {w}")
        if args.preview_at and r.status == "ready":
            ctx = resolve_dataset(definition.sample_template, definition.sample_schema,
                                  {k: v.value for k, v in r.facts.items()},
                                  now=datetime.fromisoformat(args.preview_at), tz="Asia/Kolkata",
                                  skip_weekdays=definition.skip_weekdays)
            for s in ctx.data["free_visit_slots"]:
                print(f"        slot: {s['spoken']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report.as_json(), indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nfull report written to {args.json}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
