"""Double-booking concurrency test — THE Stage 3 safety proof.

Fires N parallel booking requests for the SAME slot (capacity 1).
Pass: exactly 1 succeeds (201), the rest get 409. Zero deps (stdlib only).

Usage:
    python scripts/test_booking_concurrency.py --base http://localhost:8000 \
        --token <admin_access_token> --centre <centre_id> --slot <iso_datetime> \
        --phones 7000000001,7000000002,...   (N phone numbers = N parallel attempts)
"""
import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def call(base: str, token: str, method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(base + path, method=method,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--centre", required=True)
    ap.add_argument("--slot", required=True, help="ISO datetime of the slot to fight over")
    ap.add_argument("--phones", required=True, help="comma-separated phone numbers, one per attempt")
    args = ap.parse_args()

    phones = [p.strip() for p in args.phones.split(",") if p.strip()]
    print(f"Preparing {len(phones)} customers...")
    customer_ids = []
    for ph in phones:
        code, out = call(args.base, args.token, "POST", "/customers",
                         {"full_name": f"Load Test {ph[-4:]}", "phone": ph})
        if code == 201:
            customer_ids.append(out["id"])
        elif code == 409:  # already exists from a previous run — look it up
            code2, out2 = call(args.base, args.token, "GET", f"/customers/lookup?phone=%2B91{ph[-10:]}")
            if code2 == 200:
                customer_ids.append(out2["id"])
            else:
                print(f"  cannot prepare customer for {ph}: {code2} {out2}")
        else:
            print(f"  cannot create customer {ph}: {code} {out}")
    if len(customer_ids) < 2:
        print("\nABORT — need at least 2 prepared customers to test concurrency.")
        print("Most common cause: the --token value is wrong or expired.")
        print("Copy the FULL token printed by verify_stage3.ps1 (it is ~300 chars, no '...').")
        raise SystemExit(1)
    print(f"{len(customer_ids)} customers ready. Booking slot {args.slot} in parallel...")

    def attempt(cid: str):
        return call(args.base, args.token, "POST", "/appointments",
                    {"customer_id": cid, "centre_id": args.centre, "starts_at": args.slot})

    with ThreadPoolExecutor(max_workers=len(customer_ids)) as pool:
        results = list(pool.map(attempt, customer_ids))

    successes = [r for r in results if r[0] == 201]
    conflicts = [r for r in results if r[0] == 409]
    others = [r for r in results if r[0] not in (201, 409)]

    print(f"\nResults: {len(successes)} succeeded, {len(conflicts)} got 409, {len(others)} other")
    for code, body in others:
        print("  UNEXPECTED:", code, body)

    if len(successes) == 1 and not others:
        print("\nPASS — exactly one booking won; capacity protected under concurrency.")
    else:
        print("\nFAIL — double-booking protection is broken. Investigate before proceeding.")


if __name__ == "__main__":
    main()
