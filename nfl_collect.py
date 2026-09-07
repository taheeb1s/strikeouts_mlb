#!/usr/bin/env python3
"""
Snapshot NFL reception prop lines and archive them.

    python3 nfl_collect.py                 # capture games kicking off soon
    python3 nfl_collect.py --dry-run       # show what it would do, spend nothing
    python3 nfl_collect.py --window 6      # games starting within 6 hours
    python3 nfl_collect.py --recapture     # allow a second look at a game

By default each game is captured once, shortly before kickoff, and never
again - that's the sharpest line and costs one credit per game per week.
A manifest tracks what's been done, so running hourly is safe and cheap.

Why this exists: player stats are published retroactively by nflverse, so
there's nothing to collect there. Odds are not - historical player props are
a paid feature, so any line not captured while it's live is gone for good.

This deliberately does no modelling and makes no recommendations. It stores
every book's raw quote with a timestamp, because consensus can be computed
later from raw data but raw data can't be recovered from consensus.

Key comes from $ODDS_API_KEY or ~/.odds_api_key.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKET = "player_receptions"
OUTDIR = "snapshots/receptions"
MANIFEST = f"{OUTDIR}/_captured.json"
KEY_FILE = os.path.expanduser("~/.odds_api_key")

session = requests.Session()


def load_key():
    env = os.environ.get("ODDS_API_KEY", "").strip()
    if env:
        return env
    if not os.path.exists(KEY_FILE):
        sys.exit(f"No API key. Set ODDS_API_KEY or create {KEY_FILE}")
    key = open(KEY_FILE).read().strip()
    if not key:
        sys.exit(f"{KEY_FILE} is empty")
    return key


def read_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    try:
        return json.load(open(MANIFEST))
    except Exception:
        return {}


def write_manifest(m):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)


def due(key, window_hours, done, recapture):
    """Games kicking off inside the window that haven't been captured yet."""
    r = session.get(f"{API}/events", params={"apiKey": key}, timeout=20)
    r.raise_for_status()
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=window_hours)
    out, skipped = [], 0
    for ev in r.json():
        try:
            t = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if t > cutoff or t < now - timedelta(hours=1):
            continue
        if ev["id"] in done and not recapture:
            skipped += 1
            continue
        out.append((t, ev))
    return sorted(out), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=4,
                    help="capture games kicking off within this many hours")
    ap.add_argument("--budget", type=int, default=0,
                    help="stop after this many credits (0 = no limit)")
    ap.add_argument("--recapture", action="store_true",
                    help="re-snapshot games already captured (costs more)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = load_key()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

    done = read_manifest()

    try:
        events, skipped = due(key, args.window, done, args.recapture)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        sys.exit(f"Events request failed ({code}). 401 = bad key, 429 = out of credits.")

    print(f"{len(events)} game(s) due within {args.window}h"
          + (f", {skipped} already captured" if skipped else "")
          + f" -> {len(events)} credit(s)")

    if args.dry_run:
        for t, ev in events:
            print(f"  {t:%a %d %b %H:%M}Z  "
                  f"{ev.get('away_team')} at {ev.get('home_team')}")
        print("\nDry run - nothing fetched, no credits spent.")
        return 0

    if not events:
        print("Nothing due. Nothing spent.")
        return 0

    games, spent, remaining = [], 0, None
    for t, ev in events:
        if args.budget and spent >= args.budget:
            print(f"  budget of {args.budget} reached, stopping")
            break
        try:
            r = session.get(
                f"{API}/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": "us", "markets": MARKET,
                        "oddsFormat": "american"}, timeout=20)
            if r.status_code == 422:
                print(f"  {ev.get('away_team')} at {ev.get('home_team')}: "
                      f"market not offered")
                continue
            if r.status_code == 429:
                print("  out of credits - stopping and saving what we have")
                break
            r.raise_for_status()
            spent += 1
            remaining = r.headers.get("x-requests-remaining", remaining)
            data = r.json()
        except Exception as e:
            print(f"  {ev.get('id')}: {e}")
            continue

        # Count players seen, purely so the log is readable.
        players = set()
        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    if out.get("description"):
                        players.add(out["description"])

        games.append(data)
        done[ev["id"]] = {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kickoff": ev.get("commence_time"),
            "matchup": f"{ev.get('away_team')} at {ev.get('home_team')}",
        }
        print(f"  {ev.get('away_team')} at {ev.get('home_team')}: "
              f"{len(data.get('bookmakers', []))} book(s), {len(players)} player(s)")

    if not games:
        # Normal early in the week - books haven't posted props yet. Not an
        # error, so exit clean and let the hourly job try again later.
        print("\nNothing captured. Lines may not be posted yet.")
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/{stamp}.json"
    n = 2
    while os.path.exists(path):        # never clobber an earlier snapshot
        path = f"{OUTDIR}/{stamp}-{n}.json"
        n += 1
    with open(path, "w") as f:
        json.dump({
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "market": MARKET,
            "sport": "americanfootball_nfl",
            "games_captured": len(games),
            "credits_spent": spent,
            "credits_remaining": remaining,
            "games": games,            # raw, exactly as the API returned it
        }, f, indent=1)

    write_manifest(done)

    print(f"\nWrote {path}")
    print(f"{spent} credit(s) spent"
          + (f", {remaining} left this month" if remaining else ""))
    print("\nRaw snapshot only - no projections, no recommendations. Each game"
          "\nis captured once near kickoff; the manifest stops repeat spending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
