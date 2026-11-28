#!/usr/bin/env python3
"""
Capture college basketball spreads and totals, twice a day.

    python3 cbb_odds.py --phase open
    python3 cbb_odds.py --phase close

Why two captures. A win rate needs hundreds of bets before it says
anything. Closing line value says it in dozens: if the market drifts
toward the side the model picked, the model saw something before the
market did. That is the cleanest evidence of an edge there is, and the
only way to get it is to record where the line started and where it
finished.

One request covers every game on the board, so the cost does not grow
with the slate: two markets, one region, twice a day - about 120 credits
a month. Stopping the reception captures pays for it.

A manifest stops the same phase being bought twice in a day.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

SPORT = "basketball_ncaab"
OUTDIR = "snapshots/cbb"
MANIFEST = f"{OUTDIR}/_captured.json"
KEY_FILE = os.path.expanduser("~/.odds_api_key")


def api_key():
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k.strip()
    try:
        return open(KEY_FILE).read().strip()
    except Exception:
        return None


def load_manifest():
    try:
        return json.load(open(MANIFEST))
    except Exception:
        return {}


def save_manifest(m):
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(m, open(MANIFEST, "w"), indent=1)


def condense(events):
    """Keep the consensus and the best price, not every book's every number."""
    out = []
    for e in events:
        rec = {"id": e.get("id"), "commence_time": e.get("commence_time"),
               "home": e.get("home_team"), "away": e.get("away_team")}
        spreads, totals = [], []
        for b in e.get("bookmakers", []):
            book = b.get("key")
            for m in b.get("markets", []):
                if m.get("key") == "spreads":
                    for o in m.get("outcomes", []):
                        if o.get("point") is not None:
                            spreads.append({"book": book, "team": o.get("name"),
                                            "point": o["point"],
                                            "price": o.get("price")})
                elif m.get("key") == "totals":
                    for o in m.get("outcomes", []):
                        if o.get("point") is not None:
                            totals.append({"book": book, "side": o.get("name"),
                                           "point": o["point"],
                                           "price": o.get("price")})
        if not spreads and not totals:
            continue

        # the consensus number, which is what moves
        home_pts = [s["point"] for s in spreads if s["team"] == rec["home"]]
        tot_pts = [t["point"] for t in totals if t["side"] == "Over"]
        rec["spread_home"] = (round(sum(home_pts) / len(home_pts), 2)
                              if home_pts else None)
        rec["total"] = (round(sum(tot_pts) / len(tot_pts), 2)
                        if tot_pts else None)
        rec["books"] = len({s["book"] for s in spreads}
                           | {t["book"] for t in totals})
        # the best number available on each side, for line shopping
        if home_pts:
            rec["best_spread_home"] = max(home_pts)
            away_pts = [s["point"] for s in spreads if s["team"] == rec["away"]]
            rec["best_spread_away"] = max(away_pts) if away_pts else None
        if tot_pts:
            rec["best_over"] = min(tot_pts)
            und = [t["point"] for t in totals if t["side"] == "Under"]
            rec["best_under"] = max(und) if und else None
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["open", "close"], required=True)
    ap.add_argument("--force", action="store_true",
                    help="capture even if this phase is already recorded today")
    args = ap.parse_args()

    key = api_key()
    if not key:
        print("No ODDS_API_KEY. Nothing captured, nothing spent.")
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    man = load_manifest()
    tag = f"{today}:{args.phase}"
    if tag in man and not args.force:
        print(f"{args.phase} already captured for {today}. Nothing spent.")
        return 0

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {"apiKey": key, "regions": "us", "markets": "spreads,totals",
              "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        # Never fail the workflow over a third-party outage.
        print(f"Odds API unavailable ({type(e).__name__}: {str(e)[:80]})")
        return 0

    left = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    games = condense(events)
    if not games:
        print(f"No {SPORT} games priced right now. "
              f"{left or '?'} credits left.")
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/{today}_{args.phase}.json"
    json.dump({"captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "phase": args.phase, "date": today, "games": games},
              open(path, "w"), indent=1)
    man[tag] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "games": len(games)}
    save_manifest(man)

    print(f"{args.phase}: {len(games)} games priced, {len(events)} returned")
    print(f"  wrote {path}")
    print(f"  credits used {used or '?'}, remaining {left or '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
