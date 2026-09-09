#!/usr/bin/env python3
"""
Score the reception lines you've collected against what actually happened.

    pip install nflreadpy pandas numpy pyarrow
    python3 nfl_score.py
    python3 nfl_score.py --season 2025    # score a completed season instead

There is no reception model, so this doesn't grade one. It measures the
market: how accurate the books' lines are, whether their prices are
calibrated, and how much of the week-to-week variance is simply not
predictable by anyone.

That last number is the point. It sets the bar a model would have to clear
to be worth building, measured on your own data rather than assumed.
"""

import argparse
import glob
import json
import os
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

INDIR = "snapshots/receptions"


def norm(name):
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def key_of(name):
    p = norm(name).split()
    return (p[-1], p[0][0]) if len(p) >= 2 else (norm(name), "")


def american_to_prob(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def devig(over, under):
    o, u = american_to_prob(over), american_to_prob(under)
    return o / (o + u)


def load_snapshots():
    """One record per player per game: the last line we captured for them."""
    files = sorted(f for f in glob.glob(f"{INDIR}/*.json")
                   if not os.path.basename(f).startswith("_"))
    if not files:
        return []

    latest = {}
    for path in files:
        try:
            snap = json.load(open(path))
        except Exception:
            continue
        when = snap.get("captured_at", "")
        for g in snap.get("games", []):
            kick = g.get("commence_time")
            quotes = defaultdict(list)
            for bm in g.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "player_receptions":
                        continue
                    sides = defaultdict(dict)
                    for o in mkt.get("outcomes", []):
                        if o.get("description"):
                            sides[o["description"]][o["name"]] = o
                    for who, sd in sides.items():
                        if "Over" in sd and "Under" in sd:
                            if abs(sd["Over"]["price"]) < 100:
                                continue
                            quotes[who].append((sd["Over"]["point"],
                                                sd["Over"]["price"],
                                                sd["Under"]["price"]))
            for who, qs in quotes.items():
                # modal line, mean devigged probability at it
                counts = defaultdict(int)
                for ln, _, _ in qs:
                    counts[ln] += 1
                line = max(counts, key=lambda k: counts[k])
                at = [(o, u) for ln, o, u in qs if ln == line]
                p = float(np.mean([devig(o, u) for o, u in at]))
                rec = {"player": who, "line": line, "p_over": p,
                       "books": len(at), "kickoff": kick, "captured": when,
                       "matchup": f"{g.get('away_team')} at {g.get('home_team')}"}
                # keep the latest capture per player per game
                k = (key_of(who), kick)
                if k not in latest or when > latest[k]["captured"]:
                    latest[k] = rec
    return list(latest.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    snaps = load_snapshots()
    if not snaps:
        print(f"No snapshots in {INDIR}/ yet. Lines get captured a few hours")
        print("before each kickoff; come back after some games have played.")
        return 1
    print(f"{len(snaps)} player-game line(s) captured")

    try:
        import nflreadpy as nfl
        import polars as pl
        import pandas as pd
    except ImportError:
        print("\nNeeds: pip install nflreadpy pandas numpy pyarrow")
        return 1

    season = args.season or max(
        int(s["kickoff"][:4]) for s in snaps if s.get("kickoff"))
    # A January game belongs to the previous NFL season.
    months = [int(s["kickoff"][5:7]) for s in snaps if s.get("kickoff")]
    if months and max(months) <= 2:
        season -= 1

    print(f"Scoring against the {season} season\n")

    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    day_to_week = {}
    for _, g in sched.iterrows():
        d = str(g.get("gameday", ""))[:10]
        if d:
            day_to_week[d] = int(g["week"])

    stats = nfl.load_player_stats(seasons=[season])
    d = (stats.filter(pl.col("season_type") == "REG")
              .select(["player_display_name", "week", "team",
                       "receptions", "targets"])
              .to_pandas())
    d["receptions"] = pd.to_numeric(d["receptions"], errors="coerce").fillna(0)
    actual = {}
    for _, r in d.iterrows():
        if not isinstance(r.player_display_name, str):
            continue
        actual[(key_of(r.player_display_name), int(r.week))] = int(r.receptions)

    rows, unmatched = [], 0
    for s in snaps:
        kick = s.get("kickoff") or ""
        day = kick[:10]
        wk = day_to_week.get(day)
        if wk is None:                      # late kickoff rolls past midnight UTC
            try:
                dt = datetime.fromisoformat(kick.replace("Z", "+00:00"))
                prev = str((dt.date().toordinal() - 1))
                from datetime import date as _d
                day2 = str(_d.fromordinal(dt.date().toordinal() - 1))
                wk = day_to_week.get(day2)
            except Exception:
                wk = None
        if wk is None:
            unmatched += 1
            continue
        got = actual.get((key_of(s["player"]), wk))
        if got is None:
            unmatched += 1
            continue
        rows.append({**s, "week": wk, "actual": got})

    if not rows:
        print(f"Nothing could be matched to a played game yet "
              f"({unmatched} unmatched).")
        return 1

    line = np.array([r["line"] for r in rows], float)
    act = np.array([r["actual"] for r in rows], float)
    p = np.array([r["p_over"] for r in rows], float)
    went_over = act > line

    print(f"{'='*58}")
    print(f"{len(rows)} scored, {unmatched} unmatched\n")

    print("How good is the market?")
    print(f"  Line vs actual, mean absolute error   {np.abs(line - act).mean():.3f}")
    print(f"  Spread of actual receptions       SD  {act.std():.3f}")
    print(f"  Overs hit                             {went_over.mean():.1%}"
          f"   (an efficient market lands near 50%)")
    print(f"  Average line                          {line.mean():.2f}")
    print(f"  Average actual                        {act.mean():.2f}")

    # Are their prices honest? Bucket the devigged probability, compare to
    # how often the over actually landed.
    buckets = defaultdict(lambda: [0.0, 0, 0])
    for pi, o in zip(p, went_over):
        b = buckets[round(pi * 10) / 10]
        b[0] += pi
        b[1] += int(o)
        b[2] += 1
    usable = {k: v for k, v in buckets.items() if v[2] >= 8}
    if usable:
        print("\nAre the books' prices calibrated?")
        print(f"  {'they say':>9} {'happened':>9} {'n':>5}")
        for k in sorted(usable):
            s_, h, n = usable[k]
            print(f"  {s_/n:>9.0%} {h/n:>9.0%} {n:>5}")

    print(f"\n{'='*58}")
    mae = float(np.abs(line - act).mean())
    print("What this means for a model")
    print(f"  A reception model would have to beat {mae:.2f} average error")
    print(f"  to add anything over simply taking the book's number.")
    print(f"  The 2025 backtest put a season-average baseline at 1.49 and")
    print(f"  the best model I could build at 1.51 - both worse than that.")
    if len(rows) < 200:
        print(f"\n  Only {len(rows)} scored. Treat all of this as provisional")
        print(f"  until you have a few hundred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
