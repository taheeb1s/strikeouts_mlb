#!/usr/bin/env python3
"""
Turn the raw reception snapshots into one file the page can read.

    python3 nfl_build.py

Reads every file in snapshots/receptions/ and writes receptions.json:
consensus across books with the vig stripped, the best price on each side
and who has it, how far apart the books are, and how the line has moved if
a game was captured more than once.

No projections. This reports what the market is doing, nothing more.
"""

import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from add_odds import devig, key_of, to_decimal

INDIR = "snapshots/receptions"
OUT = "receptions.json"


def valid(q):
    """American odds are never strictly between -100 and +100."""
    return abs(q["over_price"]) >= 100 and abs(q["under_price"]) >= 100


def summarise_quotes(quotes):
    """Consensus at the line most books are using, plus best prices."""
    quotes = [q for q in quotes if valid(q)] or quotes
    counts = defaultdict(int)
    for q in quotes:
        counts[q["line"]] += 1
    line = max(counts, key=lambda k: (counts[k], -abs(k)))
    at = [q for q in quotes if q["line"] == line]

    fairs, vigs = [], []
    for q in at:
        f, v = devig(q["over_price"], q["under_price"])
        fairs.append(f)
        vigs.append(v)

    best_o = max(at, key=lambda q: to_decimal(q["over_price"]))
    worst_o = min(at, key=lambda q: to_decimal(q["over_price"]))
    best_u = max(at, key=lambda q: to_decimal(q["under_price"]))
    worst_u = min(at, key=lambda q: to_decimal(q["under_price"]))

    # What line shopping is actually worth here, per $100 staked.
    shop_o = (to_decimal(best_o["over_price"])
              - to_decimal(worst_o["over_price"])) * 100
    shop_u = (to_decimal(best_u["under_price"])
              - to_decimal(worst_u["under_price"])) * 100

    return {
        "line": line,
        "books": len(at),
        "other_lines": sorted(k for k in counts if k != line),
        "consensus_over": round(sum(fairs) / len(fairs), 4),
        "book_spread": round(max(fairs) - min(fairs), 4),
        "avg_vig": round(sum(vigs) / len(vigs), 4),
        "best_over": {"price": best_o["over_price"], "book": best_o["book_title"]},
        "best_under": {"price": best_u["under_price"], "book": best_u["book_title"]},
        "shop_over": round(shop_o, 1),
        "shop_under": round(shop_u, 1),
        "quotes": sorted(at, key=lambda q: q["book_title"]),
    }


def main():
    files = sorted(glob.glob(f"{INDIR}/*.json"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        print(f"No snapshots in {INDIR}/ yet.")
        return 1

    # game id -> list of (captured_at, {player key -> quotes})
    timeline = defaultdict(list)
    meta = {}

    for path in files:
        try:
            snap = json.load(open(path))
        except Exception:
            continue
        when = snap.get("captured_at", "")
        for g in snap.get("games", []):
            gid = g.get("id")
            if not gid:
                continue
            meta[gid] = {
                "matchup": f"{g.get('away_team')} at {g.get('home_team')}",
                "kickoff": g.get("commence_time"),
            }
            byplayer = defaultdict(list)
            for bm in g.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "player_receptions":
                        continue
                    sides = defaultdict(dict)
                    for o in mkt.get("outcomes", []):
                        who = o.get("description", "")
                        if who:
                            sides[who][o["name"]] = o
                    for who, sd in sides.items():
                        if "Over" in sd and "Under" in sd:
                            byplayer[who].append({
                                "book": bm.get("key", "?"),
                                "book_title": bm.get("title", "?"),
                                "line": sd["Over"]["point"],
                                "over_price": sd["Over"]["price"],
                                "under_price": sd["Under"]["price"],
                            })
            if byplayer:
                timeline[gid].append((when, dict(byplayer)))

    games = []
    for gid, snaps in timeline.items():
        snaps.sort(key=lambda x: x[0])
        latest_when, latest = snaps[-1]

        players = []
        for name, quotes in latest.items():
            info = summarise_quotes(quotes)
            info["player"] = name

            # movement, if this game was captured more than once
            hist = []
            for when, byp in snaps:
                if name in byp:
                    h = summarise_quotes(byp[name])
                    hist.append({"at": when, "line": h["line"],
                                 "consensus_over": h["consensus_over"]})
            info["history"] = hist
            if len(hist) > 1:
                info["moved_line"] = round(hist[-1]["line"] - hist[0]["line"], 1)
                info["moved_prob"] = round(
                    hist[-1]["consensus_over"] - hist[0]["consensus_over"], 4)
            else:
                info["moved_line"] = None
                info["moved_prob"] = None
            players.append(info)

        players.sort(key=lambda p: -p["line"])
        games.append({
            **meta[gid],
            "captured_at": latest_when,
            "snapshots": len(snaps),
            "players": players,
        })

    games.sort(key=lambda g: g["kickoff"] or "")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": "player_receptions",
        "games": games,
        "totals": {
            "games": len(games),
            "players": sum(len(g["players"]) for g in games),
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print(f"Wrote {OUT}: {out['totals']['games']} game(s), "
          f"{out['totals']['players']} player(s)")

    # Where is shopping worth the most?
    allp = [(p["shop_over"], p["shop_under"], p["player"], g["matchup"])
            for g in games for p in g["players"]]
    allp.sort(key=lambda x: -max(x[0], x[1]))
    if allp:
        print("\nBiggest price gaps between books (per $100 staked):")
        for so, su, name, match in allp[:6]:
            side = "over" if so >= su else "under"
            print(f"  {name:24} {side:5} ${max(so, su):5.2f}   {match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
