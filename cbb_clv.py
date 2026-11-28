#!/usr/bin/env python3
"""
Does the college basketball model see things before the market does?

    python3 cbb_clv.py --json

Two questions, and the second is the one that matters.

Against the closing line: did the side the model preferred actually cover?
That needs hundreds of games before it says anything, because a coin flip
lands outside 45-55% surprisingly often.

Closing line value: did the line move toward the model's side between the
morning and tip-off? That says something in dozens of games rather than
hundreds. If the model likes the home team at -4 and the number drifts to
-6 by tip, the model agreed with the money before the money arrived -
whether or not the team covered. A model with no edge moves the line the
wrong way about as often as the right way.

CLV is the better evidence precisely because it does not depend on the
outcome of any single game.
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_snaps():
    """{date: {phase: {game key: row}}}"""
    out = {}
    for f in sorted(glob.glob("snapshots/cbb/*_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        by = out.setdefault(d["date"], {})
        by[d["phase"]] = {key(g["away"], g["home"]): g for g in d["games"]}
    return out


def key(away, home):
    n = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    return f"{n(away)}@{n(home)}"


def load_picks():
    """Archived model predictions, by date."""
    out = {}
    for f in sorted(glob.glob("proj_history/mbb_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for g in d.get("games", []):
            out.setdefault(g["date"], {})[key(g["away"], g["home"])] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    snaps, picks = load_snaps(), load_picks()
    if not snaps:
        print("No odds captured yet. They accumulate twice a day.")
        card = {"graded": 0, "note": "no odds captured yet"}
    elif not picks:
        print("No archived model picks yet.")
        card = {"graded": 0, "note": "no archived picks yet"}
    else:
        rows = []
        for date, phases in snaps.items():
            o, c = phases.get("open"), phases.get("close")
            day_picks = picks.get(date, {})
            if not o or not c or not day_picks:
                continue
            for k, pick in day_picks.items():
                a, b = o.get(k), c.get(k)
                if not a or not b:
                    continue
                if a.get("spread_home") is None or b.get("spread_home") is None:
                    continue
                # The model's margin is from the home side, and so is the
                # spread - but a spread is quoted as the handicap, so a
                # home favourite by 6 is -6. Flip to compare like with like.
                model = pick["pred_margin"]
                open_line = -a["spread_home"]
                close_line = -b["spread_home"]
                rows.append({
                    "date": date, "away": pick["away"], "home": pick["home"],
                    "model": model, "open": open_line, "close": close_line,
                    "side": "home" if model > open_line else "away",
                    "edge": model - open_line,
                    "move": close_line - open_line,
                    "open_total": a.get("total"), "close_total": b.get("total"),
                    "model_total": pick.get("pred_total"),
                })
        r = pd.DataFrame(rows)
        if not len(r):
            print("Nothing lines up yet - needs an open, a close and a pick "
                  "for the same game.")
            card = {"graded": 0, "note": "no matched games yet"}
        else:
            # did the line move toward the side the model liked?
            toward = np.where(r.side == "home", r.move > 0, r.move < 0)
            moved = r.move != 0
            card = {
                "graded": int(len(r)),
                "dates": sorted(r.date.unique().tolist()),
                "moved": int(moved.sum()),
                "toward_model": int(toward[moved].sum()),
                "clv_rate": (round(float(toward[moved].mean()), 4)
                             if moved.sum() else None),
                "avg_move_toward": round(float(np.where(
                    r.side == "home", r.move, -r.move).mean()), 3),
                "avg_edge": round(float(r.edge.abs().mean()), 2),
                "recent": [
                    {"date": x.date, "away": x.away, "home": x.home,
                     "model": round(float(x.model), 1),
                     "open": round(float(x.open), 1),
                     "close": round(float(x.close), 1),
                     "side": x.side,
                     "toward": (None if x.move == 0 else
                                bool((x.move > 0) if x.side == "home"
                                     else (x.move < 0)))}
                    for _, x in r.sort_values("date", ascending=False).head(120).iterrows()
                ],
            }
            print(f"{len(r)} games with an open, a close and a model pick\n")
            print(f"  the line moved at all on {int(moved.sum())} of them")
            if moved.sum():
                print(f"  moved toward the model on {int(toward[moved].sum())}"
                      f"  ({toward[moved].mean():.1%})")
                n = int(moved.sum())
                se = (0.25 / n) ** 0.5
                print(f"  that is {abs(toward[moved].mean()-0.5)/se:.1f} "
                      f"standard errors from a coin flip")
            print(f"  average move toward the model's side: "
                  f"{card['avg_move_toward']:+.2f} points")
            print(f"  average gap between model and opener: "
                  f"{card['avg_edge']:.2f} points")
            if moved.sum() < 100:
                print(f"\n  Under a hundred moves this is still noise. "
                      f"Keep collecting.")

    if args.json:
        card["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        json.dump(card, open("cbb_clv.json", "w"), indent=1)
        print("\n  Wrote cbb_clv.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
