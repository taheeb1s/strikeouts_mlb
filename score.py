#!/usr/bin/env python3
"""
Grade past projections against what actually happened.

    python3 score.py

Reads every archived slate in history/, looks up the real result for each
start, and reports three things:

  1. Error vs the naive baseline. If this model doesn't beat "his season
     strikeouts per start," the extra machinery isn't earning its keep.
  2. Bias - is it systematically high or low, on strikeouts and on batters
     faced. Bias is fixable; random error mostly isn't.
  3. Calibration - when it says 30%, does that happen 30% of the time?
     A projection can have good average error and still lie about its
     own confidence.

Only scores games that have finished. Today's slate is skipped.
"""

import glob
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import requests

API = "https://statsapi.mlb.com/api/v1"
session = requests.Session()
session.headers["User-Agent"] = "strikeout-projections/1.0"

_cache = {}


def actual_line(pid, day):
    """Real strikeouts and batters faced for one pitcher on one date."""
    season = day[:4]
    key = (pid, season)
    if key not in _cache:
        try:
            r = session.get(f"{API}/people/{pid}/stats",
                            params={"stats": "gameLog", "group": "pitching",
                                    "season": season}, timeout=20)
            r.raise_for_status()
            data = r.json()
            splits = data["stats"][0]["splits"] if data.get("stats") else []
        except Exception:
            splits = []
        _cache[key] = splits

    for sp in _cache[key]:
        if sp.get("date") == day:
            s = sp["stat"]
            if int(s.get("gamesStarted", 0)) != 1:
                return None  # relief appearance, not the start we projected
            return int(s.get("strikeOuts", 0)), int(s.get("battersFaced", 0))
    return None  # postponed, scratched, or not yet played


def _now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def tail(pmf, n):
    return sum(pmf[n:])


def main():
    files = sorted(glob.glob("history/*.json"))
    today = str(date.today())
    files = [f for f in files
             if os.path.basename(f)[:-5] < today]

    if not files:
        if "--json" in sys.argv:
            with open("scorecard.json", "w") as f:
                json.dump({"generated_at": _now(), "days": 0, "starts": 0,
                           "verdict": "collecting"}, f, indent=1)
        print("No finished slates to score yet.")
        print("history/ fills up as you run build_projections.py each day.")
        print("Come back after a few days.")
        return 1

    model, naive, actual = [], [], []
    bf_proj, bf_actual = [], []
    buckets = defaultdict(lambda: {"p": 0.0, "hit": 0, "n": 0})
    mkt_model, mkt_line, mkt_actual = [], [], []
    detail = []
    scored, missing = 0, 0

    print(f"Scoring {len(files)} slate(s)...\n")

    for path in files:
        day = os.path.basename(path)[:-5]
        slate = json.load(open(path))
        hits = 0

        for s in slate["starters"]:
            got = actual_line(s["pitcher_id"], day)
            if got is None or s.get("naive_k") is None:
                missing += 1
                continue
            k_act, bf_act = got
            hits += 1

            model.append(s["proj_k"])
            naive.append(s["naive_k"])
            actual.append(k_act)
            bf_proj.append(s["proj_bf"])
            bf_actual.append(bf_act)

            m = s.get("market")
            if m:
                mkt_model.append(s["proj_k"])
                mkt_line.append(m["line"])
                mkt_actual.append(k_act)

            row = {
                "date": day,
                "pitcher": s["pitcher"],
                "opponent": s.get("opponent", ""),
                "proj": s["proj_k"],
                "naive": s["naive_k"],
                "actual": k_act,
                "err": round(s["proj_k"] - k_act, 2),
                "proj_bf": s.get("proj_bf"),
                "bf": bf_act,
            }
            if m:
                went_over = k_act > m["line"]
                row.update(
                    line=m["line"],
                    model_over=m["model_over"],
                    market_over=m.get("consensus_over"),
                    side="Over" if m["edge"] > 0 else "Under",
                    edge=m["edge"],
                    hit=bool((m["edge"] > 0) == went_over),
                    closer=bool(abs(s["proj_k"] - k_act) < abs(m["line"] - k_act)),
                )
            detail.append(row)

            # Calibration across thresholds near the projection.
            base = int(round(s["proj_k"]))
            for n in range(max(1, base - 2), base + 3):
                p = tail(s["pmf"], n)
                if 0.02 < p < 0.98:
                    b = buckets[round(p * 10) / 10]
                    b["p"] += p
                    b["hit"] += 1 if k_act >= n else 0
                    b["n"] += 1

        scored += hits
        print(f"  {day}  {hits} of {len(slate['starters'])} starts")

    if not model:
        print("\nNothing could be matched to a finished game yet.")
        return 1

    model = np.array(model, float)
    naive = np.array(naive, float)
    actual = np.array(actual, float)
    bfp = np.array(bf_proj, float)
    bfa = np.array(bf_actual, float)

    mae_m = np.abs(model - actual).mean()
    mae_n = np.abs(naive - actual).mean()
    gap = mae_n - mae_m

    print(f"\n{'='*52}")
    print(f"{scored} starts scored, {missing} skipped\n")

    print(f"  This model        MAE {mae_m:.3f}")
    print(f"  Naive baseline    MAE {mae_n:.3f}")
    print(f"  Difference        {gap:+.3f}"
          f"  ({'model is better' if gap > 0 else 'BASELINE IS BETTER'})\n")

    print(f"  Strikeout bias    {(model - actual).mean():+.2f}"
          f"   (positive = projecting too high)")
    print(f"  Batters faced     {(bfp - bfa).mean():+.2f}"
          f"   projected {bfp.mean():.1f} vs actual {bfa.mean():.1f}")
    print(f"  Actual spread     SD {actual.std():.2f} strikeouts per start")

    card = {
        "generated_at": _now(),
        "days": len(files),
        "starts": scored,
        "mae_model": round(float(mae_m), 3),
        "mae_naive": round(float(mae_n), 3),
        "gap": round(float(gap), 3),
        "bias_k": round(float((model - actual).mean()), 2),
        "bias_bf": round(float((bfp - bfa).mean()), 2),
        "sd_actual": round(float(actual.std()), 2),
        "market": None,
        # Newest first. 500 is roughly three weeks of slates and lands
        # around 120KB - still quick to load. The full archive lives in
        # history/ regardless, so nothing is lost when this rolls over.
        "results": sorted(detail, key=lambda r: (r["date"], r["pitcher"]),
                          reverse=True)[:500],
    }

    if len(mkt_model) >= 10:
        mm = np.array(mkt_model, float)
        ml = np.array(mkt_line, float)
        ma = np.array(mkt_actual, float)
        e_me, e_bk = np.abs(mm - ma), np.abs(ml - ma)
        print(f"\n  Against the book  ({len(mm)} starts with a posted line)")
        print(f"    This model      MAE {e_me.mean():.3f}")
        print(f"    Book's line     MAE {e_bk.mean():.3f}")

        # The sharpest read: when the two disagreed, who was closer?
        card["market"] = {
            "starts": int(len(mm)),
            "mae_model": round(float(e_me.mean()), 3),
            "mae_book": round(float(e_bk.mean()), 3),
            "disagreements": 0,
            "won": 0,
            "win_rate": None,
        }

        big = np.abs(mm - ml) >= 1.0
        if big.sum() >= 5:
            won = (e_me[big] < e_bk[big]).sum()
            card["market"].update(disagreements=int(big.sum()), won=int(won),
                                  win_rate=round(float(won / big.sum()), 3))
            print(f"    Disagreed by 1+ K on {big.sum()} starts;"
                  f" model was closer on {won} ({won/big.sum():.0%})")
            print("    Under about half means the disagreements are your"
                  "\n    model's blind spots, not edges.")

    if buckets:
        print(f"\n  Calibration  (over/under thresholds near each projection)")
        print(f"  {'predicted':>10} {'actual':>8} {'starts':>8}")
        for key in sorted(buckets):
            b = buckets[key]
            if b["n"] < 5:
                continue
            print(f"  {b['p']/b['n']:>9.0%} {b['hit']/b['n']:>8.0%} {b['n']:>8}")
        print("\n  Those two columns should track each other. If 'predicted'"
              "\n  reads 30% where 'actual' reads 45%, the distributions are"
              "\n  too narrow and the confidence is overstated.")

    if "--json" in sys.argv:
        card["verdict"] = (
            "collecting" if scored < 150
            else ("model_ahead" if gap > 0 else "baseline_ahead"))
        with open("scorecard.json", "w") as f:
            json.dump(card, f, indent=1)
        print("\n  Wrote scorecard.json")

    print(f"\n{'='*52}")
    if scored < 150:
        print(f"Only {scored} starts so far. Differences under about 0.15 MAE"
              "\nare noise at this sample size. Keep collecting.")
    elif gap <= 0:
        print("The naive baseline is winning. The extra modelling isn't"
              "\nhelping yet - worth fixing before adding anything new.")
    else:
        print(f"Model is ahead by {gap:.2f} strikeouts of average error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
