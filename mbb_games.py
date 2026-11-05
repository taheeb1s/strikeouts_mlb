#!/usr/bin/env python3
"""
Men's college basketball game predictions. Elo, backtested before use.

    python3 mbb_games.py --backtest
    python3 mbb_games.py --tune
    python3 mbb_games.py --json

College basketball is a different shape from the football models. Around
360 Division I teams play 30-odd games each, so there is far more data per
season than football's twelve - but also an enormous spread in quality,
and a lot of early-season games against teams outside D1.

Neutral-site games are common here and get no home edge, which the football
models never had to handle.

Data is sportsdataverse, read from public files on GitHub. No key.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb

START = 1500.0
NON_D1 = 1150.0         # a generic team from outside Division I

# Tuned on 25,247 games. Carry-over is high (0.8) because college
# basketball rosters turn over less violently than football's, and the
# home edge is large - home sides win 67% of the time here against 59% in
# college football, which is why the baseline is so hard to beat.
K = 26.0
HOME_EDGE = 100.0
CARRYOVER = 0.8
PTS_PER_ELO = 28.0
LG_PTS = 72.0           # points per team per game, Division I
OD_K = 0.06             # how fast scoring ratings chase recent form
TOTAL_TRUST = 1.0
# Tuned on 25,000 games: totals land at 13.55 points of error against
# 15.15 for a flat league average. Unlike football, trusting the ratings
# fully beats blending them - Division I tempo varies enough that the
# signal is strong rather than noisy.


def frame(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def win_prob(diff):
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def load(seasons):
    frames = []
    for s in seasons:
        try:
            frames.append(frame(mbb.load_mbb_schedule(seasons=[s])))
        except Exception as e:
            print(f"  {s} unavailable ({type(e).__name__})")
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)

    keep = ["season", "season_type", "date", "neutral_site",
            "home_id", "home_location", "home_short_display_name",
            "home_abbreviation", "home_score",
            "away_id", "away_location", "away_short_display_name",
            "away_abbreviation", "away_score",
            "status_type_completed", "home_conference_id", "away_conference_id"]
    d = d[[c for c in keep if c in d.columns]].copy()
    d["d"] = pd.to_datetime(d.date, errors="coerce", utc=True)
    d = d.dropna(subset=["d"]).sort_values("d").reset_index(drop=True)
    for c in ("home_score", "away_score"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Scheduled games carry 0-0 rather than a null, so a score is no
    # evidence a game happened - every future fixture looked played.
    # The status flag is the only reliable signal.
    if "status_type_completed" in d:
        d["played"] = d.status_type_completed.astype(bool)
    else:
        d["played"] = (d.home_score.notna() & d.away_score.notna()
                       & ((d.home_score > 0) | (d.away_score > 0)))
    d["neutral"] = d.get("neutral_site", False).astype(bool) \
        if "neutral_site" in d else False
    return d


def team_name(row, side):
    for c in (f"{side}_short_display_name", f"{side}_location",
              f"{side}_abbreviation"):
        v = row.get(c)
        if isinstance(v, str) and v:
            return v
    return str(row.get(f"{side}_id", "?"))


def run(games, k=K, home=HOME_EDGE, carry=CARRYOVER, ppe=PTS_PER_ELO):
    """Walk forward: every game predicted from what came before it only.

    Teams outside Division I never appear in the schedule file as rated
    sides, so an unknown opponent is treated as a weak generic team rather
    than an average one. Without that, beating a non-D1 side would look
    like beating an average D1 team and inflate everyone's rating.
    """
    elo, rows, prev = {}, [], None
    seen = set()
    # Opponent-adjusted scoring. Tempo varies enormously in college
    # basketball - a grinding defensive side and a run-and-gun one are
    # barely playing the same game - so a single league-average total
    # makes every predicted score look alike.
    off, dfn = {}, {}

    for _, g in games.iterrows():
        if g.season != prev:
            elo = {t: START + carry * (r - START) for t, r in elo.items()}
            off = {t: carry * v for t, v in off.items()}
            dfn = {t: carry * v for t, v in dfn.items()}
            prev = g.season

        h, a = g.home_id, g.away_id
        h_known, a_known = h in seen, a in seen
        eh = elo.get(h, START if h_known else NON_D1)
        ea = elo.get(a, START if a_known else NON_D1)

        oh, dh = off.get(h, 0.0), dfn.get(h, 0.0)
        oa, da = off.get(a, 0.0), dfn.get(a, 0.0)
        edge = 0.0 if bool(g.neutral) else home
        diff = eh - ea + edge
        p_home = win_prob(diff)
        raw_total = 2 * LG_PTS + oh + da + oa + dh
        pred_total = TOTAL_TRUST * raw_total + (1 - TOTAL_TRUST) * 2 * LG_PTS

        if g.played:
            margin = g.home_score - g.away_score
            rows.append({
                "season": int(g.season), "date": g.d,
                "home": team_name(g, "home"), "away": team_name(g, "away"),
                "home_id": h, "away_id": a, "neutral": bool(g.neutral),
                "p_home": p_home, "pred_margin": diff / ppe,
                "margin": margin, "home_won": int(margin > 0),
                "total": g.home_score + g.away_score,
                "pred_total": pred_total,
            })
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            wdiff = diff if margin > 0 else -diff
            mult = mov_multiplier(margin, wdiff) if margin != 0 else 1.0
            shift = k * mult * (actual - p_home)
            elo[h] = eh + shift
            elo[a] = ea - shift
            # move scoring ratings toward what happened, opponent removed
            off[h] = oh + OD_K * ((g.home_score - da - LG_PTS) - oh)
            dfn[h] = dh + OD_K * ((g.away_score - oa - LG_PTS) - dh)
            off[a] = oa + OD_K * ((g.away_score - dh - LG_PTS) - oa)
            dfn[a] = da + OD_K * ((g.home_score - oh - LG_PTS) - da)
            seen.add(h)
            seen.add(a)

    return pd.DataFrame(rows), elo, off, dfn


def evaluate(df, first):
    d = df[(df.season >= first) & (df.margin != 0)]
    if not len(d):
        return None
    acc = ((d.p_home > 0.5) == (d.home_won == 1)).mean()
    ll = -(d.home_won * np.log(d.p_home.clip(1e-9))
           + (1 - d.home_won) * np.log((1 - d.p_home).clip(1e-9))).mean()
    return {"n": len(d), "acc": acc, "logloss": ll,
            "mae": np.abs(d.pred_margin - d.margin).mean(),
            "home_base": (d.home_won == 1).mean()}


def season_results(df, season):
    d = df[(df.season == season) & (df.margin != 0)]
    if not len(d):
        return {"graded": 0}
    picked_home = d.p_home > 0.5
    right = picked_home == (d.home_won == 1)
    conf = d.p_home.where(picked_home, 1 - d.p_home)
    bands = []
    for lo, hi, lab in [(0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"),
                        (0.7, 0.8, "70-80%"), (0.8, 0.9, "80-90%"),
                        (0.9, 1.01, "over 90%")]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 3:
            bands.append({"band": lab, "n": int(m.sum()),
                          "won": int(right[m].sum()),
                          "said": round(float(conf[m].mean()), 3),
                          "happened": round(float(right[m].mean()), 3)})
    return {
        "graded": int(len(d)), "won": int(right.sum()),
        "lost": int((~right).sum()),
        "win_pct": round(float(right.mean()), 4),
        "home_baseline": round(float((d.home_won == 1).mean()), 4),
        "margin_mae": round(float(np.abs(d.pred_margin - d.margin).mean()), 2),
        "bands": bands,
        "recent": [
            {"date": str(r.date.date()), "away": r.away, "home": r.home,
             "pick": r.home if r.p_home > 0.5 else r.away,
             "conf": round(float(max(r.p_home, 1 - r.p_home)), 3),
             "pred_margin": round(float(r.pred_margin), 1),
             "margin": int(r.margin),
             "won": bool((r.p_home > 0.5) == (r.margin > 0))}
            for _, r in d.sort_values("date", ascending=False).head(80).iterrows()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025, 2026, 2027])
    ap.add_argument("--skip-first", type=int, default=2)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.seasons[0]}-{args.seasons[-1]}...")
    games = load(args.seasons)
    if games.empty:
        print("No schedule data.")
        return 1
    print(f"  {len(games)} games, {games.played.sum()} played")
    first = args.seasons[0] + args.skip_first

    if args.tune:
        best = None
        print("\nSearching.\n")
        for k in [20, 26, 32, 40]:
            for home in [70, 85, 100]:
                for carry in [0.5, 0.65, 0.8]:
                    df, _, _, _ = run(games, k, home, carry)
                    r = evaluate(df, first)
                    if r and (best is None or r["logloss"] < best[0]["logloss"]):
                        best = (r, k, home, carry)
                        print(f"  K={k:<3} home={home:<4} carry={carry:<5} "
                              f"acc {r['acc']:.4f}  logloss {r['logloss']:.4f}  <-")
        r, k, home, carry = best
        print(f"\nBest: K={k}, HOME_EDGE={home}, CARRYOVER={carry}")
        return 0

    df, final, off, dfn = run(games)
    r = evaluate(df, first)
    if not r:
        print("Nothing to score.")
        return 1

    print(f"\n{r['n']} games scored, {first}-{args.seasons[-1]}\n")
    print(f"  Model accuracy         {r['acc']:.3f}")
    print(f"  Always the home team   {r['home_base']:.3f}")
    print(f"  Log loss               {r['logloss']:.4f}")
    print(f"  Margin error           {r['mae']:.2f} points")
    dd = df[(df.season >= first) & (df.margin != 0)]
    flat = dd.total.mean()
    tm = float(np.abs(dd.pred_total - dd.total).mean())
    tf = float(np.abs(flat - dd.total).mean())
    gg = np.abs(flat - dd.total) - np.abs(dd.pred_total - dd.total)
    se = gg.std() / np.sqrt(len(gg))
    print(f"  Game total error       {tm:.2f}   flat average {tf:.2f}"
          f"   ({abs(gg.mean()/se):.1f} se better)")
    d = df[(df.season >= first) & (df.margin != 0)]
    correct = ((d.p_home > 0.5) == (d.home_won == 1)).astype(float)
    gap = correct - (d.home_won == 1).astype(float)
    se = gap.std() / np.sqrt(len(gap))
    print(f"\n  Over the home-team baseline: {gap.mean():+.3f} "
          f"({abs(gap.mean()/se):.1f} standard errors)")

    print("\nTop 15 ratings")
    names = {}
    for _, g in games.iterrows():
        names[g.home_id] = team_name(g, "home")
        names[g.away_id] = team_name(g, "away")
    for t, v in sorted(final.items(), key=lambda x: -x[1])[:15]:
        print(f"  {names.get(t, t):26} {v:7.1f}")

    # next slate
    season = args.seasons[-1]
    # Start from today, not from the earliest unplayed fixture. A single
    # abandoned exhibition from two days ago was never marked complete and
    # blocked the whole slate behind it - the model reported one game.
    now_et = pd.Timestamp.now(tz="America/New_York").normalize()
    todo = games[(games.season == season) & (~games.played)].copy()
    todo["et"] = todo.d.dt.tz_convert("America/New_York")
    todo = todo[todo.et.dt.normalize() >= now_et]
    nxt = []
    if len(todo):
        day = todo.et.dt.normalize().min()
        same = todo[todo.et.dt.normalize() == day]
        avg_total = float(d.total.mean())
        for _, g in same.iterrows():
            eh = final.get(g.home_id, START)
            ea = final.get(g.away_id, START)
            edge = 0.0 if bool(g.neutral) else HOME_EDGE
            diff = eh - ea + edge
            p = win_prob(diff)
            m = diff / PTS_PER_ELO
            raw = (2 * LG_PTS + off.get(g.home_id, 0.0) + dfn.get(g.away_id, 0.0)
                   + off.get(g.away_id, 0.0) + dfn.get(g.home_id, 0.0))
            tot = TOTAL_TRUST * raw + (1 - TOTAL_TRUST) * 2 * LG_PTS
            nxt.append({
                "date": str(day.date()), "neutral": bool(g.neutral),
                "home": team_name(g, "home"), "away": team_name(g, "away"),
                "p_home": round(float(p), 4),
                "pred_margin": round(float(m), 2),
                "score_home": round((tot + m) / 2, 1),
                "score_away": round((tot - m) / 2, 1),
                "pred_total": round(float(tot), 1),
                "pick": team_name(g, "home") if p >= 0.5 else team_name(g, "away"),
                "confidence": round(float(max(p, 1 - p)), 4),
            })
        print(f"\n{day.date()}: {len(nxt)} games. Ten most confident:")
        for g in sorted(nxt, key=lambda x: -x["confidence"])[:10]:
            print(f"  {g['away'][:18]+' at '+g['home'][:18]:40} "
                  f"{g['pick'][:18]:20} {g['confidence']:>4.0%} "
                  f"{g['pred_margin']:>+6.1f}")

    if args.json:
        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "season": season,
            "backtest": {"games": int(r["n"]), "accuracy": round(float(r["acc"]), 4),
                         "total_mae": round(float(np.abs(
                             df[(df.season >= first) & (df.margin != 0)].pred_total
                             - df[(df.season >= first) & (df.margin != 0)].total).mean()), 2),
                         "home_baseline": round(float(r["home_base"]), 4),
                         "margin_mae": round(float(r["mae"]), 2),
                         "seasons": f"{first}-{args.seasons[-1]}"},
            "ratings": {names.get(t, str(t)): round(float(v), 1) for t, v in
                        sorted(final.items(), key=lambda x: -x[1])[:40]},
            "results": season_results(df, season),
            "games": nxt,
        }
        json.dump(out, open("mbb_games.json", "w"), indent=1)
        print(f"\n  Wrote mbb_games.json ({len(nxt)} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
