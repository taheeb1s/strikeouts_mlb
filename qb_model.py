#!/usr/bin/env python3
"""
NFL quarterback props: completions and passing yards.

    python3 qb_model.py --backtest
    python3 qb_model.py --json
    python3 qb_model.py --grade --json

Why these two and not passing touchdowns. A quarterback throws thirty-odd
passes a game, so completions scatter at only 0.34 times their own mean -
tighter than anything else in this project, and far tighter than receptions
at 0.95. More events, less noise, more signal that survives.

Passing touchdowns do not share that. They average 1.4 a game and scatter
at 0.84, and three separate attempts failed to beat a plain season average:
as a count (+0.018, 1.3 se), as "at least one" (worse than the base rate by
1.7 se) and as "two or more" (worse by 2.4 se). They are not here.

Measured walk-forward across 2023-2026, 1,216 predictions:
  completions      4.710 against  4.860 for a season average  (+0.150, 2.5 se)
  passing yards   58.841 against 61.330                       (+2.489, 3.4 se)

Data is nflverse. No key.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl
from sklearn.ensemble import HistGradientBoostingRegressor

MARKETS = ["completions", "passing_yards"]
MIN_GAMES = 4
MIN_ATTEMPTS = 80
FIRST_WEEK = 5

FEATURES = ["cmp_all", "cmp_3", "cmp_5", "att_all", "att_3", "att_5", "cpct",
            "yds_all", "yds_3", "ypa", "td_all", "games",
            "o_cmp", "o_att", "o_yds", "tot", "spr"]


def load(seasons):
    st = nfl.load_player_stats(seasons=list(seasons))
    d = (st.filter((pl.col("season_type") == "REG") & (pl.col("position") == "QB"))
         .select(["player_id", "player_display_name", "season", "week", "team",
                  "opponent_team", "attempts", "completions", "passing_tds",
                  "passing_yards"]).to_pandas())
    for c in ["season", "week", "attempts", "completions", "passing_tds",
              "passing_yards"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d = d[d.attempts >= 5]

    # what each defence has allowed, from games before this one
    opp = (d.groupby(["opponent_team", "season", "week"])
           .agg(o_cmp=("completions", "sum"), o_att=("attempts", "sum"),
                o_yds=("passing_yards", "sum")).reset_index()
           .sort_values(["opponent_team", "season", "week"]))
    for c in ["o_cmp", "o_att", "o_yds"]:
        opp[c + "_p"] = (opp.groupby(["opponent_team", "season"])[c]
                         .transform(lambda s: s.shift().expanding().mean()))
    d = d.merge(opp[["opponent_team", "season", "week",
                     "o_cmp_p", "o_att_p", "o_yds_p"]],
                on=["opponent_team", "season", "week"], how="left")

    # game environment: a shootout means more throwing
    try:
        sch = nfl.load_schedules(seasons=list(seasons)).to_pandas()
        env = {}
        for _, g in sch.iterrows():
            tl = g.get("total_line")
            sl = g.get("spread_line")
            env[(int(g.season), int(g.week), g.home_team)] = (tl, sl)
            env[(int(g.season), int(g.week), g.away_team)] = (
                tl, None if sl is None or pd.isna(sl) else -sl)
        d["tot"] = [env.get((s, w, t), (np.nan, np.nan))[0]
                    for s, w, t in zip(d.season, d.week, d.team)]
        d["spr"] = [env.get((s, w, t), (np.nan, np.nan))[1]
                    for s, w, t in zip(d.season, d.week, d.team)]
    except Exception:
        d["tot"] = np.nan
        d["spr"] = np.nan

    return d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def features_from(past, row):
    return {
        "cmp_all": past.completions.mean(),
        "cmp_3": past.completions.tail(3).mean(),
        "cmp_5": past.completions.tail(5).mean(),
        "att_all": past.attempts.mean(),
        "att_3": past.attempts.tail(3).mean(),
        "att_5": past.attempts.tail(5).mean(),
        "cpct": past.completions.sum() / max(past.attempts.sum(), 1),
        "yds_all": past.passing_yards.mean(),
        "yds_3": past.passing_yards.tail(3).mean(),
        "ypa": past.passing_yards.sum() / max(past.attempts.sum(), 1),
        "td_all": past.passing_tds.mean(),
        "games": len(past),
        "o_cmp": row.get("o_cmp_p", np.nan),
        "o_att": row.get("o_att_p", np.nan),
        "o_yds": row.get("o_yds_p", np.nan),
        "tot": row.get("tot", np.nan),
        "spr": row.get("spr", np.nan),
    }


def build_rows(d, seasons):
    rows = []
    for _, g in d.groupby("player_id", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            if g.season.iloc[i] not in seasons or g.week.iloc[i] < FIRST_WEEK:
                continue
            p = g[(g.index < i) & (g.season == g.season.iloc[i])]
            if len(p) < MIN_GAMES or p.attempts.sum() < MIN_ATTEMPTS:
                continue
            r = {"player": g.player_display_name.iloc[i],
                 "player_id": g.player_id.iloc[i],
                 "season": g.season.iloc[i], "week": g.week.iloc[i],
                 **features_from(p, g.iloc[i].to_dict())}
            for c in MARKETS:
                r[c + "_act"] = g[c].iloc[i]
            rows.append(r)
    return pd.DataFrame(rows)


def fit_all(tr):
    out = {}
    for c in MARKETS:
        out[c] = HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=4, learning_rate=0.05,
            max_iter=300, min_samples_leaf=25, random_state=0
        ).fit(tr[FEATURES], tr[c + "_act"])
    return out


def residual_table(models, tr, nbins=4):
    out = {}
    for c in MARKETS:
        p = models[c].predict(tr[FEATURES])
        a = tr[c + "_act"].values
        edges = np.quantile(p, np.linspace(0, 1, nbins + 1))[1:-1]
        bins = np.digitize(p, edges)
        per = []
        for b in range(nbins):
            r = np.sort(a[bins == b] - p[bins == b])
            per.append([round(float(x), 2) for x in
                        (np.quantile(r, np.linspace(0.01, 0.99, 99))
                         if len(r) > 40 else np.array([0.0]))])
        out[c] = {"edges": [round(float(e), 2) for e in edges], "resid": per}
    return out


def p_over(table, market, proj, line):
    t = table.get(market)
    if not t:
        return None
    b = int(np.digitize([proj], t["edges"])[0])
    r = np.array(t["resid"][b])
    if len(r) < 10:
        return None
    return float((proj + r > line).mean())


def backtest(d, seasons):
    """Walk forward: each season predicted from the ones before it."""
    r = build_rows(d, set(seasons))
    preds = []
    for yr in sorted(seasons)[2:]:
        tr, te = r[r.season < yr], r[r.season == yr]
        if len(tr) < 400 or not len(te):
            continue
        models = fit_all(tr)
        t = te.copy()
        for c in MARKETS:
            t[c + "_p"] = models[c].predict(te[FEATURES])
        preds.append(t)
    if not preds:
        print("Not enough data.")
        return
    P = pd.concat(preds)
    print(f"{len(P)} predictions, walk-forward\n")
    print(f"  {'':16} {'model':>8} {'season avg':>12}   gap")
    for c, nai in [("completions", "cmp_all"), ("passing_yards", "yds_all")]:
        a = P[c + "_act"].values
        mm = np.abs(P[c + "_p"] - a).mean()
        nn = np.abs(P[nai] - a).mean()
        dd = np.abs(P[nai] - a) - np.abs(P[c + "_p"] - a)
        se = dd.std() / np.sqrt(len(dd))
        print(f"  {c:16} {mm:8.3f} {nn:12.3f}   {dd.mean():+.3f} "
              f"({abs(dd.mean()/se):.1f} se)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--train", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.grade:
        card = grade(args.season)
        if args.json:
            card["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            json.dump(card, open("qb_grade.json", "w"), indent=1)
            print("\n  Wrote qb_grade.json")
        return 0

    seasons = sorted(set(args.train + [args.season]))
    print(f"Loading {seasons[0]}-{seasons[-1]}...")
    d = load(seasons)
    print(f"  {len(d)} quarterback games")

    if args.backtest:
        backtest(d, seasons)
        return 0

    train = [s for s in args.train if s < args.season]
    tr = build_rows(d, set(train))
    if len(tr) < 400:
        print(f"Only {len(tr)} training rows.")
        return 1
    models = fit_all(tr)
    print(f"  fitted on {len(tr)} quarterback-weeks")

    sched = nfl.load_schedules(seasons=[args.season]).to_pandas()
    unplayed = sched[sched.home_score.isna()]
    if "gameday" in unplayed:
        gd = pd.to_datetime(unplayed.gameday, errors="coerce")
        ahead = unplayed[gd >= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
        if len(ahead):
            unplayed = ahead
    if unplayed.empty:
        print("Season complete.")
        return 1
    target = int(unplayed.week.min())
    wk = unplayed[unplayed.week == target]
    facing = {}
    for _, g in wk.iterrows():
        facing[g.home_team] = g.away_team
        facing[g.away_team] = g.home_team

    cur = d[(d.season == args.season) & (d.week < target)]
    rows = []
    for pid, g in cur.groupby("player_id", sort=False):
        g = g.sort_values("week")
        team = g.team.iloc[-1]
        if team not in facing or len(g) < MIN_GAMES:
            continue
        if g.attempts.sum() < MIN_ATTEMPTS:
            continue
        last = g.iloc[-1].to_dict()
        # the opponent's allowances, as of now
        o = cur[cur.opponent_team == facing[team]]
        last["o_cmp_p"] = o.completions.mean() if len(o) else np.nan
        last["o_att_p"] = o.attempts.mean() if len(o) else np.nan
        last["o_yds_p"] = o.passing_yards.mean() if len(o) else np.nan
        rows.append({"player": g.player_display_name.iloc[-1],
                     "player_id": pid, "team": team, "opponent": facing[team],
                     **features_from(g, last)})
    if not rows:
        print("Nobody qualifies yet.")
        return 1

    P = pd.DataFrame(rows)
    # Projections only. The over/under probabilities this model produced
    # were not calibrated - on held-out seasons a stated 24% came in at
    # 47% and a stated 77% at 46%, which is to say they carried no
    # information at all. The edge over a season average is 0.15
    # completions against a spread of 4.7, far too small to say anything
    # useful about a specific line. Shipping them would have looked
    # authoritative and been worthless.
    for c in MARKETS:
        P[c] = models[c].predict(P[FEATURES])
    P = P.sort_values("completions", ascending=False)

    print(f"\nWeek {target}: {len(P)} quarterbacks\n")
    print(f"  {'player':24} {'vs':5} {'completions':>12} {'yards':>8}")
    for _, r in P.iterrows():
        print(f"  {r.player[:24]:24} {r.opponent:5} {r.completions:>12.1f} "
              f"{r.passing_yards:>8.0f}")

    if args.json:
        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "season": args.season, "week": target, "trained_on": train,
            "honesty": {
                "mae": {"completions": 4.71, "passing_yards": 58.84},
                "mae_naive": {"completions": 4.86, "passing_yards": 61.33},
                "note": ("Both beat a season average walk-forward across "
                         "four seasons: completions by 0.15 at 2.5 standard "
                         "errors, yards by 2.5 at 3.4. No over/under "
                         "probabilities: the ones this model produced were "
                         "not calibrated, so they are deliberately absent. "
                         "Passing touchdowns are deliberately absent - they "
                         "failed to beat a season average three separate "
                         "ways. No odds are collected for these markets."),
            },
            "players": [
                {"player": r.player, "player_id": r.player_id,
                 "team": r.team, "opponent": r.opponent,
                 "games": int(r.games),
                 **{c: round(float(r[c]), 1) for c in MARKETS}}
                for _, r in P.iterrows()
            ],
        }
        json.dump(out, open("qb_model.json", "w"), indent=1)
        os.makedirs("proj_history", exist_ok=True)
        json.dump(out, open(f"proj_history/qb_{args.season}_wk{target}.json",
                            "w"), indent=1)
        print(f"\n  Wrote qb_model.json ({len(P)} quarterbacks)")
    return 0


def norm_name(s):
    return " ".join(w for w in str(s).lower().replace(".", "").replace("'", "").split()
                    if w not in ("jr", "sr", "ii", "iii", "iv"))


def grade(season):
    files = sorted(glob.glob(f"proj_history/qb_{season}_wk*.json"))
    if not files:
        print("No archived quarterback projections yet.")
        return {"graded": 0}
    d = load([season])
    actual = {}
    for _, r in d.iterrows():
        actual[(norm_name(r.player_display_name), int(r.week))] = {
            c: float(r[c]) for c in MARKETS}
    rows = []
    for f in files:
        try:
            snap = json.load(open(f))
        except Exception:
            continue
        for p in snap.get("players", []):
            got = actual.get((norm_name(p["player"]), snap["week"]))
            if not got:
                continue
            rows.append({"week": snap["week"], "player": p["player"],
                         **{c + "_proj": p[c] for c in MARKETS},
                         **{c + "_act": got[c] for c in MARKETS}})
    if not rows:
        print("Nothing matched a played game yet.")
        return {"graded": 0}
    r = pd.DataFrame(rows)
    out = {"graded": int(len(r)),
           "weeks": sorted(int(w) for w in r.week.unique()), "markets": {}}
    print(f"\n{len(r)} graded over weeks {out['weeks']}\n")
    for c in MARKETS:
        a = r[c + "_act"].values
        mm = float(np.abs(r[c + "_proj"] - a).mean())
        out["markets"][c] = {"mae": round(mm, 3),
                             "bias": round(float((r[c + "_proj"] - a).mean()), 3),
                             "actual_mean": round(float(a.mean()), 2)}
        print(f"  {c:16} error {mm:7.3f}   actual avg {a.mean():7.2f}")
    out["recent"] = sorted(
        [{"week": int(x.week), "player": x.player,
          **{c: [round(float(x[c + "_proj"]), 1), float(x[c + "_act"])]
             for c in MARKETS}} for _, x in r.iterrows()],
        key=lambda z: -z["week"])[:200]
    return out


if __name__ == "__main__":
    sys.exit(main())
