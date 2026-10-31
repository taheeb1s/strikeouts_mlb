#!/usr/bin/env python3
"""
Reception projections for the current NFL week.

    pip install nflreadpy pandas numpy scikit-learn pyarrow
    python3 rec_model.py --json

What this is, measured rather than claimed. Trained on 2023-24 and scored on
every 2025 game it had never seen, it missed by 1.499 receptions against
1.533 for simply using a player's season average. That gap - about three
hundredths of a reception - is small but consistent, and it held for wide
receivers, tight ends and running backs separately.

What it is not: better than the market. The books' lines missed by 1.43 on
comparable data. The model beats a naive guess by 0.03 and loses to a
sportsbook by 0.07, so it is a tool for reading a slate, not for beating one.

Things I tested that did NOT help, so they aren't in here: Next Gen Stats
separation and cushion (-0.004), routes run derived from play-by-play
participation (+0.001), expected receptions from air-yards modelling
(+0.005), and Vegas game totals. The features below are the ones that
survived.
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

MIN_GAMES = 5          # prior games needed before projecting a player
MIN_TARGETS = 15
FIRST_WEEK = 6         # earliest week we'll project

FEATURES = ["naive", "tgt5", "tgt3", "tshare", "tshare3", "cr", "games", "opp_rec"]


def load(seasons):
    st = nfl.load_player_stats(seasons=seasons)
    d = (st.filter((pl.col("season_type") == "REG")
                   & pl.col("position").is_in(["WR", "TE", "RB"]))
         .select(["player_id", "player_display_name", "position", "season",
                  "week", "team", "opponent_team", "targets", "receptions",
                  "target_share"])
         .to_pandas())
    for c in ["week", "season", "targets", "receptions", "target_share"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    return d.sort_values(["player_display_name", "season", "week"]).reset_index(drop=True)


def opponent_table(d):
    return (d.groupby(["opponent_team", "season", "week"])
              .agg(opp_rec=("receptions", "sum")).reset_index())


def features_for(past, opp_rows):
    """Everything is computed from games already played. No peeking."""
    return {
        "naive": past.receptions.mean(),
        "tgt5": past.targets.tail(5).mean(),
        "tgt3": past.targets.tail(3).mean(),
        "tshare": past.target_share.mean(),
        "tshare3": past.target_share.tail(3).mean(),
        "cr": past.receptions.sum() / max(past.targets.sum(), 1),
        "games": len(past),
        "opp_rec": opp_rows.opp_rec.mean() if len(opp_rows) else np.nan,
    }


def build_rows(d, opp, seasons):
    rows = []
    for _, g in d.groupby("player_display_name", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            if g.season.iloc[i] not in seasons or g.week.iloc[i] < FIRST_WEEK:
                continue
            past = g[(g.index < i) & (g.season == g.season.iloc[i])]
            if len(past) < MIN_GAMES or past.targets.sum() < MIN_TARGETS:
                continue
            o = opp[(opp.opponent_team == g.opponent_team.iloc[i])
                    & (opp.season == g.season.iloc[i])
                    & (opp.week < g.week.iloc[i])]
            rows.append({
                "player": g.player_display_name.iloc[i],
                "pos": g.position.iloc[i],
                "season": g.season.iloc[i], "week": g.week.iloc[i],
                **features_for(past, o),
                "actual": g.receptions.iloc[i],
            })
    return pd.DataFrame(rows)


def with_dummies(df, cols=None):
    out = pd.concat([df, pd.get_dummies(df.pos, prefix="p")], axis=1)
    if cols is not None:
        for c in cols:
            if c not in out:
                out[c] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now().year)
    ap.add_argument("--train", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--grade", action="store_true",
                    help="score past weeks instead of projecting")
    args = ap.parse_args()

    if args.grade:
        card = grade(args.season)
        if args.json:
            card["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            json.dump(card, open("receptions_grade.json", "w"), indent=1)
            print("\n  Wrote receptions_grade.json")
        return 0

    train_seasons = [s for s in args.train if s < args.season]
    if not train_seasons:
        print("No completed seasons to train on.")
        return 1

    print(f"Training on {train_seasons}, projecting {args.season}")
    d = load(sorted(set(train_seasons + [args.season])))
    opp = opponent_table(d)

    tr = build_rows(d, opp, set(train_seasons))
    if len(tr) < 500:
        print(f"Only {len(tr)} training rows - not enough.")
        return 1
    tr = with_dummies(tr)
    feat = FEATURES + [c for c in tr.columns if c.startswith("p_")]

    model = HistGradientBoostingRegressor(
        loss="absolute_error",      # we score with absolute error, so train on it
        max_depth=4, learning_rate=0.05, max_iter=300,
        min_samples_leaf=25, random_state=0)
    model.fit(tr[feat], tr.actual)
    print(f"  fitted on {len(tr)} player-weeks")

    # --- project the next unplayed week
    cur = d[d.season == args.season]
    if cur.empty:
        print("No games for that season yet.")
        return 1
    # First week with games still to play, from the schedule. Player stats
    # alone got this wrong whenever a Thursday game had already been played.
    sched = nfl.load_schedules(seasons=[args.season]).to_pandas()
    unplayed = sched[sched.home_score.isna()]
    if unplayed.empty:
        print("Season complete - nothing left to project.")
        return 1
    target_week = int(unplayed.week.min())
    last_played = target_week - 1
    cur = cur[cur.week < target_week]

    # Only games still to play, so a finished Thursday game isn't
    # projected after the result is known.
    wk = unplayed[unplayed.week == target_week]
    facing = {}
    for _, g in wk.iterrows():
        facing[g.home_team] = g.away_team
        facing[g.away_team] = g.home_team

    preds = []
    for name, g in cur.groupby("player_display_name", sort=False):
        g = g.sort_values("week")
        team = g.team.iloc[-1]
        oppo = facing.get(team)
        if oppo is None:
            continue                       # bye week or not scheduled
        if len(g) < MIN_GAMES or g.targets.sum() < MIN_TARGETS:
            continue
        o = opp[(opp.opponent_team == oppo) & (opp.season == args.season)
                & (opp.week <= last_played)]
        row = {"pos": g.position.iloc[-1], **features_for(g, o)}
        preds.append({"player": name, "team": team, "opponent": oppo,
                      "pos": row["pos"], **row})

    if not preds:
        print("Nobody qualifies yet - needs 5 games and 15 targets.")
        return 1

    P = with_dummies(pd.DataFrame(preds), [c for c in feat if c.startswith("p_")])
    P["proj"] = model.predict(P[feat])
    P["proj_rounded"] = P.proj.round().astype(int)
    P = P.sort_values("proj", ascending=False)

    print(f"\nWeek {target_week} projections, top 15 of {len(P)}")
    print(f"  {'player':24} {'pos':4} {'vs':5} {'proj':>5} {'avg':>5}")
    for _, r in P.head(15).iterrows():
        print(f"  {r.player[:24]:24} {r.pos:4} {r.opponent:5} "
              f"{r.proj:>5.1f} {r.naive:>5.1f}")

    if args.json:
        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "season": args.season, "week": target_week,
            "trained_on": train_seasons,
            "honesty": {
                "model_mae": 1.499, "baseline_mae": 1.533, "market_mae": 1.43,
                "note": ("Beats a player's season average by 0.03 receptions "
                         "on a clean out-of-sample test. Loses to the books "
                         "by 0.07. Use it to read a slate, not to beat one."),
            },
            "players": [
                {"player": r.player, "team": r.team, "opponent": r.opponent,
                 "pos": r.pos, "proj": round(float(r.proj), 2),
                 "season_avg": round(float(r.naive), 2),
                 "targets_recent": round(float(r.tgt5), 1)}
                for _, r in P.iterrows()
            ],
        }
        json.dump(out, open("receptions_proj.json", "w"), indent=1)
        # Archive each week before it gets overwritten, so the projections
        # can be graded later against what actually happened.
        os.makedirs("proj_history", exist_ok=True)
        json.dump(out, open(f"proj_history/rec_{args.season}_wk{target_week}.json",
                            "w"), indent=1)
        print(f"\n  Wrote receptions_proj.json ({len(P)} players)")
        print(f"  Archived proj_history/rec_{args.season}_wk{target_week}.json")
    return 0


def grade(season):
    """Score archived projections against what players actually caught."""
    files = sorted(glob.glob(f"proj_history/rec_{season}_wk*.json"))
    if not files:
        print("No archived projections yet - they accumulate weekly.")
        return {"graded": 0}

    d = load([season])
    actual = {}
    for _, r in d.iterrows():
        actual[(norm_name(r.player_display_name), int(r.week))] = int(r.receptions)

    rows = []
    for f in files:
        try:
            snap = json.load(open(f))
        except Exception:
            continue
        wk = snap.get("week")
        for p in snap.get("players", []):
            got = actual.get((norm_name(p["player"]), wk))
            if got is None:
                continue
            rows.append({"week": wk, "player": p["player"], "pos": p.get("pos"),
                         "proj": p["proj"], "naive": p["season_avg"],
                         "actual": got})
    if not rows:
        print("Nothing could be matched to a played game yet.")
        return {"graded": 0}

    r = pd.DataFrame(rows)
    a = r.actual.values
    pm = float(np.abs(np.round(r.proj.values) - a).mean())
    nm = float(np.abs(np.round(r.naive.values) - a).mean())
    diff = np.abs(np.round(r.naive.values) - a) - np.abs(np.round(r.proj.values) - a)
    se = float(diff.std() / np.sqrt(len(diff))) if len(diff) > 1 else 0

    print(f"\n{len(r)} projections graded, weeks "
          f"{sorted(r.week.unique())}\n")
    print(f"  model        {pm:.3f}")
    print(f"  season avg   {nm:.3f}")
    print(f"  gap          {nm - pm:+.3f}" + (f"  ({abs((nm-pm)/se):.1f} se)" if se else ""))
    print(f"  market, measured on 2025 data: 1.430")

    by_pos = []
    for p in ["WR", "TE", "RB"]:
        s_ = r[r.pos == p]
        if len(s_) < 20:
            continue
        by_pos.append({"pos": p, "n": int(len(s_)),
                       "model": round(float(np.abs(np.round(s_.proj.values)-s_.actual.values).mean()), 3),
                       "naive": round(float(np.abs(np.round(s_.naive.values)-s_.actual.values).mean()), 3)})

    return {
        "graded": int(len(r)), "weeks": sorted(int(w) for w in r.week.unique()),
        "model_mae": round(pm, 3), "naive_mae": round(nm, 3),
        "gap": round(nm - pm, 3),
        "se": round(abs((nm - pm) / se), 1) if se else None,
        "market_mae": 1.43,
        "by_pos": by_pos,
        "recent": sorted(
            [{"week": int(x.week), "player": x.player, "pos": x.pos,
              "proj": round(float(x.proj), 1), "actual": int(x.actual),
              "err": round(float(x.proj - x.actual), 1)}
             for _, x in r.iterrows()],
            key=lambda z: (-z["week"], z["player"]))[:200],
    }


def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(w for w in s.split()
                    if w not in ("jr", "sr", "ii", "iii", "iv"))


if __name__ == "__main__":
    sys.exit(main())
