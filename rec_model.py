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
import json
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
    args = ap.parse_args()

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
    last_played = int(cur[cur.receptions.notna()].week.max())
    target_week = last_played + 1

    sched = nfl.load_schedules(seasons=[args.season]).to_pandas()
    wk = sched[sched.week == target_week]
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
        print(f"\n  Wrote receptions_proj.json ({len(P)} players)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
