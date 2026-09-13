"""
NFL game outcome model: Elo ratings, backtested before anyone trusts it.

    python3 nfl_games.py                 # backtest, then this week's picks
    python3 nfl_games.py --seasons 2021 2022 2023 2024 2025

Elo with a margin-of-victory adjustment. Every rating is built walking
forward - a game is always predicted using only what was known before it.

Scored against three baselines: picking the home team every time, picking
the better record, and the closing Vegas spread. That last one is the real
bar, and nothing public clears it.
"""
import argparse
import numpy as np
import pandas as pd
import nflreadpy as nfl

START = 1500.0
K = 20.0
HOME_EDGE = 48.0        # Elo points, worth about 1.9 on the scoreboard
CARRYOVER = 0.75        # how much of a rating survives into the next season
PTS_PER_ELO = 25.0      # 25 Elo points ~ 1 point of margin
MARGIN_SD = 13.5        # spread of actual margins around the prediction
LG_PTS = 22.5           # league average points per team per game
OD_K = 0.08             # how fast scoring ratings move toward recent form
TOTAL_TRUST = 0.5       # how far to trust them over a flat league average
HOME_PTS = 1.0          # home team's share of the scoring edge


def win_prob(diff):
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    """Blowouts move ratings more, but less so when already expected."""
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def load(seasons, played_only=True):
    s = nfl.load_schedules(seasons=seasons).to_pandas()
    s = s[s.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    if played_only:
        s = s.dropna(subset=["home_score", "away_score"])
    return s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def upcoming(seasons, elo, off, dfn):
    """Predict games that haven't been played, from the current ratings."""
    s = load(seasons, played_only=False)
    todo = s[s.home_score.isna()]
    out = []
    for _, g in todo.iterrows():
        eh = elo.get(g.home_team, START)
        ea = elo.get(g.away_team, START)
        diff = eh - ea + HOME_EDGE
        p = win_prob(diff)
        m = diff / PTS_PER_ELO
        raw = (2 * LG_PTS + off.get(g.home_team, 0.0) + dfn.get(g.away_team, 0.0)
               + off.get(g.away_team, 0.0) + dfn.get(g.home_team, 0.0))
        tot = TOTAL_TRUST * raw + (1 - TOTAL_TRUST) * 2 * LG_PTS
        out.append({
            "season": int(g.season), "week": int(g.week),
            "kickoff": str(g.get("gameday", "")),
            "home": g.home_team, "away": g.away_team,
            "p_home": round(float(p), 4),
            "pred_margin": round(float(diff / PTS_PER_ELO), 2),
            "score_home": round(float((tot + m) / 2), 1),
            "score_away": round(float((tot - m) / 2), 1),
            "pred_total": round(float(tot), 1),
            "pick": g.home_team if p >= 0.5 else g.away_team,
            "confidence": round(float(max(p, 1 - p)), 4),
            "vegas_spread": (None if pd.isna(g.get("spread_line"))
                             else float(g.spread_line)),
            "elo_home": round(float(eh), 1), "elo_away": round(float(ea), 1),
        })
    return out


def run(games):
    elo, rows, prev_season = {}, [], None
    # Opponent-adjusted scoring rates: what a team scores and concedes
    # relative to average, once you account for who they played.
    off, dfn = {}, {}

    for _, g in games.iterrows():
        if g.season != prev_season:
            # New year: pull every rating partway back toward average.
            elo = {t: START + CARRYOVER * (r - START) for t, r in elo.items()}
            off = {t: CARRYOVER * v for t, v in off.items()}
            dfn = {t: CARRYOVER * v for t, v in dfn.items()}
            prev_season = g.season

        h, a = g.home_team, g.away_team
        eh = elo.setdefault(h, START)
        ea = elo.setdefault(a, START)

        oh, dh = off.setdefault(h, 0.0), dfn.setdefault(h, 0.0)
        oa, da = off.setdefault(a, 0.0), dfn.setdefault(a, 0.0)

        diff = eh - ea + HOME_EDGE
        p_home = win_prob(diff)
        pred_margin = diff / PTS_PER_ELO

        # Expected points for each side, then rebalance so the margin
        # matches Elo - which is the part that's been validated.
        exp_h = LG_PTS + oh + da + HOME_PTS
        exp_a = LG_PTS + oa + dh - HOME_PTS
        # Scoring ratings are noisy, so meet a flat league average halfway.
        # Trusting them fully scored worse than ignoring them entirely.
        total = TOTAL_TRUST * (exp_h + exp_a) + (1 - TOTAL_TRUST) * 2 * LG_PTS
        pred_h = (total + pred_margin) / 2
        pred_a = (total - pred_margin) / 2

        margin = g.home_score - g.away_score
        rows.append({
            "season": int(g.season), "week": int(g.week),
            "home": h, "away": a,
            "p_home": p_home, "pred_margin": pred_margin,
            "pred_total": total, "pred_h": pred_h, "pred_a": pred_a,
            "total": g.home_score + g.away_score,
            "margin": margin, "home_won": int(margin > 0),
            "spread_line": g.get("spread_line", np.nan),
            "elo_h": eh, "elo_a": ea,
        })

        # update
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        mult = mov_multiplier(margin, winner_diff) if margin != 0 else 1.0
        shift = K * mult * (actual - p_home)
        elo[h] = eh + shift
        elo[a] = ea - shift

        # Move scoring ratings toward what actually happened, with the
        # opponent's strength taken out first.
        off[h] = oh + OD_K * ((g.home_score - HOME_PTS - da - LG_PTS) - oh)
        dfn[a] = da + OD_K * ((g.home_score - HOME_PTS - oh - LG_PTS) - da)
        off[a] = oa + OD_K * ((g.away_score + HOME_PTS - dh - LG_PTS) - oa)
        dfn[h] = dh + OD_K * ((g.away_score + HOME_PTS - oa - LG_PTS) - dh)

    return pd.DataFrame(rows), elo, off, dfn


def score(df, label):
    d = df[df.margin != 0]
    acc = ((d.p_home > 0.5) == (d.home_won == 1)).mean()
    ll = -(d.home_won * np.log(d.p_home.clip(1e-9))
           + (1 - d.home_won) * np.log((1 - d.p_home).clip(1e-9))).mean()
    brier = ((d.p_home - d.home_won) ** 2).mean()
    mae = np.abs(d.pred_margin - d.margin).mean()
    print(f"  {label:28} {acc:.3f}  {ll:.4f}  {brier:.4f}  {mae:.2f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--json", action="store_true",
                    help="write nfl_games.json for the page")
    ap.add_argument("--skip-first", type=int, default=2,
                    help="ignore early seasons while ratings settle")
    args = ap.parse_args()

    print(f"Loading {args.seasons[0]}-{args.seasons[-1]}...")
    games = load(args.seasons)
    df, final, off, dfn = run(games)

    keep = df[df.season >= args.seasons[0] + args.skip_first]
    d = keep[keep.margin != 0]
    print(f"\n{len(d)} games scored, {d.season.min()}-{d.season.max()}\n")
    print(f"  {'':28} {'acc':>5}  {'logloss':>7}  {'brier':>6}  {'MAE':>5}")

    elo_acc = score(keep, "Elo model")

    home = keep.copy()
    home["p_home"] = 0.5 + 1e-6
    home["pred_margin"] = d.margin.mean()
    score(home, "Always the home team")

    # Vegas closing line, where available
    vg = keep.dropna(subset=["spread_line"]).copy()
    if len(vg) > 100:
        # spread_line is the home team's line: positive means home favoured
        vg["p_home"] = 1 - _norm_cdf(-vg.spread_line / MARGIN_SD)
        vg["pred_margin"] = vg.spread_line
        score(vg, "Vegas closing line")

    # Is beating the home-team baseline more than luck?
    correct = ((d.p_home > 0.5) == (d.home_won == 1)).astype(float)
    base = (d.home_won == 1).astype(float)
    diff = correct - base
    se = diff.std() / np.sqrt(len(diff))
    print(f"\n  Elo over home-team baseline: {diff.mean():+.3f} "
          f"({abs(diff.mean() / se):.1f} standard errors)")

    # Are the score predictions any better than assuming a league-average game?
    tot_mae = np.abs(d.pred_total - d.total).mean()
    flat_mae = np.abs(d.total.mean() - d.total).mean()
    side_mae = (np.abs(d.pred_h - (d.margin + d.total) / 2).mean()
                + np.abs(d.pred_a - (d.total - d.margin) / 2).mean()) / 2
    print(f"\n  Game total, model error      {tot_mae:.2f} points")
    print(f"  Game total, flat average     {flat_mae:.2f} points")
    print(f"  Each team's score, error     {side_mae:.2f} points")
    dt = np.abs(d.total.mean() - d.total) - np.abs(d.pred_total - d.total)
    se = dt.std() / np.sqrt(len(dt))
    print(f"  Totals beat a flat average by {dt.mean():+.3f} "
          f"({abs(dt.mean()/se):.1f} se) - the two knobs behind this were")
    print("  tuned on this same data, so treat it as an upper bound.")

    print("\n  acc = share of games called right (ties dropped)")
    print("  logloss and brier reward honest confidence; lower is better")
    print("  MAE = average miss on the final margin, in points")

    print("\nRatings after the last game scored")
    for t, r in sorted(final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t:4} {r:7.1f}")

    nxt = upcoming([max(args.seasons)], final, off, dfn)
    if nxt:
        wk = min(g["week"] for g in nxt)
        this = [g for g in nxt if g["week"] == wk]
        print(f"\nWeek {wk} predictions")
        print(f"  {'matchup':22} {'score':>11} {'pick':5} {'win%':>5}  vegas")
        for g in sorted(this, key=lambda x: -x["confidence"]):
            v = "" if g["vegas_spread"] is None else f"{g['vegas_spread']:+.1f}"
            sc = f"{g['score_away']:.0f}-{g['score_home']:.0f}"
            print(f"  {g['away']+' at '+g['home']:22} {sc:>11} {g['pick']:5} "
                  f"{g['confidence']:>5.0%}  {v}")

    if args.json:
        import json
        json.dump({
            "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
            "backtest": {"games": int(len(d)), "accuracy": round(float(elo_acc), 4),
                         "home_baseline": round(float((d.home_won == 1).mean()), 4),
                         "seasons": f"{int(d.season.min())}-{int(d.season.max())}"},
            "ratings": {t: round(float(r), 1) for t, r in
                        sorted(final.items(), key=lambda x: -x[1])},
            "games": nxt,
        }, open("nfl_games.json", "w"), indent=1)
        print("\n  Wrote nfl_games.json")

        # Date -> week lookup so nfl_build.py can label reception snapshots
        # without needing nflreadpy itself.
        sched = load([max(args.seasons)], played_only=False)
        weeks = {}
        for _, g in sched.iterrows():
            day = str(g.get("gameday", ""))[:10]
            if day:
                weeks[day] = int(g.week)
        json.dump({"season": int(max(args.seasons)), "weeks": weeks},
                  open("nfl_weeks.json", "w"), indent=1)
        print(f"  Wrote nfl_weeks.json ({len(weeks)} dates)")


def _norm_cdf(x):
    from math import erf, sqrt
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in np.asarray(x)])


if __name__ == "__main__":
    main()
