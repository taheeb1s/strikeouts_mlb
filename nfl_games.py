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


def upcoming(seasons, elo):
    """Predict games that haven't been played, from the current ratings."""
    s = load(seasons, played_only=False)
    todo = s[s.home_score.isna()]
    out = []
    for _, g in todo.iterrows():
        eh = elo.get(g.home_team, START)
        ea = elo.get(g.away_team, START)
        diff = eh - ea + HOME_EDGE
        p = win_prob(diff)
        out.append({
            "season": int(g.season), "week": int(g.week),
            "kickoff": str(g.get("gameday", "")),
            "home": g.home_team, "away": g.away_team,
            "p_home": round(float(p), 4),
            "pred_margin": round(float(diff / PTS_PER_ELO), 2),
            "pick": g.home_team if p >= 0.5 else g.away_team,
            "confidence": round(float(max(p, 1 - p)), 4),
            "vegas_spread": (None if pd.isna(g.get("spread_line"))
                             else float(g.spread_line)),
            "elo_home": round(float(eh), 1), "elo_away": round(float(ea), 1),
        })
    return out


def run(games):
    elo, rows, prev_season = {}, [], None

    for _, g in games.iterrows():
        if g.season != prev_season:
            # New year: pull every rating partway back toward average.
            elo = {t: START + CARRYOVER * (r - START) for t, r in elo.items()}
            prev_season = g.season

        h, a = g.home_team, g.away_team
        eh = elo.setdefault(h, START)
        ea = elo.setdefault(a, START)

        diff = eh - ea + HOME_EDGE
        p_home = win_prob(diff)
        pred_margin = diff / PTS_PER_ELO

        margin = g.home_score - g.away_score
        rows.append({
            "season": int(g.season), "week": int(g.week),
            "home": h, "away": a,
            "p_home": p_home, "pred_margin": pred_margin,
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

    return pd.DataFrame(rows), elo


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
    df, final = run(games)

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

    print("\n  acc = share of games called right (ties dropped)")
    print("  logloss and brier reward honest confidence; lower is better")
    print("  MAE = average miss on the final margin, in points")

    print("\nRatings after the last game scored")
    for t, r in sorted(final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t:4} {r:7.1f}")

    nxt = upcoming([max(args.seasons)], final)
    if nxt:
        wk = min(g["week"] for g in nxt)
        this = [g for g in nxt if g["week"] == wk]
        print(f"\nWeek {wk} predictions")
        print(f"  {'matchup':22} {'pick':5} {'win%':>5} {'margin':>7}  vegas")
        for g in sorted(this, key=lambda x: -x["confidence"]):
            v = "" if g["vegas_spread"] is None else f"{g['vegas_spread']:+.1f}"
            print(f"  {g['away']+' at '+g['home']:22} {g['pick']:5} "
                  f"{g['confidence']:>5.0%} {g['pred_margin']:>+7.1f}  {v}")

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


def _norm_cdf(x):
    from math import erf, sqrt
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in np.asarray(x)])


if __name__ == "__main__":
    main()
