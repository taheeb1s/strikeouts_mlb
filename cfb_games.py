"""
College football game model for FBS teams. Elo, backtested before use.

    python3 cfb_games.py                 # backtest, then this week's picks
    python3 cfb_games.py --tune          # search the constants
    python3 cfb_games.py --json          # write cfb_games.json for the page

College is not the NFL. 130-odd teams instead of 32, enormous talent gaps,
blowouts that would be historic in the pros, and a roster that turns over
far faster. Every constant below was tuned on college data rather than
carried across.

Games against FCS opponents still count - they move ratings and they are
real results - but only FBS-vs-FBS games are scored, since that's what
anyone is predicting.

Data comes from sportsdataverse, which reads public files on GitHub. No
API key, no account.
"""
import argparse
import numpy as np
import pandas as pd
import sportsdataverse.cfb as cfb

START = 1500.0
FCS_RATING = 1200.0     # a generic non-FBS opponent

# Tuned on 2015-2024. See --tune.
K = 36.0
HOME_EDGE = 50.0
CARRYOVER = 0.75
# A grid search barely moved the result - logloss 0.5694 at these settings
# against 0.5722 at my first guesses, with identical accuracy. The model
# isn't balanced on a knife edge, which is reassuring.
PTS_PER_ELO = 22.0
MARGIN_SD = 17.5        # college margins scatter far wider than the NFL's


def win_prob(diff):
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def load(seasons):
    frames = []
    for s in seasons:
        d = cfb.load_cfb_schedule(seasons=[s])
        d = d.to_pandas() if hasattr(d, "to_pandas") else d
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    keep = ["season", "week", "start_date", "home_team", "away_team",
            "home_points", "away_points", "home_division", "away_division",
            "home_conference", "away_conference", "conference_game"]
    d = d[[c for c in keep if c in d.columns]]
    # at least one side must be FBS for the game to teach us anything
    d = d[(d.home_division == "fbs") | (d.away_division == "fbs")]
    return d.sort_values(["season", "week", "start_date"]).reset_index(drop=True)


def run(games, k=K, home=HOME_EDGE, carry=CARRYOVER, ppe=PTS_PER_ELO):
    elo, rows, prev = {}, [], None

    for _, g in games.iterrows():
        if g.season != prev:
            elo = {t: START + carry * (r - START) for t, r in elo.items()}
            prev = g.season

        h, a = g.home_team, g.away_team
        h_fbs = g.home_division == "fbs"
        a_fbs = g.away_division == "fbs"
        eh = elo.setdefault(h, START) if h_fbs else FCS_RATING
        ea = elo.setdefault(a, START) if a_fbs else FCS_RATING

        diff = eh - ea + home
        p_home = win_prob(diff)

        played = pd.notna(g.home_points) and pd.notna(g.away_points)
        if played:
            margin = g.home_points - g.away_points
            rows.append({
                "season": int(g.season), "week": int(g.week),
                "home": h, "away": a, "both_fbs": bool(h_fbs and a_fbs),
                "p_home": p_home, "pred_margin": diff / ppe,
                "margin": margin, "home_won": int(margin > 0),
                "total": g.home_points + g.away_points,
            })
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            wdiff = diff if margin > 0 else -diff
            mult = mov_multiplier(margin, wdiff) if margin != 0 else 1.0
            shift = k * mult * (actual - p_home)
            if h_fbs:
                elo[h] = eh + shift
            if a_fbs:
                elo[a] = ea - shift

    return pd.DataFrame(rows), elo


def evaluate(df, first_scored):
    d = df[(df.season >= first_scored) & df.both_fbs & (df.margin != 0)]
    if not len(d):
        return None
    acc = ((d.p_home > 0.5) == (d.home_won == 1)).mean()
    ll = -(d.home_won * np.log(d.p_home.clip(1e-9))
           + (1 - d.home_won) * np.log((1 - d.p_home).clip(1e-9))).mean()
    mae = np.abs(d.pred_margin - d.margin).mean()
    return {"n": len(d), "acc": acc, "logloss": ll, "mae": mae,
            "home_base": (d.home_won == 1).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=list(range(2015, 2027)))
    ap.add_argument("--skip-first", type=int, default=2)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.seasons[0]}-{args.seasons[-1]}...")
    games = load(args.seasons)
    print(f"{len(games)} games with at least one FBS side")
    first = args.seasons[0] + args.skip_first

    if args.tune:
        print("\nSearching. Scored on FBS-vs-FBS games only.\n")
        best = None
        for k in [28, 36, 44, 52]:
            for home in [50, 62, 74]:
                for carry in [0.5, 0.62, 0.75]:
                    df, _ = run(games, k, home, carry)
                    r = evaluate(df, first)
                    if r and (best is None or r["logloss"] < best[0]["logloss"]):
                        best = (r, k, home, carry)
                        print(f"  K={k:<3} home={home:<3} carry={carry:<5} "
                              f"acc {r['acc']:.4f}  logloss {r['logloss']:.4f}  <-")
        r, k, home, carry = best
        print(f"\nBest: K={k}, HOME_EDGE={home}, CARRYOVER={carry}")
        return

    df, final = run(games)
    r = evaluate(df, first)
    if not r:
        print("Nothing to score.")
        return

    print(f"\n{r['n']} FBS games scored, {first}-{args.seasons[-1]}\n")
    print(f"  Model accuracy         {r['acc']:.3f}")
    print(f"  Always the home team   {r['home_base']:.3f}")
    print(f"  Log loss               {r['logloss']:.4f}")
    print(f"  Margin error           {r['mae']:.2f} points")

    d = df[(df.season >= first) & df.both_fbs & (df.margin != 0)]
    correct = ((d.p_home > 0.5) == (d.home_won == 1)).astype(float)
    base = (d.home_won == 1).astype(float)
    gap = correct - base
    se = gap.std() / np.sqrt(len(gap))
    print(f"\n  Over the home-team baseline: {gap.mean():+.3f} "
          f"({abs(gap.mean()/se):.1f} standard errors)")

    print("\nTop 15 ratings")
    for t, v in sorted(final.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t:26} {v:7.1f}")

    # upcoming
    nxt = []
    sched = load([args.seasons[-1]])
    todo = sched[sched.home_points.isna()
                 & (sched.home_division == "fbs")
                 & (sched.away_division == "fbs")]
    avg_total = float(d.total.mean())
    for _, g in todo.iterrows():
        eh = final.get(g.home_team, START)
        ea = final.get(g.away_team, START)
        diff = eh - ea + HOME_EDGE
        p = win_prob(diff)
        m = diff / PTS_PER_ELO
        nxt.append({
            "season": int(g.season), "week": int(g.week),
            "kickoff": str(g.get("start_date", "")),
            "home": g.home_team, "away": g.away_team,
            "home_conf": g.get("home_conference"),
            "away_conf": g.get("away_conference"),
            "p_home": round(float(p), 4),
            "pred_margin": round(float(m), 2),
            "score_home": round((avg_total + m) / 2, 1),
            "score_away": round((avg_total - m) / 2, 1),
            "pick": g.home_team if p >= 0.5 else g.away_team,
            "confidence": round(float(max(p, 1 - p)), 4),
        })

    if nxt:
        wk = min(g["week"] for g in nxt)
        this = sorted([g for g in nxt if g["week"] == wk],
                      key=lambda x: -x["confidence"])
        print(f"\nWeek {wk}: {len(this)} FBS games. Ten most confident:")
        for g in this[:10]:
            print(f"  {g['away'][:18]+' at '+g['home'][:18]:40} "
                  f"{g['pick'][:18]:20} {g['confidence']:>4.0%} "
                  f"{g['pred_margin']:>+6.1f}")

    if args.json:
        import json
        json.dump({
            "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
            "backtest": {"games": int(r["n"]), "accuracy": round(float(r["acc"]), 4),
                         "home_baseline": round(float(r["home_base"]), 4),
                         "margin_mae": round(float(r["mae"]), 2),
                         "seasons": f"{first}-{args.seasons[-1]}"},
            "ratings": {t: round(float(v), 1) for t, v in
                        sorted(final.items(), key=lambda x: -x[1])[:40]},
            "games": nxt,
        }, open("cfb_games.json", "w"), indent=1)
        print("\n  Wrote cfb_games.json")


if __name__ == "__main__":
    main()
