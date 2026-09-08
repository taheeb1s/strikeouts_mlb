#!/usr/bin/env python3
"""
Backtest the strikeout model against every start this season.

    python3 mlb_backtest.py              # this season
    python3 mlb_backtest.py 2025         # a completed season

Same walk-forward rule as the live model: for each start, use only what was
known before that day. No peeking. Replicates build_projections.py exactly -
same shrinkage constants, same rolling window, same odds ratio.

Answers the question score.py needs a month to answer: does this beat simply
predicting each pitcher's season strikeouts per start?

It cannot test against the market - historical odds are a paid feature - so
"did we beat the books" still needs score.py and live collection.

Takes a few minutes and caches, so re-runs are fast. Nothing is written
except the cache.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np
import requests

API = "https://statsapi.mlb.com/api/v1"
CACHE = ".backtest_cache"

# Must match build_projections.py or this measures a different model.
K_PRIOR_BF = 100
RECENT_STARTS = 8
BF_PRIOR = 4
MIN_PRIOR_STARTS = 3
MIN_PRIOR_BF = 40

session = requests.Session()
session.headers["User-Agent"] = "strikeout-backtest/1.0"


def cached(name, fetch):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{name}.json"
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    data = fetch()
    try:
        json.dump(data, open(path, "w"))
    except Exception:
        pass
    return data


def get(path, **params):
    r = session.get(f"{API}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def starters(season):
    """Every pitcher who started a game this season.

    playerPool defaults to qualified only - roughly 48 pitchers, the ones
    with enough innings for a leaderboard. That silently excluded every
    spot starter, callup and opener, which is exactly the population where
    shrinkage is supposed to matter. ALL plus paging fixes it.
    """
    def fetch():
        out, offset, page = [], 0, 1000
        while True:
            d = get("stats", stats="season", group="pitching", season=season,
                    sportId=1, playerPool="ALL", limit=page, offset=offset)
            splits = d["stats"][0]["splits"] if d.get("stats") else []
            if not splits:
                break
            for sp in splits:
                st, p = sp["stat"], sp.get("player", {})
                if int(st.get("gamesStarted", 0)) >= 1 and p.get("id"):
                    out.append({"id": p["id"], "name": p.get("fullName", "?")})
            if len(splits) < page:
                break
            offset += page
        # de-dupe: traded players appear once per team
        seen, uniq = set(), []
        for p in out:
            if p["id"] not in seen:
                seen.add(p["id"])
                uniq.append(p)
        return uniq
    return cached(f"starters_all_{season}", fetch)


def pitcher_log(pid, season):
    def fetch():
        d = get(f"people/{pid}/stats", stats="gameLog",
                group="pitching", season=season)
        rows = []
        for sp in (d["stats"][0]["splits"] if d.get("stats") else []):
            s = sp["stat"]
            opp = sp.get("opponent", {})
            rows.append({
                "date": sp.get("date"),
                "gs": int(s.get("gamesStarted", 0)),
                "k": int(s.get("strikeOuts", 0)),
                "bf": int(s.get("battersFaced", 0)),
                "opp_id": opp.get("id"),
            })
        return sorted(rows, key=lambda r: r["date"] or "")
    return cached(f"p{pid}_{season}", fetch)


def pitcher_hands(season):
    """id -> 'L' or 'R'. One call for the whole league."""
    def fetch():
        d = get("sports/1/players", season=season)
        return {str(p["id"]): (p.get("pitchHand") or {}).get("code")
                for p in d.get("people", []) if p.get("pitchHand")}
    return cached(f"hands_{season}", fetch)


def team_hand_ratio(season):
    """team -> {'L': mult, 'R': mult} on their overall strikeout rate.

    The API only serves season totals for splits, not game logs, so this
    ratio uses the full season - a small leak of future information. It
    makes the handedness test optimistic, which is the point: if it can't
    help even with the leak, it won't help without it.
    """
    def fetch():
        teams = get("teams", sportId=1, season=season)["teams"]
        out = {}
        for t in teams:
            try:
                d = get(f"teams/{t['id']}/stats", stats="statSplits",
                        group="hitting", season=season, sitCodes="vl,vr")
            except Exception:
                continue
            rates = {}
            for blk in d.get("stats", []):
                code = (blk.get("split") or {}).get("code")
                for sp in blk.get("splits", []):
                    st = sp["stat"]
                    pa = int(st.get("plateAppearances", 0) or 0)
                    k = int(st.get("strikeOuts", 0) or 0)
                    c = code or (sp.get("split") or {}).get("code")
                    if pa > 200 and c in ("vl", "vr"):
                        rates[c] = k / pa
            if "vl" in rates and "vr" in rates:
                # weight by typical exposure to get the blended rate back
                overall = 0.28 * rates["vl"] + 0.72 * rates["vr"]
                if overall > 0:
                    out[str(t["id"])] = {"L": rates["vl"] / overall,
                                         "R": rates["vr"] / overall}
        return out
    return cached(f"handratio_{season}", fetch)


def team_hitting(season):
    """Each team's cumulative strikeout rate, day by day."""
    def fetch():
        teams = get("teams", sportId=1, season=season)["teams"]
        out = {}
        for t in teams:
            d = get(f"teams/{t['id']}/stats", stats="gameLog",
                    group="hitting", season=season)
            rows = []
            for sp in (d["stats"][0]["splits"] if d.get("stats") else []):
                s = sp["stat"]
                rows.append({"date": sp.get("date"),
                             "k": int(s.get("strikeOuts", 0)),
                             "pa": int(s.get("plateAppearances", 0))})
            out[str(t["id"])] = sorted(rows, key=lambda r: r["date"] or "")
        return out
    return cached(f"teams_{season}", fetch)


def running_k_rate(rows):
    """date -> (K, PA) accumulated strictly BEFORE that date."""
    acc, k, pa = {}, 0, 0
    for r in rows:
        acc[r["date"]] = (k, pa)
        k += r["k"]
        pa += r["pa"]
    return acc, (k, pa)


def odds(p):
    return p / (1 - p)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year
    print(f"Backtesting {season}. First run downloads a few hundred game "
          f"logs; later runs use the cache.\n")

    ps = starters(season)
    print(f"{len(ps)} pitchers with at least one start")
    if len(ps) < 150:
        print("  WARNING: a full season is normally 250-350 starters.")
        print("  This looks capped - the numbers below may not represent"
              " the whole league.")

    teams = team_hitting(season)
    print(f"{len(teams)} team hitting logs")

    hands = pitcher_hands(season)
    hratio = team_hand_ratio(season)
    print(f"{len(hands)} pitcher hands, {len(hratio)} teams with usable splits")

    team_acc, team_tot = {}, {}
    for tid, rows in teams.items():
        team_acc[tid], team_tot[tid] = running_k_rate(rows)

    # League rate, accumulated by date, so early-season predictions don't
    # get to use a baseline computed from the whole year.
    daily = defaultdict(lambda: [0, 0])
    for rows in teams.values():
        for r in rows:
            daily[r["date"]][0] += r["k"]
            daily[r["date"]][1] += r["pa"]
    league_by_date, k, pa = {}, 0, 0
    for d in sorted(daily):
        league_by_date[d] = (k / pa) if pa > 2000 else 0.223
        k += daily[d][0]
        pa += daily[d][1]

    rows = []
    for i, p in enumerate(ps, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(ps)} pitchers")
        try:
            log = pitcher_log(p["id"], season)
        except Exception:
            continue

        k_tot = bf_tot = 0
        # Strikeouts in STARTS only. k_tot includes relief outings, so
        # dividing it by the number of starts inflates the baseline for
        # anyone who works out of the bullpen between starts.
        k_starts = 0
        n_games = 0
        prior_starts = []
        for g in log:
            day = g["date"]
            if g["gs"] == 1 and len(prior_starts) >= MIN_PRIOR_STARTS \
                    and bf_tot >= MIN_PRIOR_BF and day:
                lg = league_by_date.get(day, 0.223)

                # --- rate, exactly as the live model computes it
                k_rate = (k_tot + lg * K_PRIOR_BF) / (bf_tot + K_PRIOR_BF)

                # --- volume
                recent = np.array(prior_starts[-RECENT_STARTS:], float)
                season_mean = float(np.mean(prior_starts))
                w = len(recent) / (len(recent) + BF_PRIOR)
                exp_bf = w * float(recent.mean()) + (1 - w) * season_mean

                # --- opponent, as of that date
                oid = str(g.get("opp_id"))
                ok = lg
                if oid in team_acc:
                    tk, tpa = team_acc[oid].get(day, (0, 0))
                    if tpa > 300:
                        ok = tk / tpa

                o = odds(k_rate) * odds(ok) / odds(lg)
                pmatch = o / (1 + o)

                # same thing, but with the opponent rate adjusted for which
                # hand the pitcher throws with
                hand = hands.get(str(p["id"]))
                ok_h = ok
                if hand in ("L", "R") and oid in hratio:
                    ok_h = min(0.45, max(0.10, ok * hratio[oid][hand]))
                oh = odds(k_rate) * odds(ok_h) / odds(lg)
                pmatch_h = oh / (1 + oh)

                rows.append({
                    "date": day,
                    "pitcher": p["name"],
                    "model": exp_bf * pmatch,
                    "model_hand": exp_bf * pmatch_h,
                    "naive": k_starts / len(prior_starts),
                    "rate_only": season_mean * k_rate,   # no opponent adj
                    "actual": g["k"],
                    "n_prior": len(prior_starts),
                    "relief_share": 1 - len(prior_starts) / max(1, n_games),
                    "exp_bf": exp_bf,
                    "act_bf": g["bf"],
                })

            k_tot += g["k"]
            bf_tot += g["bf"]
            n_games += 1
            if g["gs"] == 1 and g["bf"]:
                prior_starts.append(g["bf"])
                k_starts += g["k"]

    if not rows:
        print("\nNo starts could be scored. Season may be too young.")
        return 1

    rows.sort(key=lambda r: r["date"])

    m = np.array([r["model"] for r in rows])
    n = np.array([r["naive"] for r in rows])
    ro = np.array([r["rate_only"] for r in rows])
    mh = np.array([r["model_hand"] for r in rows])
    a = np.array([r["actual"] for r in rows], float)
    ebf = np.array([r["exp_bf"] for r in rows])
    abf = np.array([r["act_bf"] for r in rows], float)

    def mae(x):
        return float(np.abs(x - a).mean())

    print(f"\n{'='*56}")
    print(f"{len(rows)} starts, {rows[0]['date']} to {rows[-1]['date']}\n")
    print("Mean absolute error, strikeouts per start")
    base = mae(n)
    for label, arr in [("Season-to-date average (baseline)", n),
                       ("Rate only, no opponent adjustment", ro),
                       ("Full model", m),
                       ("Full model + handedness", mh)]:
        print(f"  {label:36} {mae(arr):.4f}   {base - mae(arr):+.4f}")

    print(f"\n  Actual spread          SD {a.std():.2f}")
    print(f"  Strikeout bias         {(m - a).mean():+.3f}")
    print(f"  Batters faced bias     {(ebf - abf).mean():+.3f}"
          f"   (projected {ebf.mean():.1f} vs actual {abf.mean():.1f})")

    # Where shrinkage should matter most: pitchers with little history.
    nprior = np.array([r["n_prior"] for r in rows])
    relief = np.array([r["relief_share"] for r in rows])
    print(f"\n  Relief appearances made up {relief.mean():.1%} of prior games"
          f" on average")

    print("\nBy how many prior starts the pitcher had")
    for lo, hi, label in [(3, 6, "3-6"), (7, 14, "7-14"), (15, 99, "15+")]:
        idx = np.where((nprior >= lo) & (nprior <= hi))[0]
        if len(idx) < 30:
            continue
        mm = np.abs(m[idx] - a[idx]).mean()
        nn = np.abs(n[idx] - a[idx]).mean()
        print(f"  {label:6} n={len(idx):5}  model {mm:.3f}   naive {nn:.3f}"
              f"   {nn - mm:+.3f}")

    # Does it hold up month by month, or is it one hot stretch?
    print("\nBy month")
    bym = defaultdict(list)
    for i, r in enumerate(rows):
        bym[r["date"][:7]].append(i)
    for mo in sorted(bym):
        idx = bym[mo]
        print(f"  {mo}  n={len(idx):5}  model {np.abs(m[idx]-a[idx]).mean():.3f}"
              f"   naive {np.abs(n[idx]-a[idx]).mean():.3f}"
              f"   {np.abs(n[idx]-a[idx]).mean()-np.abs(m[idx]-a[idx]).mean():+.3f}")

    print(f"\n{'='*56}")
    gap = base - mae(m)
    se = float(np.std(np.abs(n - a) - np.abs(m - a)) / np.sqrt(len(rows)))
    print(f"Gap {gap:+.4f}, standard error {se:.4f} "
          f"-> {abs(gap/se):.1f} standard errors")

    # Is handedness worth adding? Compare it against the model, not the baseline.
    hgap = mae(m) - mae(mh)
    hse = float(np.std(np.abs(m - a) - np.abs(mh - a)) / np.sqrt(len(rows)))
    if hse > 0:
        print(f"Handedness adds {hgap:+.4f} over the model "
              f"({abs(hgap/hse):.1f} standard errors)"
              + ("  <- worth keeping" if hgap > 2 * hse
                 else "  <- not worth it" if abs(hgap) < 2 * hse
                 else "  <- actively hurts"))
        print("  Note: this uses full-season splits, so it flatters itself"
              "\n  slightly. Treat it as a ceiling.")
    if abs(gap) < 2 * se:
        print("Inside noise. No evidence the model beats the baseline.")
    elif gap > 0:
        print("The model beats the baseline by more than noise explains.")
    else:
        print("The baseline beats the model by more than noise explains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
