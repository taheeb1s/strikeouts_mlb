#!/usr/bin/env python3
"""
Build today's strikeout projections and write projections.json.

    pip install requests numpy
    python build_projections.py             # today
    python build_projections.py 2026-09-07  # a specific date

Model, in two parts:

  1. Rate  - combine the pitcher's strikeout rate with the opposing lineup's
             using the odds ratio (log5), against the league baseline.
  2. Volume- estimate how many batters he'll face from his recent starts.

Then simulate: draw batters faced, draw strikeouts | BF ~ Binomial(BF, p).
That gives a full distribution rather than a single number.
"""

import json
import os
import sys
from datetime import date

import numpy as np
import requests

API = "https://statsapi.mlb.com/api/v1"
SEASON = date.today().year
MAX_K = 16
SIMS = 200_000

# Prior strength for shrinking a pitcher's K rate toward league average,
# in batters faced. A pitcher with this many TBF gets half his own rate,
# half the league's. K rate is one of the faster-stabilizing stats.
K_PRIOR_BF = 100

# Fallback when a pitcher has no start history (debut, September callup).
DEFAULT_BF, DEFAULT_BF_SD = 21.0, 4.5

rng = np.random.default_rng()
session = requests.Session()
session.headers["User-Agent"] = "strikeout-projections/1.0"


def get(path, **params):
    r = session.get(f"{API}/{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def odds(p):
    return p / (1 - p)


# ----------------------------------------------------------------- league

def team_k_rates(season):
    """Strikeout rate for every team's hitters, plus the league baseline."""
    data = get("teams/stats", stats="season", group="hitting",
               season=season, sportId=1)
    rates, tot_k, tot_pa = {}, 0, 0
    for split in data["stats"][0]["splits"]:
        s, tid = split["stat"], split["team"]["id"]
        k, pa = int(s["strikeOuts"]), int(s["plateAppearances"])
        if pa:
            rates[tid] = k / pa
            tot_k += k
            tot_pa += pa
    return rates, tot_k / tot_pa


# ----------------------------------------------------------------- pitcher

def pitcher_profile(pid, league_k):
    """Shrunk K rate, plus mean and spread of batters faced in recent starts."""
    log = get(f"people/{pid}/stats", stats="gameLog",
              group="pitching", season=SEASON)

    splits = log["stats"][0]["splits"] if log.get("stats") else []
    starts, k_tot, bf_tot = [], 0, 0

    for sp in splits:
        s = sp["stat"]
        bf = int(s.get("battersFaced", 0))
        k_tot += int(s.get("strikeOuts", 0))
        bf_tot += bf
        if int(s.get("gamesStarted", 0)) == 1 and bf:
            starts.append(bf)

    # Shrink toward league average by how much we've actually seen.
    k_rate = (k_tot + league_k * K_PRIOR_BF) / (bf_tot + K_PRIOR_BF)

    # Volume: recent starts carry the most signal, but pull toward his own
    # season mean so one short outing doesn't dominate.
    if starts:
        recent = np.array(starts[-8:], dtype=float)
        season_mean = float(np.mean(starts))
        w = len(recent) / (len(recent) + 4)
        mean_bf = w * float(recent.mean()) + (1 - w) * season_mean
        sd_bf = float(np.std(recent)) if len(recent) > 2 else DEFAULT_BF_SD
        sd_bf = max(sd_bf, 2.0)
    else:
        mean_bf, sd_bf = DEFAULT_BF, DEFAULT_BF_SD

    # Naive baseline to beat: his season strikeouts per start, as of today.
    naive = k_tot / len(starts) if starts else None

    return k_rate, mean_bf, sd_bf, bf_tot, len(starts), naive


# ----------------------------------------------------------------- model

def simulate(k_rate, opp_k, league_k, mean_bf, sd_bf):
    o = odds(k_rate) * odds(opp_k) / odds(league_k)
    p = o / (1 + o)
    bf = np.clip(np.round(rng.normal(mean_bf, sd_bf, SIMS)), 3, 36).astype(int)
    k = rng.binomial(bf, p)
    pmf = np.bincount(k, minlength=MAX_K + 1)[: MAX_K + 1] / SIMS
    return p, pmf, float(bf.mean())


# ----------------------------------------------------------------- build

def build(day):
    print(f"Building {day}")
    opp_rates, league_k = team_k_rates(SEASON)
    print(f"  league K rate {league_k:.1%} across {len(opp_rates)} teams")

    sched = get("schedule", sportId=1, date=day,
                hydrate="probablePitcher,team,venue,lineups")

    games = [g for d in sched.get("dates", []) for g in d.get("games", [])]
    if not games:
        print("  no games scheduled")
        return None

    starters = []
    for g in games:
        venue = g.get("venue", {}).get("name", "")
        start = g.get("gameDate", "")
        for side, other in (("home", "away"), ("away", "home")):
            t = g["teams"][side]
            pit = t.get("probablePitcher")
            if not pit:
                continue

            opp_id = g["teams"][other]["team"]["id"]
            opp_k = opp_rates.get(opp_id, league_k)

            try:
                k_rate, mbf, sdbf, seen_bf, n_starts, naive = pitcher_profile(
                    pit["id"], league_k)
            except Exception as e:
                print(f"  skipped {pit.get('fullName')}: {e}")
                continue

            p, pmf, bf = simulate(k_rate, opp_k, league_k, mbf, sdbf)
            proj = float(sum(i * v for i, v in enumerate(pmf)))

            starters.append({
                "pitcher": pit.get("fullName", "Unknown"),
                "pitcher_id": pit["id"],
                "team": t["team"].get("abbreviation", ""),
                "opponent": g["teams"][other]["team"].get("abbreviation", ""),
                "home": side == "home",
                "venue": venue,
                "game_time": local_time(start),
                "proj_k": round(proj, 2),
                "proj_bf": round(bf, 1),
                "naive_k": round(naive, 2) if naive is not None else None,
                "k_rate": round(p, 4),
                "pitcher_k_pct": round(k_rate, 4),
                "opp_k_pct": round(opp_k, 4),
                "batters_faced_ytd": seen_bf,
                "starts_ytd": n_starts,
                "lineup_confirmed": bool(
                    g.get("lineups", {}).get(f"{other}Players")),
                "pmf": [round(float(v), 5) for v in pmf],
            })
            print(f"  {starters[-1]['pitcher']:24} {proj:5.2f}  "
                  f"BF {bf:5.1f}  rate {p:.3f}")

    starters.sort(key=lambda s: -s["proj_k"])
    return {
        "date": str(day),
        "generated_at": _now(),
        "league_k_pct": round(league_k, 4),
        "sample": False,
        "starters": starters,
    }


def local_time(iso):
    if not iso:
        return ""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%-I:%M %p")
    except ValueError:
        return ""


def _now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    result = build(day)
    if result:
        with open("projections.json", "w") as f:
            json.dump(result, f, indent=1)

        # Keep a dated copy so accuracy can be scored later.
        os.makedirs("history", exist_ok=True)
        with open(f"history/{day}.json", "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nWrote projections.json — {len(result['starters'])} starters")
    else:
        sys.exit(1)
