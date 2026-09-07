#!/usr/bin/env python3
"""
Add DraftKings strikeout prop lines to projections.json.

    python3 add_odds.py

Reads the projections you just built, looks up each pitcher's strikeout line,
strips the bookmaker's margin out of the two-way price to get a fair
probability, and writes the result back into projections.json.

Your API key lives in ~/.odds_api_key, never in this file. Create it with:

    echo "YOUR_KEY_HERE" > ~/.odds_api_key
    chmod 600 ~/.odds_api_key

Credits: the events list is free, then one credit per game. A 15-game slate
costs 15. The free tier is 500 a month, so once a day fits and twice doesn't.
This script prints your remaining balance every run.
"""

import json
import os
import sys
import unicodedata
from zoneinfo import ZoneInfo
from datetime import datetime

import requests

API = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
KEY_FILE = os.path.expanduser("~/.odds_api_key")
FEATURED = "draftkings"   # shown by name; all US books feed the consensus

session = requests.Session()


def load_key():
    if not os.path.exists(KEY_FILE):
        print(f"No API key found at {KEY_FILE}")
        print('Create it with:  echo "YOUR_KEY" > ~/.odds_api_key')
        sys.exit(1)
    key = open(KEY_FILE).read().strip()
    if not key:
        print(f"{KEY_FILE} is empty.")
        sys.exit(1)
    return key


def norm(name):
    """Normalise a player name so 'Walbert Ureña' matches 'Walbert Urena'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv")]
    return " ".join(parts)


def key_of(name):
    """Last name plus first initial - survives most nickname differences."""
    p = norm(name).split()
    return (p[-1], p[0][0]) if len(p) >= 2 else (norm(name), "")


def american_to_prob(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def devig(over_price, under_price):
    """Two-way prices carry the book's margin. Strip it to get fair odds."""
    o, u = american_to_prob(over_price), american_to_prob(under_price)
    total = o + u
    return o / total, total - 1  # fair probability of the over, and the vig


def event_date(commence):
    """MLB game dates are US Eastern, the API returns UTC."""
    try:
        dt = datetime.fromisoformat(commence.replace('Z', '+00:00'))
        return str(dt.astimezone(ZoneInfo('America/New_York')).date())
    except Exception:
        return None


def fetch_lines(key, want_date):
    """Map of pitcher key -> {line, fair_over, vig, over_price, under_price}."""
    r = session.get(f"{API}/events", params={"apiKey": key}, timeout=20)
    r.raise_for_status()
    events = r.json()

    # The events feed returns whatever is upcoming, which may be a different
    # day than the slate being priced. Without this filter, tomorrow's lines
    # get matched onto today's pitchers by name.
    kept = [e for e in events if event_date(e.get("commence_time", "")) == want_date]
    skipped = len(events) - len(kept)
    print(f"{len(events)} games on the board, {len(kept)} on {want_date}"
          + (f" ({skipped} on other dates, skipped)" if skipped else ""))
    if not kept:
        print("No games for that date have odds posted - they may have already"
              "\nstarted, or be too far out for the book to price.")
    events = kept

    lines, remaining = {}, None
    for i, ev in enumerate(events, 1):
        try:
            r = session.get(
                f"{API}/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": "us",
                        "markets": "pitcher_strikeouts",
                        "oddsFormat": "american"},
                timeout=20)
            if r.status_code == 422:
                continue  # market not offered for this game
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining", remaining)
            data = r.json()
        except Exception as e:
            print(f"  game {i}: {e}")
            continue

        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "pitcher_strikeouts":
                    continue
                sides = {}
                for out in mkt.get("outcomes", []):
                    who = out.get("description", "")
                    sides.setdefault(who, {})[out["name"]] = out
                for who, sd in sides.items():
                    if "Over" not in sd or "Under" not in sd:
                        continue
                    lines.setdefault(key_of(who), []).append({
                        "book": bm.get("key", "?"),
                        "book_title": bm.get("title", bm.get("key", "?")),
                        "line": sd["Over"]["point"],
                        "over_price": sd["Over"]["price"],
                        "under_price": sd["Under"]["price"],
                        "player": who,
                    })

    if remaining is not None:
        print(f"{remaining} API credits left this month")
    return lines


def tail(pmf, n):
    return sum(pmf[n:])


def to_decimal(price):
    return 1 + price / 100 if price > 0 else 1 + 100 / -price


def model_fair_line(pmf):
    """The half-point line where the model sits closest to a coin flip."""
    best, best_gap = None, 9
    for n in range(1, len(pmf)):
        gap = abs(tail(pmf, n) - 0.5)
        if gap < best_gap:
            best, best_gap = n, gap
    return (best - 0.5) if best else None


def expected_value(p, price):
    """Profit per $1 staked, if p is the true probability."""
    return p * (to_decimal(price) - 1) - (1 - p)


def summarise(quotes):
    """Consensus across books, at whichever line most of them are using."""
    counts = {}
    for q in quotes:
        counts[q["line"]] = counts.get(q["line"], 0) + 1
    line = max(counts, key=lambda k: (counts[k], -abs(k)))
    at_line = [q for q in quotes if q["line"] == line]

    fairs, vigs = [], []
    for q in at_line:
        f, v = devig(q["over_price"], q["under_price"])
        fairs.append(f)
        vigs.append(v)

    best_over = max(at_line, key=lambda q: to_decimal(q["over_price"]))
    best_under = max(at_line, key=lambda q: to_decimal(q["under_price"]))
    featured = next((q for q in at_line if q["book"] == FEATURED), None)

    return {
        "line": line,
        "books": len(at_line),
        "books_other_lines": len(quotes) - len(at_line),
        "consensus_over": round(sum(fairs) / len(fairs), 4),
        "book_spread": round(max(fairs) - min(fairs), 4),
        "avg_vig": round(sum(vigs) / len(vigs), 4),
        "best_over": {"price": best_over["over_price"],
                      "book": best_over["book_title"]},
        "best_under": {"price": best_under["under_price"],
                       "book": best_under["book_title"]},
        "featured": ({"book": featured["book_title"],
                      "over_price": featured["over_price"],
                      "under_price": featured["under_price"]}
                     if featured else None),
    }


def main():
    if not os.path.exists("projections.json"):
        print("No projections.json - run build_projections.py first.")
        return 1

    key = load_key()
    slate = json.load(open("projections.json"))

    try:
        lines = fetch_lines(key, slate['date'])
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code == 401:
            print("Key rejected (401). Check ~/.odds_api_key.")
        elif code == 429:
            print("Out of credits (429). Resets monthly.")
        else:
            print(f"Odds request failed: {e}")
        return 1

    matched = 0
    for s in slate["starters"]:
        got = lines.get(key_of(s["pitcher"]))
        if not got:
            s["market"] = None
            continue
        matched += 1

        info = summarise(got)

        # A 5.5 line means the Over needs 6 or more.
        threshold = int(info["line"]) + 1
        mine = tail(s["pmf"], threshold)

        ev_o = expected_value(mine, info["best_over"]["price"])
        ev_u = expected_value(1 - mine, info["best_under"]["price"])
        side = "Over" if ev_o >= ev_u else "Under"

        s["market"] = {
            **info,
            "threshold": threshold,
            "model_over": round(mine, 4),
            "model_line": model_fair_line(s["pmf"]),
            "edge": round(mine - info["consensus_over"], 4),
            "ev_over": round(ev_o, 4),
            "ev_under": round(ev_u, 4),
            "best_side": side,
            "best_ev": round(max(ev_o, ev_u), 4),
        }

    slate["odds_books"] = "all US books"
    with open("projections.json", "w") as f:
        json.dump(slate, f, indent=1)

    day = slate["date"]
    if os.path.exists(f"history/{day}.json"):
        with open(f"history/{day}.json", "w") as f:
            json.dump(slate, f, indent=1)

    print(f"\nMatched {matched} of {len(slate['starters'])} pitchers to a line")

    ranked = sorted((s for s in slate["starters"] if s.get("market")),
                    key=lambda x: -x["market"]["best_ev"])
    if ranked:
        print(f"\n{'pitcher':22} {'line':>5} {'mine':>5} {'mkt':>5} "
              f"{'me':>5}  {'best price':>22}  {'EV':>6}")
        for s in ranked:
            m = s["market"]
            b = m["best_over"] if m["best_side"] == "Over" else m["best_under"]
            px = f"{m['best_side']} {b['price']:+d} {b['book']}"
            print(f"  {s['pitcher']:20} {m['line']:>5} "
                  f"{str(m['model_line']):>5} "
                  f"{m['consensus_over']:>5.0%} {m['model_over']:>5.0%}  "
                  f"{px:>22}  {m['best_ev']:>+6.1%}")
        print("\n  EV is what my model says the price is worth. The model has"
              "\n  not been graded against results yet, so treat the column as"
              "\n  a list of disagreements to investigate, not a shopping list."
              "\n  Run score.py once you have a few weeks of history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
