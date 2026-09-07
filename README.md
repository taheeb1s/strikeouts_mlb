# Strikeout projections

A morning read-out of today's probable starters, each with a projected
strikeout total and the distribution behind it.

## Setup, once

```
pip install requests numpy
```

## Each morning

```
python build_projections.py     # writes projections.json
python -m http.server 8000      # then open localhost:8000
```

The page has to be served, not opened as a file. Browsers block a page loaded
from `file://` from reading other local files, so double-clicking `index.html`
will fall back to the sample slate. Running it through `http.server` costs one
extra command and avoids the whole problem.

Pass a date to project a different day: `python build_projections.py 2026-09-07`.

## What the model does

Expected strikeouts splits into two independent questions:

**How many batters will he face?** Estimated from his recent starts, weighted
toward the last eight but pulled back toward his season mean so a single short
outing doesn't swing it.

**What share of them will he strike out?** His strikeout rate combined with the
opposing team's using the odds ratio, against the league baseline:

```
odds(x) = x / (1 - x)
matchup = odds(pitcher) * odds(opponent) / odds(league)
```

His rate is shrunk toward league average based on how many batters he's
actually faced this season, so a pitcher with three starts doesn't get treated
as though his numbers are settled.

Then it simulates 200,000 games — draw batters faced, draw strikeouts from a
binomial — which gives the full distribution instead of a point estimate.

## Known gaps

Roughly in order of how much they'd improve accuracy:

- **Batters faced is the weak link.** It ignores announced innings limits,
  post-injury ramp-ups, openers, and managers who pull starters early on
  principle. This is where the model loses the most, and where hand-checking
  the news each morning beats any statistic.
- **No handedness splits.** It uses the opponent's overall strikeout rate, not
  their rate against lefties or righties. The Stats API exposes splits via
  `stats=statSplits&sitCodes=vl,vr` — worth adding.
- **No park, umpire, or catcher framing.** Each is worth a couple of percent.
- **Team rate, not the posted lineup.** Once lineups post a few hours before
  first pitch, rebuilding with the nine actual hitters is more accurate than
  the team aggregate.

## Before trusting it

Build the naive baseline — each pitcher's season strikeouts per start — and
backtest both over past dates. Split chronologically, never randomly. Mean
absolute error around 2.0 is what the naive version gets; if this doesn't beat
that, the extra machinery isn't earning its keep.

A start has a standard deviation of roughly 2.2 to 2.5 strikeouts. That's the
floor no model gets under, which is why the page shows the distribution rather
than just the number.
