#!/usr/bin/env python3
"""
Decide whether today's build should run right now.

Exit 0  -> run the build
Exit 1  -> skip, nothing to do this hour

Runs at the first hourly wake at or after RUN_AFTER local time, once a day.
GitHub's scheduler is best-effort and sometimes runs late or skips a slot,
so the workflow wakes every hour and this decides. Checking costs nothing;
only the odds fetch spends API credits.
"""

import json
import os
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

API = "https://statsapi.mlb.com/api/v1"
LOCAL = ZoneInfo("America/Chicago")   # your clock, for the 10am trigger
GAME_TZ = ZoneInfo("America/New_York")  # MLB labels game dates in Eastern
RUN_AFTER = time(10, 0)               # 10:00 am local


def already_done(day):
    """True if today's archive exists and already has market data in it."""
    path = f"history/{day}.json"
    if not os.path.exists(path):
        return False
    try:
        slate = json.load(open(path))
    except Exception:
        return False
    return any(s.get("market") for s in slate.get("starters", []))


def games_today(day):
    r = requests.get(f"{API}/schedule", params={"sportId": 1, "date": day},
                     timeout=20)
    r.raise_for_status()
    return sum(len(d.get("games", [])) for d in r.json().get("dates", []))


def emit(day):
    """Hand the slate date to the workflow so the builder uses the same one."""
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"day={day}\n")


def main():
    now = datetime.now(timezone.utc).astimezone(LOCAL)

    # The slate date must come from Eastern, not from the runner's UTC clock
    # and not from Central. Otherwise an evening run labels its output with
    # tomorrow's date and the next morning thinks the work is already done.
    day = str(datetime.now(timezone.utc).astimezone(GAME_TZ).date())
    emit(day)

    if os.environ.get("FORCE_RUN", "").lower() in ("1", "true", "yes"):
        print(f"{day}: forced, running regardless")
        return 0

    if now.time() < RUN_AFTER:
        print(f"{now:%H:%M %Z} - before {RUN_AFTER:%H:%M}, waiting")
        return 1

    if already_done(day):
        print(f"{day}: already built today, skipping")
        return 1

    try:
        n = games_today(day)
    except Exception as e:
        print(f"Couldn't read the schedule ({e}) - will retry next hour")
        return 1

    if not n:
        print(f"{day}: no games scheduled")
        return 1

    print(f"{day}: {n} games, {now:%H:%M %Z} - running now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
