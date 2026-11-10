"""Frequency handling: parse pandas-style frequency strings into resolution buckets.

The bucket id feeds the Resolution-Aware Embedding (RAE), the pooling window of
Dynamic Temporal Pooling (DTP) and the season length of the seasonal-naive baseline.
The pooling windows keep the frequency -> patch-size map of the original EZERO
data loader (seconds 64, minutes 32, hours 16, days 8, weeks 4, months 2).
"""
from __future__ import annotations

import re

# bucket name -> (bucket id, DTP window / legacy patch size, season length)
BUCKETS = {
    "unknown": (0, 8, 1),
    "second": (1, 64, 60),
    "minute": (2, 32, 288),      # 1-5 minute data, daily season = 288 x 5min
    "10min": (3, 32, 144),
    "15min": (4, 32, 96),
    "30min": (5, 32, 48),
    "hour": (6, 16, 24),
    "day": (7, 8, 7),
    "week": (8, 4, 52),
    "month": (9, 2, 12),
    "quarter": (10, 2, 4),
    "year": (11, 2, 1),
}
NUM_BUCKETS = len(BUCKETS)
BUCKET_NAMES = sorted(BUCKETS, key=lambda k: BUCKETS[k][0])

_FREQ_RE = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)")


def bucket_name(freq: str | None) -> str:
    """Map a pandas frequency string (``'H'``, ``'15T'``, ``'5min'``, ``'W-SUN'``...) to a bucket."""
    if not freq:
        return "unknown"
    m = _FREQ_RE.match(str(freq))
    if not m:
        return "unknown"
    mult = int(m.group(1)) if m.group(1) else 1
    unit = m.group(2)
    u = unit.upper()
    if u in ("S", "SEC", "SECOND", "SECONDS") or unit == "s":
        return "second" if mult < 60 else "minute"
    if u in ("T", "MIN", "MINUTE", "MINUTES"):
        if mult <= 5:
            return "minute"
        if mult <= 10:
            return "10min"
        if mult <= 15:
            return "15min"
        if mult < 60:
            return "30min"
        return "hour"
    if u in ("H", "HOUR", "HOURS", "BH"):
        return "hour" if mult < 24 else "day"
    if u in ("D", "B", "C", "DAY", "DAYS"):
        return "day" if mult < 7 else "week"
    if u.startswith("W"):
        return "week"
    if u in ("M", "MS", "ME", "BM", "BMS", "SM", "SMS", "MONTH"):
        return "month" if mult < 3 else "quarter"
    if u.startswith("Q") or u.startswith("BQ"):
        return "quarter"
    if u.startswith("A") or u.startswith("Y") or u.startswith("BA") or u.startswith("BY"):
        return "year"
    return "unknown"


def bucket_id(freq: str | None) -> int:
    return BUCKETS[bucket_name(freq)][0]


def pool_window(freq: str | None) -> int:
    return BUCKETS[bucket_name(freq)][1]


def season_length(freq: str | None) -> int:
    return BUCKETS[bucket_name(freq)][2]


def pool_window_table() -> list[int]:
    """Pooling window indexed by bucket id (used on-device by DTP)."""
    return [BUCKETS[name][1] for name in BUCKET_NAMES]


def scale_freq(freq: str | None, k: int) -> str:
    """Frequency string of a series down-sampled by factor k (e.g. '5T' x 2 -> '10T', 'H' x 4 -> '4H')."""
    m = _FREQ_RE.match(str(freq or ""))
    if not m:
        return freq or ""
    mult = int(m.group(1)) if m.group(1) else 1
    return f"{mult * k}{m.group(2)}"
