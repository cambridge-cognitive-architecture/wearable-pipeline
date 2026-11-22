# SPDX-License-Identifier: GPL-3.0-only
"""One reference clock.

The eye tracker's UTC epoch nanoseconds are the reference. Logger timestamps
are ISO-8601 UTC. Phone timestamps are naive local wall-clock time, placed
with a UTC offset derived from the recording's own metadata. Nothing depends
on the time zone of the computer running the code.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

UTC = dt.timezone.utc
NS = 1_000_000_000
INAT = np.iinfo(np.int64).min


def iso_utc_to_ns(x: str | Iterable[str]) -> int | np.ndarray:
    """ISO UTC to epoch ns. Unparseable items in a list become INAT."""
    if isinstance(x, str):
        ts = pd.Timestamp(x.replace("Z", "+00:00"))
        ts = ts.tz_localize(UTC) if ts.tz is None else ts.tz_convert(UTC)
        return int(ts.as_unit("ns").value)
    ts = pd.to_datetime(pd.Series(x), utc=True, errors="coerce", format="ISO8601")
    # pin the unit: pandas may infer microseconds, a silent 1000x error
    return ts.dt.as_unit("ns").astype("int64").to_numpy()


def local_naive_to_ns(x: str | Iterable[str], offset_s: float) -> int | np.ndarray:
    """Naive local wall-clock time to epoch ns; offset_s is local minus UTC."""
    shift = int(round(offset_s * NS))
    if isinstance(x, str):
        return int(pd.Timestamp(x).tz_localize(UTC).as_unit("ns").value) - shift
    ns = iso_utc_to_ns(x)
    out = ns - shift
    out[ns == INAT] = INAT
    return out


def derive_device_offset_s(info: dict) -> float:
    """UTC offset from a start instant recorded as both epoch and local time."""
    name = (info.get("template_data") or {}).get("recording_name") or info.get("recording_name")
    if not name:
        raise KeyError("metadata has no recording_name to derive the offset from")
    name = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", name.strip().replace("_", "T", 1))
    utc = dt.datetime.fromtimestamp(int(info["start_time"]) / NS, UTC)
    local = dt.datetime.fromisoformat(name).replace(tzinfo=UTC)
    return float(round((local - utc).total_seconds() / 60) * 60)


@dataclass
class Clock:
    start_ns: int
    end_ns: int
    offset_s: float

    @classmethod
    def from_info(cls, info: dict | str | Path) -> "Clock":
        if not isinstance(info, dict):
            info = json.loads(Path(info).read_text(encoding="utf-8"))
        start = int(info["start_time"])
        return cls(start, start + int(info.get("duration", 0)), derive_device_offset_s(info))

    def rel_s(self, ns):
        """Seconds from the start. Subtract as integers to keep nanosecond precision."""
        a = np.asarray(ns, dtype=np.int64)
        missing = a == INAT
        out = (np.where(missing, self.start_ns, a) - self.start_ns) / NS
        out = np.where(missing, np.nan, out)
        return float(out) if out.ndim == 0 else out

    def iso_to_rel(self, x):
        return self.rel_s(iso_utc_to_ns(x))

    def local_to_rel(self, x):
        return self.rel_s(local_naive_to_ns(x, self.offset_s))
