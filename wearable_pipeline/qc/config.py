# SPDX-License-Identifier: GPL-3.0-only
"""Thresholds, each with where it comes from.

registered   pre-registered for the lab's field study
operational  a working QC decision from that study, not registered
unset        neither: the figure is reported and not graded
custom       changed by whoever ran the check
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULTS = Path(__file__).with_name("defaults.json")
PAIRS = ("worn_pct", "pupil_in_range_pct", "span_s", "part_boundary_s",
         "sync_residual_ms", "pupil_after_mask_pct", "gaze_seconds_valid_pct")


@dataclass
class Config:
    values: dict
    basis: dict
    rules: dict

    def __getitem__(self, key):
        return self.values[key]


def load_config(overrides: dict | str | Path | None = None) -> Config:
    """The defaults, with any overrides marked custom. `overrides` is a dict or a JSON file.

    `streams` is merged per stream; set a stream to null when that device is not used.
    """
    values, basis, rules = {}, {}, {}
    for source, entries in json.loads(DEFAULTS.read_text(encoding="utf-8")).items():
        for key, entry in entries.items():
            values[key], basis[key], rules[key] = entry["value"], source, entry["rule"]
    if isinstance(overrides, (str, Path)):
        overrides = json.loads(Path(overrides).read_text(encoding="utf-8"))
    for key, new in (overrides or {}).items():
        if key not in values:
            raise KeyError(f"unknown threshold {key!r}")
        before = copy.deepcopy(values[key])
        if key == "streams":
            for name, spec in new.items():
                if spec is None:
                    values[key].pop(name, None)
                else:
                    values[key][name] = {**values[key].get(name, {}), **spec}
        else:
            if key in PAIRS and new is not None and set(new) != {"pass", "warn"}:
                raise ValueError(f"{key} needs both 'pass' and 'warn'")
            values[key] = new
        if values[key] != before:
            basis[key] = "custom"
    return Config(values, basis, rules)
