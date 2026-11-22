# SPDX-License-Identifier: GPL-3.0-only
"""The report as JSON, and as one page."""
from __future__ import annotations

import json
from pathlib import Path

from .checks import SECTIONS

COLOUR = {"FAIL": "#c0392b", "WARN": "#d68910", "PASS": "#2f7d4f", "INFO": "#7f8c8d"}
TITLE = dict(sync="Sync checks", worn="Worn", gaze="Gaze validity", pupil="Pupil coverage",
             gaps="Recorder gaps", breaks="Breaks and work", rates="Sampling rates",
             alignment="Timestamp alignment")
MARK = dict(registered="R", operational="o", unset="-", custom="c", structural="")


def write_json(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def write_png(report: dict, path: str | Path) -> None:
    """One A4 page: verdict, timeline, every check."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.27, 11.69), facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(0.06, 0.967, "Session quality control", fontsize=15, weight="bold", va="top")
    fig.text(0.94, 0.967, report["verdict"], fontsize=15, weight="bold", va="top", ha="right",
             color="white", bbox=dict(facecolor=COLOUR[report["verdict"]], lw=0, pad=6))
    counts = "   ".join(f"{n} {status}" for status, n in report["counts"].items())
    fig.text(0.06, 0.942, f"{report['duration_s'] / 60:.1f} min recording   |   {counts}", fontsize=9, va="top")

    bars = fig.add_axes([0.19, 0.745, 0.75, 0.17])
    details, names = report["details"], []
    for name, row in details["alignment"]["timeline"].items():
        start, end = row["start_s"] / 60, row["end_s"] / 60
        bars.barh(len(names), end - start, left=start, height=0.6, color="#aebfcd")
        for a, b in row["gaps"]:  # widened so a short gap still shows
            bars.barh(len(names), max((b - a) / 60, report["duration_s"] / 60 / 300), left=a / 60,
                      height=0.6, color=COLOUR["FAIL"])
        names.append(name)
    for a, b in details["breaks"].get("work", []):
        bars.barh(len(names), (b - a) / 60, left=a / 60, height=0.6, color="#b9d8c2")
    for a, b in details["breaks"].get("breaks", []):
        bars.barh(len(names), (b - a) / 60, left=a / 60, height=0.6, color="#555555")
    names.append("work / breaks")
    for check in details["sync"].get("checks", []):
        good = check["state"] in ("accepted", "confirmed", "set")
        bars.plot(check["at_s"] / 60, len(names), "v", ms=6, color=COLOUR["PASS" if good else "WARN"])
    names.append("sync checks")
    for x in (0, report["duration_s"] / 60):
        bars.axvline(x, color="#333333", lw=0.6, ls=":")
    bars.set_yticks(range(len(names)), names)
    bars.set_ylim(len(names) - 0.4, -0.7)
    bars.tick_params(labelsize=7, length=0)
    bars.spines[["top", "right", "left"]].set_visible(False)
    bars.set_xlabel("minutes from recording start", fontsize=7)

    per_minute = details["gaze"].get("valid_pct_per_minute", [])
    line = fig.add_axes([0.19, 0.645, 0.75, 0.055])
    line.step([i + 0.5 for i in range(len(per_minute))],
              [v if v is not None else 0 for v in per_minute], where="mid", color="#2f6db5", lw=0.8)
    line.axhline(report["thresholds"]["gaze_valid_min_pct"]["value"], color=COLOUR["WARN"], lw=0.6, ls="--")
    line.set_xlim(*bars.get_xlim())
    line.set_ylim(0, 105)
    line.set_ylabel("valid gaze\nper minute, %", fontsize=7, rotation=0, ha="right", va="center")
    line.tick_params(labelsize=7, length=0)
    line.spines[["top", "right"]].set_visible(False)

    rows = []
    for section in SECTIONS:
        checks = [c for c in report["checks"] if c["section"] == section]
        if checks:
            rows += [(section, None), *((section, c) for c in checks)]
    half = (len(rows) + 1) // 2
    # A heading never ends a column.
    if half < len(rows) and rows[half - 1][1] is None:
        half -= 1
    columns = (rows[:half], rows[half:])
    step = min(0.0175, 0.54 / max(1, *map(len, columns)))
    size = min(6.4, step * 11.69 * 72 * 0.62)
    for left, column in zip((0.06, 0.515), columns):
        y = 0.605
        for section, check in column:
            if check is None:
                fig.text(left, y, TITLE[section], fontsize=size + 1, weight="bold", va="top")
            else:
                fig.text(left, y, check["status"], fontsize=size, weight="bold", va="top",
                         color=COLOUR[check["status"]])
                fig.text(left + 0.046, y, check["label"][:36], fontsize=size, va="top")
                fig.text(left + 0.306, y, check["shown"][:22], fontsize=size, va="top", ha="right")
                fig.text(left + 0.318, y, check["expected"][:18], fontsize=size, va="top", color="#666666")
                fig.text(left + 0.425, y, MARK[check["basis"]], fontsize=size, va="top", ha="right")
            y -= step
    fig.text(0.06, 0.025, "R pre-registered   o operational, not registered   c custom   - no threshold, "
             "reported only.\nQuality figures only: no gaze, pupil-size, fixation or blink outcome is shown.",
             fontsize=7, color="#444444")
    fig.savefig(path, format="png", dpi=110, metadata={"Software": None})


def write_report(report: dict, folder: str | Path) -> tuple[Path, Path]:
    """qc_report.json and qc_summary.png in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, png_path = folder / "qc_report.json", folder / "qc_summary.png"
    write_json(report, json_path)
    write_png(report, png_path)
    return json_path, png_path
