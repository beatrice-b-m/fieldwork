"""Shared visual vocabulary: colours by meaning, type stacks, and shading ranges.

Figures are always drawn on white, so these values do not change with a report's
theme. Colour encodes an analysis result only: determined (blue) versus varying
(orange), an amount (the four-range scale), or an exclusion (red, always dashed).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

TEXT = "#000000"
INK = "#3a414a"
INK_MUTED = "#5f6670"
MUTED = "#6f7681"
SUBTLE = "#979ea8"
LINE = "#ced4db"
PANEL = "#dfe3e8"
SURFACE = "#f2f3f5"
PAPER = "#ffffff"

DETERMINED = ("#2f79d6", "#d9e9ff")  # stroke, fill: constant cells and n:1
VARIES = ("#e36927", "#ffdccc")  # varying cells and 1:n
EXCLUDED = ("#e81313", "#ffd9d9")
AMOUNT_SCALE = ("#f3efb0", "#83d494", "#06999a", "#00576e")  # four ranges, light to dark
AMOUNT = AMOUNT_SCALE[2]  # single-series bars
ON_DARKEST = "#ffffff"  # text on the darkest range and on a 1:1 cell

# fill, stroke, dashed, icon for each grain-matrix state and pair relation.
STATES: dict[str, tuple[str, str, bool, str | None]] = {
    "constant": (DETERMINED[1], DETERMINED[0], False, "positive"),
    "varying": (VARIES[1], VARIES[0], False, "partial"),
    "undefined": (PANEL, MUTED, True, "unknown"),
    "untested": (PAPER, SUBTLE, True, None),
}
RELATIONS: dict[str, tuple[str, str, bool]] = {
    "n:1": (DETERMINED[1], DETERMINED[0], False),
    "1:n": (VARIES[1], VARIES[0], False),
    "1:1": (INK, INK, False),
    "n:m": (PAPER, MUTED, True),
    "undefined": (PANEL, MUTED, True),
}

_ICONS = {
    "positive": '<path d="M3 8.5l3 3 7-7"/>',
    "partial": '<circle cx="8" cy="8" r="5.5"/><path d="M8 2.5a5.5 5.5 0 0 1 0 11z" fill="{c}"/>',
    "unknown": (
        '<circle cx="8" cy="8" r="5.5" stroke-dasharray="2 1.6"/>'
        '<path d="M6.4 6.4a1.7 1.7 0 1 1 2.3 1.6c-.5.2-.7.5-.7 1v.4"/><path d="M8 11.2v.1"/>'
    ),
}


def icon(kind: str, color: str, x: float, y: float, size: int = 14) -> str:
    """A 16-unit line icon placed in an SVG."""
    return (
        f'<svg x="{x}" y="{y}" width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{_ICONS[kind].format(c=color)}</svg>'
    )


RULES = ("equal", "log", "quantile")
DEFAULT_RULES = {"association": "equal", "counts": "log"}


def validate_shading(shading: Any) -> dict[str, Any]:
    """Shading options with defaults filled, or ValueError."""
    if shading is None:
        shading = {}
    if not isinstance(shading, Mapping):
        raise TypeError("shading must be a mapping with 'association' and/or 'counts' keys")
    unknown = set(shading) - set(DEFAULT_RULES)
    if unknown:
        raise ValueError(f"Unknown shading keys: {sorted(unknown)}")
    rules = {**DEFAULT_RULES, **shading}
    for key, rule in rules.items():
        if isinstance(rule, str):
            if rule not in RULES:
                raise ValueError(f"shading[{key!r}] must be one of {RULES} or three cut-offs")
            continue
        if (
            not isinstance(rule, Sequence)
            or len(rule) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in rule)
            or any(not math.isfinite(v) for v in rule)
            or not rule[0] < rule[1] < rule[2]
        ):
            raise ValueError(f"shading[{key!r}] cut-offs must be three increasing finite numbers")
    return rules


def _nice(value: float) -> float:
    """Round a count cut-off to one of 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 × 10^k."""
    scale = 10 ** math.floor(math.log10(value))
    return min((1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10), key=lambda k: abs(k * scale - value)) * scale


def cutoffs(values: Iterable[float], rule: Any, *, counts: bool) -> list[float] | None:
    """The three cut-offs splitting values into four ranges, or None without values.

    A value v falls in range k when it is at least the k-th cut-off; the lowest
    range holds everything below the first.
    """
    if not isinstance(rule, str):
        return [float(v) for v in rule]
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    low, high = data[0], data[-1]
    if rule == "quantile":
        cuts = [data[min(len(data) - 1, math.ceil(len(data) * k / 4))] for k in (1, 2, 3)]
    elif rule == "log":
        # Counts run from 1; other values from a thousandth of the largest.
        top = max(high, 1e-12)
        base = 1.0 if counts else top / 1000
        top = max(top, base)
        cuts = [base * (top / base) ** (k / 4) for k in (1, 2, 3)]
    else:
        cuts = [low + (high - low) * k / 4 for k in (1, 2, 3)]
    if counts:
        # Whole counts that still increase; log cut-offs are also rounded (2, 5, 12).
        whole: list[float] = []
        for cut in cuts:
            value = max(round(_nice(cut)) if rule == "log" else math.ceil(cut), 1)
            whole.append(float(max(value, whole[-1] + 1) if whole else value))
        return whole
    return [round(c, 2) for c in cuts]


def shade(value: float, cuts: Sequence[float]) -> int:
    """The range index, 0 (lowest) to 3, of a value."""
    return sum(value >= cut for cut in cuts)


def range_labels(cuts: Sequence[float], *, counts: bool) -> list[str]:
    """Legend labels for the four ranges."""
    text = [f"{c:g}" if counts else f"{c:.2f}" for c in cuts]
    return [f"< {text[0]}", f"{text[0]} – {text[1]}", f"{text[1]} – {text[2]}", f"≥ {text[2]}"]
