"""Escaping, width-aware clipping and wrapping, and value labels for every medium."""

from __future__ import annotations

import html
import re
import textwrap
import unicodedata
from collections.abc import Callable
from functools import cache
from typing import Any

from .._explore.encoding import display

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def controls(text: str) -> str:
    """Escape control and bidirectional override characters as \\uXXXX."""
    return _CONTROL.sub(lambda match: f"\\u{ord(match.group()):04x}", text)


def esc(value: Any) -> str:
    """HTML/SVG-safe text: controls escaped, then markup."""
    return html.escape(controls(str(value)), quote=True)


def label(value: Any, *, column: bool = False, missing_label: str = "<NA>") -> str:
    """Displayed form of a column name or exported value, controls escaped."""
    return controls(str(value) if column else display(value, missing_label))


def predicate(column: str, value: Any, missing_label: str = "<NA>") -> str:
    return f"{label(column, column=True)}={label(value, missing_label=missing_label)}"


def quantity(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def safe(text: str, unicode_mode: str) -> str:
    """Terminal-safe text; 'safe' mode also escapes non-ASCII characters."""
    text = controls(text)
    if unicode_mode == "safe":
        return text.encode("ascii", "backslashreplace").decode("ascii")
    if unicode_mode != "display":
        raise ValueError("unicode_mode must be 'safe' or 'display'")
    return text


def _fallback_width(character: str) -> int:
    if unicodedata.combining(character) or unicodedata.category(character) in {"Mn", "Me"}:
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1


@cache
def _width_function() -> Callable[[str], int]:
    try:
        from wcwidth import wcwidth
    except ImportError:
        return _fallback_width
    return lambda character: max(0, wcwidth(character))


def clip(text: str, width: int, unicode_mode: str) -> str:
    """Clip to a terminal width, measuring display cells in 'display' mode."""
    if unicode_mode != "display":
        if len(text) <= width:
            return text
        return "." * width if width <= 3 else text[: width - 3] + "..."
    cell_width = _width_function()
    if sum(cell_width(c) for c in text) <= width:
        return text
    marker = "." * min(3, width)
    target, used, output = max(0, width - len(marker)), 0, []
    for character in text:
        cells = cell_width(character)
        if used + cells > target:
            break
        output.append(character)
        used += cells
    return "".join(output) + marker


def wrap(text: str, width: int = 42) -> list[str]:
    """Wrap for SVG, counting wide glyphs conservatively without font metrics."""
    lines = []
    for line in textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False):
        part, used = "", 0.0
        for character in line:
            units = (
                2
                if unicodedata.east_asian_width(character) in {"W", "F"}
                else 1.9
                if character in "MW@%"
                else 1
            )
            if part and used + units > width:
                lines.append(part)
                part, used = "", 0.0
            part += character
            used += units
        if part:
            lines.append(part)
    return lines or [""]
