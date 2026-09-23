"""Public renderers: every result is projected once, then drawn as text, SVG or HTML."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ._present.common import clip, safe
from ._present.html import page
from ._present.project import project
from ._present.svg import figure
from ._present.text import lines
from .result import Result

Detail = Literal["full", "topology"]


def _data(result: Result | Mapping[str, Any]) -> Mapping[str, Any]:
    """A saved result as a mapping, read from the payload without re-exporting it."""
    if isinstance(result, Result):
        return {"kind": result.kind, "schema_version": result.schema_version, **result.payload}
    return result


def _limit(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _figure_section(data: Mapping[str, Any], section: str | None) -> str | None:
    """Figures of a profile show its grain section unless another is named."""
    return "grain" if section is None and data.get("kind") == "profile" else section


def visualization_data(
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Detail = "full",
) -> dict[str, Any]:
    """Project saved evidence into the data every renderer draws from.

    Parameters
    ----------
    result : Result or mapping
        A result or its to_dict export; the source frame is not needed.
    section : str or None, optional
        Part of an overview or profile to project; default None projects the
        whole result.
    detail : {'full', 'topology'}, optional
        'full' (default) includes measurements, counts and examples. 'topology'
        keeps labels and qualitative relationships only, in canonical order: no
        quantities, positions or statistics. Topology is a disclosure filter, not
        anonymization; labels can still identify values.

    Returns
    -------
    dict[str, Any]
        A fresh allowlisted projection (not a round-trip export).

    Raises
    ------
    ValueError
        The detail or section is invalid, or the section was not requested.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.visualization_data(result, detail="topology")["detail"]
    'topology'
    """
    return project(_data(result), section=section, detail=detail)


def render_plaintext(
    result: Result | Mapping[str, Any],
    *,
    width: int = 100,
    max_lines: int = 200,
    max_nodes: int = 1000,
    detail: Detail = "full",
    missing_label: str = "<NA>",
    unicode_mode: Literal["safe", "display"] = "display",
) -> str:
    """Render saved evidence as bounded, terminal-safe text.

    Parameters
    ----------
    result : Result or mapping
        A result or its to_dict export.
    width, max_lines, max_nodes : int, optional
        Positive line width (default 100; longer lines are clipped) and line
        count (default 200; a final marker shows more output exists), and the
        nonnegative per-list display budget for findings, candidates, tests and
        census nodes (default 1000).
    detail : {'full', 'topology'}, optional
        As in visualization_data.
    missing_label : str, optional
        Text shown for missing values; default "<NA>".
    unicode_mode : {'display', 'safe'}, optional
        'display' (default) keeps Unicode (width measured with wcwidth when
        installed); 'safe' escapes non-ASCII. Control and bidirectional override
        characters are always escaped.

    Returns
    -------
    str
        The text; nothing is printed.

    Raises
    ------
    ValueError
        A limit, detail or unicode mode is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.render_plaintext(result).splitlines()[0]
    'Fieldwork · missingness'
    """
    _limit("width", width, 1)
    _limit("max_lines", max_lines, 1)
    _limit("max_nodes", max_nodes, 0)
    safe("", unicode_mode)  # Validate even when no line is rendered.
    projection = project(_data(result), detail=detail, missing_label=missing_label)
    output: list[str] = []
    for line in lines(projection, max_nodes=max_nodes):
        if len(output) == max_lines:
            output[-1] = clip("... more output not rendered (max_lines)", width, unicode_mode)
            break
        output.append(clip(safe(line, unicode_mode), width, unicode_mode))
    return "\n".join(output)


def render_svg(
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Detail = "full",
    view: Literal["map", "matrix", "mapping", "association", "bars", "tree", "heatmap", "findings"]
    | None = None,
    show_exceptions: bool = False,
    max_findings: int = 12,
) -> str:
    """Render saved evidence as a self-contained static SVG.

    Parameters
    ----------
    result : Result or mapping
        A result or its to_dict export.
    section : str or None, optional
        Part of an overview or profile to draw; a profile defaults to 'grain'.
    detail : {'full', 'topology'}, optional
        As in visualization_data.
    view : str or None, optional
        Grain: 'map' (default) or 'matrix'; pairs: 'mapping' (default) or
        'association' (full detail only); levels: 'bars'; census: 'tree';
        joint counts: 'heatmap'; other kinds: 'findings'.
    show_exceptions : bool, optional
        Grain map only: list the columns each key fails to determine.
    max_findings : int, optional
        Nonnegative per-list display limit for findings views; default 12.

    Returns
    -------
    str
        SVG markup with no external resources.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.render_svg(result).startswith("<svg")
    True
    """
    _limit("max_findings", max_findings, 0)
    data = _data(result)
    projection = project(data, section=_figure_section(data, section), detail=detail)
    return figure(projection, view, show_exceptions=show_exceptions, max_findings=max_findings)


def render_html(
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Detail = "full",
    max_findings: int = 100,
) -> str:
    """Render a standalone, offline HTML report of saved evidence.

    Parameters
    ----------
    result : Result or mapping
        A result or its to_dict export.
    section : str or None, optional
        Part of an overview or profile to show; a profile defaults to 'grain'.
    detail : {'full', 'topology'}, optional
        As in visualization_data.
    max_findings : int, optional
        Nonnegative per-list display limit for findings, candidates and tests;
        default 100. Search in the page covers the included records only.

    Returns
    -------
    str
        One HTML document with embedded styles, figures and a single script.
        Controls filter the view; they never rerun analyses or read source rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.render_html(result).startswith("<!doctype html>")
    True
    """
    _limit("max_findings", max_findings, 0)
    data = _data(result)
    projection = project(data, section=_figure_section(data, section), detail=detail)
    return page(projection, max_findings=max_findings)
