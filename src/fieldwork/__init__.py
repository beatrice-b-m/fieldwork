"""Explore unfamiliar pandas tables through patterns and inspectable evidence.

``explore(df)`` surveys a table and ranks leads across four sections, which are
also available alone: ``missingness``, ``discover_dependencies``,
``suggest_paths`` and ``value_patterns``. ``profile(df, dimensions)`` combines
``levels``, ``census``, ``grain`` and ``pairs`` for chosen columns; ``joint_counts``
and ``infer_schema`` complete the individual tools.

Every analysis returns a ``Result``, reads the source without mutating it, and
accepts the same ``scope``, ``missing`` and ``table_id`` context and runtime
controls. Search and display budgets report omissions; only scopes, missing
exclusions and census pre-selection restrict rows. Results support ``to_frame``,
verified ``inspect``/``select`` and ``recompute``; ``Recipe`` reapplies settings to
new deliveries. Renderers consume saved evidence and return strings. The public API
is this module's ``__all__``, the members of those objects, ``Path``, and the types
in ``fieldwork.typing``.
"""

from ._explore import (
    KeySpec,
    SchemaProposal,
    census,
    grain,
    infer_schema,
    joint_counts,
    levels,
    pairs,
    profile,
)
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import Scope
from .navigation import suggest_paths
from .overview import explore
from .patterns import value_patterns
from .presentation import render_html, render_plaintext, render_svg, visualization_data
from .progress import (
    AnalysisCancelled,
    AnalysisError,
    CancellationToken,
    ProgressDisplay,
    ProgressEvent,
)
from .result import Result
from .workflow import Recipe, compare

__version__ = "0.3.1"
__all__ = [
    "AnalysisCancelled",
    "AnalysisError",
    "CancellationToken",
    "KeySpec",
    "ProgressDisplay",
    "ProgressEvent",
    "Recipe",
    "Result",
    "SchemaProposal",
    "Scope",
    "census",
    "compare",
    "discover_dependencies",
    "explore",
    "grain",
    "infer_schema",
    "joint_counts",
    "levels",
    "missingness",
    "pairs",
    "profile",
    "render_html",
    "render_plaintext",
    "render_svg",
    "suggest_paths",
    "value_patterns",
    "visualization_data",
]
