"""Explore unfamiliar pandas tables through patterns and inspectable evidence.

Use ``explore(df)`` for an automatic overview, or call ``missingness``,
``discover_dependencies``, ``suggest_paths``, and ``value_patterns`` separately.
``explore(df, dimensions)`` composes explicit foundation analyses; ``levels``,
``census``, ``grain``, ``pairs``, and ``joint_counts`` expose their individual tools.
Analyses do not mutate the source. Search/display budgets report omissions;
only explicit scopes, missing exclusions, and census preselection restrict rows.

Discovery results support ``to_frame``, verified ``inspect``/``select``, and
source-bound recomputation. Use ``Recipe`` for new deliveries. Renderers consume
saved evidence and return strings, while ``to_dict`` exports analytical data.
All dataframe analyses accept progress, cancel, and timeout runtime controls.
Import reusable option dictionary types from ``fieldwork.typing``. Only this
module's __all__, the public members/returned Path interface, and the exported
types in fieldwork.typing form the documented API; underscore modules and
unexported implementation helpers are private.
"""

from ._explore import (
    ExplorerResult,
    KeySpec,
    SchemaProposal,
    census,
    grain,
    infer_schema,
    joint_counts,
    levels,
)
from ._explore.relations import pairs
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import InvestigationResult, Scope
from .navigation import PathResult, suggest_paths
from .patterns import value_patterns
from .presentation import render_html, render_plaintext, render_svg, visualization_data
from .progress import AnalysisCancelled, CancellationToken, ProgressDisplay, ProgressEvent
from .workflow import Recipe, compare, explore

__version__ = "0.1.1"
__all__ = [
    "AnalysisCancelled",
    "CancellationToken",
    "ExplorerResult",
    "InvestigationResult",
    "KeySpec",
    "PathResult",
    "ProgressDisplay",
    "ProgressEvent",
    "Recipe",
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
    "render_html",
    "render_plaintext",
    "render_svg",
    "suggest_paths",
    "value_patterns",
    "visualization_data",
]
