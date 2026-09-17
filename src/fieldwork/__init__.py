"""Explore unfamiliar data through patterns and inspectable evidence."""

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

__version__ = "0.1.0"
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
