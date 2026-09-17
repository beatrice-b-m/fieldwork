"""Standard-library result models for feature exploration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "0.3"


@dataclass(frozen=True, repr=False)
class ExplorerResult(Mapping[str, Any]):
    """Immutable top-level result with a strict-JSON-compatible payload."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    stability: str = "unstable"

    def to_dict(self, *, resolve_references: bool = False, compact: bool = False) -> dict[str, Any]:
        """Export JSON data, optionally labeling references or sharing containers.

        Resolved exports retain IDs and typed values, adding local labels and
        values for inspection. They contain all quantitative evidence; use
        ``visualization_data(detail="topology")`` for disclosure filtering.
        ``compact=True`` wraps the data in a versioned shared-container envelope;
        restore it with the appropriate result class's ``from_dict`` method.
        """
        data = {
            "schema_version": self.schema_version,
            "stability": self.stability,
            "kind": self.kind,
            **self.payload,
        }
        if resolve_references:
            from .resolved import resolve_result

            data = resolve_result(data)
        if compact:
            from .._serialization import compact_result

            return compact_result(data)
        return data

    @classmethod
    def from_dict(cls, data):
        """Restore a foundation result from its ordinary or compact JSON export."""
        from .._serialization import expand_result

        data = expand_result(data)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported foundation schema version")
        return cls(
            data["kind"],
            {k: v for k, v in data.items() if k not in {"kind", "schema_version", "stability"}},
            schema_version=data["schema_version"],
            stability=data.get("stability", "unstable"),
        )

    def __repr__(self) -> str:
        from .render import render_plaintext

        return render_plaintext(self, width=100, max_lines=40, max_nodes=100)

    def __str__(self) -> str:
        return repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """Use the same bounded, escaped text in IPython and notebooks."""
        printer.text("ExplorerResult(...)" if cycle else repr(self))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class KeySpec:
    """An explicit, named determinant; required for composite keys."""

    name: str
    columns: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KeySpec.name must be a non-empty string")
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("KeySpec.columns must not be empty")
        object.__setattr__(self, "columns", columns)
