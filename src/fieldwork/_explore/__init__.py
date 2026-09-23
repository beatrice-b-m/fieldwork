"""Foundation analyses: levels, census, grain, pairs, joint counts and roles."""

from .census import census, levels
from .grain import KeySpec, grain
from .profile import profile
from .relations import joint_counts, pairs
from .roles import SchemaProposal, infer_schema

__all__ = [
    "KeySpec",
    "SchemaProposal",
    "census",
    "grain",
    "infer_schema",
    "joint_counts",
    "levels",
    "pairs",
    "profile",
]
