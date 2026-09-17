"""Explore unfamiliar data through patterns and inspectable evidence."""
from ._explore import *
from ._explore import __all__
from ._explore.relations import pairs

__version__ = "0.1.0"
__all__ = [*__all__, "pairs"]
