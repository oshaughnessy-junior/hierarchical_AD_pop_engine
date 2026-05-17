"""ResponseFunction — per-environment merger-rate density."""

from .coagulation import CoagulationResponse
from .protocol import ResponseFunction

__all__ = ["ResponseFunction", "CoagulationResponse"]
