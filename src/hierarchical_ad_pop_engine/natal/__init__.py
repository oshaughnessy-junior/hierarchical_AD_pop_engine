"""Parametric natal (1st-generation) populations: mass, spin, redshift.

All three layers are thin wrappers around gwkokab — we reuse rather than
reinvent. See submodules for details.
"""

from .mass import build_natal_mass, log_natal_mass
from .redshift import MadauDickinsonRedshift, PowerlawRedshift
from .spin import build_natal_spin, log_natal_spin

__all__ = [
    "build_natal_mass",
    "log_natal_mass",
    "build_natal_spin",
    "log_natal_spin",
    "MadauDickinsonRedshift",
    "PowerlawRedshift",
]
