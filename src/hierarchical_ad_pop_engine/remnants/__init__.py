"""NR-calibrated remnant fits: M_rem, χ_rem, v_kick (aligned-only).

References:

- mass: Buonanno-Kidder-Lehner 2008 / Barausse-Morozova-Rezzolla 2012
  polynomial fit in (η, χ_eff).
- spin: Barausse-Rezzolla 2009 aligned-spin "final spin formula".
- kick: Lousto-Zlochower 2008/2011 aligned-spin. Retention is a Heaviside
  step in v_esc — see ``remnants.kick`` module docstring for the AD
  consequences.
"""

from .kick import retention_aligned, v_kick_aligned
from .mass import M_rem, chi_effective, E_rad_fraction, eta
from .spin import chi_rem

__all__ = [
    "M_rem",
    "chi_rem",
    "v_kick_aligned",
    "retention_aligned",
    "chi_effective",
    "E_rad_fraction",
    "eta",
]
