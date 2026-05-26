"""NR-calibrated remnant fits: M_rem, χ_rem, v_kick — aligned and vector.

References:

- mass: Buonanno-Kidder-Lehner 2008 / Barausse-Morozova-Rezzolla 2012
  polynomial fit in (η, χ_eff).
- spin (aligned): Barausse-Rezzolla 2009 "final spin formula".
- spin (vector): Barausse-Rezzolla 2009 full vector form — see
  :mod:`remnants.spin_vector`.
- kick (aligned): Lousto-Zlochower 2008/2011 aligned-spin. Retention
  is a Heaviside step in v_esc — see ``remnants.kick`` module docstring
  for the AD consequences.
- kick (vector): Lousto-Zlochower with in-plane "superkick" term —
  see :mod:`remnants.kick_vector`.  Marginalisation over a prescribed
  orientation distribution (Session B) recovers smoothness in v_esc.
"""

from .kick import retention_aligned, v_kick_aligned
from .kick_vector import v_kick_vec
from .marginal import (
    P_retain_marginalised,
    chi_rem_at_orientation,
    chi_rem_distribution,
    chi_rem_sample,
    v_kick_at_orientation,
    v_kick_distribution,
    v_kick_sample,
)
from .mass import M_rem, chi_effective, E_rad_fraction, eta
from .spin import chi_rem
from .spin_vector import chi_rem_vec

__all__ = [
    "M_rem",
    "chi_rem",
    "chi_rem_vec",
    "v_kick_aligned",
    "v_kick_vec",
    "retention_aligned",
    "chi_effective",
    "E_rad_fraction",
    "eta",
    # Session B tilt-marginalised wrappers.
    "chi_rem_at_orientation",
    "v_kick_at_orientation",
    "chi_rem_distribution",
    "v_kick_distribution",
    "P_retain_marginalised",
    "chi_rem_sample",
    "v_kick_sample",
]
