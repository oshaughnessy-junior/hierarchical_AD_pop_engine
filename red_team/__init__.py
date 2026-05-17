"""Red-team validation: discrete-realization parallel implementation.

See ``demo/PLAN_red_team_validation.md`` for the full plan. This package
provides an algorithmically-independent forward model for the coagulation
cascade — particle Monte Carlo, not continuous-density grid quadrature —
that the engine can be validated against.

Architectural rule: this package may import the engine's lowest-level
physics primitives (``remnants.mass.M_rem``, ``remnants.spin.chi_rem``,
``remnants.kick.retention_aligned``) and may transitively use jax via
them. It must **not** import anything from
``hierarchical_ad_pop_engine.response``.
"""
