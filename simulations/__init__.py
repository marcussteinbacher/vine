# simulations/__init__.py
# Plain package — import calculation functions directly from their modules.
# No registry needed. Pass them straight to Runner(calculation_fn=...).
#
# Example:
#   from simulations.smooth import fit_smooth
#   from simulations.copula import fit_copula
#   from functools import partial
#
#   runner = Runner(calculation_fn=fit_smooth, ...)
#   runner = Runner(calculation_fn=partial(fit_copula, family="gaussian"), ...)