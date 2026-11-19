"""wakegen: body-force field that reproduces a prescribed propeller-plane wake.

The Python side owns everything that is not the flow solver: reading the target
wake, parametrising the forces, the cost function, the adjoint gradient
projection and the optimiser. The flow solver (STAR-CCM+) is driven by a Java
macro that exchanges CSV files with ``python -m wakegen``.
"""

__version__ = "0.1.0"
