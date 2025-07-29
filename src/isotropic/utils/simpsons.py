"""This module contains functions for estimating the integral of a function using Simpson's rule."""

from typing import Callable

import jax.numpy as jnp
from jax import Array


def simpsons_rule(f: Callable, a: float, b: float, C: float, tol: float) -> Array:
    """
    Estimates the integral of a function using Simpson's rule.

    Parameters
    ----------
    f : Callable
        Function to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    C : float
        Bound on 4th derivative of f.
    tol : float
        Desired tolerance for the integral estimate.

    Returns
    -------
    Array
        Estimated value of the integral.
    """
    # Estimate minimum number of intervals needed for given tolerance
    n: int = int(jnp.ceil(((180 * tol) / (C * (b - a) ** 5)) ** (-0.25)))
    if n % 2 == 1:
        n += 1  # Simpson's rule requires even n

    x: Array = jnp.linspace(a, b, n + 1)
    y: Array = f(x)

    S: Array = y[0] + y[-1] + 4 * jnp.sum(y[1:-1:2]) + 2 * jnp.sum(y[2:-2:2])
    integral: Array = (b - a) / (3 * n) * S
    return integral
