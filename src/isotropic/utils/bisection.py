"""This module contains functions for the bisection algorithm to calculate $F^{-1}$."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


# Don't jit as it takes callables as argument
def get_theta(
    F: Callable, a: float, b: float, x: float | ArrayLike, eps: float
) -> float:
    """
    Find the value of theta such that $F(\\theta) = x$ using the bisection method.

    Parameters
    ----------
    F : Callable
        Function for which to compute the inverse.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    x : float | ArrayLike
        Value for which to find the inverse.
    eps : float
        Tolerance for convergence.

    Returns
    -------
    float
        The value of $theta$ such that $F(\\theta) = x$.

    Notes
    -----
    This function assumes that $F$ is an increasing function in the interval $[a, b]$
    and that $F(a) \\leq x \\leq F(b)$. The bisection method is a root-finding method
    that repeatedly bisects an interval and then selects a subinterval in which a root exists.
    """

    def cond_fn(state):  # numpydoc ignore=GL08
        a, b = state
        return (b - a) > eps

    def body_fn(state):  # numpydoc ignore=GL08
        a, b = state
        c = (a + b) / 2.0
        Fc = F(c)
        a_new = jnp.where(Fc <= x, c, a)
        b_new = jnp.where(Fc <= x, b, c)
        return (a_new, b_new)

    a, b = jax.lax.while_loop(cond_fn, body_fn, (a, b))
    return (a + b) / 2.0
