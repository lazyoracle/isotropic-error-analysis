"""This module contains functions for the bisection algorithm to calculate $F^{-1}$"""

from typing import Callable

from jax.typing import ArrayLike


def get_theta(
    F: Callable, a: float, b: float, x: float | ArrayLike, eps: float
) -> float:
    """
    Finds the value of theta such that $F(\\theta) = x$ using the bisection method.
    This function assumes that $F$ is an increasing function in the interval $[a, b]$
    and that $F(a) \\leq x \\leq F(b)$.

    The bisection method is a root-finding method that repeatedly bisects an interval
    and then selects a subinterval in which a root exists.

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
    """
    while b - a > eps:
        c = (a + b) / 2.0
        Fc = F(c)
        if Fc <= x:
            a = c
        else:
            b = c
    return (a + b) / 2.0
