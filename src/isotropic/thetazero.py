"""This module contains functions for generating theta_0."""

import jax.numpy as jnp
from jax import Array

from isotropic.utils.bisection import get_theta


def get_theta_zero(x: Array, g: callable) -> float:
    """
    Calculate the inverse angle theta_0 with a normal distribution given a value x.

    This function finds the angle theta_0 such that the integral of g from 0 to theta_0 equals x.
    It uses Simpson's rule for numerical integration and a bisection method to find the root.

    Parameters
    ----------
    x : Array
        Value for which to find the inverse, should be uniformly distributed in [0, 1].
    g : callable
        Function g(theta) that is integrated to calculate F(theta).

    Returns
    -------
    float
        Value of theta_0.
    """

    # We wrap the function g into a callable F that integrates g from 0 to theta.
    def F(theta: float) -> Array:
        # TODO: Provide the correct value for C based on the 4th derivative bound
        # return simpsons_rule(g, 0, theta, ..., 1e-9)
        raise NotImplementedError("Simpson's rule integration is not implemented yet.")

    # Use bisection to find theta_0 such that the integral equals x
    theta_zero: float = get_theta(F, 0, jnp.pi, x, 1e-9)

    return theta_zero
