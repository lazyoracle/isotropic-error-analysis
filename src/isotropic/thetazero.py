"""This module contains functions for generating theta_0"""
from isotropic.utils.bisection import get_theta
from isotropic.utils.simpsons import simpsons_rule
import jax.numpy as jnp
from jax import Array

def get_theta_zero(g:callable,) -> float:
    """generate the angle theta_0 with a normal distribution

    Parameters
    ----------
    g: callable
        function g(theta) that is integrated to calculate F(theta)

    Returns
    -------
    float
        value of theta_0
    """
    # We generate a random number x with a uniform distribution in the interval [0, 1].
    x: Array = jnp.random.uniform(0, 1)

    # We wrap the function g into a callable F that integrates g from 0 to theta.
    def F(theta: float) -> Array:
        return simpsons_rule(g, 0, theta, ..., 1e-9) # TODO: Provide the correct value for C based on the 4th derivative bound

    # Use bisection to find theta_0 such that the integral equals x
    theta_zero: float = get_theta(F, 0, jnp.pi, x, 1e-9)
    
    return theta_zero