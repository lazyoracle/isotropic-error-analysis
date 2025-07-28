"""This module contains functions for relevant probability distributions"""

import jax.numpy as jnp
from jax import Array

def double_factorial(n: int ) -> float:
    """
    Helper function to compute double factorial:
        
        n!! = n * (n-2) * (n-4) * ... * 1 (if n is odd) or 2 (if n is even).

    Parameters
    ----------
    n : int
        The integer for which to compute the double factorial.

    Returns
    -------
    float
        The value of the double factorial n!!
    """
    return jnp.where(n <= 0, 1, jnp.prod(jnp.arange(n, 0, -2)))



def normal_integrand(theta: float | Array, d: int, sigma: float) -> jnp.ndarray:
    """
    Computes the function g(θ) that is integrated to calculate F(θ) which is the 
    distribution function for the angle θ in a normal distribution:

        g(θ) = [(d-1)!! * (1-σ²) * sin^(d-1)(θ)] / [π * (d-2)!! * (1+σ²-2σcos(θ))^((d+1)/2)]

    Parameters
    ----------
    theta : float or array-like
        Angle parameter(s).
    d : int
        Dimension parameter.
    sigma : float
        Sigma parameter (should be in valid range).

    Returns
    -------
    result : float or ndarray
        Value(s) of the function evaluated at `theta`.
    """

    # Convert inputs to JAX arrays
    theta = jnp.asarray(theta)
    d = jnp.asarray(d, dtype=jnp.int32)
    sigma = jnp.asarray(sigma)
    
    # factorial components
    numerator_factorial = double_factorial(d - 1)
    denominator_factorial = double_factorial(d - 2)
    
    # Numerator components
    one_minus_sigma_sq = 1.0 - sigma**2
    sin_theta_power = jnp.power(jnp.sin(theta), d - 1)
    
    # Denominator components  
    denominator_base = 1.0 + sigma**2 - 2.0 * sigma * jnp.cos(theta)
    denominator_power = jnp.power(denominator_base, (d + 1) / 2.0)
    
    # Combine all terms
    numerator = numerator_factorial * one_minus_sigma_sq * sin_theta_power
    denominator = jnp.pi * denominator_factorial * denominator_power
    
    result = numerator / denominator
    
    return result