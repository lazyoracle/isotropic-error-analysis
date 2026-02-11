"""This module contains functions for relevant probability distributions."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def double_factorial_jax(n: int) -> Array:
    """
    Helper function to compute double factorial.

    Parameters
    ----------
    n : int
        The integer for which to compute the double factorial.

    Returns
    -------
    Array
        The value of the double factorial n!! as a JAX array.

    Notes
    -----
    The double factorial is defined as:

        n!! = n * (n-2) * (n-4) * ... * 1 (if n is odd) or 2 (if n is even).
    """
    # works for numbers as large as 9**6
    return jnp.where(n <= 0, 1, jnp.prod(jnp.arange(n, 0, -2, dtype=jnp.uint64)))


def double_factorial_ratio(num: int, den: int) -> float:
    """
    Compute the ratio of double factorials num!! / den!! .

    Parameters
    ----------
    num : int
        The numerator double factorial.
    den : int
        The denominator double factorial.

    Returns
    -------
    float
        The ratio num!! / den!! .
    """
    num_list = list(range(num, 0, -2))
    den_list = list(range(den, 0, -2))
    # make sure both lists are the same length by padding the shorter one with 1s
    max_len = max(len(num_list), len(den_list))
    num_list += [1] * (max_len - len(num_list))
    den_list += [1] * (max_len - len(den_list))
    num_array = np.array(num_list)
    den_array = np.array(den_list)

    def ratio(a, b):
        return a / b

    result_array = np.vectorize(ratio)(num_array, den_array)
    return np.prod(result_array)


def normal_integrand(theta: float, d: int, sigma: float) -> Array:
    """
    Compute the function g(θ).

    Parameters
    ----------
    theta : float
        Angle parameter(s).
    d : int
        Dimension parameter.
    sigma : float
        Sigma parameter (should be in valid range).

    Returns
    -------
    Array
        Value(s) of the function evaluated at `theta`.

    Notes
    -----
    g(θ) is integrated to calculate F(θ) which is the
    distribution function for the angle θ in a normal distribution:

    $$g(\\theta) = \\frac{(d-1)!! \\times (1-\\sigma^2) \\times \\sin^{d-1}(\\theta)}{\\pi \\times (d-2)!! \\times (1+\\sigma^2-2\\sigma\\cos(\\theta))^{(d+1)/2}}$$.
    """

    # factorial ratio
    factorial_ratio = double_factorial_ratio(d - 1, d - 2)

    # Numerator components
    one_minus_sigma_sq = 1.0 - sigma**2
    sin_theta_power = jnp.power(jnp.sin(theta), d - 1)

    # Denominator components
    denominator_base = 1.0 + sigma**2 - 2.0 * sigma * jnp.cos(theta)
    denominator_power = jnp.power(denominator_base, (d + 1) / 2.0)

    # Combine all terms
    result = (
        factorial_ratio
        * (one_minus_sigma_sq * sin_theta_power)
        / (jnp.pi * denominator_power)
    )

    return result
