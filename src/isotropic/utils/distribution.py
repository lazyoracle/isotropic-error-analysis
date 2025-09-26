"""This module contains functions for relevant probability distributions"""

import warnings

import jax.numpy as jnp
from jax import Array
from scipy.special import factorial2


def double_factorial_ratio_scipy(num: int, den: int) -> float:
    """
    Compute the ratio of double factorials num!! / den!!.

    Parameters
    ----------
    num : int
        The numerator double factorial.
    den : int
        The denominator double factorial.

    Returns
    -------
    float
        The ratio of the double factorials.

    Notes
    -----
    This only works for numbers up to 300.

    Raises
    ------
    ValueError
        If num or den is greater than 300.
    """
    if num > 300 or den > 300:
        raise ValueError("This only works for numbers up to 300")
    return factorial2(num) / factorial2(den)


def double_factorial_jax(n: int) -> Array:
    """
    Helper function to compute double factorial:

        n!! = n * (n-2) * (n-4) * ... * 1 (if n is odd) or 2 (if n is even).

    Parameters
    ----------
    n : int
        The integer for which to compute the double factorial.

    Returns
    -------
    Array
        The value of the double factorial n!!
    """
    # works for numbers as large as 9**6
    return jnp.where(n <= 0, 1, jnp.prod(jnp.arange(n, 0, -2, dtype=jnp.uint64)))


def double_factorial_ratio_jax(num: int, den: int) -> Array:
    """
    Computes the ratio of double factorials:

        num!! / den!!

    Parameters
    ----------
    num : int
        The numerator for the double factorial.
    den : int
        The denominator for the double factorial.

    Returns
    -------
    Array
        The value of the ratio num!! / den!!

    Notes
    -----
    For very large numbers, this is numerically stable only when |num - den| ~5.
    """
    warnings.warn(
        "This is an experimental implementation. There are known issues with using this for numbers larger than 2**8",
        UserWarning,
    )
    if abs(num - den) > 4:
        raise ValueError("num and den should be close to each other")
    num_elems = jnp.arange(num, 0, -2, dtype=jnp.uint64)
    den_elems = jnp.arange(den, 0, -2, dtype=jnp.uint64)

    len_diff = den_elems.shape[0] - num_elems.shape[0]

    # Ensure both num_elems and den_elems have the same length
    if len_diff > 0:
        num_elems = jnp.concatenate((num_elems, jnp.ones(len_diff, dtype=jnp.uint64)))
    else:
        den_elems = jnp.concatenate((den_elems, jnp.ones(-len_diff, dtype=jnp.uint64)))

    num_len = num_elems.shape[0]
    den_len = den_elems.shape[0]

    ratio_elems = jnp.zeros(num_len // 2)

    for k in jnp.arange(0, num_len // 2, 1):
        ratio_elems = ratio_elems.at[k].set(
            (num_elems[k] * num_elems[num_len - 1 - k])
            / (den_elems[k] * den_elems[den_len - 1 - k])
        )
    ratio = jnp.prod(ratio_elems)
    return ratio


def normal_integrand(theta: float, d: int, sigma: float) -> Array:
    """
    Computes the function g(θ) that is integrated to calculate F(θ) which is the
    distribution function for the angle θ in a normal distribution:

    $$g(\\theta) = \\frac{(d-1)!! \\times (1-\\sigma^2) \\times \\sin^{d-1}(\\theta)}{\\pi \\times (d-2)!! \\times (1+\\sigma^2-2\\sigma\\cos(\\theta))^{(d+1)/2}}$$

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
    """

    # TODO: Convert inputs to JAX arrays once @jit works
    # theta = jnp.asarray(theta)
    # d = jnp.asarray(d, dtype=jnp.int32)
    # sigma = jnp.asarray(sigma)

    # factorial components
    numerator_factorial = factorial2(d - 1)
    denominator_factorial = factorial2(d - 2)

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
