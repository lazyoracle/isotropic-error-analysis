"""This module contains functions for generating the vector $e_2$."""

from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as random
from jax import Array
from jax.typing import ArrayLike

from isotropic.utils.bisection import get_theta
from isotropic.utils.distribution import double_factorial_ratio_scipy


def F_j(theta_j: float, j: int, d: int) -> Array:
    """
    Calculate the function $F_j$ for the given angle $\\theta_j$ and index $j$ in dimension $d$.

    Parameters
    ----------
    theta_j : float
        The angle at which to evaluate the function.
    j : int
        The index corresponding to the angle.
    d : int
        The dimension of the space.

    Returns
    -------
    Array
        The value of the function $F_j$ evaluated at $\\theta_j$.
    """
    dj = d - j
    numoverden = double_factorial_ratio_scipy(dj - 2, dj - 1)

    def F_odd(_):
        C_j = (1 / 2) * double_factorial_ratio_scipy(dj - 1, dj - 2)
        prefactor = C_j * numoverden
        k_max = (dj - 2) // 2  # upper bound for k in range
        k_vals = jnp.arange(0, k_max + 1)

        def product_term(k):
            num_factors = jnp.arange(dj - 2, 2 * k + 1, -2)
            den_factors = jnp.arange(dj - 1, 2 * k, -2)
            num_prod = jnp.prod(num_factors) if num_factors.size > 0 else 1.0
            den_prod = jnp.prod(den_factors) if den_factors.size > 0 else 1.0
            return (num_prod / den_prod) * jnp.sin(theta_j) ** (2 * k)

        # TODO: Use vectorization for better performance
        # sum_terms = jnp.sum(jnp.vectorize(product_term)(k_vals))
        sum_terms = 0.0
        for k in k_vals:
            sum_terms += product_term(k)
        return prefactor - C_j * jnp.cos(theta_j) * sum_terms

    def F_even(_):
        C_j = (1 / jnp.pi) * double_factorial_ratio_scipy(dj - 1, dj - 2)
        prefactor = C_j * numoverden * theta_j
        k_max = (dj - 1) // 2
        k_vals = jnp.arange(1, k_max + 1)

        def product_term(k):
            num_factors = jnp.arange(dj - 2, 2 * k, -2)
            den_factors = jnp.arange(dj - 1, 2 * k - 1, -2)
            num_prod = jnp.prod(num_factors) if num_factors.size > 0 else 1.0
            den_prod = jnp.prod(den_factors) if den_factors.size > 0 else 1.0
            return (num_prod / den_prod) * jnp.sin(theta_j) ** (2 * k - 1)

        # TODO: Use vectorization for better performance
        # sum_terms = jnp.sum(jnp.vectorize(product_term)(k_vals))
        sum_terms = 0.0
        for k in k_vals:
            sum_terms += product_term(k)
        return prefactor - C_j * jnp.cos(theta_j) * sum_terms

    # TODO: Use a conditional to choose between F_odd and F_even based on j
    # return lax.cond(j % 2 == 1, F_odd, F_even, operand=None)
    if j % 2 == 1:
        return F_odd(None)
    else:
        return F_even(None)


def get_e2_coeffs(
    d: int, F_j: Callable, key: ArrayLike = random.PRNGKey(0)
) -> Tuple[Array, Array]:
    """
    Generate the coefficients of the vector $e_2$.

    Parameters
    ----------
    d : int
        Dimension of the space.
    F_j : Callable
        Function to compute $F_j$ for the given angle, dimension and index.
    key : ArrayLike, optional
        Random key for reproducibility, by default random.PRNGKey(0).

    Returns
    -------
    Tuple[Array, Array]
        A tuple containing:

        - theta: Array of angles used to construct $e_2$.
        - e2: Array representing the coefficients of the vector $e_2$.
    """
    theta: Array = jnp.zeros(d - 1)

    # Generate theta_{d-1} from a uniform distribution in [0, 2*pi]
    theta = theta.at[-1].set(random.uniform(key, shape=(), minval=0, maxval=2 * jnp.pi))

    # Generate theta_j for j = 1, ..., d-2 using bisection method
    # TODO: vectorize this loop
    for j in range(0, d - 2, 1):
        # JAX PRNG is stateless, so we need to split the key
        key, subkey = random.split(key)
        x = random.uniform(key, shape=(), minval=0, maxval=1)

        theta_j = get_theta(
            F=lambda theta: F_j(theta, j, d),
            a=0,
            b=jnp.pi,
            x=x,
            eps=1e-9,
        )

        theta = theta.at[j].set(theta_j)

    # e2 has dimension d
    e2: Array = jnp.ones(d)

    # e2[1] to e2[d-1] have products of sin(theta) terms
    # TODO: vectorize this loop
    for j in range(1, d):
        e2 = e2.at[j].set(e2[j - 1] * jnp.sin(theta[j - 1]))

    theta = jnp.append(theta, 0)  # Append 0 for cos(0) of last coordinate

    # e2[d] has additional cos(theta) term in product
    e2 = e2 * jnp.cos(theta)

    return theta, e2
