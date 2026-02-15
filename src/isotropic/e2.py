"""This module contains functions for generating the vector $e_2$."""

from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from isotropic.utils.bisection import get_theta
from isotropic.utils.distribution import double_factorial_ratio


def _compute_fraction(dj: int, k: int, odd: bool) -> float:
    """
    Compute the fraction coefficient for the F_j summation at index k.

    Parameters
    ----------
    dj : int
        The value d - j.
    k : int
        The summation index.
    odd : bool
        True for the F_odd formula, False for F_even.

    Returns
    -------
    float
        The fraction value.
    """
    if odd:
        num_list = list(range(dj - 2, (2 * k + 2) - 1, -2))
        den_list = list(range(dj - 1, (2 * k + 1) - 1, -2))
    else:
        num_list = list(range(dj - 2, (2 * k + 1) - 1, -2))
        den_list = list(range(dj - 1, (2 * k) - 1, -2))
    max_len = max(len(num_list), len(den_list))
    if max_len == 0:
        return 1.0
    num_list += [1] * (max_len - len(num_list))
    den_list += [1] * (max_len - len(den_list))
    return float(np.prod(np.array(num_list) / np.array(den_list)))


# Don't jit as it has Python-level branching on static j/d;
# unrolls at trace time within get_e2_coeffs's loop
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

    if j % 2 == 1:
        C_j = (1 / 2) * double_factorial_ratio(dj - 1, dj - 2)
        k_max = (dj - 2) // 2
        # Precompute fractions (static values depending only on dj and k)
        fractions = jnp.array(
            [_compute_fraction(dj, k, odd=True) for k in range(k_max + 1)]
        )
        k_vals = jnp.arange(k_max + 1)
        sin_powers = jnp.power(jnp.sin(theta_j), 2 * k_vals)
        sum_terms = jnp.sum(fractions * sin_powers)
        return 0.5 - C_j * jnp.cos(theta_j) * sum_terms
    else:
        C_j = (1 / jnp.pi) * double_factorial_ratio(dj - 1, dj - 2)
        k_max = (dj - 1) // 2
        fractions = jnp.array(
            [_compute_fraction(dj, k, odd=False) for k in range(1, k_max + 1)]
        )
        k_vals = jnp.arange(1, k_max + 1)
        sin_powers = jnp.power(jnp.sin(theta_j), 2 * k_vals - 1)
        sum_terms = jnp.sum(fractions * sin_powers)
        return theta_j / jnp.pi - C_j * jnp.cos(theta_j) * sum_terms


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

    # Generate theta_j for j = 0, ..., d-3 using bisection method.
    # The Python for loop unrolls at trace time; each j gives a distinct
    # trace of F_j (different k_max, odd/even branch) which is fine for JIT.
    for j in range(0, d - 2, 1):
        # JAX PRNG is stateless, so we need to split the key
        key, subkey = random.split(key)
        x = random.uniform(key, shape=(), minval=0, maxval=1)

        theta_j = get_theta(
            F=lambda theta, _j=j: F_j(theta, _j, d),
            a=0,
            b=jnp.pi,
            x=x,
            eps=1e-9,
        )

        theta = theta.at[j].set(theta_j)

    # Spherical coordinate conversion: cumulative product of sines times cosine
    e2 = jnp.cumprod(jnp.concatenate([jnp.ones(1), jnp.sin(theta)]))
    e2 = e2 * jnp.cos(jnp.append(theta, 0.0))

    theta = jnp.append(theta, 0)  # Append 0 for cos(0) of last coordinate

    return theta, e2
