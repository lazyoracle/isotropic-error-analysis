"""This module contains functions for generating the vector $e_2$."""

from typing import Tuple

import jax
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


# Not jitted: contains Python-level branching on j % 2 and j-dependent shapes,
# making it unsuitable for vmap. Used as a standalone utility and in tests.
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


def get_e2_coeffs(d: int, key: ArrayLike = random.PRNGKey(0)) -> Tuple[Array, Array]:
    """
    Generate the coefficients of the vector $e_2$.

    Parameters
    ----------
    d : int
        Dimension of the space.
    key : ArrayLike, optional
        Random key for reproducibility, by default random.PRNGKey(0).

    Returns
    -------
    Tuple[Array, Array]
        A tuple containing:

        - theta: Array of angles used to construct $e_2$.
        - e2: Array representing the coefficients of the vector $e_2$.

    Notes
    -----
    PRNG stream: keys are derived via a single ``random.split(key, d-1)`` call,
    producing ``d-1`` independent subkeys upfront. This differs from the
    previous sequential ``key, subkey = random.split(key)`` chain and will
    yield numerically different draws from the same seed, while remaining
    statistically equivalent.
    """
    # Draw keys upfront: all_subkeys[0] for the azimuthal angle,
    # all_subkeys[1:] for the d-2 bisection angles.
    all_subkeys = random.split(key, d - 1)  # shape (d-1, 2)
    theta_last = random.uniform(all_subkeys[0], shape=(), minval=0, maxval=2 * jnp.pi)

    # Early return for d < 3 (no bisection needed)
    if d < 3:
        theta = jnp.array([theta_last])
        e2 = jnp.cumprod(jnp.concatenate([jnp.ones(1), jnp.sin(theta)]))
        e2 = e2 * jnp.cos(jnp.append(theta, 0.0))
        return jnp.append(theta, 0), e2

    # --- Trace-time precomputation of static arrays ---
    # max number of sum terms across all j; tight at j=0 (even) and j=1 (odd)
    max_k = (d - 1) // 2

    C_list: list = []
    fracs_list: list = []
    sinexp_list: list = []
    is_odd_list: list = []

    for j in range(d - 2):
        dj = d - j
        if j % 2 == 1:  # odd j
            C_j = 0.5 * double_factorial_ratio(dj - 1, dj - 2)
            k_max_j = (dj - 2) // 2
            fracs = [_compute_fraction(dj, k, odd=True) for k in range(k_max_j + 1)]
            sin_exps = [2 * k for k in range(k_max_j + 1)]
            is_odd = 1.0
        else:  # even j
            C_j = (1.0 / np.pi) * double_factorial_ratio(dj - 1, dj - 2)
            k_max_j = (dj - 1) // 2
            fracs = [_compute_fraction(dj, k, odd=False) for k in range(1, k_max_j + 1)]
            sin_exps = [2 * k - 1 for k in range(1, k_max_j + 1)]
            is_odd = 0.0

        # Zero-pad fracs; pad sin_exps with int 1 (keeps array integer-typed,
        # ensuring jnp.power uses integer-power dispatch and avoids the
        # exp(0*log(0))=NaN path that float 0.0 exponents can trigger).
        fracs += [0.0] * (max_k - len(fracs))
        sin_exps += [1] * (max_k - len(sin_exps))

        C_list.append(C_j)
        fracs_list.append(fracs)
        sinexp_list.append(sin_exps)
        is_odd_list.append(is_odd)

    C_arr = jnp.array(C_list)  # (d-2,)
    fracs_arr = jnp.array(fracs_list)  # (d-2, max_k)
    sin_exp_arr = jnp.array(sinexp_list)  # (d-2, max_k)
    is_odd_arr = jnp.array(is_odd_list)  # (d-2,)

    # Draw one uniform x per j via vmap over keys
    j_subkeys = all_subkeys[1:]  # (d-2, 2)
    x_vals = jax.vmap(lambda k: random.uniform(k, shape=(), minval=0, maxval=1))(
        j_subkeys
    )  # (d-2,)

    # Unified runtime function: same signature for all j, vmappable
    def compute_theta_j(x, C, fracs, sin_exps, is_odd):  # numpydoc ignore=GL08
        def F(theta):  # numpydoc ignore=GL08
            sin_sum = jnp.dot(fracs, jnp.power(jnp.sin(theta), sin_exps))
            offset = jnp.where(is_odd, 0.5, theta / jnp.pi)
            return offset - C * jnp.cos(theta) * sin_sum

        return get_theta(F, 0.0, jnp.pi, x, 1e-9)

    # vmap over j: all d-2 bisections run in parallel
    theta_vals = jax.vmap(compute_theta_j)(
        x_vals, C_arr, fracs_arr, sin_exp_arr, is_odd_arr
    )  # (d-2,)

    # Assemble full theta vector and compute e2
    theta = jnp.append(theta_vals, theta_last)  # (d-1,)

    # Spherical coordinate conversion: cumulative product of sines times cosine
    e2 = jnp.cumprod(jnp.concatenate([jnp.ones(1), jnp.sin(theta)]))
    e2 = e2 * jnp.cos(jnp.append(theta, 0.0))

    theta = jnp.append(theta, 0)  # Append 0 for cos(0) of last coordinate

    return theta, e2
