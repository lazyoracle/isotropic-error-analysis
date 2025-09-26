"""This module contains functions for transforming the quantum state"""

from math import log

import jax.numpy as jnp
import jax.random as random
from jax import Array
from jax.typing import ArrayLike

from isotropic.e2 import F_j, get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import normal_integrand


def statevector_to_hypersphere(Phi: Array) -> Array:
    """
    Generate the hypersphere from statevector $\\Phi$

    Parameters
    ----------
    Phi: ArrayLike
        statevector as a complex JAX array of dimension $2^n$, for n-qubits

    Returns
    -------
    Array
        hypersphere as a real JAX array of dimension $2^{n+1}$
    """
    S = jnp.zeros(int(2 ** (log(Phi.shape[0], 2) + 1)), dtype=float)
    for x in range(S.shape[0] // 2):
        S = S.at[2 * x].set(Phi[x].real)
        S = S.at[2 * x + 1].set(Phi[x].imag)
    return S


def hypersphere_to_statevector(S: Array) -> Array:
    """
    Generate the statevector $\\Phi$ from hypersphere $S$

    Parameters
    ----------
    S: ArrayLike
        hypersphere as a real JAX array of dimension $2^{n+1}$ for n qubits

    Returns
    -------
    Array
        statevector as a complex JAX array of dimension $2^n$
    """
    Phi = jnp.zeros(int(2 ** (log(S.shape[0], 2) - 1)), dtype=complex)
    for x in range(Phi.shape[0]):
        Phi = Phi.at[x].set(S[2 * x] + 1j * S[2 * x + 1])
    return Phi


def add_isotropic_error(Phi_sp: Array, e2: Array, theta_zero: float) -> Array:
    """
    Add isotropic error to state $\\Phi$ given $e_2$ and $\\theta_0$

    Parameters
    ----------
    Phi_sp : ArrayLike
        state to which isotropic error is added (in spherical form)
    e2 : ArrayLike
        vector $e_2$ in $S_{d-1}$ with uniform distribution
    theta_zero : float
        angle $\\theta_0$ in $[0,\\pi]$ with density function $f(\\theta_0)$

    Returns
    -------
    Array
        statevector in spherical form after adding isotropic error
    """
    Psi_sp = (Phi_sp * jnp.cos(theta_zero)) + (
        (jnp.sum(e2, axis=0)) * jnp.sin(theta_zero)
    )
    return Psi_sp


def generate_and_add_isotropic_error(
    Phi: ArrayLike, sigma: float = 0.9, key: ArrayLike = random.PRNGKey(0)
) -> Array:
    """
    Generate and add isotropic error to a given statevector.

    Parameters
    ----------
    Phi : ArrayLike
        The input statevector as a complex JAX array of dimension $2^n$, for n-qubits.
    sigma : float, optional
        The standard deviation for the isotropic error, by default 0.9.
    key : ArrayLike, optional
        Random key for reproducibility, by default random.PRNGKey(0).

    Returns
    -------
    Array
        The perturbed statevector after adding isotropic error.
    """
    Phi_spherical = statevector_to_hypersphere(Phi)
    basis = get_orthonormal_basis(
        Phi_spherical
    )  # gives d vectors with d+1 elements each
    _, coeffs = get_e2_coeffs(
        d=basis.shape[0],  # gives d coefficients for the d vectors above
        F_j=F_j,
        key=key,
    )
    e2 = jnp.expand_dims(coeffs, axis=-1) * basis

    def g(theta):
        return normal_integrand(theta, d=Phi_spherical.shape[0], sigma=sigma)

    x = random.uniform(key, shape=(), minval=0, maxval=1)
    theta_zero = get_theta_zero(x=x, g=g)
    Psi_spherical = add_isotropic_error(Phi_spherical, e2=e2, theta_zero=theta_zero)
    Psi = hypersphere_to_statevector(Psi_spherical)
    return Psi
