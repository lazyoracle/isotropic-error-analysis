"""This module contains functions for transforming the quantum state"""

import jax.numpy as jnp
from jax import Array


def statevector_to_hypersphere(Phi: Array) -> Array:
    """
    Generate the hypersphere Phi from statevector Psi

    Parameters
    ----------
    psi: ArrayLike
        statevector as a JAX array of dimension 2^n, for n-qubits

    Returns
    -------
    Array
        hypersphere as a JAX array of dimension 2^{n+1} - 1
    """
    S = jnp.zeros((2 ** (jnp.log2(Phi.shape[0]).astype(int) + 1),), dtype=complex)
    for x in range(S.shape[0] // 2):
        S = S.at[2 * x].set(Phi[x].real)
        S = S.at[2 * x + 1].set(Phi[x].imag)
    return S


def hypersphere_to_statevector(S: Array) -> Array:
    """
    Generate the statevector Psi from hypersphere Phi

    Parameters
    ----------
    S: ArrayLike
        hypersphere as a JAX array of dimension 2^{n+1} - 1 for n qubits

    Returns
    -------
    Array
        statevector as a JAX array of dimension 2^n
    """
    Phi = jnp.zeros((2 ** (jnp.log2(S.shape[0]).astype(int)),), dtype=complex)
    for x in range(Phi.shape[0]):
        Phi = Phi.at[x].set(S[2 * x] + 1j * S[2 * x + 1])
    return Phi


def add_isotropic_error(Phi: Array, e2: Array, theta_zero: float) -> Array:
    """
    Add isotropic error to state Phi given e2 and theta_zero

    Parameters
    ----------
    Phi : ArrayLike
        state to which isotropic error is added
    e2 : ArrayLike
        vector e2 in S_{d-1} with uniform distribution
    theta_zero : float
        angle θ_0 in [0,π] with density function f(θ_0)

    Returns
    -------
    Array
        statevector after adding isotropic error
    """
    # TODO: fix dimension mismatch between Phi and e2
    Psi = (Phi * jnp.cos(theta_zero)) + (e2 * jnp.sin(theta_zero))
    return Psi
