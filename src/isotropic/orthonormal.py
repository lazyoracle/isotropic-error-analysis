"""This module contains functions for constructing orthonormal basis of Pi."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from isotropic.utils.linalg import jax_null_space


def get_orthonormal_basis(Phi: ArrayLike) -> Array:
    """
    Construct an orthonormal basis given a point $\\Phi$ on a unit sphere.

    The point $\\Phi$ is given by a d+1 dimensional vector and the orthonormal basis consists of d vectors
    each of dimension d+1, which are orthogonal to $\\Phi$ and to each other.

    Parameters
    ----------
    Phi : ArrayLike
        A point on the unit sphere, should be a normalized vector.

    Returns
    -------
    Array
        An orthonormal basis of dimension (d, d+1).
    """
    Phi = jnp.array(Phi)
    dim = len(Phi)  # d+1

    # Verify Phi is normalized (within numerical precision)
    norm_phi = jnp.linalg.norm(Phi)
    Phi = jnp.where(jnp.abs(norm_phi - 1.0) > 1e-10, Phi / norm_phi, Phi)

    if Phi[0] != 0 or Phi[1] != 0:
        v1 = jnp.array([Phi[1], -Phi[0]])
    else:
        v1 = jnp.array([1.0, 0.0])

    v1 = v1 / jnp.linalg.norm(v1)
    v1 = jnp.pad(v1, (0, dim - 2), mode="constant", constant_values=0)

    basis_vectors = v1.reshape(1, -1)

    for i in range(2, dim):
        A = jnp.vstack([basis_vectors[:, : i + 1], Phi[: i + 1]])
        x = jax_null_space(A).squeeze()
        x = x / jnp.linalg.norm(x)
        x = jnp.pad(x, (0, dim - len(x)), mode="constant", constant_values=0)
        basis_vectors = jnp.vstack([basis_vectors, x])

    return basis_vectors
