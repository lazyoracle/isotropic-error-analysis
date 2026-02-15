"""This module contains functions for constructing orthonormal basis of Pi."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@jax.jit
def get_orthonormal_basis(Phi: ArrayLike) -> Array:
    """
    Construct an orthonormal basis given a point $\\Phi$ on a unit sphere.

    The point $\\Phi$ is given by a d+1 dimensional vector and the orthonormal basis consists of d vectors
    each of dimension d+1, which are orthogonal to $\\Phi$ and to each other.

    Uses a Householder reflection to map $\\Phi$ onto $e_0$. The last d columns
    of the resulting orthogonal matrix form the desired basis.

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

    # Householder vector: v = Phi + sign(Phi[0]) * e_0
    # Using sign(Phi[0]) avoids catastrophic cancellation in the first component.
    e0 = jnp.zeros(dim).at[0].set(1.0)
    sign = jnp.where(Phi[0] >= 0, 1.0, -1.0)
    v = Phi + sign * e0
    v = v / jnp.linalg.norm(v)

    # The Householder reflector is H = I - 2*v*v^T.
    # H is orthogonal and H @ Phi = -sign * e_0, so H[:, 0] is proportional to Phi.
    # Columns 1..d of H are orthonormal and orthogonal to Phi.
    # We compute only those d columns: H[:, 1:] = I[:, 1:] - 2 * outer(v, v[1:])
    basis_vectors = (jnp.eye(dim)[:, 1:] - 2.0 * jnp.outer(v, v[1:])).T

    return basis_vectors
