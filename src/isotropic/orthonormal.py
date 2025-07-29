"""This module contains functions for constructing orthonormal basis of Pi."""

import jax.numpy as jnp
from jax import Array


def get_orthonormal_basis(Phi: Array) -> Array:
    """
    Construct an orthonormal basis for the hyperplane Π tangent to the sphere at point Φ.

    The hyperplane tangent to the unit sphere at point Φ consists of all vectors
    orthogonal to Φ. We construct an orthonormal basis for this space using QR decomposition.

    Parameters
    ----------
    Phi : Array
        A point on the unit sphere, should be a normalized vector.

    Returns
    -------
    Array
        An orthonormal basis for the hyperplane Π, excluding the original vector Φ.
        The shape of the output is (d, d) where d = len(Phi) - 1 is the dimension of the hyperplane.
    """
    Phi = jnp.array(Phi)
    dim = len(Phi)

    # Verify Phi is normalized (within numerical precision)
    norm_phi = jnp.linalg.norm(Phi)
    Phi = jnp.where(jnp.abs(norm_phi - 1.0) > 1e-10, Phi / norm_phi, Phi)

    # Create a matrix where Phi is the first column
    A = jnp.zeros((dim, dim))
    A = A.at[:, 0].set(Phi)
    # Fill remaining columns with standard basis vectors
    eye_matrix = jnp.eye(dim)
    A = A.at[:, 1:].set(eye_matrix[:, 1:])

    # Perform QR decomposition
    Q, _ = jnp.linalg.qr(A)

    # Ensure the first column of Q is aligned with Phi
    # (QR might return -Phi in the first column)
    sign_correction = jnp.sign(jnp.dot(Q[:, 0], Phi))
    Q = Q.at[:, 0].set(sign_correction * Q[:, 0])

    # The remaining columns form an orthonormal basis for the hyperplane
    basis_vectors = Q[:, 1:].T  # Take columns 1 to d, transpose to get rows as vectors

    return basis_vectors
