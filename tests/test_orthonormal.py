import jax.numpy as jnp
import pytest
from jax.random import PRNGKey, uniform

from isotropic.orthonormal import get_orthonormal_basis

test_Phis = [uniform(PRNGKey(0), (int(10),)), jnp.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9])]


@pytest.mark.parametrize("Phi", test_Phis)
def test_get_orthonormal_basis(Phi):
    Phi = Phi / jnp.linalg.norm(Phi)

    # Get the orthonormal basis
    basis_vectors = get_orthonormal_basis(Phi)

    # Check if the basis vectors are unit vectors
    assert jnp.allclose(jnp.linalg.norm(basis_vectors, axis=1), 1.0), (
        "Basis vectors should be unit vectors"
    )

    # Check if the basis vectors are orthogonal to each other
    assert jnp.allclose(
        jnp.dot(basis_vectors, basis_vectors.T),
        jnp.eye(basis_vectors.shape[0]),
        atol=1e-6,
    ), f"<A, A.T> should be I, got {jnp.dot(basis_vectors, basis_vectors.T)}"

    # Check if bases is orthogonal to Phi
    assert jnp.allclose(jnp.dot(basis_vectors, Phi), 0.0, atol=1e-6), (
        "Basis vectors should be orthogonal to Phi"
    )

    # Check if the basis vectors are linearly independent
    assert jnp.linalg.matrix_rank(basis_vectors) == len(basis_vectors), (
        "Basis vectors should be linearly independent"
    )
