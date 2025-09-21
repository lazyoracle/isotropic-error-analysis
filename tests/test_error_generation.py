import jax
import jax.numpy as jnp

from isotropic.e2 import F_j, get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import normal_integrand
from isotropic.utils.state_transforms import (
    add_isotropic_error,
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


def test_add_isotropic_error():
    Phi = jnp.ones(4, dtype=complex)
    Phi = Phi / jnp.linalg.norm(Phi)
    Phi_spherical = statevector_to_hypersphere(Phi)
    basis = get_orthonormal_basis(
        Phi_spherical
    )  # gives d vectors with d+1 elements each
    _, coeffs = get_e2_coeffs(
        d=basis.shape[0],  # gives d coefficients for the d vectors above
        F_j=F_j,
        key=jax.random.PRNGKey(0),
    )
    e2 = jnp.expand_dims(coeffs, axis=-1) * basis

    # orthogonality check
    assert jnp.allclose(jnp.abs(jnp.dot(e2, Phi_spherical)), 0.0), (
        f"Expected 0.0, got {jnp.abs(jnp.dot(basis, Phi_spherical))}"
    )

    def g(theta):
        return normal_integrand(theta, d=Phi_spherical.shape[0], sigma=0.96)

    theta_zero = get_theta_zero(x=0.5, g=g)
    Psi_spherical = add_isotropic_error(Phi_spherical, e2=e2, theta_zero=theta_zero)
    Psi = hypersphere_to_statevector(Psi_spherical)

    # normalization check
    assert jnp.isclose(jnp.linalg.norm(Psi), 1.0), (
        f"Expected 1.0, got {jnp.linalg.norm(Psi)}"
    )
