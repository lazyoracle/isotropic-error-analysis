import jax
import jax.numpy as jnp

from isotropic.e2 import F_j, get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import normal_integrand
from isotropic.utils.state_transforms import (
    add_isotropic_error,
    statevector_to_hypersphere,
)


def test_add_isotropic_error():
    Psi = jnp.ones(2) / jnp.sqrt(2)  # n = 1, d = 3
    Phi = statevector_to_hypersphere(Psi)  # d+1 = 4
    basis = get_orthonormal_basis(Phi)  # gives d vectors with d+1 elements each

    _, coeffs = get_e2_coeffs(
        d=basis.shape[0],  # gives d coefficients for the d vectors above
        F_j=F_j,
        key=jax.random.PRNGKey(0),
    )
    e2 = jnp.expand_dims(coeffs, axis=-1) * basis

    def g(theta):
        return normal_integrand(theta, d=Phi.shape[0], sigma=0.9)

    theta_zero = get_theta_zero(x=0.5, g=g)
    Phi_error = add_isotropic_error(Phi, e2=e2, theta_zero=theta_zero)  # noqa
