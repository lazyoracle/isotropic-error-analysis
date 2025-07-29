import jax.numpy as jnp

from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import normal_integrand
from isotropic.utils.simpsons import simpsons_rule


def test_get_theta_zero():
    # Test for d=3 and sigma=0.5
    def g(theta):
        return normal_integrand(theta, 3, 0.5)

    x = 0.5  # Example value for x
    theta_zero = get_theta_zero(x, g)

    # Check if the returned theta_zero is within the expected range
    assert 0 <= theta_zero <= jnp.pi, "theta_zero should be in [0, pi]"

    # Check if the integral of g from 0 to theta_zero is approximately equal to x
    integral_value = simpsons_rule(g, 0, theta_zero, 1, 1e-9)
    assert jnp.isclose(integral_value, x, atol=1e-5), "Integral value does not match x"
