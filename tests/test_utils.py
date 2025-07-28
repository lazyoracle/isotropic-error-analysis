import jax.numpy as jnp

from isotropic.utils.bisection import get_theta
from isotropic.utils.distribution import double_factorial, normal_integrand
from isotropic.utils.simpsons import simpsons_rule


def test_simpsons_rule():
    # Define a simple function to integrate
    def f(x):
        return jnp.sin(x)

    # Set integration limits and parameters
    a = 0.0
    b = jnp.pi
    C = 1.0  # Bound on the 4th derivative of sin(x)
    tol = 1e-5

    # Call the Simpson's rule function
    integral_estimate = simpsons_rule(f, a, b, C, tol)

    # Check if the estimate is close to the expected value
    expected_value = 2.0  # Integral of sin(x) from 0 to pi is 2
    assert jnp.isclose(integral_estimate, expected_value, atol=tol), (
        f"Expected {expected_value}, got {integral_estimate}"
    )


def test_get_theta():
    # Define a simple increasing function
    def F(theta):
        return theta**2

    # Set parameters for the bisection method
    a = 0.0
    b = 10.0
    x = 25.0  # We want to find theta such that F(theta) = 25, which is theta = 5
    eps = 1e-5

    # Call the bisection method
    theta_estimate = get_theta(F, a, b, x, eps)

    # Check if the estimate is close to the expected value
    expected_value = 5.0
    assert jnp.isclose(theta_estimate, expected_value, atol=eps), (
        f"Expected {expected_value}, got {theta_estimate}"
    )


def test_double_factorial():
    # Test even double factorial
    n_even = 6
    result_even = double_factorial(n_even)
    expected_even = 48.0  # 6!! = 6 * 4 * 2 = 48
    assert jnp.isclose(result_even, expected_even), (
        f"Expected {expected_even}, got {result_even}"
    )

    # Test odd double factorial
    n_odd = 5
    result_odd = double_factorial(n_odd)
    expected_odd = 15.0  # 5!! = 5 * 3 * 1 = 15
    assert jnp.isclose(result_odd, expected_odd), (
        f"Expected {expected_odd}, got {result_odd}"
    )

    # Test zero double factorial
    n_zero = 0
    result_zero = double_factorial(n_zero)
    expected_zero = 1.0
    assert jnp.isclose(result_zero, expected_zero), (
        f"Expected {expected_zero}, got {result_zero}"
    )


def test_normal_integrand():
    theta = jnp.pi / 4  # 45 degrees
    d = 5  # Dimension
    sigma = 0.5  # Sigma value
    result_g = normal_integrand(theta, d, sigma)

    # Calculate expected output manually
    expected_num = (4 * 2) * (1 - (sigma**2)) * (jnp.sin(theta) ** (d - 1))
    expected_den = (
        jnp.pi
        * (3 * 1)
        * ((1 + (sigma**2) - (2 * sigma * jnp.cos(theta))) ** ((d + 1) / 2.0))
    )
    expected_g = expected_num / expected_den
    assert jnp.isclose(result_g, expected_g), f"Expected {expected_g}, got {result_g}"
