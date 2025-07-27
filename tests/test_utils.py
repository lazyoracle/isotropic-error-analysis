from isotropic.utils.simpsons import simpsons_rule
from isotropic.utils.bisection import get_theta

import jax.numpy as jnp

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
    assert jnp.isclose(integral_estimate, expected_value, atol=tol), f"Expected {expected_value}, got {integral_estimate}"

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
    assert jnp.isclose(theta_estimate, expected_value, atol=eps), f"Expected {expected_value}, got {theta_estimate}"