import jax.numpy as jnp
import pytest
from scipy.linalg import null_space
from scipy.special import factorial2

from isotropic.utils.bisection import get_theta
from isotropic.utils.distribution import (
    double_factorial_jax,
    double_factorial_ratio_jax,
    double_factorial_ratio_scipy,
    normal_integrand,
)
from isotropic.utils.linalg import jax_null_space
from isotropic.utils.simpsons import simpsons_rule
from isotropic.utils.state_transforms import (
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


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


def test_double_factorial_jax():
    # Test even double factorial
    n_even = 6
    result_even = double_factorial_jax(n_even)
    expected_even = factorial2(n_even)
    assert jnp.isclose(result_even, expected_even), (
        f"Expected {expected_even}, got {result_even}"
    )

    # Test odd double factorial
    n_odd = 5
    result_odd = double_factorial_jax(n_odd)
    expected_odd = factorial2(n_odd)
    assert jnp.isclose(result_odd, expected_odd), (
        f"Expected {expected_odd}, got {result_odd}"
    )

    # Test zero double factorial
    n_zero = 0
    result_zero = double_factorial_jax(n_zero)
    expected_zero = factorial2(n_zero)
    assert jnp.isclose(result_zero, expected_zero), (
        f"Expected {expected_zero}, got {result_zero}"
    )


def test_double_factorial_ratio_jax():
    num, den = (2**8) - 1, (2**8) - 2
    ratio_received = double_factorial_ratio_jax(num, den)
    ratio_expected = factorial2(num) / factorial2(den)
    assert jnp.isclose(ratio_received, ratio_expected), (
        f"Expected {ratio_expected}, got {ratio_received}"
    )

    num, den = (2**8) - 3, (2**8) - 1
    ratio_received = double_factorial_ratio_jax(num, den)
    ratio_expected = factorial2(num) / factorial2(den)
    assert jnp.isclose(ratio_received, ratio_expected), (
        f"Expected {ratio_expected}, got {ratio_received}"
    )

    with pytest.raises(ValueError):  # check for error on inputs not close enough
        _ = double_factorial_ratio_jax(300, 290)


def test_double_factorial_ratio_scipy():
    with pytest.raises(ValueError):
        _ = double_factorial_ratio_scipy(302, 301)


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


def test_state_transforms():
    Psi = jnp.asarray([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
    Psi_result = hypersphere_to_statevector(statevector_to_hypersphere(Psi))
    assert jnp.allclose(Psi, Psi_result), f"Expected {Psi}, got {Psi_result}"

    S = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    S_result = statevector_to_hypersphere(hypersphere_to_statevector(S))
    assert jnp.allclose(S, S_result), f"Expected {S}, got {S_result}"


def test_jax_null_space():
    A = jnp.array([[1, 2, 3], [4, 5, 6]])
    null_space_result = jax_null_space(A)
    null_space_expected = null_space(A)
    assert jnp.allclose(null_space_result, null_space_expected), (
        f"Expected {null_space_expected}, got {null_space_result}"
    )
