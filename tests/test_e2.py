import jax.numpy as jnp
from jax import Array, random
from scipy.special import factorial2

from isotropic.e2 import F_j, get_e2_coeffs


def test_get_e2():
    """Test get_e2 for dim d=5"""
    d: int = 5

    def mock_F_j(theta_j: float, j: int, d: int) -> Array:
        """Dummy function for F_j"""
        return theta_j

    theta, e2 = get_e2_coeffs(d=d, F_j=mock_F_j, key=random.PRNGKey(2441139))

    # calculate e2 manually for theta values
    e2_expected = jnp.ones_like(theta)
    e2_expected = e2_expected.at[0].set(jnp.cos(theta[0]))
    e2_expected = e2_expected.at[1].set(jnp.sin(theta[0]) * jnp.cos(theta[1]))
    e2_expected = e2_expected.at[2].set(
        jnp.sin(theta[0]) * jnp.sin(theta[1]) * jnp.cos(theta[2])
    )
    e2_expected = e2_expected.at[3].set(
        jnp.sin(theta[0]) * jnp.sin(theta[1]) * jnp.sin(theta[2]) * jnp.cos(theta[3])
    )
    e2_expected = e2_expected.at[4].set(
        jnp.sin(theta[0]) * jnp.sin(theta[1]) * jnp.sin(theta[2]) * jnp.sin(theta[3])
    )

    assert jnp.allclose(e2, e2_expected), f"Expected {e2_expected}, got {e2}"


def test_F_j_even():
    """Test F_j for even j in dim d=7"""
    d: int = 7
    theta_j = jnp.pi / 4  # Example angle
    j = 2

    result = F_j(theta_j, j, d)

    # manually calculate expected result
    C_j = factorial2(d - j - 1) / (jnp.pi * factorial2(d - j - 2))
    num = factorial2(d - j - 2)
    den = factorial2(d - j - 1)
    prefactor = C_j * (num / den) * theta_j

    # k goes from 1 to (d - j - 1) // 2, i.e., 1 to 2
    k_val_1 = ((d - j - 2) / ((d - j - 1) * (d - j - 3))) * jnp.sin(theta_j)
    k_val_2 = (1.0 / (d - j - 1)) * jnp.sin(theta_j) ** 3

    expected_result = prefactor - (C_j * jnp.cos(theta_j) * (k_val_1 + k_val_2))
    assert jnp.isclose(result, expected_result), (
        f"Expected {expected_result}, got {result}"
    )


def test_F_j_odd():
    """Test F_j for odd j in dim d=9"""
    d: int = 9
    theta_j = jnp.pi / 4  # Example angle
    j = 3

    result = F_j(theta_j, j, d)

    # manually calculate expected result
    C_j = factorial2(d - j - 1) / (2 * factorial2(d - j - 2))
    num = factorial2(d - j - 2)
    den = factorial2(d - j - 1)
    prefactor = C_j * num / den

    # k goes from 0 to (d - j - 2) // 2, i.e., 0 to 2
    k_val_0 = (
        ((d - j - 2) * (d - j - 4)) / ((d - j - 1) * (d - j - 3) * (d - j - 5))
    ) * (jnp.sin(theta_j) ** 0)
    k_val_1 = ((d - j - 2) / ((d - j - 1) * (d - j - 3))) * (jnp.sin(theta_j) ** 2)
    k_val_2 = (1.0 / (d - j - 1)) * (jnp.sin(theta_j) ** 4)
    expected_result = prefactor - (
        C_j * jnp.cos(theta_j) * (k_val_0 + k_val_1 + k_val_2)
    )
    assert jnp.isclose(result, expected_result), (
        f"Expected {expected_result}, got {result}"
    )
