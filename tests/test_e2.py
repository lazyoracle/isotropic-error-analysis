import jax.numpy as jnp
from jax import Array, random

from isotropic.e2 import get_e2


def test_get_e2():
    """Test get_e2 for dim d=5"""
    d: int = 5

    def mock_F_j(theta_j: float, j: int, d: int) -> Array:
        """Dummy function for F_j"""
        return theta_j

    theta, e2 = get_e2(d=d, F_j=mock_F_j, key=random.PRNGKey(2441139))

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
