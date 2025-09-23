import jax
import jax.numpy as jnp

from isotropic.utils.state_transforms import generate_and_add_isotropic_error


def test_add_isotropic_error():
    Phi = jnp.ones(4, dtype=complex)
    Phi = Phi / jnp.linalg.norm(Phi)

    Psi = generate_and_add_isotropic_error(
        Phi=Phi,
        sigma=0.9,
        key=jax.random.PRNGKey(0),
    )

    # normalization check
    assert jnp.isclose(jnp.linalg.norm(Psi), 1.0), (
        f"Expected 1.0, got {jnp.linalg.norm(Psi)}"
    )
