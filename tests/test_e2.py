import jax
import jax.numpy as jnp
from jax import random
from scipy.special import factorial2

from isotropic.e2 import F_j, _compute_fraction, get_e2_coeffs


def test_get_e2():
    """Test get_e2 for dim d=5"""
    d: int = 5

    theta, e2 = get_e2_coeffs(d=d, key=random.PRNGKey(2441139))

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
    # e2 represents a point on the unit (d-1)-sphere: its norm must be exactly 1
    assert jnp.allclose(jnp.linalg.norm(e2), 1.0, atol=1e-6), (
        f"e2 is not unit-norm: ‖e2‖ = {jnp.linalg.norm(e2)}"
    )


def test_get_e2_distributional():
    """Weak distributional test: CDF inversion correctness for get_e2_coeffs.

    If the bisection inversion is correct, applying the CDF F_j back to each
    sampled theta_j must yield a Uniform[0, 1] random variable.  In particular
    the empirical mean of F_j(theta_j) over N independent samples should be
    close to 0.5.  An implementation that ignores the CDF (e.g. draws theta_j
    uniformly from [0, pi]) would produce a different mean and fail this check.

    Tests both an even j (j=0) and an odd j (j=1) to exercise both branches of
    compute_theta_j.
    """
    d = 8  # d=8 gives j=0 (even) and j=1 (odd) with non-trivial CDF shapes
    N = 500  # SE of mean ≈ sqrt(1/12)/sqrt(500) ≈ 0.013; atol=0.05 gives ~4σ

    keys = random.split(random.PRNGKey(42), N)
    # vmap over keys: all N samples are compiled in a single XLA call
    thetas = jax.vmap(lambda k: get_e2_coeffs(d=d, key=k)[0])(keys)  # (N, d)

    for j in (0, 1):  # even j=0 and odd j=1
        theta_j_samples = thetas[:, j]  # (N,)
        # F_j is Python-level branched on j (static), so vmap over theta is fine
        cdf_vals = jax.vmap(lambda t: F_j(t, j=j, d=d))(theta_j_samples)  # (N,)
        empirical_mean = jnp.mean(cdf_vals)
        assert jnp.abs(empirical_mean - 0.5) < 0.05, (
            f"j={j}: empirical mean of F_j(theta_j) = {empirical_mean:.4f}, "
            f"expected ~0.5 — CDF inversion may be incorrect"
        )


def test_get_e2_d2():
    """Test get_e2_coeffs for the d=2 early-return branch (unit circle, no bisection).

    For d=2 the only angle is the azimuthal theta_last ~ Uniform[0, 2pi], so
    e2 = [cos(theta_last), sin(theta_last)] — the standard unit-circle formula.
    """
    theta, e2 = get_e2_coeffs(d=2, key=random.PRNGKey(7))

    assert theta.shape == (2,), f"Expected theta shape (2,), got {theta.shape}"
    assert e2.shape == (2,), f"Expected e2 shape (2,), got {e2.shape}"
    # The appended sentinel must be 0
    assert jnp.isclose(theta[1], 0.0), f"Expected theta[1]=0, got {theta[1]}"
    # Formula: e2 = [cos(theta[0]), sin(theta[0])]
    assert jnp.isclose(e2[0], jnp.cos(theta[0])), (
        f"Expected e2[0]=cos(theta[0])={jnp.cos(theta[0]):.6f}, got {e2[0]:.6f}"
    )
    assert jnp.isclose(e2[1], jnp.sin(theta[0])), (
        f"Expected e2[1]=sin(theta[0])={jnp.sin(theta[0]):.6f}, got {e2[1]:.6f}"
    )
    # Unit norm
    assert jnp.allclose(jnp.linalg.norm(e2), 1.0, atol=1e-6), (
        f"e2 is not unit-norm: ‖e2‖ = {jnp.linalg.norm(e2)}"
    )


def test_compute_fraction_empty_lists():
    """When both num_list and den_list are empty, _compute_fraction returns 1.0."""
    # dj=2, k=1, odd=False → range(0,2,-2) and range(1,1,-2) are both empty
    assert _compute_fraction(dj=2, k=1, odd=False) == 1.0


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
