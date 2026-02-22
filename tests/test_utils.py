import os
import shutil
from glob import glob

import jax.numpy as jnp
import pytest
import xarray as xr
from scipy.linalg import null_space
from scipy.special import factorial2
from typer.testing import CliRunner

from isotropic.utils.bisection import get_theta
from isotropic.utils.data_generation import (
    app,
    cli,
    generate_data,
    run_experiment_batch,
)
from isotropic.utils.distribution import (
    double_factorial_jax,
    double_factorial_ratio,
    normal_integrand,
)
from isotropic.utils.linalg import jax_null_space
from isotropic.utils.simpsons import simpsons_rule
from isotropic.utils.state_transforms import (
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


def test_cli():
    # add a test for the CLI using typer's testing utilities
    test_dir = "test_data"

    # Clean up test directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "3",  # min_qubits
                "3",  # max_qubits
                "2",  # min_iterations
                "2",  # max_iterations
                "0.9",  # min_sigma
                "0.95",  # max_sigma
                "--num-sigma-points",
                "2",
                "--data-dir",
                test_dir,
            ],
        )
        assert result.exit_code == 0, f"CLI failed with error: {result.output}"
        # Check if files are created
        files = glob(os.path.join(test_dir, "*.nc"))
        assert len(files) > 0, "No data files created by CLI"
        # Check if the file has the expected structure
        ds = xr.open_dataset(files[0])
        assert "num_qubits" in ds.attrs, "num_qubits attribute missing"
        assert "iterations" in ds.data_vars, "iterations dimension missing"
        assert "sigma" in ds.coords, "sigma dimension missing"
    finally:
        # Clean up test directory after test
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_cli_with_sigma_values():
    test_dir = "test_data_sigma_values"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "3",  # min_qubits
                "3",  # max_qubits
                "2",  # min_iterations
                "2",  # max_iterations
                "--sigma-values",
                "0.9,0.95",
                "--data-dir",
                test_dir,
            ],
        )
        assert result.exit_code == 0, f"CLI failed with error: {result.output}"
        files = glob(os.path.join(test_dir, "*.nc"))
        assert len(files) > 0, "No data files created by CLI"
        ds = xr.open_dataset(files[0])
        # sigma coord should contain [0.9, 0.95, 1.0] (explicit values + error-free)
        assert len(ds.coords["sigma"]) == 3
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_generate_data_raises_without_sigmas():
    with pytest.raises(
        ValueError, match="sigma_values or both min_sigma and max_sigma"
    ):
        generate_data(
            min_qubits=3,
            max_qubits=3,
            min_iterations=1,
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_values": [0.0]},  # lower boundary (invalid)
        {"sigma_values": [-0.5]},  # negative (invalid)
        {"sigma_values": [1.0]},  # upper boundary (invalid)
        {"sigma_values": [1.5]},  # above 1 (invalid)
        {"sigma_values": [0.5, 1.0]},  # mixed: one value at upper boundary
        {
            "min_sigma": 0.0,
            "max_sigma": 0.5,
            "num_sigma_points": 2,
        },  # lower boundary via linspace
        {
            "min_sigma": 0.5,
            "max_sigma": 1.0,
            "num_sigma_points": 2,
        },  # upper boundary via linspace
    ],
)
def test_generate_data_raises_on_invalid_sigma(kwargs):
    with pytest.raises(ValueError, match="Sigma values must be in the range"):
        generate_data(
            min_qubits=3,
            max_qubits=3,
            min_iterations=1,
            max_iterations=1,
            **kwargs,
        )


def test_cli_entry_point_no_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate-data"])
    with pytest.raises(SystemExit) as exc_info:
        cli()
    assert exc_info.value.code == 0


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


def test_double_factorial_ratio():
    num, den = (2**8) - 1, (2**8) - 2
    ratio_received = double_factorial_ratio(num, den)
    ratio_expected = factorial2(num) / factorial2(den)
    assert jnp.isclose(ratio_received, ratio_expected), (
        f"Expected {ratio_expected}, got {ratio_received}"
    )

    num, den = (2**8) - 3, (2**8) - 1
    ratio_received = double_factorial_ratio(num, den)
    ratio_expected = factorial2(num) / factorial2(den)
    assert jnp.isclose(ratio_received, ratio_expected), (
        f"Expected {ratio_expected}, got {ratio_received}"
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


def _make_run_experiment_batch_inputs():
    """Minimal valid inputs for run_experiment_batch (3 qubits, 2 iterations)."""
    Phi = jnp.ones(8, dtype=jnp.complex64) / jnp.sqrt(8.0)
    Phi_batch = jnp.stack([Phi, Phi])
    marked_item = "011"
    sigmas = jnp.array([0.9, 0.95])
    gate_counts = jnp.array([10.0, 20.0])
    return Phi_batch, marked_item, sigmas, gate_counts


def test_run_experiment_batch_reproducible():
    """Same random_key produces identical results."""
    Phi_batch, marked_item, sigmas, gate_counts = _make_run_experiment_batch_inputs()
    r1 = run_experiment_batch(Phi_batch, marked_item, sigmas, gate_counts, random_key=7)
    r2 = run_experiment_batch(Phi_batch, marked_item, sigmas, gate_counts, random_key=7)
    assert jnp.allclose(r1, r2), "Same random_key should give identical results"


def test_run_experiment_batch_different_keys():
    """Different random_key values produce different results."""
    Phi_batch, marked_item, sigmas, gate_counts = _make_run_experiment_batch_inputs()
    r1 = run_experiment_batch(Phi_batch, marked_item, sigmas, gate_counts, random_key=0)
    r2 = run_experiment_batch(Phi_batch, marked_item, sigmas, gate_counts, random_key=1)
    assert not jnp.allclose(r1, r2), (
        "Different random_key values should give different results"
    )


@pytest.mark.slow
def test_n_samples_reduces_variance():
    """Higher n_samples should produce more consistent results across different seeds."""
    Phi_batch, marked_item, sigmas, gate_counts = _make_run_experiment_batch_inputs()

    results_single = jnp.stack(
        [
            run_experiment_batch(
                Phi_batch, marked_item, sigmas, gate_counts, random_key=k, n_samples=1
            )
            for k in range(5)
        ]
    )
    results_avg = jnp.stack(
        [
            run_experiment_batch(
                Phi_batch, marked_item, sigmas, gate_counts, random_key=k, n_samples=20
            )
            for k in range(5)
        ]
    )

    var_single = float(jnp.var(results_single, axis=0).mean())
    var_avg = float(jnp.var(results_avg, axis=0).mean())
    assert var_avg < var_single, (
        f"n_samples=20 should give lower variance than n_samples=1: "
        f"{var_avg:.4f} vs {var_single:.4f}"
    )


@pytest.mark.slow
def test_single_sample_is_seed_dependent():
    """n_samples=1 results vary across seeds; large n_samples converges across seeds."""
    Phi_batch, marked_item, sigmas, gate_counts = _make_run_experiment_batch_inputs()

    r1 = run_experiment_batch(
        Phi_batch, marked_item, sigmas, gate_counts, random_key=1, n_samples=1
    )
    r2 = run_experiment_batch(
        Phi_batch, marked_item, sigmas, gate_counts, random_key=99, n_samples=1
    )
    assert not jnp.allclose(r1, r2), (
        "Single-sample results with different seeds should differ"
    )

    r3 = run_experiment_batch(
        Phi_batch, marked_item, sigmas, gate_counts, random_key=1, n_samples=50
    )
    r4 = run_experiment_batch(
        Phi_batch, marked_item, sigmas, gate_counts, random_key=99, n_samples=50
    )
    assert jnp.allclose(r3, r4, atol=0.05), (
        f"n_samples=50 should give consistent results across seeds; "
        f"max diff = {float(jnp.abs(r3 - r4).max()):.4f}"
    )


def test_normal_integrand_normalizes():
    """normal_integrand with d=d_phi-1 integrates to 1 over [0, π].

    d_phi is always even (2*2^n), so d_phi-1 is always odd. The formula
    only normalizes correctly for odd d; passing d=d_phi (even) gives 2/π.
    """
    for d_phi in [4, 8, 16]:  # 1-, 2-, 3-qubit state space dimensions
        d = d_phi - 1  # correct dimension: always odd
        log_fr = jnp.log(double_factorial_ratio(d - 1, d - 2))
        for sigma in [0.0, 0.5, 0.9]:
            if sigma == 0.0:
                # log(1-sigma^2) = log(1) = 0, no issue
                sigma_val = 1e-9  # use near-zero instead of exact 0
            else:
                sigma_val = sigma

            def g(theta, _d=d, _sigma=sigma_val, _lfr=log_fr):  # numpydoc ignore=GL08
                return normal_integrand(
                    theta, d=_d, sigma=_sigma, log_factorial_ratio=_lfr
                )

            integral = float(simpsons_rule(g, 0.0, jnp.pi, 1.0, 1e-10))
            assert abs(integral - 1.0) < 1e-4, (
                f"normal_integrand(d={d}, sigma={sigma_val:.1g}) integrates to "
                f"{integral:.6f}, expected 1.0"
            )


@pytest.mark.slow
def test_success_prob_monotone_in_sigma():
    """E[P_success] is non-decreasing in sigma for a fixed iteration count.

    Uses small gate_counts so sigma_iter = sigma^gate_count stays in a
    numerically well-behaved range, and n_samples large enough to suppress
    sampling noise.
    """
    Phi_batch, marked_item, _, _ = _make_run_experiment_batch_inputs()
    # Small gate counts so sigma_iter is not near 0 for any sigma
    gate_counts_small = jnp.array([5.0, 5.0])
    sigmas_ordered = jnp.array([0.5, 0.8, 0.95])

    results = run_experiment_batch(
        Phi_batch,
        marked_item,
        sigmas_ordered,
        gate_counts_small,
        random_key=42,
        n_samples=50,
    )
    # For each iteration, success prob should be non-decreasing across sigmas
    for iter_idx in range(results.shape[0]):
        p = results[iter_idx]  # shape (3,) — one per sigma
        assert float(p[1]) >= float(p[0]) - 0.05, (
            f"iter={iter_idx}: P(sigma=0.8)={float(p[1]):.3f} < P(sigma=0.5)={float(p[0]):.3f}"
        )
        assert float(p[2]) >= float(p[1]) - 0.05, (
            f"iter={iter_idx}: P(sigma=0.95)={float(p[2]):.3f} < P(sigma=0.8)={float(p[1]):.3f}"
        )


def test_cli_random_key_option():
    """--random-key CLI option is accepted and propagated without error."""
    test_dir = "test_data_random_key"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "3",  # min_qubits
                "3",  # max_qubits
                "2",  # min_iterations
                "2",  # max_iterations
                "0.9",  # min_sigma
                "0.95",  # max_sigma
                "--num-sigma-points",
                "2",
                "--data-dir",
                test_dir,
                "--random-key",
                "99",
            ],
        )
        assert result.exit_code == 0, f"CLI failed with error: {result.output}"
        files = glob(os.path.join(test_dir, "*.nc"))
        assert len(files) > 0, "No data files created by CLI"
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
