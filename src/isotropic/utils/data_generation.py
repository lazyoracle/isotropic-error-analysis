"""
This module generates data for Grover's algorithm with isotropic error.
"""

import os
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import typer
import xarray as xr
from jax import Array, random
from qiskit.quantum_info import Operator, Statevector

from isotropic.algos.grover import get_grover_circuit
from isotropic.e2 import get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import double_factorial_ratio, normal_integrand
from isotropic.utils.state_transforms import (
    add_isotropic_error,
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


def _run_batch_inner(
    Phi_sp_batch: Array,
    basis_batch: Array,
    sigmas: Array,
    gate_counts: Array,
    sigma_keys: Array,
    log_factorial_ratio: Array,
    marked_index: int,
    n_samples: int,
) -> Array:
    """
    Inner batched computation, stable at module level for JIT cache reuse.

    Parameters
    ----------
    Phi_sp_batch : Array
        Hypersphere representations of the statevectors, shape ``(num_iters, d_phi)``.
    basis_batch : Array
        Orthonormal bases for each iteration, shape ``(num_iters, d_basis, d_phi)``
        where ``d_basis = d_phi - 1``.
    sigmas : Array
        Base noise levels to evaluate, shape ``(num_sigmas,)``.
    gate_counts : Array
        Total gate count per iteration used to scale sigma, shape ``(num_iters,)``.
    sigma_keys : Array
        Independent PRNG subkeys, one per sigma value, shape ``(num_sigmas, 2)``.
    log_factorial_ratio : Array
        Precomputed ``log((d_phi-1)!! / (d_phi-2)!!)``, scalar.
    marked_index : int
        Index of the marked basis state; declared static so XLA compiles a
        specialised kernel per target state.
    n_samples : int
        Number of independent error samples to average per ``(sigma, iter)`` point.
        Declared static because ``random.split`` requires a compile-time count.
        Use ``n_samples=1`` for a single draw (high variance); use larger values
        (e.g. 500) for smooth, reliable estimates.

    Returns
    -------
    Array
        Success probabilities, shape ``(num_sigmas, num_iters)``.

    Notes
    -----
    ``d_phi = Phi_sp_batch.shape[1]`` is the hypersphere dimension (``2 * 2**n``
    for n-qubit statevectors). ``d_basis = d_phi - 1`` is the number of
    orthonormal basis vectors used for the isotropic error.

    Both ``marked_index`` and ``n_samples`` are declared static
    (``static_argnums=(6, 7)``). All other arguments are traced. ``d_phi`` and
    ``d_basis`` are derived from array shapes at trace time and are therefore
    always concrete integers.
    """
    d_basis = Phi_sp_batch.shape[1] - 1  # concrete: derived from static shape
    d_phi = Phi_sp_batch.shape[1]  # concrete: derived from static shape

    def get_success_for_sigma(sigma, sigma_key):  # numpydoc ignore=PR01,RT01
        """Compute success probability for one sigma across all iterations."""

        def get_success_for_iter(
            Phi_sp, basis, sigma_iter, iter_key
        ):  # numpydoc ignore=GL08
            # Average over n_samples independent error draws to reduce Monte Carlo
            # variance. Each sample gets its own (e2_key, theta_key) pair.
            sample_keys = random.split(iter_key, num=n_samples)

            def single_sample(sample_key):  # numpydoc ignore=GL08
                e2_key, theta_key = random.split(sample_key)
                _, coeffs = get_e2_coeffs(d=d_basis, key=e2_key)
                e2 = jnp.expand_dims(coeffs, axis=-1) * basis

                def g(theta):  # numpydoc ignore=GL08
                    return normal_integrand(
                        theta,
                        d=d_phi - 1,
                        sigma=sigma_iter,
                        log_factorial_ratio=log_factorial_ratio,
                    )

                x = random.uniform(theta_key, shape=(), minval=0, maxval=1)
                theta_zero = get_theta_zero(x=x, g=g)
                Psi_sp = add_isotropic_error(Phi_sp, e2=e2, theta_zero=theta_zero)
                Psi = hypersphere_to_statevector(Psi_sp)
                return jnp.abs(Psi[marked_index]) ** 2

            return jnp.mean(jax.vmap(single_sample)(sample_keys))

        sigma_iter_batch = jnp.pow(
            sigma, gate_counts
        )  # scale sigma by gate count per iteration
        iter_keys = random.split(
            sigma_key, num=Phi_sp_batch.shape[0]
        )  # one key per iteration
        return jax.vmap(get_success_for_iter)(
            Phi_sp_batch, basis_batch, sigma_iter_batch, iter_keys
        )

    return jax.vmap(get_success_for_sigma)(sigmas, sigma_keys)


# Stable module-level compiled object: JIT cache persists across all calls
# that share the same (marked_index, n_samples, array shapes/dtypes).
_run_batch_compiled = jax.jit(_run_batch_inner, static_argnums=(6, 7))


# TODO: add an algo parameter which for now only supports "grover"
def generate_data(
    min_qubits: int,
    max_qubits: int,
    min_iterations: int,
    max_iterations: int,
    min_sigma: Optional[float] = None,
    max_sigma: Optional[float] = None,
    num_sigma_points: int = 2,
    sigma_values: Optional[list[float]] = None,
    data_dir: str = "data",
    random_key: int = 42,
    n_samples: int = 1,
) -> None:
    """
    Generate data for Grover's algorithm with isotropic error and save to xarray files.

    Parameters
    ----------
    min_qubits : int
        Minimum number of qubits.
    max_qubits : int
        Maximum number of qubits.
    min_iterations : int
        Minimum number of Grover iterations to simulate.
    max_iterations : int
        Maximum number of Grover iterations to simulate.
    min_sigma : float, optional
        Minimum sigma value for isotropic error. Required if sigma_values is not provided.
    max_sigma : float, optional
        Maximum sigma value for isotropic error. Required if sigma_values is not provided.
    num_sigma_points : int, optional
        Number of sigma points to evaluate between min_sigma and max_sigma. Default is 2.
    sigma_values : list[float], optional
        Explicit list of sigma values. If provided, min_sigma/max_sigma/num_sigma_points
        are ignored.
    data_dir : str, optional
        Directory to save the generated data files. Default is "data".
    random_key : int, optional
        Integer seed for the JAX PRNG root key. Default is 42.
    n_samples : int, optional
        Number of independent error samples to average per ``(sigma, iter)`` point.
        Default is 1 (single draw, high variance). Use larger values (e.g. 50)
        for smooth, reliable estimates of the expected success probability.

    Returns
    -------
    None
        Saves the generated data to xarray files.

    Notes
    -----
    We implement ``n_samples`` averaging via vmap to eliminate single-sample Monte
    Carlo noise which introduces unexpected variance in the results. This is crucial
    for generating smooth, monotonic results based on scaling of ``sigma``.
    """
    if sigma_values is not None:
        sigmas = jnp.array(sigma_values)
    elif min_sigma is not None and max_sigma is not None:
        sigmas = jnp.linspace(min_sigma, max_sigma, num_sigma_points)
    else:
        raise ValueError("Provide either sigma_values or both min_sigma and max_sigma.")
    if jnp.any(sigmas <= 0) or jnp.any(sigmas >= 1):
        raise ValueError("Sigma values must be in the range (0, 1).")

    os.makedirs(data_dir, exist_ok=True)

    # Loop over qubit counts (cannot vmap: each num_qubits yields different
    # array shapes, e.g. statevector length 2^n).
    for num_qubits in range(min_qubits, max_qubits + 1):
        # TODO: change hardcoded grover oracle
        oracle = jnp.eye(2**num_qubits).tolist()
        oracle[3][3] = -1
        U_w = Operator(oracle)
        marked_item = "0" * (num_qubits - 2) + "11"

        # Pre-compute all statevectors via Qiskit (not JAX-traceable).
        iterations_range = list(range(min_iterations, max_iterations + 1))
        statevectors = []
        # model: per-gate error; effective sigma after k gates is sigma^k
        total_gate_counts = []  # required for scaling sigma
        error_free_probs = []
        for iterations in iterations_range:
            circuit = get_grover_circuit(num_qubits, U_w, iterations)
            total_gate_count = sum(
                v for op, v in circuit.count_ops().items() if op != "barrier"
            )
            total_gate_counts.append(total_gate_count)
            sv = Statevector(circuit)
            statevectors.append(jnp.array(sv.data))
            error_free_probs.append(sv.probabilities_dict()[marked_item])

        Phi_batch = jnp.stack(statevectors)  # (num_iters, 2^n) complex
        error_free_batch = jnp.array(error_free_probs)
        gate_counts_batch = jnp.array(total_gate_counts)

        # Batch JAX computation: vmap over iterations and sigmas
        results = run_experiment_batch(
            Phi_batch=Phi_batch,
            marked_item=marked_item,
            sigmas=sigmas,
            gate_counts=gate_counts_batch,
            random_key=random_key,
            n_samples=n_samples,
        )
        # results shape: (num_iterations, num_sigma_points)

        # Save per-iteration xarray files
        for i, iterations in enumerate(iterations_range):
            error_success = jnp.append(results[i], error_free_batch[i])
            data = xr.Dataset(
                {
                    "success_probability": (["sigma"], error_success),
                    "iterations": iterations,
                    "gate_count": total_gate_counts[i],
                },
                coords={
                    "sigma": jnp.append(sigmas, jnp.array([1.0])),
                },
                attrs={
                    "num_qubits": num_qubits,
                    "marked_item": marked_item,
                },
            )
            data.to_netcdf(
                f"{data_dir}/grover_{num_qubits}_qubits_{iterations}_iterations.nc"
            )


def run_experiment_batch(
    Phi_batch: Array,
    marked_item: str,
    sigmas: Array,
    gate_counts: Array,
    random_key: int = 42,
    n_samples: int = 1,
) -> Array:
    """
    Run batched experiment: vmap over iterations and sigmas.

    For a fixed num_qubits all statevectors share the same shape, so the
    pure-JAX computation is vmapped over the iteration dimension (Phi_batch)
    and the sigma dimension in a single JIT-compiled XLA program.

    Parameters
    ----------
    Phi_batch : Array
        Stack of complex statevectors, shape ``(num_iterations, 2**n)``.
    marked_item : str
        The marked item to search for in binary string format.
    sigmas : Array
        Sigma values to evaluate, shape ``(num_sigma_points,)``.
    gate_counts : Array
        Total gate counts excluding barriers for each iteration, shape ``(num_iterations,)``.
    random_key : int, optional
        Integer seed for the JAX PRNG root key. Default is 42.
    n_samples : int, optional
        Number of independent error samples to average per ``(sigma, iter)`` point.
        Default is 1 (single draw, high variance). Use larger values (e.g. 50)
        for smooth, reliable estimates of the expected success probability.

    Returns
    -------
    Array
        Success probabilities, shape ``(num_iterations, num_sigma_points)``.

    Notes
    -----
    We use an error model of per-gate error, implying the effective sigma after k gates is sigma^k.
    The gate_counts array is used to scale the sigma values for each iteration accordingly.

    We implement ``n_samples`` averaging via vmap to eliminate single-sample Monte
    Carlo noise which introduces unexpected variance in the results. This is crucial
    for generating smooth, monotonic results based on scaling of ``sigma``.
    """
    # Convert all statevectors to hypersphere (vmap over iterations)
    Phi_sp_batch = jax.vmap(statevector_to_hypersphere)(Phi_batch)

    # Orthonormal basis for each iteration (vmap)
    basis_batch = jax.vmap(get_orthonormal_basis)(Phi_sp_batch)

    d_phi = Phi_sp_batch.shape[1]
    log_factorial_ratio = jnp.log(double_factorial_ratio(d_phi - 2, d_phi - 3))
    marked_index = int(marked_item, 2)

    key = random.PRNGKey(random_key)  # root PRNG key; all randomness derives from this
    sigma_keys = random.split(key, num=sigmas.shape[0])  # one key per sigma

    # Delegate to the stable module-level compiled function so the JIT cache
    # is reused across all calls with the same (marked_index, n_samples, array shapes).
    results = _run_batch_compiled(
        Phi_sp_batch,
        basis_batch,
        sigmas,
        gate_counts,
        sigma_keys,
        log_factorial_ratio,
        marked_index,
        n_samples,
    )
    # Shape: (num_sigmas, num_iterations)

    return results.T  # (num_iterations, num_sigmas)


def _main(  # numpydoc ignore=PR01
    min_qubits: int = typer.Argument(..., help="Minimum number of qubits."),
    max_qubits: int = typer.Argument(..., help="Maximum number of qubits."),
    min_iterations: int = typer.Argument(
        ..., help="Minimum number of Grover iterations."
    ),
    max_iterations: int = typer.Argument(
        ..., help="Maximum number of Grover iterations."
    ),
    min_sigma: Optional[float] = typer.Argument(
        default=None,
        help="Minimum sigma value for isotropic error. Required unless --sigma-values is given.",
    ),
    max_sigma: Optional[float] = typer.Argument(
        default=None,
        help="Maximum sigma value for isotropic error. Required unless --sigma-values is given.",
    ),
    sigma_values: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of sigma values (alternative to min/max sigma).",
    ),
    num_sigma_points: int = typer.Option(2, help="Number of sigma points to evaluate."),
    data_dir: str = typer.Option(
        "data", help="Directory to save the generated data files."
    ),
    random_key: int = typer.Option(
        42,
        help="Integer seed for the JAX PRNG. Use different values for independent runs.",
    ),
    num_samples: int = typer.Option(
        1,
        help="Number of independent error samples to average per (sigma, iter) point. "
        "Use larger values (e.g. 50) for smooth plots.",
    ),
):
    """
    Generate data for Grover's algorithm with isotropic error.
    """
    parsed_sigmas = (
        [float(s.strip()) for s in sigma_values.split(",")] if sigma_values else None
    )

    print("Generating data with the following parameters:")
    for name, value in [
        ("min_qubits", min_qubits),
        ("max_qubits", max_qubits),
        ("min_iterations", min_iterations),
        ("max_iterations", max_iterations),
        ("min_sigma", min_sigma),
        ("max_sigma", max_sigma),
        ("sigma_values", parsed_sigmas),
        ("num_sigma_points", num_sigma_points),
        ("data_dir", data_dir),
        ("random_key", random_key),
        ("num_samples", num_samples),
    ]:
        print(f"{name}: {value}")
    generate_data(
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_values=parsed_sigmas,
        num_sigma_points=num_sigma_points,
        data_dir=data_dir,
        random_key=random_key,
        n_samples=num_samples,
    )


# for CLI entry point
app = typer.Typer()
app.command()(_main)


def cli():
    """
    Command-line interface for data generation.
    """
    if len(sys.argv) == 1:
        # No arguments provided, show help and exit
        sys.argv.append("--help")
    app()
