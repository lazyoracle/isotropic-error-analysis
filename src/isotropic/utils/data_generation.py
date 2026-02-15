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
from isotropic.e2 import F_j, get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import double_factorial_ratio, normal_integrand
from isotropic.utils.state_transforms import (
    add_isotropic_error,
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


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

    Returns
    -------
    None
        Saves the generated data to xarray files.
    """
    if sigma_values is not None:
        sigmas = jnp.array(sigma_values)
    elif min_sigma is not None and max_sigma is not None:
        sigmas = jnp.linspace(min_sigma, max_sigma, num_sigma_points)
    else:
        raise ValueError("Provide either sigma_values or both min_sigma and max_sigma.")

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
        error_free_probs = []
        for iterations in iterations_range:
            circuit = get_grover_circuit(num_qubits, U_w, iterations)
            sv = Statevector(circuit)
            statevectors.append(jnp.array(sv.data))
            error_free_probs.append(sv.probabilities_dict()[marked_item])

        Phi_batch = jnp.stack(statevectors)  # (num_iters, 2^n) complex
        error_free_batch = jnp.array(error_free_probs)

        # Batch JAX computation: vmap over iterations and sigmas
        results = run_experiment_batch(
            Phi_batch=Phi_batch,
            marked_item=marked_item,
            sigmas=sigmas,
        )
        # results shape: (num_iterations, num_sigma_points)

        # Save per-iteration xarray files
        for i, iterations in enumerate(iterations_range):
            error_success = jnp.append(results[i], error_free_batch[i])
            data = xr.Dataset(
                {
                    "success_probability": (["sigma"], error_success),
                    "iterations": iterations,
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

    Returns
    -------
    Array
        Success probabilities, shape ``(num_iterations, num_sigma_points)``.
    """
    # Convert all statevectors to hypersphere (vmap over iterations)
    Phi_sp_batch = jax.vmap(statevector_to_hypersphere)(Phi_batch)

    # Orthonormal basis for each iteration (vmap)
    basis_batch = jax.vmap(get_orthonormal_basis)(Phi_sp_batch)

    # e2 coefficients: depend only on d (same for all iterations), compute once.
    d_basis = Phi_sp_batch.shape[1] - 1
    key = random.PRNGKey(0)
    _, coeffs = get_e2_coeffs(d=d_basis, F_j=F_j, key=key)

    # e2 per iteration: broadcast coefficients across each iteration's basis
    e2_batch = jnp.expand_dims(coeffs, axis=-1) * basis_batch

    d_phi = Phi_sp_batch.shape[1]
    log_factorial_ratio = jnp.log(double_factorial_ratio(d_phi - 1, d_phi - 2))
    marked_index = int(marked_item, 2)

    def get_success_for_sigma(sigma):  # numpydoc ignore=PR01,RT01
        """Compute success probability for one sigma across all iterations."""

        def g(theta):  # numpydoc ignore=GL08
            return normal_integrand(
                theta, d=d_phi, sigma=sigma, log_factorial_ratio=log_factorial_ratio
            )

        x = random.uniform(key, shape=(), minval=0, maxval=1)
        theta_zero = get_theta_zero(x=x, g=g)

        def get_success_for_iter(Phi_sp, e2):  # numpydoc ignore=GL08
            Psi_sp = add_isotropic_error(Phi_sp, e2=e2, theta_zero=theta_zero)
            Psi = hypersphere_to_statevector(Psi_sp)
            return jnp.abs(Psi[marked_index]) ** 2

        # vmap over iterations
        return jax.vmap(get_success_for_iter)(Phi_sp_batch, e2_batch)

    # Outer vmap over sigmas, JIT the whole computation
    results = jax.jit(jax.vmap(get_success_for_sigma))(sigmas)
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
