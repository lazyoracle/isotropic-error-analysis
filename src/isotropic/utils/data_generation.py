"""
This module generates data for Grover's algorithm with isotropic error.
"""

import sys

import jax.numpy as jnp
import typer
import xarray as xr
from jax import random
from joblib import Parallel, delayed
from qiskit.quantum_info import Operator, Statevector

from isotropic.algos.grover import get_grover_circuit
from isotropic.e2 import F_j, get_e2_coeffs
from isotropic.orthonormal import get_orthonormal_basis
from isotropic.thetazero import get_theta_zero
from isotropic.utils.distribution import normal_integrand
from isotropic.utils.state_transforms import (
    add_isotropic_error,
    hypersphere_to_statevector,
    statevector_to_hypersphere,
)


# TODO: add an algo parameter which for now only supports "grover"
def generate_data(
    num_qubits: int,
    min_iterations: int,
    max_iterations: int,
    min_sigma: float,
    max_sigma: float,
    num_sigma_points: int = 2,
    num_jobs: int = 2,
    data_dir: str = "data",
) -> None:
    """
    Generate data for Grover's algorithm with isotropic error and save to xarray files.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the Grover's algorithm.
    min_iterations : int
        Minimum number of Grover iterations to simulate.
    max_iterations : int
        Maximum number of Grover iterations to simulate.
    min_sigma : float
        Minimum sigma value for isotropic error.
    max_sigma : float
        Maximum sigma value for isotropic error.
    num_sigma_points : int, optional
        Number of sigma points to evaluate between min_sigma and max_sigma. Default is 2.
    num_jobs : int, optional
        Number of parallel jobs to use for computation. Default is 2.
    data_dir : str, optional
        Directory to save the generated data files. Default is "data".

    Returns
    -------
    None
        Saves the generated data to xarray files.
    """
    # We first implement the oracle that will add a phase to our desired search item.
    # Note the negative sign on one of the diagonal entries.
    # TODO: change hardcoded grover oracle
    oracle = jnp.eye(2**num_qubits).tolist()
    oracle[3][3] = -1
    U_w = Operator(oracle)
    marked_item = "0" * (num_qubits - 2) + "11"

    import os

    os.makedirs(data_dir, exist_ok=True)
    for iterations in range(min_iterations, max_iterations + 1):
        data = run_experiment(
            num_qubits=num_qubits,
            U_w=U_w,
            iterations=iterations,
            marked_item=marked_item,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma_points=num_sigma_points,
            num_jobs=num_jobs,
        )
        # Save xarray data to file
        data.to_netcdf(
            f"{data_dir}/grover_{num_qubits}_qubits_{iterations}_iterations.nc"
        )


def run_experiment(
    num_qubits: int,
    U_w: Operator,
    iterations: int,
    marked_item: str,
    min_sigma: float,
    max_sigma: float,
    num_sigma_points: int,
    num_jobs: int,
) -> xr.Dataset:
    """
    Run Grover's algorithm experiment with isotropic error and return results as xarray Dataset.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the Grover's algorithm.
    U_w : Operator
        Oracle operator for Grover's algorithm.
    iterations : int
        Number of Grover iterations to perform.
    marked_item : str
        The marked item to search for in binary string format.
    min_sigma : float
        Minimum sigma value for isotropic error.
    max_sigma : float
        Maximum sigma value for isotropic error.
    num_sigma_points : int
        Number of sigma points to evaluate between min_sigma and max_sigma.
    num_jobs : int
        Number of parallel jobs to use for computation.

    Returns
    -------
    xr.Dataset
        Xarray Dataset containing success probabilities for different sigma values.
    """
    # Grover's Circuit
    grover_circuit = get_grover_circuit(
        num_qubits=num_qubits, U_w=U_w, iterations=iterations
    )

    # error free final statevector before measurements
    statevector = Statevector(grover_circuit)

    # The probability of measuring the $0011$ state gives us a likelihood of success for our search exercise.
    error_free_success = statevector.probabilities_dict()[marked_item]

    # Effect of error levels on success probability
    ## Pre-compute error parameters that are independent of sigma
    Phi = statevector.data
    Phi_spherical = statevector_to_hypersphere(Phi)
    basis = get_orthonormal_basis(
        Phi_spherical
    )  # gives d vectors with d+1 elements each
    key = random.PRNGKey(0)
    theta, coeffs = get_e2_coeffs(
        d=basis.shape[0],  # gives d coefficients for the d vectors above
        F_j=F_j,
        key=key,
    )
    e2 = jnp.expand_dims(coeffs, axis=-1) * basis

    # sigma specific calculations
    def get_success_after_error(sigma):
        def g(theta):
            return normal_integrand(theta, d=Phi_spherical.shape[0], sigma=sigma)

        x = random.uniform(key, shape=(), minval=0, maxval=1)
        theta_zero = get_theta_zero(x=x, g=g)
        Psi_spherical = add_isotropic_error(Phi_spherical, e2=e2, theta_zero=theta_zero)
        Psi = hypersphere_to_statevector(Psi_spherical)
        statevector_error = Statevector(Psi.tolist())
        return statevector_error.probabilities_dict()[marked_item]

    sigmas = jnp.linspace(min_sigma, max_sigma, num_sigma_points)

    TIMEOUT = 99999  # see https://stackoverflow.com/a/71977764
    error_success = Parallel(n_jobs=num_jobs, timeout=TIMEOUT)(
        delayed(get_success_after_error)(sigma) for sigma in sigmas
    )

    error_success.append(error_free_success)

    # Create xarray Dataset
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

    return data


def main(  # numpydoc ignore=PR01
    num_qubits: int = typer.Argument(..., help="Number of qubits."),
    min_iterations: int = typer.Argument(
        ..., help="Minimum number of Grover iterations."
    ),
    max_iterations: int = typer.Argument(
        ..., help="Maximum number of Grover iterations."
    ),
    min_sigma: float = typer.Argument(
        ..., help="Minimum sigma value for isotropic error."
    ),
    max_sigma: float = typer.Argument(
        ..., help="Maximum sigma value for isotropic error."
    ),
    num_sigma_points: int = typer.Option(2, help="Number of sigma points to evaluate."),
    num_jobs: int = typer.Option(2, help="Number of parallel jobs."),
    data_dir: str = typer.Option(
        "data", help="Directory to save the generated data files."
    ),
):
    """
    Generate data for Grover's algorithm with isotropic error.
    """
    print("Generating data with the following parameters:")
    for name, value in [
        ("num_qubits", num_qubits),
        ("min_iterations", min_iterations),
        ("max_iterations", max_iterations),
        ("min_sigma", min_sigma),
        ("max_sigma", max_sigma),
        ("num_sigma_points", num_sigma_points),
        ("num_jobs", num_jobs),
        ("data_dir", data_dir),
    ]:
        print(f"{name}: {value}")
    generate_data(
        num_qubits=num_qubits,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma_points=num_sigma_points,
        num_jobs=num_jobs,
        data_dir=data_dir,
    )


# for CLI entry point
app = typer.Typer()
app.command()(main)


def cli():
    if len(sys.argv) == 1:
        # No arguments provided, show help and exit
        sys.argv.append("--help")
    app()
