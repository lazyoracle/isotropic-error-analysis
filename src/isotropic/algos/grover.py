import jax.numpy as jnp
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import PrimitiveResult
from qiskit.quantum_info import Operator


def get_grover_circuit(
    num_qubits: int, U_w: Operator, iterations: int
) -> QuantumCircuit:
    """
    Create a Grover circuit with the given number of qubits, oracle, and iterations.

    Parameters
    ----------
    num_qubits : int
        The number of qubits used to encode the states.
    U_w : Operator
        The oracle that marks the solution states.
    iterations : int
        The number of Grover iterations to perform.

    Returns
    -------
    QuantumCircuit
        The constructed Grover quantum circuit.
    """

    if U_w.num_qubits != num_qubits:
        raise ValueError(
            "Oracle U_w must have the same number of qubits as num_qubits."
        )

    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qcirc = QuantumCircuit(qr, cr)

    # Initializing the circuit
    for i in range(num_qubits):
        qcirc.h(i)

    for i in range(iterations):
        qcirc.barrier()

        # Oracle to introduce negative phase (circuit of above matrix)
        qcirc.append(U_w, qr)

        qcirc.barrier()

        # Diffusion operator
        for qubit in range(num_qubits):
            qcirc.h(qubit)
        for qubit in range(num_qubits):
            qcirc.x(qubit)
        qcirc.h(num_qubits - 1)
        qcirc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qcirc.h(num_qubits - 1)
        for qubit in range(num_qubits):
            qcirc.x(qubit)
        for qubit in range(num_qubits):
            qcirc.h(qubit)

    return qcirc


def get_grover_answer(result: PrimitiveResult) -> tuple[dict, str]:
    """
    Extract the measurement counts from the result of a Grover circuit execution.

    Parameters
    ----------
    result : PrimitiveResult
        The result object returned by executing the Grover circuit.

    Returns
    -------
    tuple[dict, str]
        A tuple containing the counts dictionary and the most probable search item.
    """
    if not isinstance(result, PrimitiveResult):
        raise TypeError("The result must be an instance of PrimitiveResult.")

    counts = result[0].data.c0.get_counts()
    grover_answer = max(counts, key=counts.get)

    return counts, grover_answer


def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
    """
    Return the optimal number of iterations, if the number of solutions is known.

    Parameters
    ----------
    num_solutions : int
        The number of solutions.
    num_qubits : int
        The number of qubits used to encode the states.

    Returns
    -------
    int
        The optimal number of iterations for Grover's algorithm to succeed.
    """
    amplitude = jnp.sqrt(num_solutions / 2**num_qubits)
    return round(jnp.arccos(amplitude) / (2 * jnp.arcsin(amplitude)))
