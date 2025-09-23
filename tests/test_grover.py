import pytest
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator

from isotropic.algos.grover import (
    get_grover_answer,
    get_grover_circuit,
    optimal_num_iterations,
)

U_w = Operator(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


def test_grover_integration():
    num_qubits = 3
    optimal_iterations = optimal_num_iterations(num_solutions=1, num_qubits=num_qubits)
    grover_circuit = get_grover_circuit(num_qubits, U_w, optimal_iterations)
    grover_circuit.measure_all(add_bits=False)
    statevectorsampler = StatevectorSampler()
    result = statevectorsampler.run([grover_circuit]).result()
    _, answer = get_grover_answer(result)
    assert answer == "011", f"Expected '011', got {answer}"

    with pytest.raises(ValueError):  # check wrong number of qubits error
        get_grover_circuit(num_qubits=4, U_w=U_w, iterations=optimal_iterations)

    with pytest.raises(TypeError):  # check wrong result datatype error
        get_grover_answer(result[0])
