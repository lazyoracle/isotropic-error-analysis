# isotropic

`isotropic` is a python library that contains tools for analysis of isotropic errors in QC algorithms. See [Executive summary of concepts](#executive-summary-of-concepts) for a quick run down on isotropic errors and how they are generated and added to the quantum state.

## Installation

### Install from pip
```bash
pip install isotropic[all]
```

The `all` option installs the necessary libraries for using the qiskit simulator for algorithm analysis and running the example notebooks.

### Install from source

#### Clone the repository
```bash
git clone https://github.com/lazyoracle/isotropic-error-analysis
```

#### Install library and dependencies
```bash
cd isotropic-error-analysis
pip install -e .\[all\]
```

### Run examples
```bash
jupyter notebook notebooks/
```

## Documentation

Documentation is available online at https://lazyoracle.github.io/isotropic-error-analysis/

### Build local docs

If you want to build and view the documentation locally, follow the steps below:

1. Clone the repository using `git clone https://github.com/lazyoracle/isotropic-error-analysis`
2. Install dependencies for docs using `pip install -e .\[docs,all\]`
3. Build and serve a live reloading version of the docs using `mkdocs serve`

## Executive summary of concepts

In quantum error models, isotropic errors are those that affect all states of the system equally, without a preferred direction in Hilbert space. For example, depolarizing noise is isotropic. A depolarizing channel replaces a qubit state with the maximally mixed state with some probability $p$. No basis or axis is privileged. The error distribution is thus uniform over possible error directions. On the other hand, independent errors are assumed to act locally and independently across different qubits or gates. For example, an independent bit-flip error model applies an $X$ operator to each qubit with probability $p$, independently of what happens to other qubits. In this case, errors are uncorrelated across subsystems.

In order to model the effect of an isotropic error on a quantum state $\Phi$, we follow the steps outlined below:

1. Construct an orthonormal basis of $\Pi$ with center at $\Phi$.
2. Generate a vector $e_2$ in $S_{d−1}$ with uniform distribution.
3. Generate an angle $\theta_0$ in $[0,\pi]$ with density function $f(\theta_0)$.
4. Generate the final perturbed state $\Psi$ as a rotation of $e_1 = \Phi$ by angle $\theta_0$ in the subspace spanned by the orthonormal basis $[e_1,e_2]$ using the expression $\Psi = \Phi \cos(\theta_0) + e_2 \sin(\theta_0)$

Once we have a recipe for adding an isotropic error to a quantum state, we can study the effect of such an error on the complexity of algorithms such as Grover's or Shor's that involve repeated executions until success.

## FAQs

1. What system sizes can be analysed?

    Currently this is tested for systems with up to 7 qubits. The error generation requires the calculation of a double factorial ratio, which becomes intractable at larger system sizes unless some nifty tricks are applied. There is an experimental implementation of double factorial ratio for larger numbers, but this is somewhat buggy and not yet thoroughly tested

2. What algorithms can be analysed?

    As of now, only Grover's algorithm is implemented (using Qiskit) and analysed here. However, the library accepts an arbitrary quantum state and returns the perturbed quantum state, both as a standard complex array. Which means it can be integrated with any quantum programming library such as Pennylane or Qiskit or Cirq that has its own routines for statevector simulation of different quantum algorithms.

3. Why does this not work on Python 3.13?

    You likely ran into an issue with the build from source of the `qiskit-aer` dependency. Try installing the library without the optional `[all]` or `[algo]` flags, and then separately install `qiskit<2.0,>1.2` and `qiskit-aer<0.18,>0.15`.
