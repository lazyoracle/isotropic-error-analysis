# Welcome to `isotropic`

`isotropic` is a python package for isotropic error analysis in quantum computing. The library provides a comprehensive set of tools to model and generate isotropic errors to test their effect on various quantum algorithms.

Broadly this involves the following steps:

1. Construct an orthonormal basis of $\Pi$ with center at $\Phi$.
2. Generate a vector $e_2$ in $S_{d−1}$ with uniform distribution.
3. Generate an angle $\theta_0$ in $[0,\pi]$ with density function $f(\theta_0)$.
4. Generate the final perturbed state $\Psi$ as a rotation of $e_1 = \Phi$ by angle $\theta_0$ in the subspace spanned by the orthonormal basis $[e_1,e_2]$ using the expression
$$\Psi = \Phi \cos(\theta_0) + e_2 \sin(\theta_0)$$


This library provides modules to perform each of the above steps. See below for the API reference.

## API Reference

- [Algorithms](algos.md)
- [Coefficients of e2](e2.md)
- [Orthonormal basis](orthonormal.md)
- [Theta Zero](thetazero.md)
- [Utilities](utils.md)
