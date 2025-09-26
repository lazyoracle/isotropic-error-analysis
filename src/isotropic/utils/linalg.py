import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import svd
from jax.typing import ArrayLike


def jax_null_space(A: ArrayLike) -> Array:
    """
    Compute the null space of a matrix $A$ using JAX.

    Parameters
    ----------
    A : ArrayLike
        The input matrix for which to compute the null space.

    Returns
    -------
    Array
        The basis vectors of the null space of A.

    Notes
    ------
    See also:

    - `scipy.linalg.null_space` for the reference implementation in SciPy.
    - [https://github.com/jax-ml/jax/pull/14486](https://github.com/jax-ml/jax/pull/14486) for an old JAX implementation.
    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s, initial=0.0) * rcond
    num = jnp.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q
