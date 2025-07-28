"""This module contains functions for generating the vector e_2"""

from jax import Array
from isotropic.utils.bisection import get_theta

def get_e2(d: int) -> Array:
    """
    Generates the vector e_2 in R^d.

    Parameters
    ----------
    d : int
        Dimension of the space.
    key : jax.random.PRNGKey, optional
        Random key for reproducibility, by default random.PRNGKey(0)

    Returns
    -------
    Array
        The vector e_2 in R^d.
    """
    theta:Array = jnp.zeros(d - 1)

    # Generate theta_{d-1} from a uniform distribution in [0, 2*pi]
    theta = theta.at[-1].set(
        random.uniform(key, shape=(), minval=0, maxval=2 * jnp.pi))

    # Generate theta_j for j = 1, ..., d-2 using bisection method
    # TODO: vectorize this loop 
    for j in range(0, d - 2, 1):
        theta_j = get_theta(
            F = lambda theta: calculate_F_j(theta, j, d),
            a = 0,
            b = jnp.pi,
            x = random.uniform(key, shape=(), minval=0, maxval=1),
            tol = 1e-9
        )

        theta = theta.at[j].set(theta_j)

    # e2 has dimension d
    e2:Array = jnp.ones(d)

    # e2[1] to e2[d-1] have products of sin(theta) terms
    # TODO: vectorize this loop 
    for j in range(1, d):
        e2 = e2.at[j].set(e2[j - 1] * jnp.sin(theta[j - 1]))

    theta = jnp.append(theta, 0) # Append 0 for cos(0) of last coordinate

    # e2[d] has additional cos(theta) term in product
    e2 = e2 * jnp.cos(theta)

    return e2