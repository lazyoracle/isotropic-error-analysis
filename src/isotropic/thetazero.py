"""This module contains functions for generating theta_0"""
from isotropic.utils.bisection import get_theta
from isotropic.utils.simpsons import simpsons_rule
import jax.numpy as jnp
from jax import Array

def get_theta_zero(d: int) -> float:
    """generate the angle theta_0 with a normal distribution

    Parameters
    ----------
    d : int
        dimension of the system

    Returns
    -------
    float
        value of theta_0
    """
    raise NotImplementedError("This function is not implemented yet.")