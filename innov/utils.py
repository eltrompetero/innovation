# ====================================================================================== #
# Useful functions for innovation model.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
import pandas as pd
import os
import datetime
from warnings import warn
from multiprocess import Pool, cpu_count
from threadpoolctl import threadpool_limits
import dill as pickle
from itertools import combinations
from scipy.optimize import minimize
from scipy import sparse
from scipy.special import loggamma
from functools import cache

# JAX modules
import jax.numpy as jnp
from jax import jit, vmap, config, random, device_put, devices
from jax.lax import fori_loop, cond
import jax.experimental.sparse as jsparse


def pretty_load(fname):
    print(f"Loading {fname}...", end=' ')
    with open(fname, 'rb') as f:
        sim = pickle.load(f)
    print("Done!")
    return sim

def check_init_conditions(el, samples, K, *args): 
    """Check if initial conditions are consistent.
    """
    inn_front = args[0]
    obs_sub = args[1]
    sub = args[2]
    n = args[3]
    obs_front = args[4]

    assert inn_front.shape==(samples, el[1]*K)
    # check that the innovation front is one-dimensional and cover all branches
    assert inn_front[:,el[1]*K:(el[1]+1)*K].all() and (inn_front.sum(1)==K).all()

    # check that populated boundary coincides with innovation front
    assert jnp.where(n[0,:])[0][-1]==jnp.where(inn_front[0,:])[0][-1]

    # check that obs front does not overlap with populated subgraph
    assert (n[obs_front]==0).all()
    assert (n[obs_sub]==0).all()

def poisson(k, lam):
    return np.exp(-lam + k*np.log(lam) - loggamma(k+1))
