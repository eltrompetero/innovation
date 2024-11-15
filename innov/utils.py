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