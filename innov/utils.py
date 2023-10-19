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
import duckdb as db
from itertools import combinations
from scipy.optimize import minimize

