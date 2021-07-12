# ====================================================================================== #
# Calculation derived for MFT of 1D firm growth.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize

from .utils import *


def density_bounds(density, dt, vi, vo=.49, vg=.5):
    """Min density bound for nnovation front as derived from MFT and compared with
    simulation results.
    
    Depends on obsolescence and growth rates. Make sure that density has already been
    cached into parquet for faster runtime.
    
    Parameters
    ----------
    density : list of ndarray
    dt : int
        Number of time steps recorded in each trajectory.
    vi : float
        Innovation rate (success at innovating).
    vo : float, .49
        Obsolescence rate per unit time.
    vg : float, .5
        Firm growth rate per unit time (attempt at moving to the right).
    
    Returns
    -------
    float
        MFT min density bound for innovation front to keep moving.
    list
        Avg density for each simulation.
    """
    
    # number of firms on right boundary for each random trajectory
    right_density = [[i[-1] for i in d] for d in density]
    
    # histogram this
    y = [np.bincount(d, minlength=1) for d in right_density]
    # account for time points that do not appear in the parquet file (this happens when
    # no firms exist in the simulation)
    for y_ in y:
        y_[0] += dt - y_.sum()

    sim_density = np.array([i.dot(range(i.size))/i.sum() for i in y])
    # min density required to progress
    mft_bound = np.log(1 - vo) / np.log(1 - vi * vg)

    return mft_bound, sim_density

def exp_density_f(x, a, b):
    """
    a * (1 - np.exp(-b*(1-x)))

    Parameters
    ----------
    x : float
    a : float
    b : float
    """

    return a * (1 - np.exp(-b*(1-x)))

def fit_density(x, ydata, full_output=False):
    """Fit MFT prediction of firm density.
    
    Parameters
    ----------
    x : ndarray
        Normalized lattice coordinate.
    ydata : ndarray

    Returns
    -------
    ndarray
        (a,b) from a * (1-exp(-b*(1-x)))
    """

    def cost(args):
        a, b = np.exp(args)
        return np.linalg.norm(exp_density_f(x, a, b) - ydata)

    soln = minimize(cost, (np.log(ydata[0]), np.log(10)))
    if full_output:
        return np.exp(soln['x']), soln
    return np.exp(soln['x'])

