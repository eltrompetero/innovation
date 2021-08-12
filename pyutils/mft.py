# ====================================================================================== #
# Calculation derived for MFT of 1D firm growth.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize

from .utils import *


def density_bounds(density, wi,
                   vo=.49,
                   ve=.5,
                   dt=.1,
                   exact=False):
    """THIS IS WRONG. THIS IS A BOUND, NOT THE ACTUAL VELOCITY.

    Min density bound for nnovation front as derived from MFT and compared with
    simulation results.
    
    Depends on obsolescence and growth rates. Make sure that density has already been
    cached into parquet for faster runtime.
    
    Parameters
    ----------
    density : list of ndarray
    wi : float
        Innovation probability at each attempt.
    vo : float, .49
        Obsolescence rate per unit time.
    ve : float, .5
        Firm growth rate per unit time (attempt at moving to the right).
    dt : float, .1
        Size of simulation time step.
    exact : bool, False
        If True, use exact formulation. Else use small dt approximation. Make sure that dt
        corresponds to the actual time step in the simulation and not the rate at which
        samples were recorded.
    
    Returns
    -------
    float
        MFT min density bound for innovation front to keep moving.
    list
        Ave density for each simulation.
    """
    
    # number of firms on right boundary for each random trajectory
    right_density = [[i[-1] for i in d] for d in density]
    
    # histogram this
    y = [np.bincount(d, minlength=1) for d in right_density]

    sim_density = np.array([i.dot(range(i.size))/i.sum() for i in y])
    # min density required to progress
    if exact:
        assert not dt is None
        mft_bound = np.log(1 - vo * dt) / np.log(1 - wi * ve * dt)
    else: 
        mft_bound = vo / wi / ve

    return mft_bound, sim_density

def exp_density_f(x, a, b, wi):
    """
    a * (1 - np.exp(-b*(1-x)))

    Parameters
    ----------
    x : float
    a : float
    b : float
    wi : float
        Probability of successful innovation.
    """

    return np.exp(-b*(1-x)) / wi + a * (1 - np.exp(-b*(1-x)))

def fit_density(x, ydata, wi, full_output=False):
    """Fit MFT prediction of firm density.
    
    Parameters
    ----------
    x : ndarray
        Normalized lattice coordinate.
    ydata : ndarray
    wi : ndarray
        Probability of successful innovation.

    Returns
    -------
    ndarray
        (a,b) from a * (1-exp(-b*(1-x)))
        b is normalized by the typical width
    """

    def cost(args):
        a, b = np.exp(args)
        return np.linalg.norm(exp_density_f(x, a, b, wi) - ydata)

    soln = minimize(cost, (np.log(ydata[0]), np.log(10)))
    if full_output:
        return np.exp(soln['x']), soln
    return np.exp(soln['x'])

