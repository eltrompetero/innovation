# Module for plotting.
# Author: Eddie Lee, edlee@csh.ac.at
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from .utils import *

from .simple_calculations import CompartmentModel

from .simple_calculations import CompartmentModel
from .utils import *

def density_snapshot(n, el, K, t, mean=False, replica_ix=0, **kwargs):
    """Plot density snapshots from automaton simulation. Show the first two
    branches separately and either one replica over mean over replicas.

    Parameters
    ----------
    n : np.ndarray
        Density.
    el : tuple
        (length of initial branch, length of later branches
    K : int
    t : list-like
        List of time-indices to show.
    mean : bool or str, False
        If False show only the first replica. Else 'replica' or 'branch+replica'
    **kwargs : dict
    """
    fig, ax = plt.subplots(figsize=(6,2))

    if mean=='replica': 
        for tix in t:
            ax.plot(n[tix,:,::K].mean(0))
            if K>1:  # show a second branch
                ax.plot(n[tix,:,1::K].mean(0))
    elif mean=='branch+replica':
        for tix in t:
            # mean over replicas then mean over branches
            ax.plot(n[tix].mean(0).reshape(el[1], K).mean(1))
    elif mean is False:
        for tix in t:
            ax.plot(n[tix,replica_ix,::K])
            if K>1:
                ax.plot(n[tix,replica_ix,1::K])
    else: raise NotImplementedError

    ax.set(**kwargs)
    return fig

def low_density_region(vo_plot, r0, I, K, gamma):
    """Return low density region where N<L, rd as a function of vo.
    
    This has been fixed to work specifically for the values that are plotted in paper. 
    For other values, some fine-tuning in solving the boundary condition may be 
    necessary.

    Returns
    -------
    ndarray
        Solved rd corresponding to input range.
    """
    vo_range = np.linspace(.001, 1.4, 200)
    rd = np.zeros_like(vo_range)
    err = np.zeros_like(vo_range)
    model = CompartmentModel(r0, I=I, gamma=gamma, K=K)
    
    for i, vo in enumerate(vo_range):
        def cost(logrd):
            rd = np.exp(logrd)
            if rd>5 or rd<1: return 1e10
            return np.abs(model.L(vo=vo, rd=rd, quadratic_form=0)[0] - model.N(vo=vo, rd=rd, quadratic_form=0)[0])**2
        sol = minimize(cost, 1., method='powell', tol=1e-10)
        rd[i] = np.exp(sol['x'][0])
        err[i] = sol['fun']
    ix = (err<1e-5) & (~np.isnan(rd))
    x, y = vo_range[ix], rd[ix]
    spline = CubicSpline(x, y)
    y = spline(vo_plot)
    #y[y<1] = 1
    return y
