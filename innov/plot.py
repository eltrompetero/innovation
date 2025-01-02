# Module for plotting.
# Author: Eddie Lee, edlee@csh.ac.at
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


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
    vo_range = 2 - np.logspace(-3, .2, 100)[::-1]
    critical_rd = np.zeros_like(vo_range)
    err = np.zeros_like(vo_range)
    model = CompartmentModel(r0, I=I, gamma=gamma, K=K)
    
    for i, vo in enumerate(vo_range):
        def cost(logrd):
            rd = np.exp(logrd)
            if vo<1.8 and rd>4:
                return 1e10
            return np.abs(model.L(vo=vo, rd=rd)[0] - model.N(vo=vo, rd=rd)[0])**2
        sol = minimize(cost, .75, method='powell', tol=1e-10)
        critical_rd[i] = np.exp(sol['x'][0])
        err[i] = sol['fun']

    x, y = vo_range[err<1e-10], critical_rd[err<1e-10]
    spline = CubicSpline(x, y, extrapolate=True)
    y = spline(vo_plot)
    y[y<1] = 1
    return y
