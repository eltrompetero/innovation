# Module for plotting.
# Author: Eddie Lee, edlee@csh.ac.at
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

from .simple_calculations import CompartmentModel, pde_pseudogap
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
    vo_range = np.linspace(.05, .8, 200)
    rd = np.zeros_like(vo_range)
    err = np.zeros_like(vo_range)
    model = CompartmentModel(r0, I=I, gamma=gamma, K=K)
    
    for i, vo in enumerate(vo_range):
        def cost(logrd):
            rd = np.exp(logrd)
            if rd>5 or rd<1: return 1e10
            return np.abs(model.L(vo=vo, rd=rd, quadratic_form=1)[0] - model.N(vo=vo, rd=rd, quadratic_form=1)[0])**2
        sol = minimize(cost, 1., method='powell', tol=1e-10)
        rd[i] = np.exp(sol['x'][0])
        err[i] = sol['fun']
    ix = (err<1e-5) & (~np.isnan(rd))
    x, y = vo_range[ix], rd[ix]
    spline = CubicSpline(x, y)
    y = spline(vo_plot)
    return y

@cache
def structure_low_density(vo):
    """For plots in paper, vo should be either .5 or 1."""
    assert vo==.5 or vo==1

    if vo==1: 
        gamma_range = np.linspace(.7, 1, 30)
    else:
        gamma_range = np.logspace(-3, 0, 60)
    r = .4

    r0 = 10/r
    I = 2
    rd = .5/r

    K = np.zeros_like(gamma_range)
    err = np.zeros_like(gamma_range)

    model = CompartmentModel(r0, I, rd, vo)
    for i, g in enumerate(gamma_range):
        def cost(logK):
            K = np.exp(logK[0]) + 1
            if g>.1 and K>25: return 1e20
            elif .01<=g<=.1 and K>1e2: return 1e20
            elif g<1e-2 and K>1e3: return 1e20
            return np.abs(model.L(gamma=g, K=K, quadratic_form=1)[0].real -
                          model.N(gamma=g, K=K, quadratic_form=1)[0].real)**2
        sol = minimize(cost, 0., tol=1e-10)
        K[i] = np.exp(sol['x'][0]) + 1
        err[i] = sol['fun']

    ix = (err<1e-4) & (~np.isnan(K))
    K, gamma_range = K[ix], gamma_range[ix]

    return K, gamma_range

@cache
def critical_K_line():
    """Critical K and gamma relation for phase diagram in Figure 2.
    
    This returns the necessary values to plot the lines for the two values of vo shown."""
    r0 = 10/.52
    I = 2
    rd = .5/.52
    
    # for vo=.5
    vo = .5
    model = CompartmentModel(r0=r0, I=I, rd=rd, vo=vo)

    gamma_range = np.linspace(0, 1, 50), np.linspace(0, 1, 100)
    K_critical = np.zeros_like(gamma_range[0]), np.zeros_like(gamma_range[1])

    K_critical[0][:] = np.array([model.K_runaway(gamma=g, f_threshold=1e-5, K_max=1e4)
                             for g in gamma_range[0]])
    dcounter = 0
    while dcounter<10:
        counter = 0
        while counter<100 or (~np.isnan(K_critical[0])).all():
            for i in range(K_critical[0].size-1):
                if np.isnan(K_critical[0][i]) and not np.isnan(K_critical[0][i+1]):
                    K_critical[0][i] = model.K_runaway(gamma=gamma_range[0][i], f_threshold=1e-5, K_max=1e4,
                                                    K0=np.log(K_critical[0][i+1]) + dcounter*.1)
            counter += 1
        dcounter += 1
        
    # for vo=1.
    vo = 1.
    model = CompartmentModel(r0=r0, I=I, rd=rd, vo=vo)
    K_critical[1][:] = np.array([model.K_runaway(gamma=g, f_threshold=1e-5, K_max=1e3) for g in gamma_range[1]])
    dcounter = 0
    while dcounter<25:
        counter = 0
        while counter<100 or (~np.isnan(K_critical[1])).all():
            for i in range(K_critical[1].size-1):
                if np.isnan(K_critical[1][i]) and not np.isnan(K_critical[1][i-1]):
                    K_critical[1][i] = model.K_runaway(gamma=gamma_range[1][i], f_threshold=1e-5, K_max=1e3,
                                                       K0=np.log(K_critical[1][i-1]) - dcounter*.1)
                if np.isnan(K_critical[1][i]) and not np.isnan(K_critical[1][i+1]):
                    K_critical[1][i] = model.K_runaway(gamma=gamma_range[1][i], f_threshold=1e-5, K_max=1e3,
                                                       K0=np.log(K_critical[1][i+1]) + dcounter*.1)
            counter += 1
        dcounter += 1

    return K_critical, gamma_range

def find_bistable_l1(r0, I, rd, vo, gamma, K, tmax,
                     initial_guess=100, tol=1e-6, dtol=1e-13):
    def y_end(l1):
        y0 = np.array([l1, l1, 1, 1])
        t = np.linspace(0, tmax, int(1e4)*2)
        y = odeint(pde_pseudogap, y0, t, args=(r0, I, rd, vo, gamma, K))
        return y[-1]

    l1 = initial_guess
    assert y_end(l1)[0]<2
    
    d = 1
    prev_val = np.inf
    val = y_end(l1)[0]
    while (val<2 or val>1e10 or abs(prev_val-val)>tol) and d>dtol:
        prev_val = val
        while val<2:  # keep incrementing til we diverge
            l1 += d
            val = y_end(l1)[0]
        # undo divergence step and shrink increment step size
        l1 -= d
        val = y_end(l1)[0]
        d /= 10

    return l1
