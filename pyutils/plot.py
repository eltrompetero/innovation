# ====================================================================================== #
# Plotting helper functions.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import colors


def jangili_params(ix=0):
    if ix==0:
        return {'G':30,
                'ro':.53,
                're':.43,
                'rd':.414,
                'I':.189,
                'dt':.1}
    elif ix==1:
        return {'G':20,
                'ro':.53,
                're':.43,
                'rd':.414,
                'I':.189,
                'dt':.1}
    else:
        raise Exception

def prb_params(ix=0):
    if ix==0:
        return {'G':90,
                'ro':.68,
                're':.43,
                'rd':.42,
                'I':1.8,
                'dt':.1}
    elif ix==1:
        return {'G':150,
                'ro':.6,
                're':.43,
                'rd':.422,
                'I':.7,
                'dt':.1}
    else:
        raise Exception

def covid_params(ix=0):
    if ix==0:
        return {'G':70,
                'ro':1.1,
                're':.58,
                'rd':.23,
                'I':1.3,
                'dt':.1,
                'Q':2.3271742356782428}
    elif ix==1:
        return {'G':70,
                'ro':1.1,
                're':.58,
                'rd':.23,
                'I':1.3,
                'dt':.1,
                'Q':2.3174342105263157}
    else:
        raise Exception

def patent_params(ix=0):
    if ix==0:
        return {'G':23,
                'ro':.54,
                're':.43,
                'rd':.37,
                'I':.8,
                'dt':.1}
    else:
        raise Exception

def phase_space_example_params():
    return {'G':40,
            'ro':.53,
            're':.43,
            'rd':.414,
            'I':.189,
            'dt':.1}

def phase_space_variations(ax, plot_range=[0,1,2,3], set_ticklabels=False):
    """Helper function for plotting phase space for cooperative and Bethe lattice
    variations using 1st order approximation.
    
    Parameters
    ----------
    ax : list of mpl.Axes
    plot_range : list, [0,1,2,3]
        Indices of plots to include.
    set_ticklabels : bool, False
    """
    
    # set default params
    G, ro, re, rd, I, dt = jangili_params(1).values()
    ro /= re
    rd /= re
    G /= re
    rd_range = np.linspace(0, 4, 100)
    
    def collapse_cost_1st(ro, rd):
        """This is 0 for 1st order approximation of collapse boundary."""
        return (((ro/I)**(1/a) * (rd + ro - 2)) - G)**2
    
    plot_counter = 0

    if 0 in plot_range:
        # anti-cooperative
        a = .5
        ro_range = 2 - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = [minimize(collapse_cost_1st, ro_range[i]+.2, args=(rd_,), bounds=[(1e-3,np.inf),],
                             constraints=[{'type':'ineq',
                                           'fun':lambda ro, rd: rd-2+ro,
                                           'jac':lambda ro, rd: 1,
                                           'args':(rd_,)}])['x'][0]
                    for i, rd_ in enumerate(rd_range)]
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*4, fc='C0', alpha=.4)
        plot_counter += 1
    
    # cooperative
    if 1 in plot_range:
        a = 1.5
        ro_range = 2 - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = [minimize(collapse_cost_1st, ro_range[i]+.2, args=(rd_,), bounds=[(1e-3,np.inf),],
                             constraints=[{'type':'ineq',
                                           'fun':lambda ro, rd: rd-2+ro,
                                           'jac':lambda ro, rd: 1,
                                           'args':(rd_,)}])['x'][0]
                    for i, rd_ in enumerate(rd_range)]
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*4, fc='C0', alpha=.4)
        plot_counter += 1

    # bethe lattice Q=1.5
    if 2 in plot_range:
        Q = 1.5
        ro_range = 2/(Q-1) - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = 1/(Q-1) - rd_range/2 + np.sqrt((1/(Q-1)-rd_range/2)**2 + G*I)
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*4, fc='C0', alpha=.4)
        plot_counter += 1

    # bethe lattice Q=3
    if 3 in plot_range:
        Q = 3
        ro_range = 2/(Q-1) - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = 1/(Q-1) - rd_range/2 + np.sqrt((1/(Q-1)-rd_range/2)**2 + G*I)
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*4, fc='C0', alpha=.4)

    for a in ax:
        a.set(xlim=(0,4), ylim=(0,4))
        if set_ticklabels:
            a.set(xticklabels=[], yticklabels=[])
    if set_ticklabels:
        ax[0].set(yticklabels=[0,2,4])
        ax[2].set(yticklabels=[0,2,4])
        ax[2].set(xticklabels=[0,2,4])
        ax[3].set(xticklabels=[0,2,4])

def phase_space_ODE2(fig, ax, ro_bar, rd_bar, L,
                     vmin=1, vmax=50):
    """Fast way of showing results of numerical solution to phase space from 2nd
    order approximation.

    Parameters
    ----------
    fig : plt.Figure
    ax : plt.Axes
    ro_bar : ndarray
    rd_bar : ndarray
    L : ndarray
    vmin : float, 1
    vmax : float, 50
    """

    dro = (ro_bar[1] - ro_bar[0])/2
    drd = (rd_bar[1] - rd_bar[0])/2

    # static regime
    L_ = L.copy()
    cm = plt.cm.copper.copy()
    cax = ax.imshow(L_, vmin=vmin, vmax=vmax, origin='lower',
                    extent=[ro_bar[0]-dro, ro_bar[-1]+dro, rd_bar[0]-drd, rd_bar[-1]+drd],
                    cmap=cm, zorder=0)

    cmap = colors.ListedColormap(['#E1918B', '#B6CDE3'])
    bounds=[0,1,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # collapse regime
    L_ = L.copy()
    L_[L>1] = np.nan
    L_[L<=1] = 1.5
    ax.imshow(L_, origin='lower',
              extent=[ro_bar[0]-dro, ro_bar[-1]+dro, rd_bar[0]-drd, rd_bar[-1]+drd],
              cmap=cmap, norm=norm, zorder=1)

    # growth regime
    L_ = L.copy()
    L_[L==1e5] = .5
    L_[L<1e5] = np.nan
    ax.imshow(L_, origin='lower',
              extent=[ro_bar[0]-dro, ro_bar[-1]+dro, rd_bar[0]-drd, rd_bar[-1]+drd],
              cmap=cmap, norm=norm, zorder=2)

    fig.colorbar(cax, label=r'lattice length $L$')
    ax.set(xlabel=r'obsolescence $\bar r_o$', ylabel=r'death $\bar r_d$',
           xlim=(ro_bar[0]-1.5*dro, ro_bar[-1]+1.5*dro),
           ylim=(rd_bar[0]-drd, rd_bar[-1]+drd))
