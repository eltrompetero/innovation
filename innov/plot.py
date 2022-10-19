# ====================================================================================== #
# Plotting helper functions.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import colors
import dill as pickle

from .simple_model import FlowMFT



def fit_sol(name, rev=False, offset=0, sol_ix=0, **model_kw):
    with open(f'cache/{name}.p', 'rb') as f:
        odata = pickle.load(f)
    fit_results = odata['fit_results']

    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]

    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    model = FlowMFT(G, ro, 1, rd, I, dt=.1, **model_kw)
    model.solve_stationary()
    if rev:
        return lambda x, a=a, b=b: model.n(model.L - x*a - offset) * b
    return lambda x, a=a, b=b: model.n(x*a - offset) * b

def iwai_params(ix=0, sol_ix=0):
    with open(f'cache/iwai_{ix}.p', 'rb') as f:
        data = pickle.load(f)
    
    fit_results = data['fit_results']

    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]
    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    return {'G':G,
            'ro':ro,
            'rd':rd,
            'I':I,
            'dt':.1,
            'a':a,
            'b':b}

def jangili_params(ix=0, sol_ix=0):
    with open(f'cache/jangili_{ix}.p', 'rb') as f:
        data = pickle.load(f)
    
    fit_results = data['fit_results']

    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]
    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    return {'G':G,
            'ro':ro,
            'rd':rd,
            'I':I,
            'dt':.1,
            'a':a,
            'b':b}

def prb_params(ix=0, sol_ix=0):
    with open(f'cache/prb_citations_{ix}.p', 'rb') as f:
        data = pickle.load(f)
    
    fit_results = data['fit_results']

    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]
    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    return {'G':G,
            'ro':ro,
            'rd':rd,
            'I':I,
            'dt':.1,
            'a':a,
            'b':b}

def covid_params(ix=0, sol_ix=0):
    from .genome import covid_clades

    covx, covy, br = covid_clades()
    if ix==0:
        with open(f'cache/covid_europe.p', 'rb') as f:
            data = pickle.load(f)
    elif ix==1:
        with open(f'cache/covid_northam.p', 'rb') as f:
            data = pickle.load(f)
    else:
        raise NotImplementedError

    fit_results = data['fit_results']

    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]
    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    return {'G':G,
            'ro':ro,
            'rd':rd,
            'I':I,
            'dt':.1,
            'a':a,
            'b':b,
            'Q':br[ix]+1}

def patent_params(ix=0, tech_class=5, sol_ix=0):
    """Model fit parameters.

    Parameters
    ----------
    ix : int, 0
        Citation class index.
    tech_class : int, 5
    sol_ix : int, 0
        Solution index when ordered by closeness of fit.

    Returns
    -------
    dict
    """
    
    with open(f'cache/fit_patent_cites_{tech_class}_1990_{ix}.p', 'rb') as f:
        data = pickle.load(f)
    
    fit_results = data['fit_results']
    
    k = list(fit_results.keys())[np.argsort([i[2]['fun'] for i in fit_results.values()])[sol_ix]]
    G, ro, rd, I = k
    a, b = fit_results[k][:2]

    return {'G':G,
            'ro':ro,
            'rd':rd,
            'I':I,
            'dt':.1,
            'a':a,
            'b':b}

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
    G, ro, rd, I, dt, a, b = jangili_params(1).values()
    rd_range = np.linspace(0, 10, 100)
    
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
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*10, fc='C0', alpha=.4)
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
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*10, fc='C0', alpha=.4)
        plot_counter += 1

    # bethe lattice Q=1.5
    if 2 in plot_range:
        Q = 1.5
        ro_range = 2/(Q-1) - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = 1/(Q-1) - rd_range/2 + np.sqrt((1/(Q-1)-rd_range/2)**2 + G*I/4)
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*10, fc='C0', alpha=.4)
        plot_counter += 1

    # bethe lattice Q=3
    if 3 in plot_range:
        Q = 3
        ro_range = 2/(Q-1) - rd_range
        ax[plot_counter].fill_between(ro_range, np.zeros_like(rd_range), rd_range, fc='C3', alpha=.4)

        ro_range = 1/(Q-1) - rd_range/2 + np.sqrt((1/(Q-1)-rd_range/2)**2 + G*I/4)
        ax[plot_counter].fill_betweenx(rd_range, ro_range, np.ones_like(ro_range)*10, fc='C0', alpha=.4)

    for a in ax:
        a.set(xlim=(0,10), ylim=(0,10))
        if set_ticklabels:
            a.set(xticklabels=[], yticklabels=[])
    if set_ticklabels:
        ax[0].set(yticklabels=[0,5,10])
        ax[2].set(yticklabels=[0,5,10])
        ax[2].set(xticklabels=[0,5,10])
        ax[3].set(xticklabels=[0,5,10])

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
