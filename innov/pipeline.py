# ====================================================================================== #
# Pipeline analysis.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np

from workspace.utils import save_pickle, increment_name
from .simple_model import UnitSimulator, ODE2, FlowMFT
from .utils import *
from .plot import phase_space_example_params



# ============== #
# Main functions #
# ============== #
def critical_ro():
    comparator = []
    spacing = np.linspace(-.1, .2, 10)

    ro = .9
    G = 80
    re = .5
    rd = .51
    I = .8
    ro_range = spacing + 2 * re - rd

    comparator.append(Comparator(G, ro, re, rd, I))
    comparator[-1].run_ro(ro_range, 10_000, 32);

    ro = .9
    G = 80
    re = .5
    rd = .51
    I = .4
    ro_range = spacing + 2 * re - rd

    comparator.append(Comparator(G, ro, re, rd, I))
    comparator[-1].run_ro(ro_range, 10_000, 32);

    save_pickle(['comparator','ro_range'], increment_name('cache/comparator_ro'))

def critical_re():
    comparator = []
    spacing = np.linspace(.05, .3, 10)

    # small I
    ro = .9
    G = 80
    re = .5  # just some arbitrary value
    rd = .5
    I = .1
    re_range = -spacing[::-1] + (ro + rd)/2  # only below the critical point

    comparator.append(Comparator(G, ro, re, rd, I))
    comparator[-1].run_re(re_range, 40_000, 64);

    # medium I
    ro = .9
    G = 20
    re = .5
    rd = .5
    I = .4
    re_range = -spacing[::-1] + (ro + rd)/2  # only below the critical point

    comparator.append(Comparator(G, ro, re, rd, I))
    comparator[-1].run_re(re_range, 40_000, 64);

    # large I
    ro = .9
    G = 20
    re = .5
    rd = .5
    I = .8
    re_range = -spacing[::-1] + (ro + rd)/2  # only below the critical point

    comparator.append(Comparator(G, ro, re, rd, I))
    comparator[-1].run_re(re_range, 40_000, 64);

    save_pickle(['comparator','re_range'], 'cache/comparator_re.p')

def phase_space_ODE2(G_bar=None,
                     re=None,
                     I=None,
                     n_space=128,
                     fname='cache/phase_space_ODE2.p'):
    """Map out regime phase space along rescaled rates ro/re and rd/re according to
    second-order approximation, which must be solved numerically.

    Parameters
    ----------
    G_bar : float, None
    re : float, None
    I : float, None
    n_space : int, 128
    fname : str, 'cache/phase_space_ODE2.p'

    Returns
    -------
    ndarray
        ro/re (y-axis) 
    ndarray
        rd/re (x-axis)
    ndarray
        L for each ro,rd pair
    """

    if G_bar is None:
        G, ro, re, rd, I, dt = phase_space_example_params().values()
        G_bar = G/re
        re = 1

    def loop_wrapper(args):
        ro_bar, rd_bar = args
        if ro_bar <= (2-rd_bar):
            return 1e5
        odemodel = ODE2(G_bar, ro_bar, 1, rd_bar, I)
        sol = odemodel.solve_L(full_output=True, method=3)[1]
        if np.isnan(sol['fun']):
            sol = odemodel.solve_L(L0=odemodel.L*1.1, full_output=True, method=3)[1]
        return sol['x'][0]

    ro_bar = np.linspace(0, 4, n_space)
    rd_bar = np.linspace(0, 4, n_space)

    ro_bar_grid, rd_bar_grid = np.meshgrid(ro_bar, rd_bar)

    with Pool() as pool:
        L = np.array(list(pool.map(loop_wrapper, zip(ro_bar_grid.ravel(), rd_bar_grid.ravel()))))

    L = np.reshape(L, (ro_bar.size, rd_bar.size))
    if fname:
        save_pickle(['G_bar','I','ro_bar','rd_bar','L'], fname, True)

    return ro_bar, rd_bar, L

def automaton_rescaling():
    """For comparison of automaton model with MFT under rescaling."""

    params = [{'G':15,
               'obs_rate':.55,
               'expand_rate':.4351,
               'death_rate':.435,
               'innov_rate':.4},
              {'G':10,
               'obs_rate':.45,
               'expand_rate':.43,
               'death_rate':.435,
               'innov_rate':.4}]

    c = []
    occupancy = []
    for i, p in enumerate(params):
        c_, occupancy_ = _automaton_rescaling(p)
        c.append(c_)
        occupancy.append(occupancy_)
        
        save_pickle(['c','occupancy','params'], 'cache/automaton_rescaling.p', True)
        print(f"Done with parameter set {i}.")

def cooperativity_comparison():
    """Comparing mean-field flow with automaton for different cooperativities.
    """

    G = 10
    ro = .5
    re = .4
    rd = .41
    I = .9
    dt = .1
    alpha_range = [.5, 1, 1.5]

    dmftmodel = []
    for alpha in alpha_range:
        dmftmodel.append(FlowMFT(G, ro, re, rd, I, dt, alpha=alpha))
        flag, mxerr = dmftmodel[-1].solve_stationary()

    # automaton model (dt must be small for good convergence)
    dt = 1e-3
    automaton = []
    for alpha in alpha_range:
        automaton.append(UnitSimulator(G, ro, re, rd, I,
                                       alpha=alpha,
                                       dt=dt))
        automaton[-1].parallel_simulate(1_000, 3_000)
        
    save_pickle(['alpha_range', 'dmftmodel', 'automaton'],
                'plotting/cooperativity_ex.p', True)



# ================ #
# Helper functions #
# ================ #
def _automaton_rescaling(params):
    """For comparison of automaton model with MFT under rescaling.
    
    Parameters
    ----------
    params : dict
        Containing 'G', 'obs_rate', 'expand_rate', 'death_rate', 'innov_rate' in
        this order (not all dicts preserve entry order).
        
    Returns
    -------
    float
        Rescaling factor.
    list
        Occupancy list per sample.
    """
    
    assert 'G' in params.keys()
    assert 'obs_rate' in params.keys()
    assert 'expand_rate' in params.keys()
    assert 'death_rate' in params.keys()
    assert 'innov_rate' in params.keys()

    simulator = UnitSimulator(*params.values(), dt=1e-3/2)
    c, occupancy = simulator.rescale_factor(2_000)
    return c, occupancy
