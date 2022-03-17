# ====================================================================================== #
# Pipeline analysis.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np

from workspace.utils import save_pickle
from .simple_model import UnitSimulator, ODE2
from .utils import *
from .plot import jangili_params



# ============== #
# Main functions #
# ============== #
def phase_space_ODE2(G_bar=None, re=None, I=None,
                     n_space=128,
                     fname='cache/phase_space_ODE2.p'):
    """Map out regime phase space along rescaled rates ro/re and rd/re according to second-order
    approximation, which must be solved numerically.

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
        G, ro, re, rd, I, dt = jangili_params(1).values()
        G_bar = G/re
        re = 1

    def loop_wrapper(args):
        ro_bar, rd_bar = args
        if ro_bar <= (2-rd_bar):
            return 1e5
        odemodel = ODE2(G_bar*re, ro_bar*re, re, rd_bar*re, I)
        sol = odemodel.solve_L(full_output=True)[1]
        if np.isnan(sol['fun']):
            sol = odemodel.solve_L(L0=odemodel.L*2, full_output=True)[1]
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
        
    L = np.mean([len(i) for i in automaton[0].occupancy])

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
