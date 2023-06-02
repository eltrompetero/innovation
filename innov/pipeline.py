# ====================================================================================== #
# Pipeline analysis.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
from scipy.interpolate import interp1d

from workspace.utils import save_pickle, increment_name
from .simple_model import UnitSimulator, ODE2, FlowMFT, GridSearchFitter, L_1ode
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
        odemodel = ODE2(G_bar, ro_bar, rd_bar, I)
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

def automaton_rescaling(iprint=True):
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
        if iprint: print(f"Done with parameter set {i}.")

def cooperativity_comparison(iprint=True):
    """Comparing mean-field flow with automaton for different cooperativities.

    Parameters
    ----------
    iprint : bool, True
    """
    G = 10
    ro = .5
    re = .4
    rd = .41
    I = .9
    dt = .1
    alpha_range = [.5, 1, 1.5]
    
    if iprint: print("Starting DMFT calculation...")
    dmftmodel = []
    for alpha in alpha_range:
        dmftmodel.append(FlowMFT(G/re, ro/re, rd/re, I, dt, alpha=alpha))
        # try alternative L solution method if it is too short for computation
        if dmftmodel[-1].L<1:
            L0 = L_1ode(G/re, ro/re, rd/re, I, alpha=alpha)
            dmftmodel[-1].L = ODE2(G/re, ro/re, rd/re, I, alpha=alpha).solve_L(L0=L0, method=2)
        flag, mxerr = dmftmodel[-1].solve_stationary()
    if iprint: print("Done.")

    # automaton model (dt must be small for good convergence)
    if iprint: print("Starting automaton calculation...")
    dt = 1e-3
    automaton = []
    for alpha in alpha_range:
        automaton.append(UnitSimulator(G, ro, re, rd, I,
                                       alpha=alpha,
                                       dt=dt))
        automaton[-1].parallel_simulate(1_000, 3_000)
    if iprint: print("Done.")
        
    save_pickle(['alpha_range', 'dmftmodel', 'automaton'],
                'cache/cooperativity_ex.p', True)

def fit_iwai():
    data1958 = pd.read_csv('cache/iwai1958.csv', header=None).values[:,:2]
    data1963 = pd.read_csv('cache/iwai1963.csv', header=None).values[:,:2]

    x = data1958[:,0]
    y = data1958[:,1]
    ix = np.unique(x, return_index=True)[1]

    yfun = interp1d(x[ix], y[ix], kind='cubic')

    y = yfun(np.linspace(x.min(), x.max(), 30))
    x = np.arange(30)

    fitter = GridSearchFitter(y)

    # pre-selected fitting range from previous manual fits
    G_range = np.arange(50, 100, 2)
    ro_range = np.linspace(1., 3., 40)
    rd_range = np.linspace(.4, 1.2, 20)
    I_range = np.logspace(-2, -.5, 20)

    fit_results = fitter.scan(G_range, ro_range, rd_range, I_range,
                              L_scale=2)
    del_poor_fits(fit_results)

    save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                f'cache/iwai_0.p', True)

def fit_jangili(fit_ix):
    assert fit_ix==0 or fit_ix==1
    
    if fit_ix==0:
        data2014 = pd.read_csv('cache/jangili2014_small.csv', header=None).values
    else:
        data2014 = pd.read_csv('cache/jangili2014_large.csv', header=None).values

    x = data2014[:,0]
    y = data2014[:,1]
    ix = np.unique(x, return_index=True)[1]

    yfun = interp1d(x[ix], y[ix], kind='cubic')

    y = yfun(np.linspace(x.min(), x.max(), 30))
    x = np.arange(30)

    fitter = GridSearchFitter(y)

    # pre-selected fitting range from previous manual fits
    G_range = np.arange(40, 90, 2)
    if fit_ix==0:
        ro_range = np.linspace(1., 3., 40)
    else:
        ro_range = np.linspace(2.5, 4.5, 40)
    rd_range = np.linspace(.2, 1., 20)
    if fit_ix==1:
        I_range = np.logspace(-1, .5, 20)
    else:
        I_range = np.logspace(-3, 0, 20)

    fit_results = fitter.scan(G_range, ro_range, rd_range, I_range,
                              L_scale=2)
    del_poor_fits(fit_results)

    save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                 f'cache/jangili_{fit_ix}.p', True)

def fit_covid(fit_ix):
    from .genome import covid_clades

    covx, covy, br = covid_clades()
    x = covx[fit_ix]
    y = covy[fit_ix]
    x = x[y>0]
    y = y[y>0]

    fitter = GridSearchFitter(y, x)

    # pre-selected fitting range from previous manual fits
    G_range = np.arange(80, 130, 2)
    ro_range = np.linspace(1., 3., 40)
    rd_range = np.logspace(-2, -1, 20)
    I_range = np.linspace(3, 5, 20)

    fit_results = fitter.scan(G_range, ro_range, rd_range, I_range,
                              L_scale=.5, log=True, ignore_nan=True, rev=True, Q=br[fit_ix]+1)
    del_poor_fits(fit_results)

    if fit_ix==0:
        save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                    f'cache/covid_europe.p', True)
    else:
        save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                    f'cache/covid_northam.p', True)

def fit_prb():
    with open('cache/prb_citations_by_class.p', 'rb') as f:
        corrected_citation_rate_prb = pickle.load(f)['corrected_citation_rate_prb']

    for i in range(len(corrected_citation_rate_prb)):
        # only fit to the first decade
        y = corrected_citation_rate_prb[i][:11]
        x = np.arange(y.size)
        fitter = GridSearchFitter(y)

        # pre-selected fitting range from previous manual fits
        G_range = np.arange(100, 205, 5)
        ro_range = np.linspace(1., 2., 20)
        rd_range = np.linspace(.75, 1.5, 20)
        I_range = np.logspace(0, 1, 30)

        fit_results = fitter.scan(G_range, ro_range, rd_range, I_range,
                                  rev=True, log=True, L_scale=.25, offset=1)
        del_poor_fits(fit_results)
        
        save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                    f'cache/prb_citations_{i}.p', True)
        print(f"Done with cache/prb_citations_{i}.p")

def fit_patents():
    factor = 2
    fit_year_count = 6
    start_year = 1990

    for tech_class in [5]:
        with open(f'cache/patent_cites_{tech_class}_{start_year}.p', 'rb') as f:
            data = pickle.load(f)
            cite_traj = data['cite_traj']
            n = data['n']

        # separate trajectories out by total number of citations in the first 20 years
        sum_cites = np.array([i[:21].sum() for i in cite_traj])
        y = []
        for i in range(1, 8):
            ix = (factor*i<=sum_cites) & (sum_cites<factor*(i+1))
            y.append(cite_traj[ix].mean(0)[:fit_year_count] / n[:fit_year_count] * n[0])
        
        for j in range(len(y)):
            # only fit the first five years after publication
            fitter = GridSearchFitter(y[j])

            # pre-selected fitting range from previous manual fits
            G_range = np.arange(70, 110, 2)
            ro_range = np.linspace(1.5, 3.5, 20)
            rd_range = np.logspace(-3, -1.5, 20)
            I_range = np.logspace(-1.5, 0, 20)

            fit_results = fitter.scan(G_range, ro_range, rd_range, I_range,
                                      rev=True, log=True, L_scale=6/51, offset=1)
            del_poor_fits(fit_results)

            save_pickle(['fit_results','y','fitter','G_range','ro_range','rd_range','I_range'],
                        f'cache/fit_patent_cites_{tech_class}_{start_year}_{j}.p', True)
            print(f"Done with cache/fit_patent_cites_{tech_class}_{start_year}_{j}.p")

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


if __name__=='__main__':
    #fit_iwai()
    fit_jangili(0)
    fit_jangili(1)
    fit_covid(0)
    fit_covid(1)
    fit_patents()
    fit_prb()
