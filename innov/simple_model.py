# ====================================================================================== #
# Minimal innovation model implementations and solutions.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from numba import njit, jit
from numba.typed import List
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
import warnings
from scipy.special import logsumexp

from workspace.utils import save_pickle
from .utils import *



def L_linear(G, ro, rd, I, alpha=1., Q=2):
    """Lattice length given linear pseudogap approximation.

    Parameters
    ----------
    G : float
    ro : float
    rd : float
    I : float
    alpha : float, 1.
    Q : float, 2

    Returns
    -------
    float
        Approximate lattice length.
    """
    G_bar = G
    ro_bar = ro
    rd_bar = rd
    n0 = ro_bar / I
    
    if (ro_bar * (rd_bar + ro_bar - 2/(Q-1)))==0:
        return np.inf
    return G_bar * I / (ro_bar * (rd_bar + I*n0**alpha - 2/(Q-1)))
    return G_bar * I / (ro_bar * (rd_bar + ro_bar - 2/(Q-1)))

def L_1ode(G, ro, rd, I, alpha=1., Q=2, return_denom=False):
    """Calculate stationary lattice width accounting the first order correction
    (from Firms II pg. 140, 238).

    This is equivalent to

    (np.exp((1-rd_bar)/(1-ro_bar)) - 1) / np.exp((1-rd_bar)/(1-ro_bar)) * -G_bar * I
    / ro_bar / (1-rd_bar)
    
    This matches numerical solution to first-order equation with x=-1 boundary
    condition.

    Parameters
    ----------
    G : float
    ro : float
    rd : float
    I : float
    alpha : float, 1.
    Q : float 2.
    return_denom : bool, False
    
    Returns
    -------
    float
        Estimated lattice width.
    float (optional)
        Denominator for L calculation.
    """
        
    G_bar = G
    ro_bar = ro
    rd_bar = rd
    
    assert not hasattr(G_bar, '__len__')

    z = -(1 - rd_bar*(Q-1)) / (1 - ro_bar*(Q-1))
    if not hasattr(z, '__len__'):
        is_ndarray = False
        z = np.array([z])
    else:
        is_ndarray = True
    C = np.zeros_like(z)

    # handle numerical precision problems with z
    # analytic limit
    zeroix = z==0

    # large z approximation
    largeix = z < -200
    C[largeix] = np.inf
    
    remainix = (~zeroix) & (~largeix)
    C[remainix] = (np.exp(-z[remainix]) - 1 + z[remainix]) / z[remainix]
    
    denom = ro_bar/I * ((1+1/(1-C))/(Q-1) - rd_bar - ro_bar/(1-C))
    L = -G_bar / denom

    # analytic form
    #L = G_bar / (1-rd_bar) * I/ro_bar * (np.exp(-(1-rd_bar)/(1-ro_bar)) - 1)
    
    if return_denom:
        if is_ndarray:
            return L, denom
        return L[0], denom
    if is_ndarray:
        return L
    return L[0]

def match_length(y1, y2, side1='l', side2='r'):
    """Fill zeros to match two vectors. Pad zeros on either left or right sides.

    Parameters
    ----------
    y1 : ndarray
    y2 : ndarray
    side1 : str, 'l'
    side2 : str, 'r'

    Returns
    -------
    ndarray
    ndarray
    """
    
    if side1=='l':
        y1 = y1[::-1]
    if side2=='l':
        y2 = y2[::-1]

    if y1.size > y2.size:
        y2 = np.concatenate((y2, np.zeros(y1.size-y2.size)))
    elif y2.size > y1.size:
        y1 = np.concatenate((y1, np.zeros(y2.size-y1.size)))

    if side1=='l':
        y1 = y1[::-1]
    if side2=='l':
        y2 = y2[::-1]
    
    return y1, y2

def _solve_G(G0, L, ro, re, rd, I):
    """Solve for G that return appropriate L."""

    z = (re - rd) / (re * (ro/re-1))
    C = (np.exp(-z)-1+z) / z

    def cost(logG):
        G = np.exp(logG)
        return (L - G * I * re / (ro * (rd + ro - re*(1+1/(1-C)))))**2

    soln = minimize(cost, np.log(G0))
    return np.exp(soln['x']), soln

def fit_dmft(data, L, dt, initial_params, **params_kw):
    """Best fit of dmft model at stationarity.

    Parameters
    ----------
    dt : float
    initial_params : list
        G shouldn't be specified.
    **params_kw

    Returns
    -------
    dict
    """

    def cost(params):
        ro, re, rd, I = np.exp(params)
        G = _solve_G(30, L, ro, re, rd, I)
        G = G[0]
        #print(f'{G=}')

        model = FlowMFT(G, ro, re, rd, I, dt, **params_kw)
        print(f'{model.L=}')
        if not np.isclose(model.L, L, atol=1e-2): return 1e30
        
        flag, mxerr = model.solve_stationary()
        if np.any(model.n<0): return 1e30

        try:
            c = np.linalg.norm(data-model.n)
        except ValueError:
            return 1e30
        return c

    soln = minimize(cost, np.log(initial_params))
    return np.exp(soln['x']), soln

def _fit_dmft(data, dt, initial_params, **params_kw):
    """Best fit of dmft model at stationarity.

    Parameters
    ----------
    dt : float
    initial_params
    **params_kw

    Returns
    -------
    dict
    """
    
    def cost(params):
        model = FlowMFT(*np.exp(params), dt, **params_kw)
        if model.L<1 or ~np.isfinite(model.L): return 1e30
        
        flag, mxerr = model.solve_stationary()
        if np.any(model.n<0): return 1e30

        y1, y2 = match_length(data, model.n, 'l', 'l')
        c = np.linalg.norm(y1-y2)
        return c

    soln = minimize(cost, np.log(initial_params))
    return np.exp(soln['x']), soln

def fit_flow(x, data, initial_params, full_output=False, reverse=False, **params_kw):
    """Best fit of ODE model at stationarity.

    Parameters
    ----------
    x : ndarray
    data : ndarray
    initial_params : list
    full_output : bool, False
        If True, return soln dict.
    reverse : bool, False
    **params_kw

    Returns
    -------
    ndarray
        Estimates for (G, ro, re, rd, I).
    dict
        If full_output is True. From scipy.optimize.minimize.
    """

    data_fun = interp1d(x, data, bounds_error=False, fill_value=0)

    def cost(params):
        A, B, G, ro, rd, I = params
        
        try:
            model = FlowMFT(G, ro, rd, I, dt=.1, L_method=2, **params_kw)
            model.solve_stationary()
        except (AssertionError, ValueError):  # e.g. problem with stationarity and L
            return 1e30
        
        if reverse:
            modx = np.linspace(x.max()-model.n.size, x.max(), model.n.size)
            c = np.linalg.norm(data_fun(modx/A) - model.n*B)
        else:
            modx = np.arange(model.n.size)
            c = np.linalg.norm(data_fun(modx/A) - model.n*B)

        # handle overflow
        if np.isnan(c): c = 1e30
        return c
    
    soln = minimize(cost, initial_params, bounds=[(0,np.inf),(0,np.inf),(1e-3, np.inf)]+[(1e-3,30)]*3,
                    method='SLSQP',
                    constraints=({'type':'ineq', 'fun':lambda args: args[3] - 2 + args[4] - 1e-3,
                                  'jac':lambda args: np.array([0,0,0,1,0,1])},))

    if full_output:
        return soln['x'], soln
    return soln['x']


def fit_ode(x, data, initial_params, full_output=False, **params_kw):
    """Best fit of ODE model at stationarity.

    Parameters
    ----------
    x : ndarray
    data : ndarray
    initial_params : list
    full_output : bool, False
        If True, return soln dict.
    **params_kw

    Returns
    -------
    ndarray
        Estimates for (G, ro, re, rd, I).
    dict
        If full_output is True. From scipy.optimize.minimize.
    """

    def cost(params):
        G, ro, rd, I = params
        
        try:
            model = ODE2(G, ro, 1, rd, I, **params_kw)
        except AssertionError:  # e.g. problem with stationarity and L
            return 1e30
        
        c = np.linalg.norm(data - model.n(x))
        return c

    soln = minimize(cost, initial_params, bounds=[(1e-3, np.inf)]*4, method='SLSQP',
                    constraints=({'type':'ineq', 'fun':lambda args: args[1] - 2 + args[2] - 1e-3,
                                  'jac':lambda args: np.array([0,1,0,1])},))

    if full_output:
        return soln['x'], soln
    return soln['x']

def fit_piecewise_ode(x, y, initial_guess,
                      full_output=False):
    """Heuristic algorithm for fitting to density function.

    Parameters
    ----------
    full_output : bool, False
    """

    def cost(args):
        try:
            # first three args are x offset, width, and height units
            xoff = args[0]
            args = np.exp(args[1:])
            xunit, yunit = args[:2]
            args = np.insert(args[2:], 2, 1)
            odemodel = ODE2(*args)
        except AssertionError:
            return 1e30
        
        peakx = odemodel.peak()
        if peakx<=0: return 1e30

        mody = odemodel.n(x / xunit - xoff) / yunit
        return np.linalg.norm(mody[-10:] - y[-10:])
        
    # convert parameters to log space to handle cutoff at 0
    soln = minimize(cost, [initial_guess[0]]+np.log(initial_guess[1:]).tolist(), method='powell')
    if full_output:
        return np.exp(soln['x']), soln
    return np.exp(soln['x'])

def _fit_piecewise_ode(peakx, peaky, n0, s0, initial_guess,
                       full_output=False):
    """Heuristic algorithm for fitting to density function.

    Parameters
    ----------
    full_output : bool, False
    """

    def cost(args):
        try:
            # first three args are x offset, width, and height units
            xoff = args[0]
            args = np.exp(args[1:])
            xunit, yunit = args[:2]
            args = np.insert(args[2:], 2, 1)
            odemodel = ODE2(*args)
        except AssertionError:
            return 1e30
        
        x = odemodel.peak() / xunit
        if x<=0: return 1e30
        y = odemodel.n(x) / yunit
        if y<=0: return 1e30
        
        # weight location and height of peak, innov density, slope at innov
        return ((y-peaky)**2 + (x+xoff-peakx)**2 +
                (odemodel.n0/yunit - n0)**2 +
                (odemodel.d_complex(0).real*xunit/yunit - s0)**2)
    
    # convert parameters to log space to handle cutoff at 0
    soln = minimize(cost, [initial_guess[0]]+np.log(initial_guess[1:]).tolist(), method='powell')
    if full_output:
        return np.exp(soln['x']), soln
    return np.exp(soln['x'])

def solve_min_rd(G, ro, re, I, Q=2, a=1.,
                 tol=1e-10,
                 initial_guess=None,
                 full_output=False,
                 return_neg=False):
    """Solve for minimum rd that leads to divergent lattice, i.e. when denominator
    for L goes to 0 for a fixed re.
    
    Parameters
    ----------
    G : float
    ro : float
    re : float
    I : float
    Q : float, 2
    a : float, 1.
    tol : float, 1e-10
    initial_guess : float, None
    full_output : bool, False
    return_neg : bool, False
    
    Returns
    -------
    float
    dict (optional)
    """
    
    # use linear guess as default starting point
    initial_guess = initial_guess or (2*re+ro)
    
    # analytic continuation from the collapsed curve
    if re==0:
        if full_output:
            return 0., {}
        return 0.

    def cost(rd):
        n0 = (ro / re / I)**(1/a)
        z = (re/(Q-1)-rd) / re / (I*n0**a - 1/(Q-1)) 
        C = z**-1 * (np.exp(-z) - 1 + z)
        return (rd + ro  - re / (Q-1) * (1 + 1/(1-C)))**2
    soln = minimize(cost, initial_guess)
    
    # if it didn't converge, return nan
    if soln['fun'] > tol:
        if full_output:
            return np.nan, soln
        return np.nan
    # neg values should be rounded up to 0
    elif not return_neg and soln['x'][0] < 0:
        if full_output:
            return 0., soln
        return 0.
    
    if full_output:
        return soln['x'][0], soln
    return soln['x'][0]

def solve_max_rd(G, ro, re, I, Q=2, a=1.,
                 tol=1e-10,
                 initial_guess=None,
                 full_output=False,
                 return_neg=False):
    """Solve for max rd that precedes collapsed lattice, i.e. when we estimate L~1.
    
    Parameters
    ----------
    G : float
    ro : float
    re : float
    I : float
    Q : float, 2
    a : float, 1.
    tol : float, 1e-10
    initial_guess : float, None
    full_output : bool, False
    return_neg : bool, False
    
    Returns
    -------
    float
    dict (optional)
    """
    
    # use linear guess as default starting point
    initial_guess = initial_guess or (G * (ro/re/I)**(-1/a) - ro + 2*re)
    
    # analytic continuation from the collapsed curve
    if re==0:
        if full_output:
            return 0., {}
        return 0.

    def cost(rd):
        n0 = (ro / re / I)**(1/a)
        z = (re/(Q-1)-rd) / re / (I*n0**a - 1/(Q-1)) 
        C = z**-1 * (np.exp(-z) - 1 + z)
        return (rd + ro  - re / (Q-1) * (1 + 1/(1-C)) - G*(re*I)**(1./a))**2
    soln = minimize(cost, initial_guess)
    
    # if it didn't converge, return nan
    if soln['fun'] > tol:
        if full_output:
            return np.nan, soln
        return np.nan
    # neg values should be rounded up to 0
    elif not return_neg and soln['x'][0] < 0:
        if full_output:
            return 0., soln
        return 0.
    
    if full_output:
        return soln['x'][0], soln
    return soln['x'][0]

def flatten_phase_boundary(G, ro, re, I, Q, a,
                           re_data, rd_data,
                           poly_order=5):
    """Fit polynomial to min rd growth curve and use inverse transform to map
    relative to 1:1 line.

    Parameters
    ----------
    G: float
    ro : float
    re : ndarray
    I : float
    Q : float
    a : float
    re_data : ndarray
    rd_data : ndarray
    poly_order : int, 5

    Returns
    -------
    float
    float
    """

    if not hasattr(re_data, '__len__'):
        re_data = np.array([re_data])
    if not hasattr(rd_data, '__len__'):
        rd_data = np.array([rd_data])

    y = np.array([solve_min_rd(G, ro, re_, I, Q=Q, return_neg=True) for re_ in re])
    y = y[1:]
    x = re[1:][~np.isnan(y)]
    y = y[~np.isnan(y)]

    # rescale x to interval [0,1]
    newx = re_data / x[-1]
    x /= x[-1]

    p = np.poly1d(np.polyfit(x, y, poly_order))
    
    newy = np.zeros_like(rd_data)
    for i, y_ in enumerate(rd_data):
        roots = (p-y_).roots
        ny = roots[np.abs(roots-y_).argmin()]
        if not ny.imag==0:
            newy[i] = np.nan
        else:
            newy[i] = ny
    return newx, newy.real

def L_denominator(ro, rd, Q=2):
    """Denominator for 2nd order calculation of L at stationarity. Parameters are
    rescaled rates.

    When denom goes negative, we have no physical solution for L, i.e. it is
    either non-stationary or it diverges there is a weird boundary at ro=1.

    Parameters
    ----------
    ro : float or ndarray
        Rescaled obsolescence rate ro/re
    rd : float or ndarray
        Rescaled death rate rd/re
    Q : float, 2
        Branching parameter. Q=2 is linear model.

    Returns
    -------
    ndarray
    """
    
    if not hasattr(ro, '__len__'):
        ro = np.array([ro])
    assert (ro>=0).all()
    if not hasattr(rd, '__len__'):
        rd = np.array([rd])
    assert (rd>=0).all()
    assert Q>=2

    z = -(1/(Q-1) - rd) / (1/(Q-1) - ro)
    C = np.zeros_like(z)

    # handle numberical precision problems with z
    # analytic limit
    zeroix = z==0

    # large z approximation
    largeix = z < -200
    C[largeix] = np.inf
    
    remainix = (~zeroix) & (~smallix) & (~largeix)
    C[remainix] = (np.exp(-z[remainix]) - 1 + z[remainix]) / z[remainix]

    return ro/(1-C) + rd - (1+1/(1-C)) / (Q-1)

def collapse_condition(ro, rd, G, I, Q=2, allow_negs=False):
    """When this goes to 0, we are at a collapsed boundary. Coming from first order
    ODE approximation.

    Parameters
    ----------
    ro : float or ndarray
        ro/re
    rd : float or ndarray
        rd/re
    G : float
        G/re
    I : float
    Q : float, 2
    allow_negs : bool, False

    Returns
    -------
    ndarray
    """
    
    if not hasattr(ro, '__len__'):
        ro = np.array([ro])
    if not hasattr(rd, '__len__'):
        rd = np.array([rd])
    if not allow_negs:
        assert (ro>=0).all()
        assert (rd>=0).all()
    assert Q>=2

    z = -(1/(Q-1) - rd) / (1/(Q-1) - ro)
    C = np.zeros_like(z)

    # handle numerical precision problems with z
    # analytic limit
    zeroix = z==0

    # large z approximation
    largeix = z < -200
    C[largeix] = np.inf
    
    remainix = (~zeroix) & (~largeix)
    C[remainix] = (np.exp(-z[remainix]) - 1 + z[remainix]) / z[remainix]

    return rd + ro / (1-C) - (1+1/(1-C)) / (Q-1) - G*I/ro

def weibull_b_k(el, n=100):
    """Average obsolescence tail from Weibull process as a function of exponent k.
    
    k=1 corresponds to exponential decay.
    
    Parameters
    ----------
    el : float
        Decay rate. el=0 is infinitely slow decay.
    n : int, 100
        Number of terms to include in series approximation.
    
    Returns
    -------
    func
    """
    from scipy.special import gamma, gammaincc
    func = lambda k: np.sum([b * np.exp(-(b*el)**k) *
                             np.exp(-gamma(1/k) * gammaincc(1/k, (b*el)**k)/(el*k))
                             for b in range(n)])
    return np.vectorize(func)

def weibull_b_el(k, n=100):
    """Average obsolescence tail from Weibull process as a function of correlation
    time el.
    
    Parameters
    ----------
    k : float
        Exponent. k=1 corresponds to exponential decay.
    n : int, 100
        Number of terms to include in series approximation.
    
    Returns
    -------
    func
    """
    from scipy.special import gamma, gammaincc

    func = lambda el: np.sum([b * np.exp(-(b*el)**k) *
                              np.exp(-gamma(1/k) * gammaincc(1/k, (b*el)**k)/(el*k))
                              for b in range(n)])
    return np.vectorize(func)



# ======= #
# Classes #
# ======= #
class IterativeMFT():
    def __init__(self, G, ro, rd, I, alpha=1., Q=2):
        """Class for calculating discrete MFT quantities.

        Parameters
        ----------
        G : float
        ro : float
        rd : float
        I : float
        alpha : float, 1.
            Cooperativity.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        
        assert alpha==1
        assert Q>=2

        self.ro = ro
        self.G = G
        self.rd = rd
        self.I = I
        self.alpha = alpha
        self.Q = Q
        self.n0 = ro/I  # stationary density
        
        # where is this criterion from?
        #assert (2/(Q-1) * re - rd) * self.n0 - re*I*self.n0**2 <= 0, "Stationary criterion unmet."
        
        # MFT guess for L, which we will refine using tail convergence criteria
        try:
            self.L0 = G * I / (ro * (rd + ro - 2/(Q-1)))
        except ZeroDivisionError:
            self.L0 = np.inf
        
        # for handling infinite L
        if ~np.isfinite(self.L0) or self.L0<0:
            self.L0 = 100_000
            self.L = self.L0
            self.n = self.iterate_n()
        else:
            self.min_L(self.L0)

    def min_L(self, L0, mx_decimal=10):
        """Lower L slower and slower while keeping tail positive in order to
        find value of L that solves iterative solution.

        As a heuristic, we keep the numerically calculated value of the tail
        positive instead of the self-consistent iterative value, which seems to
        behave worse (probably because it depends on the estimate of n[-2],
        which itself can be erroneous).
        
        Parameters
        ----------
        L0 : float
            L value to start with as a guess.
        mx_decimal : int, 10
            No. of decimal places to fit to.
        
        Returns
        -------
        float
            Refined estimate of lattice width L.
        """
        
        assert mx_decimal < 14, "Exceeding floating point precision."
        assert L0 > 2, L0

        ro = self.ro
        G = self.G
        rd = self.rd
        I = self.I
        Q = self.Q
        n0 = self.n0
        
        L = np.ceil(L0)
        decimal = 1

        nfun = self.iterate_n(L)
        # check that tail is positive; in case it is not, increase starting guess for L a few times
        counter = 0
        while nfun[-1] < 0: #(I * n0 * nfun[-2] + G/re/L / (I * n0 + rd/re)) < 0:
            L += 1
            nfun = self.iterate_n(L)
            counter += 1
            assert counter < 1e3
        
        while decimal <= mx_decimal:
            # ratchet down til the tail goes the wrong way
            while nfun[-1] > 0 and L>0: #(I * n0 * nfun[-2] + G/re/L / (I * n0 + rd/re)) > 0:
                L -= 10**-decimal
                nfun = self.iterate_n(L)
            L += 10**-decimal  # oops, go back up
            nfun = self.iterate_n(L)
            decimal += 1

        self.L, self.n = L, nfun
        return L, nfun

    def iterate_n_high_prec(self, L=None, iprint=False):
        """Iterative solution to occupancy number with high precision. See NB II pg. 118."""
        
        import mpmath as mp
        mp.dps = 30

        ro = self.ro
        G = self.G
        rd = self.rd
        I = self.I
        n0 = self.n0
        L = L or self.L
        Q = self.Q

        n = mp.matrix(int(L)+1, 1)
        n[0] = n0
        if len(n) > 1:
            # assumption about n[-1]=0 gives n[1]
            n[1] = (Q-1) * (I * n0**2 + (rd * n0 - G / L))

            for i in range(2, len(n)):
                n[i] = (Q-1) * (I * n0 * (n[i-1] - n[i-2]) + (rd * n[i-1] - G / L))

        return np.array([float(i) for i in n])

    def iterate_n(self, L=None, iprint=False):
        """Iterative solution to occupancy number. See NB II pg. 118."""
        
        ro = self.ro
        G = self.G
        rd = self.rd
        I = self.I
        n0 = self.n0
        L = L or self.L
        Q = self.Q

        n = np.zeros(int(L)+1)
        n[0] = n0
        if n.size > 1:
            # assumption about n[-1]=0 gives n[1]
            n[1] = (Q-1) * (I * n0**2 + (rd * n0 - G / L))
            
            # handle overflow separately
            overflow = False
            for i in range(2, n.size):
                n[i] = (Q-1) * (I * n0 * (n[i-1] - n[i-2]) + (rd * n[i-1] - G / L))
                if abs(n[i]) > 1e200:
                    overflow = True
                    break
            if overflow:
                n[i:] = np.nan

        return n

    def estimate_L(self, x=2):
        """Invert stationary equation to solve for L.

        Parameters
        ----------
        x : int, 2
            Lattice point to use for estimating L. Too close to the right side
            boundary leads to large numerical errors.
        """

        ro = self.ro
        G = self.G
        rd = self.rd
        I = self.I
        Q = self.Q
        n0 = self.n0
        n = self.n
        
        return G / (I * n[0] * (n[x-1]-n[x-2]) + rd * n[x-1] - n[x]/(Q-1))
#end IterativeMFT



class FlowMFT():
    def __init__(self,
                 G=None,
                 ro=None,
                 rd=None,
                 I=None,
                 dt=None,
                 L_method=2,
                 alpha=1.,
                 Q=2):
        """Class for calculating discrete MFT quantities by running dynamics.

        Parameters
        ----------
        G : float
        ro : float
        rd : float
        I : float
        dt : float
        L_method : int
            0, naive
            1, corrected
            2, ODE2
        alpha : float, 1.
            Cooperativity parameter.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        assert alpha>0
        assert Q>=2

        self.G = G
        self.ro = ro
        self.rd = rd
        self.I = I
        self.dt = dt
        self.alpha = float(alpha)
        self.Q = Q

        self.n0 = (ro/I)**(1/alpha)
        
        try:
            if L_method==0:
                self.L = G / (self.n0 * (rd - (1+1/(Q-1)) + I*self.n0**self.alpha))
            elif L_method==1:
                self.L = L_1ode(G, ro, rd, I, alpha=alpha, Q=Q)
            elif L_method==2:
                self.L = ODE2(G, ro, rd, I, alpha=alpha, Q=Q).L
            else: raise NotImplementedError
        except ZeroDivisionError:
            self.L = np.inf
        if self.L <= 0:
            self.L = np.inf
        else:
            self.L = min(self.L, 10_000)
    
    def update_n(self, L=None):
        """Update occupancy number for a small time step self.dt.

        Parameters
        ----------
        L : float, None
        """
        L = L or self.L

        # entrance
        dn = self.G / L 

        # innovation shift
        dn -= self.I * self.n_[0]**self.alpha * self.n_
        dn[1:] += self.I * self.n_[0]**self.alpha * self.n_[:-1]

        # expansion
        dn[:-1] += 1 / (self.Q-1) * self.n_[1:]

        # death
        dn -= self.rd * self.n_

        self.n_ += dn * self.dt
        return dn

    def solve_stationary(self,
                         tol=1e-5,
                         T=5e4,
                         L=None,
                         n0=None,
                         iprint=False):
        """Run til stationary state is met using convergence criterion.

        Parameters
        ----------
        tol : float, 1e-5
            Max absolute change permitted per lattice site per unit time.
        T : float, 5e4
        L : int, None
        n0 : ndarray, None
        iprint : bool, False

        Returns
        -------
        int
            Flag indicating if problem converged
                (0) to stationary solution with correct innov. density
                (1) to stationary soln with wrong innov. density
                (2) or did not converge
        float
            Maximum absolute difference between last two steps of simulation.
        """
        L = L or self.L
        # simply impose a large upper limit for infinite L
        if not np.isfinite(L):
            warnings.warn("Imposing upper limit on L to be 1e4.")
            L = 10_000
        elif L<1:
            raise Exception("Lattice L<1.")
        
        if n0 is None and 'n' in self.__dict__ and self.n_.size==(L+1):
            n0 = self.n_[:-1]
        elif n0 is None:
            n0 = np.ones(int(L))/2
        self.n_ = n0.copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                # fast version of loop til stationarity, uses both jit_update_n and jit_while_loop
                dn, counter = jit_while_loop(self.n_, self.G, self.ro, self.rd, 1,
                                             self.I, self.alpha, self.Q, self.dt, tol, T, L)
                
                if (self.dt*counter) >= T:
                    flag = 2
                elif np.abs(self.n_[0]-self.n0)/self.n_.max() < 1e-5:  # relative error
                    flag = 0
                else:
                    flag = 1
                
                mx_err = np.abs(dn).max()
                self.n_ = np.append(self.n_, 0)
                # interpolate discrete lattice solution
                self.n = interp1d(np.arange(self.n_.size), self.n_,
                                  kind='cubic',
                                  bounds_error=False,
                                  fill_value=0)
            except RuntimeWarning:
                self.n_[:] = np.nan
                self.n = lambda x: np.zeros(x.size) + np.nan if hasattr(x, '__len__') else np.nan
                flag = 2
                mx_err = np.nan

        return flag, mx_err

    def run(self, T, save_every, L=None, iprint=False):
        """
        Parameters
        ----------
        T : int
        save_every : int
        L : int, None
        iprint : bool, False

        Returns
        -------
        list of list
        """
        
        L = L or self.L
        # simply impose a large upper limit for infinite L
        if not np.isfinite(L):
            L = 10_000

        t = []
        n0 = np.ones(int(L))/2
        self.n_ = n0.copy()
        snapshot_n = []
        counter = 0
        while (self.dt*counter) < T:
            self.update_n(L)
            
            if np.isclose((self.dt*counter)%save_every, 0, atol=self.dt/10, rtol=0):
                if iprint: print(f"Recording {dt*counter}...")
                t.append(counter * self.dt)
                snapshot_n.append(self.n_.copy())
            counter += 1

        return snapshot_n, t

    def solve_n0(self, L):
        """Quadratic equation solution for n0."""
            
        assert self.alpha==1, "This does not apply for alpha!=1."
        assert self.Q==2
        G = self.G
        rd = self.rd
        I = self.I

        return ((2-rd) + np.sqrt((2-rd)**2 + 4*I*G/L)) / 2 / I

    def corrected_n0(self):
        """Stil figuring out the logic of this.
        """
        assert self.alpha==1 and self.Q==2
        G = self.G
        I = self.I
        re = 1
        rd = self.rd
        ro = self.ro
        n0 = self.n0
        
        n02 = -(re-rd) * n0 / ((re-rd)/2 - re*I*n0 + re)
        # this is our estimate of the correction to the slope, which gives corrections to the intercept
        z = - (re - rd) / ((re-rd)/2 + re*(1 - I*n0))

        return n0 + (n02 * z**-2. * (-1 + z + np.exp(-z)) if z!=0 else 0.)

@njit
def jit_update_n(n, G, ro, rd, re, I, alpha, Q, L, dt):
    """Update occupancy number for a small time step self.dt.

    Parameters
    ----------
    n : ndarray
    G : float
    ro : float
    rd : float
    re : float
    I : float
    alpha : float
    Q : float
    L : float, None
    dt : float
    """
    
    # entrance
    dn = np.zeros(n.size) + G/L

    # innovation shift
    dn -= re * I * n[0]**alpha * n
    dn[1:] += re * I * n[0]**alpha * n[:-1]

    # expansion
    dn[:-1] += re / (Q-1) * n[1:]

    # death
    dn -= rd * n

    n += dn * dt
    return dn

@njit
def jit_while_loop(n, G, ro, rd, re, I, alpha, Q, dt, tol, T, L):
    counter = 0
    dn = np.zeros(n.size) + dt*tol*1.1
    while (dt*counter) < T and np.abs(dn).max() > (dt*tol):
        dn = jit_update_n(n, G, ro, rd, re, I, alpha, Q, L, dt)
        counter += 1
    return dn, counter
#end FlowMFT



class ODE2():
    def __init__(self, G, ro, rd, I, L=None, alpha=1., Q=2):
        """Class for second-order analytic solution to MFT.

        Parameters
        ----------
        G : float
            Rescaled growth rate.
        ro : float
            Rescaled obsolescence rate.
        rd : float
            Rescaled death rate.
        I : float
            Innovation rate.
        L : float, None
        alpha : float, 1.
            Cooperativity.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        self.G = float(G)
        self.ro = float(ro)
        self.rd = float(rd)
        self.I = float(I)
        self.alpha = alpha
        self.Q = Q
        self.n0 = (ro/I)**(1/alpha)  # stationary density
        
        self.L = self.solve_L(L)
    
    def n(self, x, L=None, return_im=False, method=2, zero_bound=True):
        """Interpolated occupancy function.

        Parameters
        ----------
        x : ndarray
            Lattice values for which to compute density.
        return_im : bool, False
        method : int, 2
            Either 1 or 2.
        zero_bound : bool, True
            If True, bound all function values outside of the range [0, L] to 0.

        Returns
        -------
        ndarray
        """
        L = L if not L is None else self.L
        if not hasattr(x, '__len__'):
            x = np.array([x])

        if method==1:
            # cleaned up output from mathematica
            assert self.alpha==1

            ro = self.ro
            G = self.G
            rd = self.rd
            I = self.I
            re = 1

            z = -re**2 - 4*re*ro + ro**2 + 2*rd*(re+ro)
            sol = (G / ((np.exp(2*sqrt(z)/(re+ro))-1) * L * (re-rd)) *
                   (1 - np.exp(2*sqrt(z)/(re+ro)) + np.exp(-(sqrt(z)*(x-1)+re*x-ro*(1+x)+re)/(re+ro)) -
                    np.exp((-re*x+ro*(1+x)+sqrt(z)*(1+x)-re)/(re+ro)) +
                    (L*(re-rd)*ro/(G*re*I)+1) * (np.exp((-re*x+ro*x+sqrt(z)*(2+x))/(re+ro)) -
                                                 np.exp((-re*x+(ro-sqrt(z))*x)/(re+ro)))))
            y = sol.real
            if hasattr(y, '__len__'):
                y[(x<0)|(x>L)] = 0
            else:
                if x<0 or x>L:
                    y = 0
            return y
        
        elif method==2:
            # hand-written soln
            # rescale params in units of re
            ro = self.ro
            G = self.G
            rd = self.rd
            I = self.I
            Q = self.Q
            
            # compute eigenvalues of characteristic soln
            z = (1/(Q-1)-ro)**2 - 2 * (1/(Q-1)-rd) * (1/(Q-1)+ro)
            lp = (ro-1/(Q-1) + sqrt(z)) / (1/(Q-1)+ro)
            lm = (ro-1/(Q-1) - sqrt(z)) / (1/(Q-1)+ro)
             
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # constants for homogenous terms
                # divergences occur at special values of ro and rd
                try:
                    A = ((G * np.exp((-sqrt(z)+ro-1/(Q-1)) / (1/(Q-1)+ro)) / (L * (1/(Q-1)-rd)) -
                         (2*G*(1/(Q-1)+ro) / (L*((1/(Q-1)-ro)**2 - z)) + (ro/I)**(1./self.alpha))) / 
                        (np.exp(-2*sqrt(z) / (1/(Q-1)+ro)) - 1))
                except ZeroDivisionError:
                    if np.exp((ro - 1 - sqrt(z)) / (1 + ro)) > 1:
                        A = np.inf
                    elif np.exp((ro - 1 - sqrt(z)) / (1 + ro)) < 1:
                        A = -np.inf
                    else:
                        A = 0
                try:
                    B = ((G * np.exp(( sqrt(z)+ro-1/(Q-1)) / (1/(Q-1)+ro)) / (L * (1/(Q-1)-rd)) -
                         (2*G*(1/(Q-1)+ro) / (L*((1/(Q-1)-ro)**2 - z)) + (ro/I)**(1./self.alpha))) / 
                        (np.exp( 2*sqrt(z) / (1/(Q-1)+ro)) - 1))
                except ZeroDivisionError:
                    if np.exp((ro - 1 + sqrt(z)) / (1 + ro)) > 1:
                        B = np.inf
                    elif np.exp((ro - 1 + sqrt(z)) / (1 + ro)) < 1:
                        B = -np.inf
                    else:
                        B = 0
 
                # particular soln using numerical precision tricks with logsumexp
                #y = A * np.exp(lp * x) + B * np.exp(lm * x) - G/L/(1/(Q-1)-rd)
                logexp1 = np.log(A * np.exp(lp * x))
                logexp2 = np.log(B * np.exp(lm * x))
                logexp3 = np.zeros_like(x) + np.log(G) - np.log(L)  # constant over all x
                if (1/(Q-1)-rd)==0:
                    s3 = -np.inf
                else:
                    s3 = -1/(1/(Q-1)-rd)
                y = logsumexp(np.vstack([logexp1, logexp2, logexp3]),
                              axis=0,
                              b=np.array([1, 1, s3])[:,None],
                              return_sign=True)
                y = np.exp(y[0]) * y[1]
            
            if hasattr(y, '__len__'):
                y[(x<0)|(x>L)] = 0
            else:
                if x<0 or x>L:
                    y = 0 + 0j
            
            if return_im:
                return y.real, y.imag
            return y.real
        else: raise NotImplementedError

    def solve_L(self, L0=None, full_output=False, method=3):
        """Solve for stationary value of L that matches self-consistency condition,
        i.e. analytic solution for L should be equal to the posited value of L.

        Parameters
        ----------
        L0 : float, None
            Initial guess.
        full_output : bool, False
        method : int, 3
            1: 'mathematica'
            2: 'hand'
            3: 'hand' but w/ boundary condition at x=L
            Use formulation returned by mathematica or hand written solution. Hand
            written solution is more numerically stable.

        Returns
        -------
        float
        dict (optional)
        """
        G = self.G
        ro = self.ro
        rd = self.rd
        I = self.I
        Q = self.Q
        re = 1

        # if not provided, use the iterative method to initialize the search
        L0 = L0 or L_linear(G, ro, rd, I, alpha=self.alpha, Q=Q)
        # this is infinite limit, don't expect a good solution
        if L0 < 0: L0 = 2e5
        if np.isinf(L0): L0 = 2e5

        if method==1:
            assert self.alpha==1
            assert self.Q==2

            # analytic eq for L solved from continuum formulation in Mathematica
            # this formulation has much bigger numerical errors (sometimes)
            a = -re**2 - 4*re*ro + ro**2 + 2*rd*(re+ro)
            num = lambda x:(np.exp(-(re-ro)/(re+ro)) * (np.exp(-sqrt(a)*(x-1)/(re+ro)) -
                                                        np.exp(sqrt(a)*(x+1)/(re+ro))) + 
                            np.exp((re-ro)*x/(re+ro)) * (1-np.exp(2*sqrt(a)/(re+ro))) -
                            np.exp(-sqrt(a)*x/(re+ro)) +
                            np.exp(sqrt(a)*(2+x)/(re+ro)))
            den = lambda x:(np.exp(-sqrt(a)*x/(re+ro)) -
                            np.exp(sqrt(a)*(2+x)/(re+ro))) * (re-rd)*ro / (G*re*I)

            statL = lambda x:num(x) / den(x)
            soln = minimize(lambda x:(statL(x).real - x)**2, L0, tol=1e-10, bounds=[(0,np.inf)])
            if full_output:
                return soln['x'][0], soln
            return soln['x'][0]

        elif method==2: 
            def cost(args):
                L = args[0]
                return self.n(-1, L, method=2)**2
            
            soln = minimize(cost, L0, tol=1e-10, bounds=[(0,np.inf)])

            if full_output:
                return soln['x'][0], soln
            return soln['x'][0]

        elif method==3: 
            def cost(args):
                L = args[0]
                return self.n(L, L, method=2)**2
            
            soln = minimize(cost, L0, tol=1e-10, bounds=[(0,np.inf)])

            if full_output:
                return soln['x'][0], soln
            return soln['x'][0]

        else: raise NotImplementedError
    
    def check_stat(self, x=1):
        """Violation of stationarity condition by checking accuracy of iterative
        solution.

        Returns
        -------
        float
            n(x) - [n(x) calculated with n(x-1)]
        """
        
        assert x>=1

        G = self.G
        ro = self.ro
        rd = self.rd
        I = self.I
        L = self.L
        
        n0 = self.n(0) 
        if x==1:
            return self.n(x) - I * n0**2 - rd*n0 + G/L
        return self.n(x) - (I * n0 * (self.n(x-1) - self.n(x-2)) + rd*self.n(x-1) - G/L)
    
    def slope(self):
        """Slope at x=0."""

        ro = self.ro
        G = self.G
        rd = self.rd
        I = self.I
        L = self.L
 
        return ro**2 / I + (rd-1) * ro / I - G / L

    def d_complex(self, x):
        """Complex derivative.
        
        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """
        
        # transform into normal parameters
        rd = self.rd
        ro = self.ro
        G = self.G
        I = self.I
        L = self.L
        a = -1 + (ro - 4) * ro + 2 * rd * (1 + ro) 
        sqrta = sqrt(a)

        return ((G/L/(rd-1) * ((-1 + ro + sqrta)/(1+ro) *
                                      np.exp((-1 + sqrta + x*(sqrta-1) + ro*(1+x)) / (1+ro)) -
                               (-1 + ro - sqrta)/(1+ro) *
                                      np.exp((-1 + sqrta - x*(1+sqrta) + ro*(1+x)) / (1+ro))) + 
                        np.exp((-1 + ro - sqrta) * x / (1+ro)) * (G/L/(rd-1) - ro/I) * (-1+ro-sqrta)/(1+ro)+ 
                        np.exp((2 * sqrta + (ro-1+sqrta)*x) / (1+ro)) * (-G/L/(rd-1) + ro/I) * (-1+ro+sqrta) /
                                (1+ro)) / 
                      (-1 + np.exp(2 * sqrta/(1+ro))))

    def peak(self, initial_guess=None, full_output=False):
        """Solve for peak by finding root in derivative.
        
        Parameters
        ----------
        full_output : bool, False

        Returns
        -------
        float
        dict (optional)
        """
       
        if initial_guess is None:
            initial_guess = 2/3 * self.L

        # transform into normal parameters
        rd = self.rd
        ro = self.ro
        G = self.G
        I = self.I
        L = self.L
        a = -1 + (ro - 4) * ro + 2 * rd * (1 + ro) 
        sqrta = sqrt(a)

        # need a clever transformation of coordinates to allow the minimizer to find the soln
        soln = minimize(lambda y:np.abs(self.d_complex(L-np.exp(y)).real), np.log(L-initial_guess),
                        bounds=[(-np.inf,np.log(L/2))],
                        method='powell')

        if full_output:
            return L - np.exp(soln['x'][0]), soln
        return L - np.exp(soln['x'][0])
#end ODE2


class UnitSimulator(FlowMFT):
    def __init__(self,
                 G=None,
                 ro=None,
                 re=None,
                 rd=None,
                 I=None,
                 dt=None,
                 L_method=2,
                 alpha=1.,
                 Q=2,
                 rng=np.random):
        """Independent unit simulation innovating agents. This is the simplest
        implementation possible that only keeps track of the occupancy number and
        processes to order dt.

        Parameters
        ----------
        G : float
            Growth rate.
        re : float
            Replication rate.
        rd : float
            Death rate.
        I : float
            Innovation rate.
        dt : float
            Time step.
        L_method : int
            0, naive
            1, corrected
            2, ODE2
        alpha : float, 1.
            Cooperativity parameter.
        Q : int, 2
            Bethe lattice branching ratio.
        rng : RandomState, np.random
        """
        assert alpha>0
        assert Q>=2

        self.G = G
        self.ro = ro
        self.re = re
        self.rd = rd
        self.I = I
        self.dt = dt
        self.alpha = float(alpha)
        self.Q = Q
        self.rng = rng

        self.n0 = (ro/re/I)**(1/alpha)
        
        try:
            if L_method==0:
                self.L = G / (self.n0 * (rd - (1+1/(Q-1))*re + re*I*self.n0**self.alpha))
            elif L_method==1:
                self.L = L_1ode(G, ro, re, rd, I, alpha=alpha, Q=Q)
            elif L_method==2:
                self.L = ODE2(G, ro, re, rd, I, alpha=alpha, Q=Q).L
            else: raise NotImplementedError
        except ZeroDivisionError:
            self.L = np.inf
        if self.L <= 0:
            self.L = np.inf
        else:
            self.L = min(self.L, 10_000)

    def simulate(self, T,
                 reset_rng=False,
                 jit=True,
                 occupancy=None,
                 no_expansion=False,
                 return_fronts=False):
        """
        NOTE: dt must be small enough to ignore coincident events.

        Parameters
        ----------
        T : int
            Simulation time to run.
        reset_rng : bool, False
        jit : bool, True
        occupancy : list, None
            Feed in a starting occupancy on which to run dynamics.
        no_expansion : bool, False
            When True, expansion term is removed from simulation. This only
            works without occupancy.
        return_fronts : bool, False
            If True, return twople of obs and innov front location at end of
            simulation.

        Returns
        -------
        list
            occupancy at each site
        """
        G = float(self.G)
        ro = float(self.ro)
        rd = float(self.rd)
        re = float(self.re)
        I = float(self.I)
        a = float(self.alpha)
        dt = float(self.dt)
        
        assert (G * dt)<1
        assert (rd * dt)<1
        assert (re * dt)<1
        assert (ro * dt)<1
        
        if jit and occupancy is None:
            if reset_rng: np.random.seed()
            if no_expansion:
                sim_output = jit_unit_sim_loop_no_expand(T, dt, G, ro, re, rd, I, a)
            else:
                sim_output = jit_unit_sim_loop(T, dt, G, ro, re, rd, I, a)
            if return_fronts:
                return sim_output
            return sim_output[0]
        elif jit and not occupancy is None:
            if reset_rng: np.random.seed()
            occupancy = List(occupancy)
            sim_output = jit_unit_sim_loop_with_occupancy(occupancy, T, dt, G, ro, re, rd, I, a)
            if return_fronts:
                return list(sim_output[0]), sim_output[1], sim_output[2]
            return list(sim_output[0])
 
        if reset_rng: self.rng.seed()
        counter = 0
        occupancy = [0]
        obs_loc = 0  # anything below this value is empty 
        innov_loc = 1  # anything at this value or greater is the adjacent possible
        while (counter * dt) < T:
            # innov
            innov = False
            if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
                occupancy.append(1)
                innov_loc += 1
                innov = True

            # obsolescence
            if len(occupancy) > 1 and np.random.rand() < (ro * dt):
                occupancy.pop(0)
                obs_loc += 1
            
            # from right to left b/c of expansion
            for x in range(len(occupancy)-1, -1, -1):
                # expansion (fast approximation)
                if x < (len(occupancy)-1-innov):
                    if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                        occupancy[x+1] += 1

               # death (fast approximation)
                if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                    occupancy[x] -= 1

                # start up (remember that L is length of lattice on x-axis, s.t. L=0 means lattice has one site)
                if len(occupancy)==1:
                    occupancy[x] += 1
                elif np.random.rand() < (G / len(occupancy) * dt):
                    occupancy[x] += 1

            counter += 1
        if return_fronts:
            return occupancy, (obs_loc, innov_loc)
        return occupancy

    def parallel_simulate(self, n_samples, T, **kwargs):
        """
        Parameters
        ----------
        n_samples : int
            Number of parallel simulations to run.
        T : int
            Simulation time.
        **kwargs
        
        Returns
        -------
        list of lists
            Each inner list is an occupancy list.
        """
        with Pool() as pool:
            if 'return_fronts' in kwargs.keys() and kwargs['return_fronts']:
                self.occupancy, obs_loc, innov_loc = list(zip(*pool.map(lambda args: self.simulate(T,
                                                                                                   True,
                                                                                                   **kwargs),
                                                                        range(n_samples))))
                return self.occupancy, obs_loc, innov_loc

            self.occupancy = list(pool.map(lambda args: self.simulate(T, True, **kwargs), range(n_samples)))
            return self.occupancy

    def mean_occupancy(self,
                       occupancy=None,
                       width=np.inf,
                       norm_err=True,
                       rescale=False):
        """
        Parameters
        ----------
        occupancy : list of list, None
        width : tuple, np.inf
            Range of permitted widths to average. When this is a float or int, then
            only sim results with exactly that width are returned.
        norm_err : bool, True
        rescale : bool, False
            If True, rescale density by cooperativity before taking mean.

        Returns
        -------
        ndarray
            Average occupancy.
        ndarray
            Standard deviation as error bars.
        """
        if occupancy is None:
            occupancy = self.occupancy
        if not hasattr(width, '__len__'):
            width = width, width
        assert width[0]<=width[1]

        if width[0]==np.inf:
            maxL = max([len(i) for i in occupancy])
            y = np.zeros(maxL)
            yerr = np.zeros(maxL)

            # first calculate the means
            counts = np.zeros(maxL, dtype=int)
            if rescale:
                for i in occupancy:
                    y[:len(i)] += np.array(i[::-1])**self.alpha
                    counts[:len(i)] += 1
            else:
                for i in occupancy:
                    y[:len(i)] += i[::-1]
                    counts[:len(i)] += 1
            y = y / counts

            # then calculate the std
            if rescale:
                for i in occupancy:
                    yerr[:len(i)] += (np.array(i[::-1])**self.alpha - y[:len(i)])**2
            else:
                for i in occupancy:
                    yerr[:len(i)] += (i[::-1] - y[:len(i)])**2

            yerr /= counts
            yerr = np.sqrt(yerr)
            if norm_err:
                return y, yerr / np.sqrt(counts)
            return y, yerr

        # case where there is some finite range of L to average over
        y = np.vstack([i[-width[0]:] for i in occupancy if width[0]<=len(i)<=width[1]])[:,::-1]

        if rescale:
            if norm_err:
                return (y**self.alpha).mean(0), (y**self.alpha).std(0) / np.sqrt(y.shape[0])
            return (y**self.alpha).mean(0), (y**self.alpha).std(0)

        if norm_err:
            return y.mean(0), y.std(0) / np.sqrt(y.shape[0])
        return y.mean(0), y.std(0)

    def rescale_factor(self, T, sample_size=1_000):
        """Rescaling factor needed to correct for bias in mean L. This comes from the
        fact that the mean field theory assumes that the average of the inverse of L
        is the same as the inverse of the average, which is not true given
        fluctuations in its value.

        The returned factor c can be used to modify the automaton model with the set
        of transformations
            x -> x * c
            G -> G * c
            n -> n / c
            I -> I / c
        Equivalently, we can transform the MFT with the inverse set of
        transformations
            x -> x / c
            G -> G / c
            n -> n * c
            I -> I * c
        
        Parameters
        ----------
        T : float
            Run time before sampling.
        sample_size : int
            No. of indpt. trajectories to use to estimate ratio.

        Returns
        -------
        float
        """
        G = float(self.G)
        ro = float(self.ro)
        re = float(self.re)
        rd = float(self.rd)
        I = float(self.I)
        a = float(self.alpha)
        dt = float(self.dt)
 
        occupancy = self.parallel_simulate(sample_size, T)
        L = np.array([(len(i)-1) for i in occupancy])

        odemodel = ODE2(G/re, ro/re, rd/re, I)

        return odemodel.L / L.mean(), occupancy
#end UnitSimulator

@njit
def jit_unit_sim_loop(T, dt, G, ro, re, rd, I, a):
    """
    Parameters
    ----------
    occupancy : numba.typed.ListType[int64]
    T : int
    dt : float
    ro : float
    G : float
    re : float
    rd : float
    I : float
    a : float
    """
    counter = 0
    occupancy = [0]
    obs_loc = 0
    innov_loc = 1
    while (counter * dt) < T:
        # innov
        innov = False
        if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
            occupancy.append(1)
            innov = True
            innov_loc += 1

        # obsolescence
        if len(occupancy) > 1 and np.random.rand() < (ro * dt):
            occupancy.pop(0)
            obs_loc += 1
        
        # from right to left b/c of expansion
        for x in range(len(occupancy)-1, -1, -1):
            # expansion (fast approximation)
            if x < (len(occupancy)-1-innov):
                if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                    occupancy[x+1] += 1

           # death (fast approximation)
            if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                occupancy[x] -= 1

            # start up (remember that L is length of lattice on x-axis, s.t. L=0 means lattice has one site)
            if np.random.rand() < (G / len(occupancy) * dt):
                occupancy[x] += 1

        counter += 1
    return occupancy, obs_loc, innov_loc

@njit
def jit_unit_sim_loop_with_occupancy(occupancy, T, dt, G, ro, re, rd, I, a):
    """
    Parameters
    ----------
    occupancy : numba.typed.ListType[int64]
    T : int
    dt : float
    ro : float
    G : float
    re : float
    rd : float
    I : float
    a : float
    """
    counter = 0
    obs_loc = 0
    innov_loc = 1
    while (counter * dt) < T:
        # innov
        innov = False
        if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
            occupancy.append(1)
            innov = True

        # from right to left b/c of expansion
        for x in range(len(occupancy)-1, -1, -1):
            # death (fast approximation)
            if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                occupancy[x] -= 1

            # expansion (fast approximation)
            if x < (len(occupancy)-1-innov):
                if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                    occupancy[x+1] += 1

            # start up
            if np.random.rand() < (G / len(occupancy) * dt):
                occupancy[x] += 1

        # obsolescence
        if len(occupancy) > 1 and np.random.rand() < (ro * dt):
            occupancy.pop(0)
        
        counter += 1
    return occupancy, obs_loc, innov_loc

@njit
def jit_unit_sim_loop_no_expand(T, dt, G, ro, re, rd, I, a):
    """Special case for testing. Without expansion.
    
    Parameters
    ----------
    occupancy : numba.typed.ListType[int64]
    T : int
    dt : float
    ro : float
    G : float
    re : float
    rd : float
    I : float
    a : float
    """
    counter = 0
    occupancy = [0]
    obs_loc = 0
    innov_loc = 1
    while (counter * dt) < T:
        # innov
        innov = False
        if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
            occupancy.append(1)
            innov = True
            innov_loc += 1

        # obsolescence
        if len(occupancy) > 1 and np.random.rand() < (ro * dt):
            occupancy.pop(0)
            obs_loc += 1
        
        # from right to left b/c of expansion
        for x in range(len(occupancy)-1, -1, -1):
           # death (fast approximation)
            if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                occupancy[x] -= 1

            # start up (remember that L is length of lattice on x-axis, s.t. L=0 means lattice has one site)
            if len(occupancy)==1:
                occupancy[x] += 1
            elif np.random.rand() < (G / (len(occupancy)-1) * dt):
                occupancy[x] += 1

        counter += 1
    return occupancy



class GridSearchFitter():
    def __init__(self, y, x=None):
        """Class for fitting ODE2 and FlowMFT to data using grid search.

        Parameters
        ----------
        y : ndarray
            Data y.
        x : ndarray, None
            Data x.
        """
        self.y = y
        self.x = x if not x is None else np.arange(y.size)
    
    def fit_length_scales(self, G, ro, rd, I,
                          primary='flow', log=False, L_scale=.5, T=5e4,
                          **model_kw):
        """Find optimal length scales for one set of model parameter values.

        Parameters
        ----------
        G : float
        ro : float
        rd : float
        I : float
        primary : str, 'ode'
            'ode' or 'flow'
        log : bool, False
            If True, fit on log scale.

        Returns
        -------
        float
            Optimal x-scaling
        float
            Optimal y-scaling.
        float
            Fit error.
        float
            Error between ODE and flow solutions. When this is too large,
            then the model solutions may not be trustworthy.
        """
        # scan thru parameter range, for each parameter combo,
        # find scaling of x and y axes that is optimal
        assert ro > 2-rd
        assert ro < 1 - rd/2 + np.sqrt((1-rd/2)**2 + G*I/4)
        assert 0 < L_scale

        # ODE model
        model1 = ODE2(G, ro, rd, I, **model_kw)
        assert model1.L > 5 and model1.L < 10_000, model1.L
        # flow model
        model2 = FlowMFT(G, ro, rd, I, dt=.1, **model_kw)
        model2.solve_stationary(T=T)
        if primary=='flow':
            temp = model1
            model1 = model2
            model2 = temp
        elif primary=='ode':
            pass
        else: raise NotImplementedError

        x_mod = np.arange(model1.L)
        y_mod1 = model1.n(x_mod)
        y_mod2 = model2.n(x_mod)

        # mismatch between two solution methods for the given parameter values
        mod_err = np.linalg.norm(y_mod1[:min(y_mod1.size, y_mod2.size)] -
                                 y_mod2[:min(y_mod1.size, y_mod2.size)])

        def cost(args):
            """
            Parameters
            ----------
            a : float
                Horizontal length scale.
            b : float
                Vertical length scale.
            """

            a, b = args
            if log:
                c = np.linalg.norm(np.log(self.y) - np.log(model1.n(self.x*a) * b))
                if np.isnan(c):
                    return 1e30
                return c
            return np.linalg.norm(self.y - model1.n(self.x*a) * b)

        # bounds on a such that model goes through at least half the data
        a_mx = model1.L / self.y.size * L_scale

        sol = minimize(cost, [a_mx/2,1], bounds=[(0, a_mx), (0, np.inf)])
        a, b = sol['x']

        return a, b, sol, mod_err
    
    def fit_length_scales_rev(self, G, ro, rd, I,
                              primary='flow',
                              log=False,
                              offset=0,
                              L_scale=.5,
                              ignore_nan=False,
                              **model_kw):
        """Find optimal length scales for one set of model parameter values but rescaling
        relative to the obsolescence front instead of the innovation front.
        
        Parameters
        ----------
        G : float
        ro : float
        rd : float
        I : float
        primary : str, 'ode'
            'ode' or 'flow'
        log : bool, False
            If True, fit on log scale.
        offset : float, 0.
            Amount to offset fixed point at obsolescence front to, i.e. when this is
            0, then the function starts at 0 on the left side (the obs front). This
            is a decision about whether the dnensity starts at 0 or not.
        L_scale : float, .5
            Fraction of model width that corresponds to data width, i.e. when
            L_scale=1/2, it means that the model length will be twice as long as the
            data width as is counted by the discrete lattice.
        ignore_nan : bool, False
        **model_kw

        Returns
        -------
        float
            Optimal x-scaling
        float
            Optimal y-scaling.
        float
            Fit error.
        float
            Error between ODE and flow solutions. When this is too large, then the
            model solutions may not be trustworthy.
        """
        # scan thru parameter range, for each parameter combo,
        # find scaling of x and y axes that is optimal
        assert ro > 2 - rd
        assert ro < 1 - rd/2 + np.sqrt((1-rd/2)**2 + G*I/4)
        assert offset >= 0
        assert 0 < L_scale

        # setup ODE and flow models
        model1 = ODE2(G, ro, rd, I, **model_kw)
        assert model1.L > 5 and model1.L < 10_000, model1.L
        model2 = FlowMFT(G, ro, rd, I, dt=.1, **model_kw)
        model2.solve_stationary()
        
        if primary=='flow':
            temp = model1
            model1 = model2
            model2 = temp
        elif primary=='ode':
            pass
        else: raise NotImplementedError

        x_mod = np.arange(model1.L)
        y_mod1 = model1.n(x_mod)
        y_mod2 = model2.n(x_mod)

        # mismatch between two solution methods
        mod_err = np.linalg.norm(y_mod1[:min(y_mod1.size, y_mod2.size)] -
                                 y_mod2[:min(y_mod1.size, y_mod2.size)])

        def cost(args):
            """
            Parameters
            ----------
            a : float
                Horizontal length scale.
            b : float
                Vertical length scale.
            """
            a, b = args
            if log:
                if ignore_nan:
                    if offset==0:
                        # dropping first value b/c fixing x=0 at 0 means that it doesn't matter in cost
                        c = np.sqrt(np.nansum((np.log(self.y[1:]) -
                                               np.log(model1.n(model1.L-self.x[1:]*a)) - b)**2))
                    else:
                        c = np.sqrt(np.nansum((np.log(self.y) -
                                               np.log(model1.n(model1.L-offset-self.x*a)) - b)**2))
                    if c==0 or np.isnan(c).all():
                        return 1e30
                else:
                    if offset==0:
                        # dropping first value b/c fixing x=0 at 0 means that it doesn't matter in cost
                        c = np.linalg.norm(np.log(self.y[1:]) - np.log(model1.n(model1.L-self.x[1:]*a)) - b)
                    else:
                        c = np.linalg.norm(np.log(self.y) - np.log(model1.n(model1.L-offset-self.x*a)) - b)
                    # must decide how to handle cases where log of a negative or zero value may be taken
                    # by making it high cost, implicitly adds min length to lattice condition
                    if np.isnan(c):
                        return 1e30
                return c
            c = np.linalg.norm(self.y - model1.n(model1.L-offset-self.x*a) * b)
            if np.isnan(c).all():
                return 1e30
            return c

        # bounds on a such that model is always at least as wide as data
        a_mx = model1.L / self.y.size * L_scale

        sol = minimize(cost, [a_mx/2,0], bounds=[(0, a_mx), (-np.inf, np.inf)])
        a, b = sol['x']
        
        if log:
            # b is solved for in log terms, otherwise there are problems w/ convergence
            return a, np.exp(b), sol, mod_err
        return a, b, sol, mod_err

    def scan(self, G_range, ro_range, rd_range, I_range, rev=False, **fitter_kw):
        """
        Parameters
        ----------
        G_range : ndarray
        ro_range : ndarray
        rd_range : ndarray
        I_range : ndarray
        rev : bool, False
        **fitter_kw
            E.g. 'primary' and 'log'

        Returns
        -------
        dict
        """
        from itertools import product

        def loop_wrapper(params):
            try:
                if rev:
                    result = self.fit_length_scales_rev(*params, **fitter_kw)
                else:
                    result = self.fit_length_scales(*params, **fitter_kw)
                # ignore cases where the distance measure is nan (indicating problems w/ computing density)
                if not np.isnan(result[2]['fun']):
                    return result
                return
            except AssertionError:
                return
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with Pool() as pool:
                output = pool.map(loop_wrapper, product(G_range, ro_range, rd_range, I_range), chunksize=100)

        # read out fit results
        fit_results = {}
        for params, output in zip(product(G_range, ro_range, rd_range, I_range), output):
            if not output is None:
                fit_results[params] = output

        return fit_results
#end GridSearchFitter
