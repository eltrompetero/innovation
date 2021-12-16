# ====================================================================================== #
# For analyzing model results.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from datetime import datetime
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.special import erfc

from .simple_model import UnitSimulator
from .utils import *



def fit_density(density, wi, fraction_fit=1., return_xy=False):
    """Fit average density curve from a set of density samples.

    This is an easier interface than mft.fit_density

    Parameters
    ----------
    density : list of ndarray
        Density from leftmost front to right over multiple snapshots.
    wi : float
        Innovation success probability.
    fraction_fit : float
        Fraction of curve to fit to starting from innovation front.
    return_xy : bool, False
        If True, return the evaluated density.

    Returns
    -------
    list of ndarray
        Fit parameters.
    """

    from .mft import fit_density, exp_density_f
    
    assert all([i[0]==0 for i in density]), "leftmost density should be given"
    assert 0<=wi<=1

    # must be careful to determine the fitting region scaled to [0,1] interval
    mxdx = max([i.size for i in density]) - 1
    x_interp = np.linspace(1/mxdx, 1, mxdx)
    dx = x_interp[1] - x_interp[0]
    
    # pad density with zeros so we can easily take a vertical mean
    y = np.vstack([np.concatenate((np.zeros(mxdx-y.size+1)+np.nan, y[1:]))
                   for i, y in enumerate(density)])
    ym = np.nanmean(y, 0)

    # fit total density profile
    nfit = int(fraction_fit * ym.size)
    x = np.linspace(1/mxdx, 1, mxdx)[-nfit:]
    soln = fit_density(x, ym[-nfit:], wi)

    if return_xy: 
        x = np.linspace(1/mxdx, 1, mxdx)
        return soln, (x, exp_density_f(x, *soln, wi), y)
    return soln

def group_by_vel(X, windows, velsign,
                 burn_in=5000,
                 dx=13,
                 force=False):
    """Group the given set of observations by whether velocity is positive, negative, or
    zero within the given windows.
    
    Parameters
    ----------
    X : list of vectors
    windows : list of twoples
    velsign : list of int
    burn_in : int, 5_000
        This number of time steps to discount to reduce seeding effects.
    dx : int, 13
        Skip entries in X by this number to reduce autocorrelation of samples.
    force : bool, False
        If True, skip checks.
    
    Returns
    -------
    list of list
    """
    
    if not force:
        assert len(windows)==len(velsign)
        assert burn_in<windows[-1][0], "burn_in period skips all windows"
        assert windows[-1][-1]==(len(X)-1), "window bounds do not line up with X"
        assert dx>=1

    Y = [[],[],[]]
    
    if isinstance(X, np.ndarray) and X.ndim==1:
        # account for burn in 
        startix = np.argmax([w[1]>burn_in for w in windows])
        
        for w, s in zip(windows[startix:], velsign[startix:]):
            if s==-1:
                Y[0].extend(X[w[0]:w[1]:dx])
            elif s==0:
                Y[1].extend(X[w[0]:w[1]:dx])
            else:
                Y[2].extend(X[w[0]:w[1]:dx])
        return Y
        
    # account for burn in 
    startix = np.argmax([w[1]>burn_in for w in windows])
    
    for w, s in zip(windows[startix:], velsign[startix:]):
        if s==-1:
            Y[0].extend(np.concatenate(X[w[0]:w[1]:dx]))
        elif s==0:
            Y[1].extend(np.concatenate(X[w[0]:w[1]:dx]))
        else:
            Y[2].extend(np.concatenate(X[w[0]:w[1]:dx]))
    return Y

def _growth_std_scaling(grate, wealth, wbins):
    s = np.zeros(wbins.size-1)

    for i in range(wbins.size-1):
        ix = (wealth>=wbins[i]) & (wealth<wbins[i+1])
        if ix.any():
            thisgrate = grate[ix]
            s[i] = thisgrate.std()
    
    return s

def growth_std_scaling(grate, wealth, nbins, bootstrap=False):
    """Scaling of std of wealth growth.

    Parameters
    ----------
    grate : ndarray
    wealth : ndarray
    nbins : int
    bootstrap : int, False
        If a positive number, then this many bootstrap calculations will be done.

    Returns
    -------
    ndarray
        Std per log spaced wealth bin.
    ndarray
        Bin edges.
    ndarray
        Bin centers.
    ndarray (optional)
        Error bars from bootstrapping.
    """

    wbins = np.logspace(np.log10(wealth).min(), np.log10(wealth).max()+1e-6, nbins)
    wcenters = np.exp((np.log(wbins[:-1])+np.log(wbins[1:])) / 2)
    
    s = _growth_std_scaling(grate, wealth, wbins)
    
    if bootstrap:
        assert bootstrap>0
        samp = np.zeros((bootstrap, s.size))
        ran = list(range(grate.size))

        for i in range(bootstrap):
            ix = np.random.choice(ran, size=len(ran), replace=True) 
            samp[i] = _growth_std_scaling(grate[ix], wealth[ix], wbins)
        return s, wbins, wcenters, samp

    return s, wbins, wcenters

def segment_by_rho_vel(rho, zero_thresh,
                       moving_window=10,
                       smooth_vel=False):
    """Given the boundaries of the 1D lattice, segment time series into pieces with +/-/0
    velocities according to the defined threshold.

    Parameters
    ----------
    rho : ndarray
    zero_thresh : float
    moving_window : int or tuple, 10
        Width of moving average window.
    smooth_vel : bool, False
        If True, smooth velocity after calculation from smoothed distance time series.
        Can specify separate moving windows by providing a tuple for moving_window.

    Returns
    -------
    list of twoples
        Boundaries of time segments. Note that this is calculated on vector that is one
        element shorter than leftright.
    list
        Sign of the velocity for each window segment.
    list
        Time-averaged velocities at each moment in time for each of the extracted
        segments.
    """


    if not hasattr(moving_window, '__len__'):
        if moving_window<5:
            warn("Window may be too small to get a smooth velocity.")
        moving_window = (moving_window, moving_window)
    
    if moving_window[0]>1:
        rho = fftconvolve(rho, np.ones(moving_window[0])/moving_window[0], mode='same')

    v = np.diff(rho)
    if smooth_vel:
        rawv = v
        v = fftconvolve(v, np.ones(moving_window[1])/moving_window[1], mode='same')

    # consider the binary velocity
    # -1 as below threshold
    # 0 as bounded by threshold
    # 1 as above threshold
    vsign = np.zeros(v.size, dtype=int)
    vsign[v>zero_thresh] = 1
    vsign[v<-zero_thresh] = -1
    
    # find places where velocity switches sign
    # offset of 1 makes sure window counts to the last element of set
    ix = np.where(vsign[1:]!=vsign[:-1])[0] + 1
    if ix.size:
        windows = [(0, ix[0])] + [(ix[i], ix[i+1]) for i in range(ix.size-1)] + [(ix[-1], vsign.size)]
    else:
        windows = [(0, vsign.size)]
    
    vel = []  # velocities for each window
    velsign = []
    for w in windows:
        vel.append(rawv[w[0]:w[1]])  
        velsign.append(vsign[w[0]])

    return windows, velsign, vel

def segment(y, zero_thresh,
            moving_window=10):
    """Given the boundaries of the 1D lattice, segment time series into pieces with +/-/0
    velocities according to the defined threshold.

    Parameters
    ----------
    y : ndarray
    zero_thresh : float
    moving_window : int or tuple, 10
        Width of moving average window.

    Returns
    -------
    list of twoples
        Boundaries of time segments. Note that this is calculated on vector that is one
        element shorter than leftright.
    list
        Sign of the velocity for each window segment.
    list
        Time-averaged velocities at each moment in time for each of the extracted
        segments.
    """


    if moving_window<5:
        warn("Window may be too small to get a smooth velocity.")
    
    rawy = y
    if moving_window>1:
        y = fftconvolve(y, np.ones(moving_window)/moving_window, mode='same')

    # consider the binary velocity
    # -1 as below threshold
    # 0 as bounded by threshold
    # 1 as above threshold
    ysign = np.zeros(y.size, dtype=int)
    ysign[y>zero_thresh] = 1
    ysign[y<-zero_thresh] = -1
    
    # find places where velocity switches sign
    # offset of 1 makes sure window counts to the last element of set
    ix = np.where(ysign[1:]!=ysign[:-1])[0] + 1
    if ix.size:
        windows = [(0, ix[0])] + [(ix[i], ix[i+1]) for i in range(ix.size-1)] + [(ix[-1], ysign.size)]
    else:
        windows = [(0, ysign.size)]
    
    vel = []  # velocities for each window
    velsign = []
    for w in windows:
        vel.append(rawy[w[0]:w[1]])  
        velsign.append(ysign[w[0]])

    return windows, velsign, vel

def time_interval(L, cutoff):
    """Time intervals when below and above cutoff.

    Parameters
    ----------
    L : list of int
    cutoff : float

    Returns
    -------
    list of two lists
        First list is periods of time below cutoff and second is that above cutoff.
    """
    
    ix = L<=cutoff

    # find first place where cutoff is exceeded
    dt = [[],[]]
    if ix[0]:
        startix = (L>int(cutoff)).argmax() - 1
    else:
        startix = 0

    switch = False  # True when below and False when above
    abovedt = 0
    belowdt = 0
    for i in np.diff(ix[startix:].astype(int)):
        if i==-1: switch = True
        elif i==1: switch = False
            
        if switch:
            abovedt += 1
            if belowdt!=0:
                dt[0].append(belowdt)
                belowdt = 0
        else:
            belowdt += 1
            if abovedt!=0:
                dt[1].append(abovedt)
                abovedt = 0
    
    assert (sum(dt[0]) + sum(dt[1]) + startix + belowdt + abovedt)==(len(L)-1)
    return dt


# ======= #
# Classes #
# ======= #
class Comparator():
    def __init__(self, ro, G, re, rd, I, dt=5e-3):
        """For comparing MFT and automaton.
        
        Parameters
        ----------
        ro : float
        G : float
        re : float
        rd : float
        I : float
        dt : float
        """
        
        self.ro = ro
        self.G = G
        self.re = re
        self.rd = rd
        self.I = I
        self.dt = dt
        
    def load_params(self):
        return self.ro, self.G, self.re, self.rd, self.I, self.dt

    def find_critical_re(self, re0, full_output=False):
        """Find divergence point for the expansion rate according to corrected MFT.
        
        This numerically solves the nonlinear relation we have for L.

        Parameters
        ----------
        re0 : float
            Initial guess for re.
        full_output : bool, False

        Returns
        -------
        float
        """
        
        ro, G, _, rd, I, dt = self.load_params()

        def cost(logre):
            re = np.exp(logre)
            z = - (re - rd) / ((re-rd)/2 + re*(1 - ro/re))
            if z==0: 
                C = 0  # determined from continuity
            else:
                C = (-1 + z + np.exp(-z)) / z

            L = G*I*re / ro / (ro+rd-re*(1+1/(1-C)))
            return L**-2.

        if full_output:
            soln = minimize(cost, np.log(re0))
            return np.exp(soln['x'])[0], soln
        return np.exp(minimize(cost, np.log(re0))['x'])[0]

    def find_critical_ro(self, ro0, full_output=False):
        """Find divergence point for the expansion rate according to corrected MFT.
        
        This numerically solves the nonlinear relation we have for L.

        Parameters
        ----------
        ro0 : float
            Initial guess for re.
        full_output : bool, False

        Returns
        -------
        float
        """
        
        _, G, re, rd, I, dt = self.load_params()

        def cost(logro):
            ro = np.exp(logro)
            z = - (re - rd) / ((re-rd)/2 + re*(1 - ro/re))
            if z==0: 
                C = 0  # determined from continuity
            else:
                C = (-1 + z + np.exp(-z)) / z

            L = G*I*re / ro / (ro+rd-re*(1+1/(1-C)))
            return L**-2.

        if full_output:
            soln = minimize(cost, np.log(ro0))
            return np.exp(soln['x'])[0], soln
        return np.exp(minimize(cost, np.log(ro0))['x'])[0]

    def run_re(self, re_range, T, n_samples, ensemble=True, t_skip=500):
        """
        Parameters
        ----------
        re_range : ndarray
        T : float
            Duration of simulation.
        n_samples : int
            No. of samples or indpt. trajectories to take.
        ensemble : bool, True
            If True, run simulation on an ensemble of random trajectories. If False,
            sample over time from a single running instance.
        t_skip : float, 500
            Only for when we are using sequential temporal sampling.
        
        Returns
        -------
        dict
            Indexed by re values. Values are occupancy snapshots.
        """
        
        ro, G, _, rd, I, dt = self.load_params()
        
        if ensemble:
            def loop_wrapper(re):
                simulator = UnitSimulator(L0=1,
                                          N0=0,
                                          g0=G,
                                          obs_rate=ro,
                                          expand_rate=re,
                                          innov_rate=I,
                                          exploit_rate=0,
                                          death_rate=rd,
                                          dt=dt)
                snapshot_n = simulator.parallel_simulate(n_samples, T)
                return snapshot_n
        else:
            def loop_wrapper(re):
                simulator = UnitSimulator(L0=1,
                                          N0=0,
                                          g0=G,
                                          obs_rate=ro,
                                          expand_rate=re,
                                          innov_rate=I,
                                          exploit_rate=0,
                                          death_rate=rd,
                                          dt=dt)
                snapshot_n = [simulator.simulate(T)]
                for i in range(n_samples):
                    snapshot_n.append(simulator.simulate(t_skip, occupancy=snapshot_n[-1]))
                return snapshot_n
        
        if not ensemble:
            with Pool() as pool:
                output = pool.map(loop_wrapper, re_range)
            snapshot_n = dict(zip(re_range, output))
        else:
            snapshot_n = {}
            for re in re_range:
                snapshot_n[re] = loop_wrapper(re)
        
        self.snapshot_n = snapshot_n
        return snapshot_n

    def run_ro(self, ro_range, T, n_samples, ensemble=True, t_skip=500):
        """
        Parameters
        ----------
        ro_range : ndarray
        T : float
            Duration of simulation.
        n_samples : int
            No. of samples or indpt. trajectories to take.
        ensemble : bool, True
            If True, run simulation on an ensemble of random trajectories. If False,
            sample over time from a single running instance.
        t_skip : float, 500
            Only for when we are using sequential temporal sampling.
        
        Returns
        -------
        dict
            Indexed by re values. Values are occupancy snapshots.
        """
        
        _, G, re, rd, I, dt = self.load_params()
        
        if ensemble:
            def loop_wrapper(ro):
                simulator = UnitSimulator(L0=1,
                                          N0=0,
                                          g0=G,
                                          obs_rate=ro,
                                          expand_rate=re,
                                          innov_rate=I,
                                          exploit_rate=0,
                                          death_rate=rd,
                                          dt=dt)
                snapshot_n = simulator.parallel_simulate(n_samples, T)
                return snapshot_n
        else:
            def loop_wrapper(ro):
                simulator = UnitSimulator(L0=1,
                                          N0=0,
                                          g0=G,
                                          obs_rate=ro,
                                          expand_rate=re,
                                          innov_rate=I,
                                          exploit_rate=0,
                                          death_rate=rd,
                                          dt=dt)
                snapshot_n = [simulator.simulate(T)]
                for i in range(n_samples):
                    snapshot_n.append(simulator.simulate(t_skip, occupancy=snapshot_n[-1]))
                return snapshot_n
        
        if not ensemble:
            with Pool() as pool:
                output = pool.map(loop_wrapper, ro_range)
            snapshot_n = dict(zip(ro_range, output))
        else:
            snapshot_n = {}
            for ro in ro_range:
                snapshot_n[ro] = loop_wrapper(ro)
        
        self.snapshot_n = snapshot_n
        return snapshot_n
#end Comparator


class WSBFitter():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def citations_curve(self, params=None):
        if params is None:
            params = self.params
        C, el, mu, s = params
        return C * np.exp(el * erfc((mu - np.log(self.x))/s)/4/np.sqrt(2) - 
                          (mu-np.log(self.x))**2/s**2) * el / s / self.x

    def cost(self, params):
        logC, logel, mu, logs = params
        C = np.exp(logC)
        el = np.exp(logel)
        s = np.exp(logs)
        self.citations_curve([C, el, mu, s])
        # fit to reversed equation
        err = np.linalg.norm(self.citations_curve([C, el, mu, s]) - self.y)
        return err if not np.isnan(err) else 1e30
    
    def solve(self):
        soln = minimize(self.cost, [4, 0, 2, 0])
        self.params = np.exp(soln['x'])
        self.params[2] = soln['x'][2]
        self.soln = soln
        return self.params
#end WSBFitter



class WeibullFitter():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def pdf(self, params=None):
        if params is None:
            params = self.params
        C, el, k = params
        return C * self.x**(k-1.) * np.exp(-(self.x / el)**k)

    def cost(self, params):
        C, el, k = np.exp(params)
        # fit to reversed equation
        err = np.linalg.norm(self.pdf([C, el, k]) - self.y)
        return err if not np.isnan(err) else 1e30
    
    def solve(self):
        soln = minimize(self.cost, [4, -2, 0],
                        bounds=[(-np.inf,np.inf),(-np.inf,np.inf),(0,np.inf)])
        self.params = np.exp(soln['x'])
        self.soln = soln
        return self.params
#end WeibullFitter
