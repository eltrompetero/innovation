# ====================================================================================== #
# For analyzing model results.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from datetime import datetime
from scipy.signal import fftconvolve

from .utils import *



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
