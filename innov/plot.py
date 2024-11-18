import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def density_snapshot(n, el, K, t, mean=False, **kwargs):
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
    mean : bool, False
        If True, mean over replicas, else show only the first replica.
    **kwargs : dict
    """
    fig, ax = plt.subplots(figsize=(6,2))

    if mean: 
        for tix in t:
            ax.plot(n[tix,:,::K].mean(0))
            if K>1:
                ax.plot(n[tix,:,1::K].mean(0))
    else:
        for tix in t:
            ax.plot(n[tix,0,::K])
            if K>1:
                ax.plot(n[tix,0,1::K])

    ax.set(**kwargs)
    return fig