import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def density_snapshot(n, el, K, t, **kwargs):
    """Plot density snapshots from automaton simulation.

    Parameters
    ----------
    n : np.ndarray
        Density.
    el : tuple
        (length of initial branch, length of later branches
    K : int
    t : list-like
        List of time-indices to show.
    """
    fig, ax = plt.subplots(figsize=(6,2))
    
    for tix in t:
        ax.plot(np.concatenate((n[tix,0,:el[0]], n[tix,0,el[0]::K])))
        ax.plot(np.concatenate((n[tix,0,:el[0]], n[tix,0,el[0]+1::K])))
    
    ax.set(**kwargs)
    return fig