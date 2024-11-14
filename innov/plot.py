import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def density_snapshot(n, el, K, t, mean=False, **kwargs):
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
    mean : bool, False
        If True, mean over replicas.
    **kwargs : dict
    """
    fig, ax = plt.subplots(figsize=(6,2))

    if mean: 
        for tix in t:
            ax.plot(np.concatenate((n[tix,:,:el[0]].mean(0), n[tix,:,el[0]::K].mean(0))))
            ax.plot(np.concatenate((n[tix,:,:el[0]].mean(0), n[tix,:,el[0]+1::K].mean(0))))
    else:
        for tix in t:
            ax.plot(np.concatenate((n[tix,0,:el[0]], n[tix,0,el[0]::K])))
            ax.plot(np.concatenate((n[tix,0,:el[0]], n[tix,0,el[0]+1::K])))

    ax.set(**kwargs)
    return fig