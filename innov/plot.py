# Module for plotting.
# Author: Eddie Lee, edlee@csh.ac.at
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def density_snapshot(n, el, K, t, mean=False, replica_ix=0, **kwargs):
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
    mean : bool or str, False
        If False show only the first replica. Else 'replica' or 'branch+replica'
    **kwargs : dict
    """
    fig, ax = plt.subplots(figsize=(6,2))

    if mean=='replica': 
        for tix in t:
            ax.plot(n[tix,:,::K].mean(0))
            if K>1:  # show a second branch
                ax.plot(n[tix,:,1::K].mean(0))
    elif mean=='branch+replica':
        for tix in t:
            # mean over replicas then mean over branches
            ax.plot(n[tix].mean(0).reshape(el[1], K).mean(1))
    elif mean is False:
        for tix in t:
            ax.plot(n[tix,replica_ix,::K])
            if K>1:
                ax.plot(n[tix,replica_ix,1::K])
    else: raise NotImplementedError

    ax.set(**kwargs)
    return fig