# ====================================================================================== #
# Useful functions for analyzing corp data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
import pandas as pd
from fastparquet import ParquetFile
import snappy
import os
import datetime as dt
from warnings import warn
from multiprocess import Pool, cpu_count
from threadpoolctl import threadpool_limits
import dill as pickle
import duckdb as db
from itertools import combinations

DATADR = os.path.expanduser('~')+'/Dropbox/Research/corporations/starter_packet' 


def db_conn():
    return db.connect(database=':memory:', read_only=False)

def snappy_decompress(data, uncompressed_size):
        return snappy.decompress(data)

def topic_added_dates():
    """Dates on which new topics were added according to the "topic taxonomy.csv".
    
    Returns
    -------
    ndarray
    """

    df = pd.read_csv('%s/topic taxonomy.csv'%DATADR)
    udates, count = np.unique(df['Active Date'], return_counts=True)
    udates = np.array([dt.datetime.strptime(d,'%m/%d/%Y').date() for d in udates])
    return udates

def bin_laplace(y, nbins, center=1):
    """Bin statistics from a laplace distribution by using log bins spaced around the center.
    
    Parameters
    ----------
    y : ndarray
    nbins : int
    center : float, 1.
    
    Returns
    -------
    ndarray
        Counts in bins.
    ndarray
        Bin edges.
    ndarray
        Bin centers.
    """
     
    logy = np.log(y)

    bins = np.linspace(0, np.abs(logy).max()+1e-6, nbins//2)
    bins = np.concatenate((-bins[1:][::-1], bins)) + np.log(center)

    n = np.histogram(logy, bins)[0]
    return n, np.exp(bins), np.exp(bins[:-1] + (bins[1] - bins[0])/2)

def log_hist(y, nbins=20):
    """Log histogram on discrete domain. Assuming min value is 1.

    Parameters
    ----------
    y : ndarray
    nbins : int, 20
    
    Returns
    -------
    ndarray
        Normalized frequency.
    ndarray
        Bin midpoints.
    ndarray
        Bin edges.
    """
    
    bins = np.unique(np.around(np.logspace(0, np.log10(y.max()+1), nbins)).astype(int))

    p = np.histogram(y, bins)[0]
    p = p / p.sum() / np.floor(np.diff(bins))

    xmid = np.exp((np.log(bins[:-1]) + np.log(bins[1:]))/2)
    
    return p, xmid, bins

def del_poor_fits(fit_results, n_keep=100):
    """Only keep best fit results from grid scan fit to data. Deleted from passed
    list in place.
    
    Parameters
    ----------
    fit_results : tuple
        As returned by GridSearchFitter.fit_length_scales().
    n_keep : int, 100
    """
    counter = 0
    keys_to_del = []
    for k, v in sorted(fit_results.items(), key=lambda i: i[1][2]['fun']):
        if counter > (n_keep-1):
            keys_to_del.append(k)
        counter += 1

    for k in keys_to_del:
        del fit_results[k]
