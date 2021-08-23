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
import duckdb

DATADR = os.path.expanduser('~')+'/Dropbox/Research/corporations/starter_packet' 



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
