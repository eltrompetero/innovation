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


# ======= #
# Classes #
# ======= #
class IterativeMFT():
    def __init__(self, vo, G, re, rd, wi):
        """Class for calculating discrete MFT quantities.

        Parameters
        ----------
        vo : float
        G : float
        re : float
        rd : float
        wi : float
        """

        self.vo = vo
        self.G = G
        self.re = re
        self.rd = rd
        self.wi = wi
        self.n0 = vo/re/wi  # stationary density
        
        # MFT guess for L, which we will refine using tail convergence criteria
        self.L0 = 4 * G * re * wi / ((2 * vo - (2*re-rd))**2 - (2*re-rd)**2)
        self.L, self.n = self.min_L(self.L0)

    def min_L(self, L0, mx_decimal=10):
        """Lower L slower and slower while keeping tail positive in order to find value
        of L that solves iterative solution.
        
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
        assert L0 > 2

        vo = self.vo
        G = self.G
        re = self.re
        rd = self.rd
        wi = self.wi
        n0 = self.n0
        
        L = np.ceil(L0)
        decimal = 1

        nfun = self.iterate_n(L)
        # check that tail is positive; in case it is not, increase starting guess for L a few times
        counter = 0
        while (wi * n0 * nfun[-2] + G/re/L / (wi * n0 + rd/re)) < 0:
            L += 1
            nfun = self.iterate_n(L)
            counter += 1
            assert counter < 1e3
        
        while decimal <= mx_decimal:
            # ratchet down til the tail goes the wrong way
            while (wi * n0 * nfun[-2] + G/re/L / (wi * n0 + rd/re)) > 0:
                L -= 10**-decimal
                nfun = self.iterate_n(L)
            L += 10**-decimal  # oops, go back up
            nfun = self.iterate_n(L)
            decimal += 1
        return L, nfun

    def iterate_n(self, L=None, iprint=False):
        """Iterative solution to occupancy number. See NB II pg. 118."""
        
        vo = self.vo
        G = self.G
        re = self.re
        rd = self.rd
        wi = self.wi
        n0 = self.n0
        L = L or self.L

        eps = wi * n0**2 + (rd/re-2) * n0 - G/re/L

        nfun = np.zeros(int(L))
        nfun[0] = n0
        nfun[1] = wi*n0**2 + rd*n0/re - G/L/re

        for i in range(2, int(L)):
            nfun[i] = wi * n0 * (nfun[i-1] - nfun[i-2]) + rd * nfun[i-1] / re - G / re / L

        if iprint: print(eps)

        return nfun
#end IterativeMFT

