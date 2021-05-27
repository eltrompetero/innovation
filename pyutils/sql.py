# ====================================================================================== #
# Wrappers for quick SQL queries.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb
import fastparquet as fp

from .organizer import SimLedger
from .utils import *



def setup_parquet(do_firms=True, do_lattice=True):
    """Set up parquet files from pickles.

    Parameters
    ----------
    do_firms : bool, True
    do_lattice : bool, True
    """
    
    from .model import reconstruct_lattice

    simledger = SimLedger()

    for ix in range(len(simledger.ledger)):
        sims = simledger.load(ix)
        loadlist = sims.load_list()

        for key in loadlist:
            sims.load(key)
            firm_snapshot, lattice_snapshot = sims.storage[key]
            lattice_snapshot = reconstruct_lattice(firm_snapshot, lattice_snapshot)
            
            if do_firms:
                parquet_firms(firm_snapshot, sims.cache_dr, key)
               
            if do_lattice:
                parquet_firms(lattice_snapshot, sims.cache_dr, key)
            
            # clear RAM
            del sims.storage[key]
        print(f"Done with ix={ix}.")

def parquet_firms(firm_snapshot,
                  cache_dr,
                  key):
    """Meat of setup_parquet().

    Parameters
    ----------
    firm_snapshot : list
    cache_dr : str
    key : str
    """
    
    # write firms to parquet file
    ids = []
    w = []
    t = []
    innov = []
    left = []
    right = []
    age = []
    for i, firms in enumerate(firm_snapshot):
        ids.extend([f.id for f in firms])
        w.extend([f.wealth for f in firms])
        t.extend([i]*len(firms))
        innov.extend([f.innov for f in firms])
        left.extend([f.sites[0] for f in firms])
        right.extend([f.sites[1] for f in firms])
        age.extend([f.age for f in firms])

    df = pd.DataFrame({'ids':np.array(ids),
                       'wealth':np.array(w),
                       'innov':np.array(innov),
                       'left':np.array(left),
                       'right':np.array(right),
                       'age':np.array(age),
                       't':np.array(t)})

    fp.write(f'{cache_dr}/{key}.parquet', df, 100_000,
             compression='SNAPPY')
    
def parquet_lattice(lattice_snapshot,
                    cache_dr,
                    key):
    """Meat of setup_parquet().

    Parameters
    ----------
    lattice_snapshot : list
    cache_dr : str
    key : str
    """
       
    # write lattice to parquet file
    occupancy = []
    index = []
    t = []
    for i, lattice in enumerate(lattice_snapshot):
        occupancy.append(lattice.occupancy)
        index.append(np.arange(lattice.left, lattice.right+1))
        t.append([i]*occupancy[-1].size)

    df = pd.DataFrame({'occupancy':np.concatenate(occupancy),
                       'index':np.concatenate(index),
                       't':np.concatenate(t)})

    fp.write(f'{cache_dr}/{key}_lattice.parquet', df, 100_000,
             compression='SNAPPY')
    


class QueryRouter():
    def __init__(self):
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.simledger = SimLedger()

    def wealth_by_firm(self, ix, tbds=None, mn_nfirms=10):
        """Set of wealth for all firms per time step.

        Parameters
        ----------
        ix : int
        tbds : twople, None
        mn_nfirms : int, 10

        Returns
        -------
        list of ndarray
            Each list is for a simulation instance.
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        wealth = []
        for thiskey in simlist:
            # wealth distribution when there are at least mn_nfirms  firms
            if tbds is None:
                query = f'''
                         SELECT otable.t, wealth
                         FROM (select t, COUNT(ids) as n_firms
                               FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet')
                               GROUP BY t) ctable
                               INNER JOIN parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') otable
                               ON ctable.t = otable.t
                         WHERE ctable.n_firms>={mn_nfirms}
                         '''
            else:
                query = f'''
                         SELECT otable.t, wealth
                         FROM (select t, COUNT(ids) as n_firms
                               FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet')
                               GROUP BY t) ctable
                               INNER JOIN parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') otable
                               ON ctable.t = otable.t
                         WHERE ctable.n_firms>={mn_nfirms} AND 
                             ctable.t>={tbds[0]} AND
                             ctable.t<{tbds[1]}
                         '''
            wealth.append(self.con.execute(query).fetchdf().values)
        return wealth

    def total_wealth(self, ix, tbds=None):
        """Total wealth summed over all firms per time step.
        """
        
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        wealth = []
        for thiskey in simlist:
            query = f'''
                     SELECT t, SUM(wealth)
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet')
                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                     GROUP BY t
                     ORDER BY t
                     '''
            wealth.append(self.con.execute(query).fetchdf().values)
        return wealth

    def lattice_width(self, ix, tbds=None):
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        width = []
        for thiskey in simlist:
            query = f'''
                     SELECT lattice.t, MAX(lattice.index) - MIN(lattice.index)
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet') lattice
                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                     GROUP BY lattice.t
                     ORDER BY lattice.t
                     '''
            width.append(self.con.execute(query).fetchdf().values)
        return width

    def total_wealth_per_lattice(self, ix, tbds=None):
        """Total wealth summed over all firms divided by lattice width per time step.
        """
        
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        wealth = []
        for thiskey in simlist:
            query = f'''
                     SELECT lattice.t, SUM(wealth)/(MAX(lattice.index) - MIN(lattice.index) + 1) as wealth
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     INNER JOIN parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet') lattice
                        ON firm.t = lattice.t
                     WHERE firm.t>={tbds[0]} AND firm.t<{tbds[1]}
                     GROUP BY lattice.t
                     ORDER BY lattice.t
                     '''
            wealth.append(self.con.execute(query).fetchdf().values)
        return wealth

    def norm_firm_wealth(self, ix, tbds=None):
        """Firm wealth normalized by firm size for all firms per time step.
        """
        
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        wealth = []
        for thiskey in simlist:
            query = f'''
                     SELECT t, firm.wealth / (firm.right - firm.left + 1) as wealth
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                     ORDER BY t
                     '''
            wealth.append(self.con.execute(query).fetchdf())
        return wealth

    def bounds(self, ix, tbds=None):
        """Left and right sides for each firm and lattice per time step.
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        bds = []
        for thiskey in simlist:
            query = f'''
                     SELECT firm.t,
                            firm.left,
                            firm.right,
                            lattice.left as lat_left,
                            lattice.right as lat_right
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     INNER JOIN (SELECT t, MIN(index) as left, MAX(index) as right
                                 FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet')
                                 GROUP BY t) lattice
                     ON firm.t = lattice.t
                     WHERE lattice.t>={tbds[0]} AND lattice.t<{tbds[1]}
                     ORDER BY lattice.t
                     '''
            bds.append(self.con.execute(query).fetchdf())
        return bds

    def query(self, ix, q):
        """A general query.
        
        Parameters
        ----------
        ix : str or int
        q : str
            SQL query.

        REturns
        -------
        list of DataFrame
        """

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        q = q.replace('CACHE_DR', simulator.cache_dr)

        output = []
        for thiskey in simlist:
            output.append(self.con.execute(q.replace('KEY', thiskey)).fetchdf())
        return output
#end QueryRouter
