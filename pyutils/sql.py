# ====================================================================================== #
# Wrappers for quick SQL queries.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb
import fastparquet as fp

from .organizer import SimLedger
from .utils import *



def setup_parquet(do_firms=True, do_lattice=True):
    """Set up parquet files.

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

                fp.write(f'{sims.cache_dr}/{key}.parquet', df, 1_000_000,
                         compression='SNAPPY')
               
            if do_lattice:
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

                fp.write(f'{sims.cache_dr}/{key}_lattice.parquet', df, 1_000_000,
                         compression='SNAPPY')
            
            # clear RAM
            del sims.storage[key]
        print(f"Done with ix={ix}.")



class QueryRouter():
    def __init__(self):
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.simledger = SimLedger()

    def wealth_by_firm(self, ix, tbds=None, mn_nfirms=10):
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = simulator.load_list()

        wealth = []
        for thiskey in simlist:
            # wealth distribution when there are at least 10 firms
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
        """Total wealth per time step.
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
