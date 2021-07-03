# ====================================================================================== #
# Wrappers for quick SQL queries.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb
import fastparquet as fp

from .organizer import SimLedger
from .utils import *



def setup_parquet(ix_range, do_firms=True, do_lattice=True):
    """Set up parquet files from pickles.

    Parameters
    ----------
    ix_range : list
    do_firms : bool, True
    do_lattice : bool, True
    """
    
    from .model import reconstruct_lattice

    simledger = SimLedger()

    for ix in ix_range:
        sims = simledger.load(ix)
        loadlist = sims.load_list()

        for key in loadlist:
            sims.load(key)
            firm_snapshot, lattice_snapshot = sims.storage[key]
            lattice_snapshot = reconstruct_lattice(firm_snapshot, lattice_snapshot)
            
            if do_firms:
                parquet_firms(firm_snapshot, sims.cache_dr, key)
               
            if do_lattice:
                parquet_lattice(lattice_snapshot, sims.cache_dr, key)
            
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
                       'fleft':np.array(left),
                       'fright':np.array(right),
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
                       'ix':np.concatenate(index),
                       't':np.concatenate(t)})

    fp.write(f'{cache_dr}/{key}_lattice.parquet', df, 100_000,
             compression='SNAPPY')

def parquet_density(ix):
    """Create parquet file storing density for specified cache in simulator.

    Parameters
    ----------
    ix : int
        Index of simulation in sim ledger.
    """
    
    from itertools import chain
    
    # set up
    simledger = SimLedger()
    simulator = simledger.load(ix)
    simlist = simulator.load_list()
    qr = QueryRouter()

    # get boundaries of lattice and firms
    bds = qr.bounds(ix)

    def loop_wrapper(args):
        """For each time step of sim.
        """

        # for each time step in sim, create a vector of size of lattice, and
        # iterate thru each firm counting up any site it occupies
        # if there are no firms, then this adds nothing to the density (this is b/c 
        # the returned bounds only have an entry if there are firms)
        t, group = args
        # lattice bounds
        lat_left = group.iloc[0]['lat_left']
        lat_right = group.iloc[0]['lat_right']
        thisdensity = np.zeros(lat_right-lat_left+1, dtype=np.int64)

        for i, row in group.iterrows():
            thisdensity[row['fleft']-lat_left:row['fright']-lat_left+1] += 1

        return list(zip([t]*(lat_right-lat_left+1),
                        range(lat_left, lat_right+1),
                        thisdensity))

    with Pool() as pool:
        # this form of the loop is more memory efficient than par-looping over bds (if
        # slower)
        for bds_, key in zip(bds, simlist):
            density = list(chain(*pool.map(loop_wrapper, bds_.groupby('t'))))
            density = pd.DataFrame(density, columns=['t','ix','density'])
            fp.write(f'{simulator.cache_dr}/{key}_density.parquet', density, 1_000_000,
                     compression='SNAPPY')
  
def death_rate(df, fill_in_missing_t=False):
    """Death rate for all firms over the entire lattice, i.e. fraction of firms that do not
    appear in the next time step.

    Parameters
    ----------
    df : pd.DataFrame
        With each firm id and time of measurement.
    fill_in_missing_t : bool, False

    Returns
    -------
    ndarray
    """

    rate_deaths = []
    
    t_group = df.groupby('t')
    for i, (t, group) in enumerate(t_group):
        if i==0:
            prevt = t
            prevgroup = group
        else:
            if (t-prevt)>1 and not fill_in_missing_t:
                # all firms died; don't count
                pass
            elif (t-prevt)>1 and fill_in_missing_t:
                rate_deaths.append(np.nan)
            else:
                # count which firms died
                counter = 0
                for fid in prevgroup.ids:
                    if not fid in group.ids.values:
                        counter += 1
                rate_deaths.append(counter/len(prevgroup))

            prevt = t
            prevgroup = group

    rate_deaths = np.array(rate_deaths)

    if fill_in_missing_t:
        temp = np.zeros(df['t'].iloc[-1] - df['t'].iloc[0]) + np.nan
        temp[np.array(list(t_group.groups.keys()))[:-1]-df['t'].min()] = rate_deaths
        rate_deaths = temp
        
    return rate_deaths




class QueryRouter():
    """Class for routing queries through SQL to access parquet database for simulation results."""
    def __init__(self):
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.simledger = SimLedger()
        self.set_subsample(list(range(40)))

    def set_subsample(self, ix):
        self.subsample_ix = ix

    def subsample(self, y):
        return [y[i] for i in self.subsample_ix if (i+1)<=len(y)]

    def wealth(self, ix, tbds=None, mn_nfirms=10):
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
        simlist = self.subsample(simulator.load_list())

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
                               WHERE t>={tbds[0]} AND t<{tbds[1]}
                               GROUP BY t) ctable
                               INNER JOIN parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') otable
                               ON ctable.t = otable.t
                         WHERE ctable.n_firms>={mn_nfirms} AND 
                             ctable.t>={tbds[0]} AND
                             ctable.t<{tbds[1]}
                         '''
            wealth.append(self.con.execute(query).fetchdf().values)
        return wealth

    def wealth_by_firm(self, ix, tbds=None, iprint=False):
        """Set of wealth for all firms per time step.

        Parameters
        ----------
        ix : int
        tbds : twople, None

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
        simlist = self.subsample(simulator.load_list())
        
        wealth = []
        for thiskey in simlist:
            if iprint: print(f"Loading {thiskey}...")
            # there seems to be a bug in duckdb with columns for firm.fleft and right
            query = f'''
                     SELECT ids, t, wealth, (fright-fleft+1) AS fsize
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     WHERE (firm.t>={tbds[0]}) AND (firm.t<{tbds[1]})
                     ORDER BY firm.ids, firm.t
                     '''
            wealth.append(self.con.execute(query).fetchdf())

        return wealth

    def total_wealth(self, ix, tbds=None):
        """Total wealth summed over all firms per time step.
        """
        
        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())

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

    def lattice_width(self, ix, tbds=None, return_lr=False):
        """Width of lattice. Not counting leftmost lattice point, which is the absorbing
        boundary.

        Parameters
        ----------
        ix : int
        tbds : tuple, None
        return_lr : bool, False

        Returns
        -------
        list of ndarray
            cols (t, width)
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__'):
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())

        width = []
        if return_lr:
            for thiskey in simlist:
                query = f'''
                         SELECT lattice.t,
                                MAX(lattice.ix) - MIN(lattice.ix) AS width,
                                MIN(lattice.ix) as lleft,
                                MAX(lattice.ix) as lright
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet') lattice
                         WHERE t>={tbds[0]} AND t<{tbds[1]}
                         GROUP BY lattice.t
                         ORDER BY lattice.t
                         '''
                width.append(self.con.execute(query).fetchdf().values)
        else:
            for thiskey in simlist:
                query = f'''
                         SELECT lattice.t, MAX(lattice.ix) - MIN(lattice.ix)
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
        simlist = self.subsample(simulator.load_list())

        wealth = []
        for thiskey in simlist:
            query = f'''
                     SELECT lattice.t, SUM(wealth)/(MAX(lattice.ix) - MIN(lattice.ix) + 1) as wealth
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
        simlist = self.subsample(simulator.load_list())

        wealth = []
        for thiskey in simlist:
            query = f'''
                     SELECT t, firm.wealth / (firm.fright - firm.fleft + 1) as wealth
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                     ORDER BY t
                     '''
            wealth.append(self.con.execute(query).fetchdf())
        return wealth

    def bounds(self, ix, tbds=None, mn_width=0):
        """Left and right sides for each firm and lattice per time step.

        Parameters
        ----------
        ix : int
        tbds : tuple, None
        mn_width : int, 0
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())

        bds = []
        if mn_width:
            for thiskey in simlist:
                query = f'''
                         SELECT lattice.t,
                                firm.fleft,
                                firm.fright,
                                lattice.lleft as lat_left,
                                lattice.lright as lat_right
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                         INNER JOIN (SELECT t, MIN(ix) as lleft, MAX(ix) as lright
                                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet')
                                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                                     GROUP BY t) lattice
                            ON firm.t = lattice.t
                         WHERE (lattice.lright-lattice.lleft+1)>={mn_width}
                         ORDER BY lattice.t
                         '''
                bds.append(self.con.execute(query).fetchdf())
        else:
            for thiskey in simlist:
                query = f'''
                         SELECT firm.t,
                                firm.fleft,
                                firm.fright,
                                lattice.lleft as lat_left,
                                lattice.lright as lat_right
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                         INNER JOIN (SELECT t, MIN(ix) as lleft, MAX(ix) as lright
                                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet')
                                     WHERE t>={tbds[0]} AND t<{tbds[1]}
                                     GROUP BY t) lattice
                            ON firm.t = lattice.t
                         ORDER BY lattice.t
                         '''
                bds.append(self.con.execute(query).fetchdf())
        return bds

    def density(self, ix, tbds=None, side=None, mn_width=0, fill_in_missing_t=False):
        """Density of firms on lattice. Options to count only left or right most sides.

        TODO: speed up the loops?

        Parameters
        ----------
        ix : int
        tbds : tuple or int, None
        side : str, None
            'left', 'right'
        mn_width : int, 0
        fill_in_missing_t : bool, False
            This only does something when side is not None.

        Returns
        -------
        list
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())
        
        if side is None:
            if not os.path.isfile(f'{simulator.cache_dr}/{simlist[0]}_density.parquet'):
                print("Caching parquet file...", end='')
                parquet_density(ix)
                print('done!')

            density = []
            for thiskey in simlist:
                query = f'''
                         SELECT t, density
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}_density.parquet')
                         WHERE t>={tbds[0]} AND t<{tbds[1]}
                         ORDER BY t, ix
                        '''
                density.append([i[1]['density'].values
                                for i in self.con.execute(query).fetchdf().groupby('t')])

        else:
            bds = self.bounds(ix, tbds, mn_width=mn_width)

            if side=='left':
                def loop_wrapper(args):
                    t, group = args
                    lat_left = group.iloc[0]['lat_left']
                    # remember that left is always set to 0
                    thisdensity = ((group['fleft']+1)==lat_left).sum()
                    return thisdensity

            elif side=='right':
                def loop_wrapper(args):
                    t, group = args
                    lat_right = group.iloc[0]['lat_right']
                    thisdensity = (group['fright']==lat_right).sum()
                    return thisdensity

            else:
                raise NotImplementedError

            with Pool() as pool:
                # this form of the loop is more memory efficient than par-looping over bds
                density = []
                for bds_ in bds:
                    t_group = bds_.groupby('t')
                    density.append(np.array(list(pool.map(loop_wrapper, t_group))))
                    
                    # create an empty array of the full length of the time series and copy
                    # in the entries that we have; the rest must be zero
                    if fill_in_missing_t:
                        temp = np.zeros(tbds[1] - tbds[0], dtype=int)
                        temp[np.array(list(t_group.groups.keys())) - tbds[0]] = density[-1]
                        density[-1] = temp

        return density
    
    def no_firms(self, ix, tbds=None):
        """Number of firms at each time step with at least one firm.
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())
        
        nfirms = []
        for thiskey in simlist:
            query = f'''
                     SELECT t, COUNT(*) AS nfirms
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet')
                     GROUP BY t
                     ORDER BY t
                     '''
            nfirms.append(self.con.execute(query).fetchdf())
        return nfirms

    def firm_ids(self, ix, tbds=None, return_lr=False):
        """Ids of all firms per time step.
        
        Parameters
        ----------
        ix : int
        tbds : tuple, None
        return_lr : bool, False

        Returns
        -------
        list of pd.DataFrame
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())
        
        ids = []
        if return_lr:
            for thiskey in simlist:
                query = f'''
                         SELECT firm.t, firm.ids, fleft, fright
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                         WHERE firm.t>={tbds[0]} AND firm.t<{tbds[1]}
                         ORDER BY firm.t
                         '''
                ids.append(self.con.execute(query).fetchdf())
        else:
            for thiskey in simlist:
                query = f'''
                         SELECT firm.t, firm.ids
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                         WHERE firm.t>={tbds[0]} AND firm.t<{tbds[1]}
                         ORDER BY firm.t
                         '''
                ids.append(self.con.execute(query).fetchdf())

        return ids

    def lattice_death_rate(self, ix, tbds=None):
        """Death rate for each lattice point.

        Parameters
        ----------

        Returns
        -------
        ndarray
        """

        lattice_width = self.lattice_width(ix, tbds=tbds, return_lr=True)
        lleft = [i[:,2] for i in lattice_width]
        lright = [i[:,3] for i in lattice_width]
        
        def loop_wrapper(args):
            df, lleft, lright = args

            death_count = []
            t_group = df.groupby('t')
            for i, (t, group) in enumerate(t_group):
                if i==0:
                    prevt = t
                    prevgroup = group
                else:
                    if (t-prevt)>1:
                        # all firms died; don't count
                        pass
                    else:
                        # count which firms died
                        thislatleft = lleft[t-tbds[0]]
                        thislatright = lright[t-tbds[0]]
                        death_count.append(np.zeros(thislatright-thislatleft, dtype=int)) 

                        for fid in prevgroup.ids:
                            if not fid in group.ids.values:
                                left = prevgroup.loc[prevgroup['ids']==fid]['fleft'].values[0]
                                right = prevgroup.loc[prevgroup['ids']==fid]['fright'].values[0]
                                for j in range(left, right+1):
                                    death_count[-1][j-thislatleft-1] -= 1

                    prevt = t
                    prevgroup = group

            return death_count
        
        with Pool() as pool:
            rate_deaths = list(pool.map(loop_wrapper, zip(self.firm_ids(ix, tbds, return_lr=True),
                                                          lleft,
                                                          lright)))

        return rate_deaths

    def query(self, ix, q, tbds=None):
        """A general query.
        
        Parameters
        ----------
        ix : str or int
        q : str
            SQL query.

        Returns
        -------
        list of DataFrame
        """

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())

        q = q.replace('CACHE_DR', simulator.cache_dr)

        output = []
        for thiskey in simlist:
            output.append(self.con.execute(q.replace('KEY', thiskey)).fetchdf())
        return output
#end QueryRouter
