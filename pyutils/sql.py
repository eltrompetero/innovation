# ====================================================================================== #
# Wrappers for quick SQL queries.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb
import fastparquet as fp
from itertools import chain
from uuid import uuid4
import shutil

from .organizer import SimLedger
from .mft import density_bounds
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
                  key,
                  t0=0):
    """Meat of setup_parquet().

    Parameters
    ----------
    firm_snapshot : list
    cache_dr : str
    key : str
    t0 : int or float, 0
        Time offset.
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
        t.extend([i+t0]*len(firms))
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
    
    if os.path.isfile(f'{cache_dr}/{key}.parquet'):
        fp.write(f'{cache_dr}/{key}.parquet', df, 1_000_000,
                 compression='SNAPPY', append=True)
    else:
        fp.write(f'{cache_dr}/{key}.parquet', df, 1_000_000,
                 compression='SNAPPY')
 
def parquet_lattice(lattice_snapshot,
                    cache_dr,
                    key,
                    t0=0):
    """Meat of setup_parquet().

    Parameters
    ----------
    lattice_snapshot : list
    cache_dr : str
    key : str
    t0 : int or float, 0
        Time offset.
    """
       
    # write lattice to parquet file
    occupancy = []
    index = []
    t = []
    for i, lattice in enumerate(lattice_snapshot):
        occupancy.append(lattice.occupancy)
        index.append(np.arange(lattice.left, lattice.right+1))
        t.append([i+t0]*occupancy[-1].size)

    df = pd.DataFrame({'occupancy':np.concatenate(occupancy),
                       'ix':np.concatenate(index),
                       't':np.concatenate(t)})
    
    if os.path.isfile(f'{cache_dr}/{key}_lattice.parquet'):
        fp.write(f'{cache_dr}/{key}_lattice.parquet', df, 1_000_000,
                 compression='SNAPPY', append=True)
    else:
        fp.write(f'{cache_dr}/{key}_lattice.parquet', df, 1_000_000,
                 compression='SNAPPY')

def parquet_density(ix, n_cpus):
    """Create parquet file storing density for specified cache in simulator.

    Parameters
    ----------
    ix : int
        Index of simulation in sim ledger.
    n_cpus : int 
    """
    
    # set up
    simledger = SimLedger()
    simulator = simledger.load(ix)
    simlist = simulator.load_list()
    qr = QueryRouter()
    tmpfile = dict([(i, f'/fastcache/corp/{uuid4()}.parquet') for i in simlist])
    assert all([not os.path.isfile(i) for i in tmpfile.values()])

    def loop_wrapper(args):
        """For each time step of sim.
        """

        # for each time step in sim, create a vector of size of lattice, and
        # iterate thru each firm counting up any site it occupies
        # if there are no firms, then this adds nothing to the density (this is b/c 
        # the returned bounds only have an entry if there are firms)
        t, group = args
        # get rid of time col
        group = group.iloc[:,1:]
        # lattice bounds
        lat_left = group.iloc[0]['lat_left']
        lat_right = group.iloc[0]['lat_right']
        thisdensity = np.zeros(lat_right-lat_left+1, dtype=np.int32)
        thisldensity = np.zeros(lat_right-lat_left+1, dtype=np.int32)
        thisrdensity = np.zeros(lat_right-lat_left+1, dtype=np.int32)

        for i, row in group.iterrows():
            thisdensity[row['fleft']-lat_left:row['fright']-lat_left+1] += 1
            thisldensity[row['fleft']-lat_left] += 1
            thisrdensity[row['fright']-lat_left] += 1

        return list(zip([t]*(lat_right-lat_left+1),
                        range(lat_left, lat_right+1),
                        thisdensity,
                        thisldensity,
                        thisrdensity))

    with Pool(n_cpus) as pool:
        # get boundaries of lattice and firms in 10_000 step increments
        bds = qr.bounds(ix, tbds=(0, 10_000))

        # this form of the loop is more memory efficient than par-looping over bds which
        # is indexed by the random trajectory (if slower)
        counter = 0 
        while any([len(i) for i in bds]):
            for bds_, key in zip(bds, simlist):
                density = list(chain(*pool.map(loop_wrapper, bds_.groupby('t'))))
                density = pd.DataFrame(density, columns=['t','ix','density','ldensity','rdensity'])
                if counter:
                    fp.write(tmpfile[key], density, 1_000_000,
                             compression='SNAPPY', append=True)
                else:
                    fp.write(tmpfile[key], density, 1_000_000,
                             compression='SNAPPY')
 
            counter += 1 
            bds = qr.bounds(ix, tbds=(10_000*counter, 10_000*(counter+1)))
    
    for key, f in tmpfile.items():
        # move from fastcache to cache
        shutil.move(f, f'./{simulator.cache_dr}/{key}_density.parquet')

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
            additional optional (left, right) cols
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
                         SELECT firm.t,
                                firm.fleft,
                                firm.fright,
                                lattice.lleft as lat_left,
                                lattice.lright as lat_right
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                         INNER JOIN (SELECT t, MIN(ix) as lleft, MAX(ix) as lright
                                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet')
                                     GROUP BY t) lattice
                            ON firm.t = lattice.t
                         WHERE (lattice.lright-lattice.lleft+1)>={mn_width} AND
                               lattice.t>={tbds[0]} AND lattice.t<{tbds[1]}
                         ORDER BY firm.t
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
                                     GROUP BY t) lattice
                            ON firm.t = lattice.t
                         WHERE lattice.t>={tbds[0]} AND lattice.t<{tbds[1]}
                         ORDER BY firm.t
                         '''
                bds.append(self.con.execute(query).fetchdf())
        return bds

    def density(self, ix, tbds=None, side=None, mn_width=0, n_cpus=None,
                fill_in_missing_t=False):
        """Density of firms on lattice. Options to count only left or right most sides.

        TODO: speed up the loops?

        Parameters
        ----------
        ix : int
        tbds : tuple or int, None
        side : str, None
            'left', 'right'
        mn_width : int, 0
        n_cpus : int, None
        fill_in_missing_t : bool, False

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
        
        if not os.path.isfile(f'{simulator.cache_dr}/{simlist[0]}_density.parquet'):
            print("Caching parquet file...", end='')
            parquet_density(ix, n_cpus)
            print('done!')
        
        density = []
        t = []

        if side is None:
            for thiskey in simlist:
                query = f'''
                         SELECT t, density
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}_density.parquet')
                         WHERE t>={tbds[0]} AND t<{tbds[1]}
                         ORDER BY t, ix
                        '''
                groups = self.con.execute(query).fetchdf().groupby('t')
                t.append(list(groups.groups.keys()))
                density.append([i[1]['density'].values for i in groups])

        elif side=='left':
            for thiskey in simlist:
                query = f'''
                         SELECT t, ldensity
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}_density.parquet')
                         WHERE t>={tbds[0]} AND t<{tbds[1]}
                         ORDER BY t, ix
                        '''
                groups = self.con.execute(query).fetchdf().groupby('t')
                t.append(list(groups.groups.keys()))
                density.append([i[1]['ldensity'].values for i in groups])

        elif side=='right':
            for thiskey in simlist:
                query = f'''
                         SELECT t, rdensity
                         FROM parquet_scan('{simulator.cache_dr}/{thiskey}_density.parquet')
                         WHERE t>={tbds[0]} AND t<{tbds[1]}
                         ORDER BY t, ix
                        '''
                groups = self.con.execute(query).fetchdf().groupby('t')
                t.append(list(groups.groups.keys()))
                density.append([i[1]['rdensity'].values for i in groups])
        else:
            raise Exception("Invalid side argument.")
        
        if fill_in_missing_t:
            # get lattice width for every single time point and use it to fill in empty density vectors
            lat_width = self.lattice_width(ix, tbds)
            
            for i in range(len(lat_width)):
                counter = 0
                for t_ in t[i]:
                    while (t_-tbds[0]) > counter:
                        density[i].insert(counter+1, np.zeros(lat_width[i][counter,1]+1, dtype=int))
                        counter += 1
                    counter += 1

                # must handle case taking us to the end of tbds
                t_ = tbds[1]
                while (t_-tbds[0]) > counter:
                    density[i].insert(counter+1, np.zeros(lat_width[i][counter,1]+1, dtype=int))
                    counter += 1
                counter += 1

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
        """Ids of all firms per time step. When there are no firms, the time is not
        included.
        
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

    def innov_front_death_rate(self, ix, tbds=None):
        """Effective death rate on innovation front. This counts the rate at which any
        single firm disappears from the right front, i.e. it may still be alive but no
        longer occupying the innovation front. This is the effective death rate term in
        the density equation.

        Parameters
        ----------
        ix : int
        tbds : twople, None

        Returns
        -------
        ndarray
            Death count for each time point it could be measured.
        """

        if tbds is None:
            tbds = 0
        if not hasattr(tbds, '__len__') or len(tbds)==1:
            tbds = (tbds, 10_000_000)

        simulator = self.simledger.load(ix)
        simlist = self.subsample(simulator.load_list())
 
        def loop_wrapper(thiskey):
            query = f'''
                     SELECT firm.t, firm.ids, firm.fleft, firm.fright, lattice.lleft, lattice.lright
                     FROM parquet_scan('{simulator.cache_dr}/{thiskey}.parquet') firm
                     INNER JOIN (SELECT t, MIN(ix) AS lleft, MAX(ix) AS lright
                                 FROM parquet_scan('{simulator.cache_dr}/{thiskey}_lattice.parquet')
                                 GROUP BY t)
                                 AS lattice
                        ON firm.t = lattice.t
                     WHERE firm.t>={tbds[0]} AND firm.t<{tbds[1]}
                     '''
            qr = QueryRouter()
            df = qr.con.execute(query).fetchdf()
            lleft = df['lleft']
            lright = df['lright']
            dt = np.diff(np.unique(df['t'])[:2])  # assuming that all time diffs are equal

            death_count = []
            t_group = df.groupby('t')
            for i, (t, group) in enumerate(t_group):
                if i==0:
                    prevt = t
                    prevgroup = group
                    prevlatright = group['lright'].iloc[0]
                else:
                    nowlatright = group['lright'].iloc[0]
                    if not np.isclose(t-prevt, dt):
                        # no firms, meaning that all firms previously died in one time step
                        # any firm touching right side should contribute to death rate
                        for fid in prevgroup.ids:
                            prev_at_right = prevgroup.loc[prevgroup['ids']==fid]['fright'].values[0]==prevlatright
                            # we only care about firms that were previously touching right front
                            if prev_at_right:
                                death_count.append(-1)
                    else:
                        # count which firms died
                        # a firm has "died" only if it was previously touching the innovation front and is no
                        # longer
                        for fid in prevgroup.ids:
                            prev_at_right = prevgroup.loc[prevgroup['ids']==fid]['fright'].values[0]==prevlatright
                            # we only care about firms that were previously touching right front
                            if prev_at_right:
                                # firm died
                                if not fid in group.ids.values:
                                    death_count.append(-1)
                                else:
                                    now_at_right = group.loc[group['ids']==fid]['fright'].values[0]==nowlatright
                                    if now_at_right:  # firm maintains innov edge
                                        death_count.append(0)
                                    else:  # firm falls behind
                                        death_count.append(-1)

                    prevt = t
                    prevgroup = group
                    prevlatright = nowlatright

            return np.array(death_count)
        
        with Pool() as pool:
            rate_deaths = list(pool.map(loop_wrapper, simlist))

        return rate_deaths

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

    def innov_density(self, simix, windows):
        """Time-averaged innovation front density.
        
        Parameters
        ----------
        simix : int
        windows : list of twoples
            Start and end times for windows over which to time-average.
        tbds : twople, None
        
        Returns
        -------
        ndarray
            MFT predicted right speed for each random trajectory.
        ndarray
            Innovation front velocities for random trajectories. Each row is a sim.
        """
        
        def loop_wrapper(window, simix=simix, vi=self.simledger.ledger.iloc[simix]['innov_rate']):
            # to make this work, must declare copy of QR instance inside loop b/c it doesn't pickle well
            qr = QueryRouter()
            density = qr.density(simix, window)

            # assuming that vo and vg are the typical ones
            mft, sim = density_bounds(density, window[1]-window[0], vi)
            return mft, sim
        
        with Pool() as pool:
            loop_wrapper(windows[1]);
            mft, sim = list(zip(*pool.map(loop_wrapper, windows)))
            # mft density is determined by parameters of sim so they're all the same
            mft = mft[0]
            sim = np.vstack(sim).T
        return mft, sim

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
