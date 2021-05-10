# ====================================================================================== #
# 1D firm innovation models
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from datetime import datetime
import string
from scipy.signal import fftconvolve
from time import perf_counter
import _pickle as cpickle
from time import sleep

from workspace.utils import save_pickle
from .model_ext import TopicLattice, LiteFirm, snapshot_firms
from .organizer import SimLedger
from .utils import *

LETTERS = list(string.ascii_letters)



# ========= #
# Functions #
# ========= #
def check_firms_occupancy(firms, lattice, iprint=False):
    """Testing function to check that firm occupancy is properly reflected in the lattice.
    
    Parameters
    ----------
    firms : list of Firm
    lattice : TopicLattice
    """
    
    occupancy = [0] * len(lattice.occupancy)
    for f in firms:
        for ix in range(f.sites[0], f.sites[1]+1):
            occupancy[ix-lattice.left] += 1
    if iprint:
        print("firms:  ", occupancy)
        print("lattice:", lattice.occupancy)
    assert occupancy==lattice.occupancy

def snapshot_occupancy(firms, xbds):
    """Measure occupancy for a particular range.
    
    Parameters
    ----------
    firms : list of Firm
    xbds : tuple
        Inclusive range.
    
    Returns
    -------
    ndarray
        Each entry counts the occupancy number within the interval xbds.
    """
    
    assert xbds[1]>=xbds[0]
    occ = np.zeros(xbds[1]-xbds[0]+1, dtype=int)
    
    for f in firms:
        # no overlap
        if f.sites[1]<xbds[0] or f.sites[0]>xbds[1]:
            pass
        # overlap but missing left side of firm
        elif f.sites[0]<xbds[0] and f.sites[1]>=xbds[0]:
            if f.sites[1]<=xbds[1]:
                for x in range(xbds[0], f.sites[1]+1):
                    occ[x-xbds[0]] += 1
            else:
                for x in range(xbds[0], xbds[1]+1):
                    occ[x-xbds[0]] += 1
        # overlap but missing right side of firm
        elif f.sites[1]>xbds[1] and f.sites[0]<=xbds[1]:
            if f.sites[0]<xbds[0]:
                for x in range(xbds[0], xbds[1]+1):
                    occ[x-xbds[0]] += 1
            else:
                for x in range(f.sites[0], xbds[1]+1):
                    occ[x-xbds[0]] += 1
        # entire firm fits inside interval
        elif f.sites[0]>=xbds[0] and f.sites[1]<=xbds[1]:
            for x in range(f.sites[0], f.sites[1]+1):
                occ[x-xbds[0]] += 1
        # entire interval is covered
        else:
            for x in range(xbds[0], xbds[1]+1):
                occ[x-xbds[0]] += 1
    return occ
    
def extract_deadfirms(fsnapshot):
    """Extract list of dead firms per time step.
    
    Parameters
    ----------
    fsnapshot : list of lists of Firm
    
    Returns
    -------
    list of lists of Firm
    """
    
    # extract firm ids
    ids = [[f.id for f in firms] for firms in fsnapshot]
    
    deadfirms = [[]]
    for i in range(len(ids)-1):
        deadfirms.append([])
        for j, thisid in enumerate(ids[i]):
            if not thisid in ids[i+1]:
                deadfirms[-1].append(fsnapshot[i][j])
    return deadfirms
    
def extract_growth_rate(fsnapshot):
    """Extract list of growth rates per firm over sequential time steps.
    
    Firms that don't survive into the next time step have growth rate of -1. This latter
    calculation may be somewhat slow.
    
    Parameters
    ----------
    fsnapshot : list of lists of Firm
    
    Returns
    -------
    list of list of floats
        Relative growth rate from previous time step, or (w_t - w_{t-1}) / w_{t-1}, the
        wealth levels used to calculate the ratio, and firm age.

        Empty elements (no companies to grow) are given infinities.
    """
    
    # extract firm ids for each snapshot
    ids = [[f.id for f in firms] for firms in fsnapshot]
    
    grate = []  # growth rate
    for i in range(1, len(ids)):
        grate.append([])
        selected = []
        
        # growth rate for all firms that survive into next time step
        for j, thisid in enumerate(ids[i]):
            if thisid in ids[i-1]:
                previx = ids[i-1].index(thisid)
                w0 = fsnapshot[i-1][previx].wealth
                w1 = fsnapshot[i][j].wealth
                t = fsnapshot[i][j].age
                grate[-1].append((w1/w0, w0, w1, t-1))
                selected.append(previx)
        # growth rate for firms that have died
        for j, thisid in enumerate(ids[i-1]):
            if not thisid in ids[i]:
                w0 = fsnapshot[i-1][j].wealth
                t = fsnapshot[i-1][j].age
                grate[-1].append((0, w0, 0, t))

        if not grate[-1]:  # label empty elements with inf
            grate[-1].append((np.inf, np.inf, np.inf, np.inf))
    return grate
    
def find_wide_segments(leftright, mn_width=50, mn_len=50):
    """Find continuous segments with differences between left and right boundaries wider
    than some min width.
    
    Parameters
    ----------
    leftright : ndarray
    mn_width : int, 100
        Min width of extents returned.
    mn_len : int, 50
        Min length of continuous sequence returned.
    
    Returns
    -------
    list of twoples
        Such that the interval can be extracted from the appropriate array by using the
        given left and right indices [left:right], i.e. the right index already contains
        the offset by 1.
    """
    
    d = np.diff(leftright, axis=1).ravel()
    d = np.diff((d>=mn_width).astype(int))
    ix = np.where(d)[0]
    bds = []
    
    if ix.size:
        # consider first element
        if d[ix[0]]==-1:
            bds.append((0, ix[0]+1))
            ix = ix[1:]
        for i in range(ix.size//2):
            bds.append((ix[i*2]+1, ix[i*2+1]+1))
        # consider last element
        if (ix.size%2):
            bds.append((ix[-1]+1, d.size+1))
        return [i for i in bds if (i[1]-i[0])>=mn_len]
    return []

def segment_by_bound_vel(leftright, zero_thresh,
                         moving_window=10,
                         smooth_vel=False):
    """Given the boundaries of the 1D lattice, segment time series into pieces with +/-/0
    velocities according to the defined threshold.

    Parameters
    ----------
    leftright : list or ndarray
    zero_thresh : float
    moving_window : int or tuple, 10
        Width of moving average window.
    smooth_vel : bool, False
        If True, smooth velocity after calculation from smoothed distance time series.
        Can specify separate moving windows by providing a tuple for moving_window.

    Returns
    -------
    list of twoples
        Boundaries of time segments. Note that this is calculated on vector that is one
        element shorter than leftright.
    list
        Sign of the velocity for each window segment.
    list
        Time-averaged velocities at each moment in time for each of the extracted
        segments.
    """
    
    if not hasattr(moving_window, '__len__'):
        if moving_window<5:
            warn("Window may be too small to get a smooth velocity.")
        moving_window = (moving_window, moving_window)
    
    if isinstance(leftright, list):
        leftright = np.vstack(leftright)
    else:
        assert leftright.ndim==2
        leftright.shape[1]==2

    width = leftright[:,1] - leftright[:,0]
    if moving_window[0]>1:
        swidth = fftconvolve(width, np.ones(moving_window[0])/moving_window[0], mode='same')
    else:
        swidth = width
    
    v = swidth[1:] - swidth[:-1]  # change width over a single time step
    rawv = v
    if smooth_vel:
        v = fftconvolve(v, np.ones(moving_window[1])/moving_window[1], mode='same')

    # consider the binary velocity
    # -1 as below threshold
    # 0 as bounded by threshold
    # 1 as above threshold
    vsign = np.zeros(v.size, dtype=int)
    vsign[v>zero_thresh] = 1
    vsign[v<-zero_thresh] = -1
    
    # find places where velocity switches sign
    # offset of 1 makes sure window counts to the last element of set
    ix = np.where(vsign[1:]!=vsign[:-1])[0] + 1
    windows = [(0, ix[0])] + [(ix[i], ix[i+1]) for i in range(ix.size-1)] + [(ix[-1], vsign.size)]
    
    vel = []  # velocities for each window
    velsign = []
    for w in windows:
        vel.append(rawv[w[0]:w[1]])  
        velsign.append(vsign[w[0]])

    return windows, velsign, vel

def reconstruct_lattice(firm_snapshot, lattice_bds):
    """Reconstruct lattice from lattice bounds.

    Parameters
    ----------
    firm_snapshot : list of LiteFirm
    lattice_bds : list of twople

    Returns
    -------
    list of LiteTopicLattice
    """
    
    lattice_snapshot = []

    for firms, bds in zip(firm_snapshot, lattice_bds):
        lattice_snapshot.append(LiteTopicLattice(bds))
        
        for f in firms:
            lattice_snapshot[-1].occupancy[f.sites[0]-bds[0]:f.sites[1]+1-bds[0]] += 1

    return lattice_snapshot



# ======= #
# Classes #
# ======= #
class Simulator():
    def __init__(self,
                 L0=20,
                 N0=20,
                 g0=.2,
                 obs_rate=.49,
                 expand_rate=.5,
                 innov_success_rate=.6,
                 depression_rate=.2,
                 inn_mut_width=.05,
                 replication_p=.95,
                 growf=.9,
                 connect_cost=0.,
                 rng=None,
                 cache_dr=None):
        """
        Parameters
        ----------
        L0 : int, 20
            Initial size of lattice.
        N0 : int, 20
            Initial population size.
        g0 : float, .2
            New firm rate per site.
        obs_rate : float, .49
            Obsolescence rate (from left).
        expand_rate : float, .5
            Rate at which firms try to expand.
        innov_success_rate : float, .6
        depression_rate : float, .2
        inn_mut_width : float, .05
            Width of the normal distribution determining mutant innovation.
        replication_p : float, .95
            Probability that new firm is a mutant replicate of an existing
            firm's innovation.
        growf : float, .9
            Growth cost fraction f. If this is higher, then it is more expensive
            to expand into new sectors.
        connect_cost : float, 0.
        """
        
        self.L0 = L0
        self.N0 = N0
        self.g0 = g0
        self.obs_rate = obs_rate
        self.expand_rate = expand_rate
        self.innov_success_rate = innov_success_rate
        self.depression_rate = depression_rate
        self.inn_mut_width = inn_mut_width
        self.replication_p = replication_p
        self.growf = growf
        self.connect_cost = connect_cost
        self.rng = rng or np.random
        self.cache_dr = cache_dr

        self.storage = {}  # store previous sim results

    def simulate(self, T, cache=True, reset_rng=False):
        """Run firm simulation for T time steps. To minimize memory and cost of saving sim
        results, only LiteFirms and lattice boundaries are stored.
        
        Parameters
        ----------
        T : int
            Number of iterations starting with an initially empty lattice.
        cache : bool, True
            If True, save simulation result into self.storage dict.
        reset_rng : bool, False
            If True, reinitialize rng. Useful for parallel processing.
            
        Returns
        -------
        list
            LiteFirm copies at each time step.
        list
            Lattice copies at each time step.
        """

        if reset_rng:
            self.rng = np.random.RandomState()
        
        # settings
        L0 = self.L0
        N0 = self.N0
        g0 = self.g0
        obs_rate = self.obs_rate
        expand_rate = self.expand_rate
        innov_success_rate = self.innov_success_rate
        depression_rate = self.depression_rate
        inn_mut_width = self.inn_mut_width
        replication_p = self.replication_p
        growf = self.growf
        c_cost = self.connect_cost
        
        # variables for tracking history
        firm_snapshot = []
        lattice_snapshot = []
        
        # updated parameters
        firms = []
        lattice = TopicLattice()
        # initialize empty lattice
        lattice.extend_right(L0-1)

        # initialize
        for i in range(N0):
            # choose innovation tendency uniformly? how to impose selection?
            firms.append(Firm(self.rng.randint(lattice.left, lattice.right+1),
                              self.rng.rand(),
                              lattice=lattice,
                              wealth=growf,
                              connection_cost=c_cost,
                              rng=self.rng))
            lattice.d_add(firms[-1].sites[0])
        lattice.push()

        # run sim
        for t in range(T):
            # spawn new firms either by mutating from a sample of existing firms or by random sampling
            # from a uniform distribution
            nNew = self.rng.poisson(g0)
            for i in range(nNew):
                if len(firms) and self.rng.rand() <= replication_p:
                    # mutate an existing firm's innov
                    newInnov = np.clip(self.rng.choice(firms).innov + self.rng.normal(scale=inn_mut_width),
                                       0, 1)
                    firms.append(Firm(self.rng.randint(lattice.left, lattice.right+1),
                                      newInnov,
                                      lattice=lattice,
                                      rng=self.rng))
                    lattice.d_add(firms[-1].sites[0])
                else:
                    firms.append(Firm(self.rng.randint(lattice.left, lattice.right+1), self.rng.rand(),
                                      lattice=lattice,
                                      rng=self.rng))
                    lattice.d_add(firms[-1].sites[0])
            lattice.push()
        #     check_firms_occupancy(firms, lattice)

            # calculate firm income and growth
            grow_lattice = 0  # switch for if lattice needs to be grown
            new_occupancy = 0  # count new firms occupying innovated area
            if depression_rate:
                # these are ordered (previous attempts at speedup)
                #ix  = self.rng.rand(lattice.right-lattice.left+1) < depression_rate
                #depressedSites = np.arange(lattice.left, lattice.right+1)[ix]
                #depressedSites = [i+lattice.left for i, el in enumerate(ix) if el]
                depressedSites = [i
                                  for i in range(lattice.left, lattice.right+1)
                                  if self.rng.rand() < depression_rate]
            else:
                depressedSites = []
                
            for f in firms:
                income = f.income(depressedSites)
                f.wealth += income
                f.age += 1

                growthcost = growf * f.wealth/f.size()
                eps = .01
                if f.wealth>(growthcost+eps) and f.rng.rand()<expand_rate:
                    out = f.grow(innov_success_rate, cost=growthcost)
                    if out[0] and not out[1]:
                        lattice.d_add(f.sites[1])
                    elif out[0] and out[1]:
                        new_occupancy += 1
                        grow_lattice += out[1]
                    elif not out[0] and out[1]:
                        grow_lattice += 1
            lattice.push()
            # if any firm innovated, then grow the lattice
            if grow_lattice:
                lattice.extend_right()
                lattice.add(lattice.right, new_occupancy)
        #     check_firms_occupancy(firms, lattice)

            # shrink topic lattice
            if self.rng.rand() < obs_rate and lattice.left < lattice.right:
                lattice.shrink_left()
                # shrink all firms that have any value on the obsolete topic
                # firms of size one will be delete by accounting for negative wealth next
                for i, f in enumerate(firms):
                    if f.sites[0]<=lattice.left:  # s.t. lattice of size 1 incurs loss!
                        # lose fraction of wealth invested in that site
                        f.wealth -= f.wealth / (f.sites[1] - f.sites[0] + 1)
                        f.sites = f.sites[0]+1, f.sites[1]
        #     check_firms_occupancy(firms, lattice)

            # kill all firms with negative wealth
            removeix = []
            for i, f in enumerate(firms):
                if f.wealth <= 0:
                    removeix.append(i)
            for count, ix in enumerate(removeix):
                f = firms.pop(ix - count)
                for i in range(f.sites[0], f.sites[1]+1):
                    lattice.remove(i)
        #     check_firms_occupancy(firms, lattice)
        
            # if there are at least two firms
            # split up any firm making up >10% of total wealth and occupying >10 sites
            # this only imposes a finite cutoff
#            if len(firms)>1:
#                w = np.array([f.wealth for f in firms])
#                totwealth = w.sum()
#                fwealth = w / totwealth
#                ix = np.where(fwealth>.1)[0]
#                if ix.size:
#                    counter = 0
#                    babyfirms = []
#                    for i in ix:
#                        if firms[i-counter].size()>10:
#                            babyfirms += firms.pop(i-counter).split()
#                            counter += 1
#                    firms += babyfirms

            # collect data on status
            firm_snapshot.append(snapshot_firms(firms))
            lattice_snapshot.append((lattice.left, lattice.right))  # lattice endpts
        
        if cache:
            self.storage[str(datetime.now())] = firm_snapshot, lattice_snapshot
        return firm_snapshot, lattice_snapshot
        
    def parallel_simulate(self, n_samples, T,
                          min_nfirms=0,
                          min_success=1,
                          iprint=True,
                          n_cpus=None):
        """Parallelize self.simulate() and save results into self.storage data member dict.
        
        Parameters
        ----------
        n_samples : int
        T : int
        min_nfirms : int, 0
            Only trajectories above this min by average no. of firms are saved.
        min_success : int, 1
            Min no. of successful sims required.
        iprint : bool, True
        n_cpus : int, None
            Defaults to the max number of CPUs.
        
        Returns
        -------
        None
        """
        
        assert len(self.storage)==0, "Avoid piping large members."

        n_cpus = n_cpus or cpu_count()
        t0 = perf_counter()

        def loop_wrapper(args, self=self, T=T, min_nfirms=min_nfirms):
            firm_snapshot, lattice_snapshot = self.simulate(T, cache=True, reset_rng=True)

            avgn = np.mean([len(f) for f in firm_snapshot])
            if avgn >= min_nfirms:
                self.save()
                return True
            return False

        if min_nfirms:
            # TODO: there is a better way to do this than by running groups at a time
            storage = {}
            with threadpool_limits(user_api='blas', limits=1):
                with Pool(n_cpus) as pool:
                    success_count = 0
                    while success_count < min_success:
                        success_count += sum(list(pool.map(loop_wrapper, range(n_samples))))

            for k, val in storage.items():
                self.storage[k] = val

        else:
            with threadpool_limits(user_api='blas', limits=1):
                with Pool(n_cpus) as pool:
                    pool.map(loop_wrapper, range(n_samples))

        if iprint: print(f"Runtime of {perf_counter()-t0} s.")
            
    def save(self, iprint=True):
        """Save simulator instance with each simulation run in a separate pickle.
        Ledger must be updated separately.
        
        Parameters
        ----------
        iprint : bool, True
        """
        
        assert not self.cache_dr is None
        if not os.path.isdir(self.cache_dr):
            os.makedirs(self.cache_dr)
        
        if not os.path.isfile(f'{self.cache_dr}/top.p'):
            while os.path.isdir(f'{self.cache_dr}/lock'):
                print(f"Directory {self.cache_dr} is locked. Waiting 1 second...")
                sleep(1)
            with open(f'{self.cache_dr}/lock', 'w') as f:
                f.write('')

            # save class info without all of storage
            storage = self.storage
            self.storage = {}
            with open(f'{self.cache_dr}/top.p', 'wb') as f:
                pickle.dump({'simulator':self}, f)

            os.remove(f'{self.cache_dr}/lock')
            self.storage = storage
        
        # save values of storage dict into separate pickles
        for k in self.storage.keys():
            with open(f'{self.cache_dr}/{k}.p', 'wb') as f:
                pickle.dump({'storage':self.storage[k]}, f)

        if iprint: print(f"Saved simulator instances in cache.")
    
    def load_list(self):
        """Show list of sims possible to load from cache directory.

        Returns
        -------
        list of str
        """

        if not 'cache_dr' in self.__dict__.keys():
            raise Exception("No cache specified.")

        return ['.'.join(i.split('.')[:-1]) for i in os.listdir(self.cache_dr) if not 'top' in i]

    def load(self, name):
        """Load an individual simulation run with given name.

        Parameters
        ----------
        name : str or tuple of str
        """
        
        if not isinstance(name, str) and hasattr(name, '__len__'):
            for n in name:
                self.load(n)
            return

        if not 'cache_dr' in self.__dict__.keys():
            raise Exception("No cache specified.")
        if name=='top':
            raise Exception("top file name reserved")
        
        if name[-2:]=='.p':
            with open(f'{self.cache_dr}/{name}', 'rb') as f:
                self.storage[name] = pickle.load(f)['storage']
        else:
            with open(f'{self.cache_dr}/{name}.p', 'rb') as f:
                self.storage[name] = pickle.load(f)['storage']

    def add_to_ledger(self, f, extra_props={}):
        """
        Parameters
        ----------
        f : str
            File name.
        extra_props : dict, {}
            Extra properties to add to ledger such as simulation duration.
        """
        
        extra_props.update({'L0':self.L0,
                            'N0':self.N0,
                            'g0':self.g0,
                            'obs_rate':self.obs_rate,
                            'expand_rate':self.expand_rate,
                            'innov_success_rate':self.innov_success_rate,
                            'depression_rate':self.depression_rate,
                            'inn_mut_width':self.inn_mut_width,
                            'replication_p':self.replication_p,
                            'growf':self.growf,
                            'connect_cost':self.connect_cost,
                            'n_sims':len(self.storage)})
        ledger = SimLedger()
        ledger.add(f, extra_props)
    
    def info(self):
        """Show parameters."""
        
        print(f'new firm rate   =\t{self.g0}')
        print(f'grow frac cost  =\t{self.growf}')
        print(f'depression rate =\t{self.depression_rate}')
        print(f'connection cost =\t{self.connect_cost}')
        print()

        print(f'This instance has {len(self.storage)} sims run on')
        for k in self.storage.keys():
            print(k)
        print()
#end Simulator



class Firm():
    """One-dimensional firm on semi-infinite line. Firms can only grow to the right!
    """
    def __init__(self, sites, innov,
                 lattice=None,
                 connection_cost=0.,
                 wealth=1.,
                 id=None,
                 rng=None):
        """
        Parameters
        ----------
        sites : tuple
            Left and right boundaries of firm. If a int, then it assumes that firm is
            localized to a single site.
        innov : float
            Innovation parameter. This determines the probability with which the firm
            will choose to expand into a new region or go into an already explored region.
            This is much more interesting in topic space beyond one dimension.
        lattice : TopicLattice
        connection_cost : float, 0.
        wealth : float, 1.
        id : str, None
        rng : np.random.RandomState, None
        """
        
        if hasattr(sites, '__len__'):
            assert sites[1]>=sites[0]
            self.sites = sites
        else:
            self.sites = sites, sites
        self.lattice = lattice
        assert 0<=innov<=1
        self.innov = innov
        self.wealth = wealth
        self.age = 0
        self.connect_cost = connection_cost
        
        self.rng = rng or np.random.RandomState()
        self.id = id or ''.join(self.rng.choice(LETTERS, size=20))
            
    def size(self):
        return self.sites[1]-self.sites[0]+1

    def income(self, depressed_sites=[]):
        """For each site, wealth grows by the amount
        (W/N) * di where di is the inverse site occupancy number. For depressed sites, wealth
        decreases by this amount.
        
        Parameters
        ----------
        depressed_sites : ndarray of type int, np.zeros()
            Sites that see decreased income.
            
        Returns
        -------
        float
            Total change in wealth for this firm over one time step.
        """

        income = 0.
        fwealth = self.wealth / self.size()
        
        # avoid iterating thru unnecessary depressed sites
        counter = 0
        while (counter < len(depressed_sites)) and (depressed_sites[counter] < self.sites[0]):
            counter += 1
        _depressed_sites = depressed_sites[counter:]
        
        go = True
        while len(_depressed_sites) and go:
            if _depressed_sites[-1] > self.sites[1]:
                _depressed_sites.pop(-1)
            else:
                go = False
        
        if len(_depressed_sites):
            for s in range(self.sites[0], self.sites[1]+1):
                if s in _depressed_sites:
                    income -= fwealth / self.lattice.get_occ(s)
                else:
                    income += fwealth / self.lattice.get_occ(s)
        else:
            for s in range(self.sites[0], self.sites[1]+1):
                income += fwealth / self.lattice.get_occ(s)
        #income -= self.wealth * self.connect_cost * np.log(self.size())
        income -= self.wealth * self.connect_cost * self.size()
        return income
        
        #dincome = (self.wealth / self.size() / 
        #           self.lattice.get_occ(np.arange(self.sites[0], self.sites[1]+1, dtype=np.int32)))

        #ix = (depressed_sites>=self.sites[0]) & (depressed_sites<=self.sites[1])
        #ix = depressed_sites[ix] - self.sites[0]

        #income = dincome.sum() - 2 * dincome[ix].sum()

        #income -= self.wealth * self.connect_cost * (self.sites[1]-self.sites[0]+1)
        #return income
        
    def grow(self, expansion_p, cost=None):
        """Attempt to expand to a neighboring site, and deplete resources whether
        or not attempt succeeded. This makes innovation expensive. If no innovation
        required, then expansion is guaranteed to succeed.
        
        Parameters
        ----------
        expansion_p : float
            Probability of extending into newly discovered region on lattice, i.e. the difficulty
            of exploiting innovation immediately.
        cost : float, None
            If None, then wealth is halved with growth.
            
        Returns
        -------
        int
            If 1, then firm expanded to right.
            If 0, then firm did not expand.
        int
            If 1, then lattice expanded to right.
            If 0, then lattice should not be extended.
        """
        
        lattice = self.lattice
        
        # always pay a cost for attempting to grow
        if cost is None:
            self.wealth /= 2
        else:
            self.wealth -= cost
            
        # consider tendency to innovate and the prob of exploiting it
        # innovation may be rewarding but not necessarily
        assert self.sites[1]<=lattice.right
        if self.sites[1]==lattice.right:
            if self.rng.random() < self.innov:  # successful innovation?
                if self.rng.random() < expansion_p:  # successful exploitation?
                    self.sites = self.sites[0], self.sites[1]+1
                    return 1, 1
                return 0, 1
            return 0, 0
        # if no innovation is necessary to grow
        self.sites = self.sites[0], self.sites[1] + 1
        return 1, 0
        
    def split(self, n_split=2):
        """Split firm into equally-sized smaller firms. Split wealth equally.
        
        Parameters
        ----------
        n_split : int, 2
        
        Returns
        -------
        list of Firm
        """
        
        ix = [int((self.sites[1]-self.sites[0]+1) * i/n_split)+self.sites[0]
              for i in range(n_split)]
        ix.append(self.sites[1])
        
        babyfirms = []
        for i in range(len(ix)-1):
            # copy everything besides splitting lattice and wealth
            babyfirms.append(Firm((ix[i],ix[i+1]),
                                  self.innov,
                                  lattice=self.lattice,
                                  connection_cost=self.connect_cost,
                                  wealth=self.wealth/n_split))
        return babyfirms
        
    def copy(self):
        return LiteFirm(self.sites,
                        self.innov,
                        self.wealth,
                        self.connect_cost,
                        self.age,
                        self.id)
#end Firm


class _LiteFirm(Firm):
    """One-dimensional firm copy.
    """
    def __init__(self,
                 sites,
                 innov,
                 connection_cost,
                 wealth,
                 age,
                 id):
        """
        Parameters
        ----------
        sites : tuple
            Left and right boundaries of firm. If a int, then it assumes that firm is
            localized to a single site.
        innov : float
            Innovation parameter. This determines the probability with which the firm
            will choose to expand into a new region or go into an already explored region.
            This is much more interesting in topic space beyond one dimension.
        connection_cost : float, 0.
        wealth : float, 1.
        age : int
        id : str
        """
        
        self.sites = sites
        self.innov = innov
        self.wealth = wealth
        self.connection_cost = connection_cost
        self.age = age
        self.id = id
#end _LiteFirm


    
class _TopicLattice():
    def __init__(self):
        """Start out one-dimensional lattice centered about 0."""
        
        self.left = 0
        self.right = 0
        self.occupancy = [0]
        self.d_occupancy = [0]
        
    # ============== #
    # Info functions #
    # ============== #
    def get_occ(self, i):
        if hasattr(i, '__len__'):
            return [self.occupancy[i_-self.left] for i_ in i]
        return self.occupancy[i-self.left]
        
    def len(self):
        return len(self.occupancy)
        
    # ============= #
    # Mod functions #
    # ============= #
    def shrink_left(self, n=1):
        assert n>=1
        assert self.left+n<=self.right
        self.left += n
        self.occupancy = self.occupancy[n:]
        self.d_occupancy = self.d_occupancy[n:]
    
    def extend_left(self, n=1):
        """
        Parameters
        ----------
        n : int, 1
            Number of spots by which to extend lattice.
        """
        
        self.left -= n
        self.occupancy = [0] * n + self.occupancy
        self.d_occupancy = [0] * n + self.d_occupancy
    
    def extend_right(self, n=1):
        """
        Parameters
        ----------
        n : int, 1
            Number of spots by which to extend lattice.
        """
        
        self.occupancy += [0] * n
        self.d_occupancy += [0] * n
        self.right += n
        
    def remove(self, i, d=1):
        """Remove one occupant from lattice site i.
        
        Parameters
        ----------
        i : int
            This is the coordinate and not the number of bins from the leftmost spot.
        d : int, 1
        """
        
        assert self.occupancy[i-self.left]
        self.occupancy[i-self.left] -= d
        
    def add(self, i, d=1):
        """Add one occupant to lattice site i.
        
        Parameters
        ----------
        i : int
            This is the coordinate and not the number of bins from the leftmost spot.
        d : int, 1
        """
        
        self.occupancy[i-self.left] += d
        
    def push(self):
        """Push stored changes to occupancy and reset self.d_occupancy.
        """
        
        for i in range(len(self.occupancy)):
            self.occupancy[i] += self.d_occupancy[i]
        self.d_occupancy = [0] * len(self.occupancy)
        
    def copy(self):
        """Copy the topic lattice.
        """
        
        copy = TopicLattice()
        copy.left = self.left
        copy.right = self.right
        copy.occupancy = self.occupancy[:]
        return copy
#end TopicLattice



class LiteTopicLattice():
    def __init__(self, bds):
        """
        Parameters
        ----------
        bds : twople
        """

        self.left = bds[0]
        self.right = bds[1]
        self.bds = bds
    
        self.occupancy = np.zeros(bds[1]-bds[0]+1, dtype=int)
#end LiteTopicLattice
