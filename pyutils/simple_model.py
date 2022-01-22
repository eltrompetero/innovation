# ====================================================================================== #
# Firm innovation models without wealth dynamics for simplified model.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from numba import njit, jit
from numba.typed import List
from scipy.optimize import minimize
from cmath import sqrt

from workspace.utils import save_pickle
from .model_ext import TopicLattice, LiteFirm, snapshot_firms
from .organizer import SimLedger
from .utils import *
from . import sql
from .model import *



def match_length(y1, y2, side1='l', side2='r'):
    """Fill zeros to match two vectors. Pad zeros on either left or right sides.

    Parameters
    ----------
    y1 : ndarray
    y2 : ndarray
    side1 : str, 'l'
    side2 : str, 'r'

    Returns
    -------
    ndarray
    ndarray
    """
    
    if side1=='l':
        y1 = y1[::-1]
    if side2=='l':
        y2 = y2[::-1]

    if y1.size > y2.size:
        y2 = np.concatenate((y2, np.zeros(y1.size-y2.size)))
    elif y2.size > y1.size:
        y1 = np.concatenate((y1, np.zeros(y2.size-y1.size)))

    if side1=='l':
        y1 = y1[::-1]
    if side2=='l':
        y2 = y2[::-1]
    
    return y1, y2

def _solve_G(G0, L, ro, re, rd, I):
    """Solve for G that return appropriate L."""

    z = (re - rd) / (re * (ro/re-1))
    C = (np.exp(-z)-1+z) / z

    def cost(logG):
        G = np.exp(logG)
        return (L - G * I * re / (ro * (rd + ro - re*(1+1/(1-C)))))**2

    soln = minimize(cost, np.log(G0))
    return np.exp(soln['x']), soln

def fit_dmft(data, L, dt, initial_params, **params_kw):
    """Best fit of dmft model at stationarity.

    Parameters
    ----------
    dt : float
    initial_params : list
        G shouldn't be specified.
    **params_kw

    Returns
    -------
    dict
    """

    def cost(params):
        ro, re, rd, I = np.exp(params)
        G = _solve_G(30, L, ro, re, rd, I)
        G = G[0]
        #print(f'{G=}')

        model = DynamicalMFT(G, ro, re, rd, I, dt, **params_kw)
        print(f'{model.L=}')
        if not np.isclose(model.L, L, atol=1e-2): return 1e30
        
        flag, mxerr = model.solve_stationary()
        if np.any(model.n<0): return 1e30

        try:
            c = np.linalg.norm(data-model.n)
        except ValueError:
            return 1e30
        return c

    soln = minimize(cost, np.log(initial_params))
    return np.exp(soln['x']), soln

def _fit_dmft(data, dt, initial_params, **params_kw):
    """Best fit of dmft model at stationarity.

    Parameters
    ----------
    dt : float
    initial_params
    **params_kw

    Returns
    -------
    dict
    """
    
    def cost(params):
        model = DynamicalMFT(*np.exp(params), dt, **params_kw)
        if model.L<1 or ~np.isfinite(model.L): return 1e30
        
        flag, mxerr = model.solve_stationary()
        if np.any(model.n<0): return 1e30

        y1, y2 = match_length(data, model.n, 'l', 'l')
        c = np.linalg.norm(y1-y2)
        return c

    soln = minimize(cost, np.log(initial_params))
    return np.exp(soln['x']), soln

def fit_ode(x, data, initial_params, full_output=False, **params_kw):
    """Best fit of ODE model at stationarity.

    Parameters
    ----------
    x : ndarray
    data : ndarray
    initial_params : list
    full_output : bool, False
        If True, return soln dict.
    **params_kw

    Returns
    -------
    ndarray
        Estimates for (G, ro, re, rd, I).
    dict
        If full_output is True. From scipy.optimize.minimize.
    """

    def cost(params):
        G, ro, re, rd, I = np.exp(params)
        
        try:
            model = Analytic(G, ro, re, rd, I, **params_kw)
        except AssertionError:  # e.g. problem with stationarity and L
            return 1e30
        
        c = np.linalg.norm(data - model.n(x))
        return c

    soln = minimize(cost, np.log(initial_params))
    if full_output:
        return np.exp(soln['x']), soln
    return np.exp(soln['x'])



# ======= #
# Classes #
# ======= #
class Simulator():
    def __init__(self,
                 L0=20,
                 N0=20,
                 G=.2,
                 obs_rate=.49,
                 expand_rate=.5,
                 death_rate=.2,
                 innov_rate=.5,
                 exploit_rate=0,
                 cooperativity=1.,
                 depressed_frac=.5,
                 dt=0.1,
                 rng=None,
                 cache_dr=None):
        """
        Parameters
        ----------
        L0 : int, 20
            Initial size of lattice.
        N0 : int, 20
            Initial population size.
        G : float, .2
            Total new firm rate.
        obs_rate : float, .49
            Obsolescence rate (from left).
        expand_rate : float, .5
            Rate at which firms try to expand.
        death_rate : float, .2
            Rate of death.
        innov_rate : float, .5
            Probability of successful innovation.
        exploit_rate : float, .5
            Probability of successfully exploiting innovated area.
        cooperativity : float, 1.
        depressed_frac : float, .5
        dt : float, .1
            Time step size.
        rng : np.RandomState
        cache_dr : str, None
        """

        assert L0>0 and N0>=0 and G>0 and dt>0
        
        self.L0 = L0
        self.N0 = N0
        self.G = G
        self.obs_rate = obs_rate
        self.expand_rate = expand_rate
        self.innov_rate = innov_rate
        self.exploit_rate = exploit_rate
        self.death_rate = death_rate
        self.cooperativity = cooperativity
        self.depressed_frac = depressed_frac
        
        self.dt = dt
        self.rng = rng or np.random
        self.cache_dr = cache_dr

        self.storage = {}  # store previous sim results

    def simulate(self, T, save_every, cache=True, reset_rng=False):
        """Run firm simulation for T time steps. To minimize memory and cost of saving sim
        results, only LiteFirms and lattice boundaries are stored.
        
        Parameters
        ----------
        T : float
            Total duration of simulation.
        save_every : int
            Number of time steps to wait before saving. Time elapsed between saves is dt *
            save_every.
        cache : bool, True
            If True, save simulation result into self.storage dict.
        reset_rng : bool, False
            If True, reinitialize rng. Useful for parallel processing.
            
        Returns
        -------
        list
            LiteFirm copies at each time step (non-empty if cache=False).
        list
            Lattice copies at each time step (non-empty if cache=False).
        """

        if reset_rng:
            self.rng = np.random.RandomState()
        
        # settings
        L0 = self.L0
        N0 = self.N0
        G = self.G
        obs_rate = self.obs_rate
        expand_rate = self.expand_rate
        innov_rate = self.innov_rate
        exploit_rate = self.exploit_rate
        death_rate = self.death_rate
        dt = self.dt
        
        # variables for tracking history
        save_key = ''
        cache_every = 1_000  # this will dynamically change depending on RAM available
        firm_snapshot = []
        lattice_snapshot = []
        
        # updated parameters
        firms = []
        lattice = TopicLattice()
        # initialize empty lattice
        lattice.extend_right(L0)

        # initialize
        for i in range(N0):
            # choose innovation tendency uniformly? how to impose selection?
            firms.append(Firm(self.rng.randint(lattice.left+1, lattice.right+1),
                              innov_rate,
                              lattice=lattice,
                              wealth=np.inf,
                              rng=self.rng))
            lattice.d_add(firms[-1].sites[0])
        lattice.push()

        # run sim
        # 1. try to grow each firm
        # 3. grow/shrink lattice
        # 4. eliminate firms by death rate
        # 5. spawn new firms (including at innov front)
        t = 0
        counter = 0
        while t < T:
            grow_lattice = 0  # switch for if lattice needs to be grown
            new_occupancy = 0  # count new firms occupying innovated area
            
            n_firms = len(firms)
            for i in range(n_firms):
                f = firms[i]
                # attempt to grow (assuming wealth is positive)
                if f.rng.rand() < (expand_rate * dt):
                    out = f.grow(exploit_rate)
                    # if the firm grows and lattice does not
                    if out[0] and not out[1]:
                        lattice.d_add(f.sites[1])

                        # new firm
                        firms.append(Firm(f.sites[1], innov_rate,
                                          lattice=lattice, wealth=np.inf, rng=self.rng))
                        f.sites = f.sites[0], f.sites[1]-1
                    # if the firm grows and the lattice grows
                    elif out[0] and out[1]:
                        new_occupancy += 1  # keep track of total frontier density
                        grow_lattice += 1

                        f.sites = f.sites[0], f.sites[1]-1
                    # if the firm does not grow and the lattice does
                    elif not out[0] and out[1]:
                        grow_lattice += 1

                f.age += dt
                i += 1
            lattice.push()
            # if any firm innovated, then grow the lattice
            if grow_lattice:
                lattice.extend_right()
                lattice.add(lattice.right, new_occupancy)
                for i in range(grow_lattice):
                    # new firm
                    firms.append(Firm(lattice.right, innov_rate,
                                      lattice=lattice, wealth=np.inf, rng=self.rng))

            # shrink topic lattice
            if (lattice.left < (lattice.right-1)) and (self.rng.rand() < (obs_rate * dt)):
                lattice.shrink_left()
                # shrink all firms that have any value on the obsolete topic since lattice
                # is now empty on obsolescent site
                # firms of size one will be deleted by accounting for negative wealth next
                # firms that have left boundary pass right will be deleted next
                for i, f in enumerate(firms):
                    if f.sites[0] <= lattice.left:  # s.t. lattice of size 1 incurs loss!
                        f.sites = f.sites[0]+1, f.sites[1]

            # kill all firms with wealth below cutoff or obsolete
            removeix = []
            for i, f in enumerate(firms):
                if (self.rng.rand() < (self.death_rate * dt)) or (f.sites[1] == lattice.left):
                    removeix.append(i)
            for count, ix in enumerate(removeix):
                f = firms.pop(ix - count)
                lattice.d_remove_range(f.sites[0], f.sites[1])
            lattice.push()
            # check_firms_occupancy(firms, lattice, iprint=True)

            # spawn new firms
            nNew = self.rng.poisson(G * dt)
            #nNew = self.rng.rand() < (G*dt)
            for i in range(nNew):
                firms.append(Firm(self.rng.randint(lattice.left+1, lattice.right+1),
                                  innov_rate,
                                  wealth=np.inf,
                                  lattice=lattice,
                                  rng=self.rng))
                lattice.d_add(firms[-1].sites[0])
            if nNew:
                lattice.push()
            # check_firms_occupancy(firms, lattice, iprint=True)
        
            # collect status
            if (counter%save_every)==0:
                firm_snapshot.append(snapshot_firms(firms))
                lattice_snapshot.append((lattice.left, lattice.right))  # lattice endpts
            
            # export results to file every few thousand steps
            if cache and (len(lattice_snapshot) > cache_every):
                if not save_key:
                    save_key = str(datetime.now())
                self.storage[save_key] = firm_snapshot, lattice_snapshot
                # this will save to file and clear the lists
                self.save(t - save_every * dt * (len(firm_snapshot) - 1), dt*save_every)
                firm_snapshot, lattice_snapshot = self.storage[save_key] 
                
                # save less frequently if less than 20% of RAM available
                # save more frequently if more than 50% of RAM available
                if (virtual_memory().available/virtual_memory().total) > .5:
                    cache_every *= 2
                    cache_every = min(cache_every, 8_000)
                elif (virtual_memory().available/virtual_memory().total) < .2:
                    cache_every /= 2
                    cache_every = max(cache_every, 500)
            
            # update time
            counter += 1
            t = counter * dt
        
        if cache and len(lattice_snapshot):
            # save any remnants of lists that were not saved to disk
            if not save_key:
                save_key = str(datetime.now())
            self.storage[save_key] = firm_snapshot, lattice_snapshot
            self.save(t - save_every * dt * len(firm_snapshot), dt*save_every)
            # this will save to file and clear the lists
            firm_snapshot, lattice_snapshot = self.storage[save_key] 
        
        return firm_snapshot, lattice_snapshot
        
    def parallel_simulate(self, n_samples, T, save_every,
                          iprint=True,
                          n_cpus=None):
        """Parallelize self.simulate() and save results into self.storage data member dict.
        
        Parameters
        ----------
        n_samples : int
            Number of random trajectories to sample.
        T : int
            Total simulation duration.
        save_every : int
            How often to snapshot simulation.
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
        thislock = Lock()

        assert not self.cache_dr is None
        if not os.path.isdir(self.cache_dr):
            try:
                os.makedirs(self.cache_dr)
            except FileExistsError:
                pass

        # save file with sim instance for info on the settings
        if not os.path.isfile(f'{self.cache_dr}/top.p'):
            # save class info without all of storage
            storage = self.storage
            self.storage = {}
            with open(f'{self.cache_dr}/top.p', 'wb') as f:
                pickle.dump({'simulator':self}, f)

            self.storage = storage

        def init(thislock):
            """Only for generating lock shared amongst processes for file writing."""
            global lock
            lock = thislock

        def loop_wrapper(args, self=self, T=T):
            firm_snapshot, lattice_snapshot = self.simulate(T, save_every, cache=True, reset_rng=True)
            assert len(firm_snapshot)==0 and len(lattice_snapshot)==0
            if iprint: print("Done with one rand trajectory.")
            #lock.acquire()  # only save one sim result at a time
            #self.save()
            #lock.release()
        
        if iprint: print("Starting simulations...")
        with threadpool_limits(user_api='blas', limits=1):
            with Pool(n_cpus, initializer=init, initargs=(thislock,)) as pool:
                pool.map(loop_wrapper, range(n_samples))

        if iprint: print(f"Runtime of {perf_counter()-t0} s.")
            
    def save(self, t0, dt, iprint=False):
        """Save simulator instance with each simulation run in a separate parquet file.
        Ledger must be updated separately.
        
        Parameters
        ----------
        t0 : float
        dt : float
        iprint : bool, False
        """
       
        # save values of storage dict into separate pickles
        for k in self.storage.keys():
            sql.parquet_firms(self.storage[k][0], self.cache_dr, k, t0, dt)
            sql.parquet_lattice(reconstruct_lattice(*self.storage[k]), self.cache_dr, k, t0, dt)

            # clear storage
            self.storage[k] = [], []

        if iprint: print(f"Saved simulator instances in cache.")
    
    def load_list(self):
        """Show list of sims possible to load from cache directory.

        Returns
        -------
        list of str
        """

        if not 'cache_dr' in self.__dict__.keys():
            raise Exception("No cache specified.")
        
        # get all unique cache names
        return sorted(set(['.'.join(i.split('.')[:-1]) for i in os.listdir(self.cache_dr)
                           if (not 'top' in i) and (not '_lattice' in i) and (not 'density' in i)]))

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
            # load from parquet file
            query = f'''SELECT *
                        FROM parquet_scan('{self.cache_dr}/{name}.parquet')
                     '''
            qr = sql.QueryRouter()
            output = qr.con.execute(query).fetchdf()
            dt = qr.dt(self.cache_dr.split('/')[-1])
            firm_snapshot = []
            
            # account for off-by-one counting bug
            if output['t'].min() < 0:
                counter = -1
            else:
                counter = 0
            for t, firms in output.groupby('t'):
                while not np.isclose(counter * dt, t):
                    firm_snapshot.append([])
                    counter += 1
                    #assert (counter * dt) <= t, (counter * dt, t)
                firm_snapshot.append([])
                counter += 1

                for i, f in firms.iterrows():
                    firm_snapshot[-1].append(LiteFirm((f.fleft, f.fright),
                                                      f.innov,
                                                      f.wealth,
                                                      self.connect_cost,
                                                      f.age,
                                                      f.ids))
            self.storage[name] = firm_snapshot

    def add_to_ledger(self, extra_props={}):
        """
        Add this simulation instance to the ledger described in organizer.py.

        Parameters
        ----------
        extra_props : dict, {}
            Extra properties to add to ledger such as simulation duration.
        """
        
        extra_props.update({'L0':self.L0,
                            'N0':self.N0,
                            'G':self.G,
                            'obs_rate':self.obs_rate,
                            'expand_rate':self.expand_rate,
                            'innov_rate':self.innov_rate,
                            'exploit_rate':self.exploit_rate,
                            'depressed_frac':self.depressed_frac,
                            'growf':self.growf,
                            'connect_cost':self.connect_cost,
                            'income_multiplier':self.income_multiplier,
                            'n_sims':(len(os.listdir(self.cache_dr))-1)//2,
                            'dt':self.dt})
        ledger = SimLedger()
        ledger.add(self.cache_dr.split('/')[-1], extra_props)
    
    def info(self):
        """Show parameters."""
        
        print(f'new firm rate   =\t{self.G}')
        print(f'innov rate      =\t{self.innov_rate}')
        print()

        print(f'This instance has {len(self.storage)} sims run on')
        for k in self.storage.keys():
            print(k)
        print()
#end Simulator


class UnitSimulator(Simulator):
    """Independent unit simulation of firms, which is the same thing as a density
    evolution equation. This is the simplest implementation possible that only
    keeps track of the occupancy number and does nothing fancy.
    """

    def simulate(self, T, reset_rng=False, jit=True, occupancy=None):
        """
        dt must be small enough to ignore coincident events.

        Parameters
        ----------
        T : int
            Simulation time to run.
        reset_rng : bool, False
        jit : bool, True
        occupancy : list, None
            Feed in a starting occupancy on which to run dynamics.

        Returns
        -------
        list
        """
       
        G = float(self.G)
        ro = float(self.obs_rate)
        rd = float(self.death_rate)
        re = float(self.expand_rate)
        I = float(self.innov_rate)
        a = float(self.cooperativity)
        dt = float(self.dt)
        
        assert (G * dt)<1
        assert (rd * dt)<1
        assert (re * dt)<1
        assert (ro * dt)<1
        
        if jit and occupancy is None:
            if reset_rng: np.random.seed()
            return jit_unit_sim_loop(T, dt, G, ro, re, rd, I, a)
        elif jit and not occupancy is None:
            if reset_rng: np.random.seed()
            occupancy = List(occupancy)
            return list(jit_unit_sim_loop_with_occupancy(occupancy, T, dt, G, ro, re, rd, I, a))
 
        if reset_rng: self.rng.seed()
        counter = 0
        occupancy = [0]
        while (counter * dt) < T:
            # innov
            innov = False
            if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
                occupancy.append(1)
                innov = True

            # obsolescence
            if len(occupancy) > 1 and np.random.rand() < (ro * dt):
                occupancy.pop(0)
            
            # from right to left b/c of expansion
            for x in range(len(occupancy)-1, -1, -1):
                # expansion (fast approximation)
                if x < (len(occupancy)-1-innov):
                    if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                        occupancy[x+1] += 1

               # death (fast approximation)
                if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                    occupancy[x] -= 1

                # start up (remember that L is length of lattice on x-axis, s.t. L=0 means lattice has one site)
                if len(occupancy)==1:
                    occupancy[x] += 1
                elif np.random.rand() < (G / (len(occupancy)-1) * dt):
                    occupancy[x] += 1

            counter += 1
        return occupancy

    def parallel_simulate(self, n_samples, T, **kwargs):
        """
        Parameters
        ----------
        n_samples : int
        T : int
        **kwargs
        
        Returns
        -------
        list of lists
            Each inner list is an occupancy list.
        """

        with Pool() as pool:
            self.occupancy = list(pool.map(lambda args: self.simulate(T, True, **kwargs), range(n_samples)))
        return self.occupancy

    def mean_occupancy(self, occupancy=None,
                       width=np.inf,
                       norm_err=True,
                       rescale=False):
        """
        Parameters
        ----------
        occupancy : list of list, None
        width : int, np.inf
        norm_err : bool, True
        rescale : bool, False
            If True, rescale density by cooperativity before taking mean.

        Returns
        -------
        ndarray
            Average occupancy.
        ndarray
            Standard deviation as error bars.
        """
        
        if occupancy is None:
            occupancy = self.occupancy

        if width==np.inf:
            maxL = max([len(i) for i in occupancy])
            y = np.zeros(maxL)
            yerr = np.zeros(maxL)

            # first calculate the means
            counts = np.zeros(maxL, dtype=int)
            if rescale:
                for i in occupancy:
                    y[:len(i)] += np.array(i[::-1])**self.cooperativity
                    counts[:len(i)] += 1
            else:
                for i in occupancy:
                    y[:len(i)] += i[::-1]
                    counts[:len(i)] += 1
            y = y / counts

            # then calculate the std
            if rescale:
                for i in occupancy:
                    yerr[:len(i)] += (np.array(i[::-1])**self.cooperativity - y[:len(i)])**2
            else:
                for i in occupancy:
                    yerr[:len(i)] += (i[::-1] - y[:len(i)])**2

            yerr /= counts
            yerr = np.sqrt(yerr)
            if norm_err:
                return y, yerr / np.sqrt(counts)
            return y, yerr

        y = np.vstack([i[-width:] for i in occupancy if len(i)>=width])[:,::-1]
        if norm_err:
            return y.mean(0), y.std(0) / np.sqrt(y.shape[0])
        return y.mean(0), y.std(0)

    def rescale_factor(self, T, sample_size=1_000):
        """Rescaling factor needed to correct for bias in mean L. The returned
        factor c can be used to modify the automaton model with the set of
        transformations
            x -> x * c
            G -> G * c
            n -> n / c
            I -> I / c
        Equivalently, we can transform the MFT with the inverse set of
        transformations
            x -> x / c
            G -> G / c
            n -> n * c
            I -> I * c
        
        Parameters
        ----------
        T : float
            Run time before sampling.
        sample_size : int
            No. of indpt. trajectories to use to estimate ratio.

        Returns
        -------
        float
        """

        G = float(self.G)
        ro = float(self.obs_rate)
        re = float(self.expand_rate)
        rd = float(self.death_rate)
        I = float(self.innov_rate)
        a = float(self.cooperativity)
        dt = float(self.dt)
 
        occupancy = self.parallel_simulate(sample_size, T)
        L = np.array([(len(i)-1) for i in occupancy])

        odemodel = Analytic(G, ro, re, rd, I)

        return odemodel.L / L.mean(), occupancy
#end UnitSimulator

@njit
def jit_unit_sim_loop(T, dt, G, ro, re, rd, I, a):
    """
    Parameters
    ----------
    occupancy : numba.typed.ListType[int64]
    T : int
    dt : float
    ro : float
    G : float
    re : float
    rd : float
    I : float
    a : float
    """
    
    counter = 0
    occupancy = [0]
    while (counter * dt) < T:
        # innov
        innov = False
        if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
            occupancy.append(1)
            innov = True

        # obsolescence
        if len(occupancy) > 1 and np.random.rand() < (ro * dt):
            occupancy.pop(0)
        
        # from right to left b/c of expansion
        for x in range(len(occupancy)-1, -1, -1):
            # expansion (fast approximation)
            if x < (len(occupancy)-1-innov):
                if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                    occupancy[x+1] += 1

           # death (fast approximation)
            if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                occupancy[x] -= 1

            # start up (remember that L is length of lattice on x-axis, s.t. L=0 means lattice has one site)
            if len(occupancy)==1:
                occupancy[x] += 1
            elif np.random.rand() < (G / (len(occupancy)-1) * dt):
                occupancy[x] += 1

        counter += 1
    return occupancy


@njit
def jit_unit_sim_loop_with_occupancy(occupancy, T, dt, G, ro, re, rd, I, a):
    """
    Parameters
    ----------
    occupancy : numba.typed.ListType[int64]
    T : int
    dt : float
    ro : float
    G : float
    re : float
    rd : float
    I : float
    a : float
    """
    
    counter = 0
    while (counter * dt) < T:
        # innov
        innov = False
        if occupancy[-1] and np.random.rand() < (re * I * occupancy[-1]**a * dt):
            occupancy.append(1)
            innov = True

        # from right to left b/c of expansion
        for x in range(len(occupancy)-1, -1, -1):
            # death (fast approximation)
            if occupancy[x] and np.random.rand() < (occupancy[x] * rd * dt):
                occupancy[x] -= 1

            # expansion (fast approximation)
            if x < (len(occupancy)-1-innov):
                if occupancy[x] and np.random.rand() < (occupancy[x] * re * dt):
                    occupancy[x+1] += 1

            # start up
            if np.random.rand() < (G / len(occupancy) * dt):
                occupancy[x] += 1

        # obsolescence
        if len(occupancy) > 1 and np.random.rand() < (ro * dt):
            occupancy.pop(0)
        
        counter += 1
    return occupancy



class MultiunitSimulator(Simulator):
    def simulate(self, T, save_every, cache=True, reset_rng=False):
        """Run firm simulation for T time steps. To minimize memory and cost of saving sim
        results, only LiteFirms and lattice boundaries are stored.
        
        Parameters
        ----------
        T : float
            Total duration of simulation.
        save_every : int
            Number of time steps to wait before saving. Time elapsed between saves is dt *
            save_every.
        cache : bool, True
            If True, save simulation result into self.storage dict.
        reset_rng : bool, False
            If True, reinitialize rng. Useful for parallel processing.
            
        Returns
        -------
        list
            LiteFirm copies at each time step (non-empty if cache=False).
        list
            Lattice copies at each time step (non-empty if cache=False).
        """

        if reset_rng:
            self.rng = np.random.RandomState()
        
        # settings
        L0 = self.L0
        N0 = self.N0
        G = self.G
        obs_rate = self.obs_rate
        expand_rate = self.expand_rate
        innov_rate = self.innov_rate
        exploit_rate = self.exploit_rate
        death_rate = self.death_rate
        dt = self.dt
        
        # variables for tracking history
        save_key = ''
        cache_every = 1_000  # this will dynamically change depending on RAM available
        firm_snapshot = []
        lattice_snapshot = []
        
        # updated parameters
        firms = []
        lattice = TopicLattice()
        # initialize empty lattice
        lattice.extend_right(L0)

        # initialize
        for i in range(N0):
            # choose innovation tendency uniformly? how to impose selection?
            firms.append(SimpleFirm(self.rng.randint(lattice.left+1, lattice.right+1),
                                    innov_rate,
                                    lattice=lattice,
                                    wealth=np.inf,
                                    rng=self.rng))
            lattice.d_add(firms[-1].sites[0])
        lattice.push()

        # run sim
        # 1. try to grow each firm
        # 3. grow/shrink lattice
        # 4. eliminate firms by death rate
        # 5. spawn new firms (including at innov front)
        t = 0
        counter = 0
        while t < T:
            grow_lattice = 0  # switch for if lattice needs to be grown
            new_occupancy = 0  # count new firms occupying innovated area
            
            n_firms = len(firms)
            for i in range(n_firms):
                f = firms[i]
                # attempt to grow (assuming wealth is positive)
                if f.rng.rand() < (expand_rate * dt):
                    out = f.grow()
                    # if the firm grows and the lattice grows
                    if out[0] and out[1]:
                        new_occupancy += 1  # keep track of total frontier density
                        grow_lattice += 1
                    # if the firm does not grow and the lattice does
                    elif not out[0] and out[1]:
                        grow_lattice += 1
                    # else neither firm grows nor lattice

                f.age += dt
                i += 1
            lattice.push()
            # if any firm innovated, then grow the lattice
            if grow_lattice:
                lattice.extend_right()
                lattice.add(lattice.right, new_occupancy)
                for i in range(grow_lattice):
                    # new firm
                    firms.append(SimpleFirm(lattice.right, innov_rate,
                                            lattice=lattice, wealth=np.inf, rng=self.rng))

            # shrink topic lattice
            if (lattice.left < (lattice.right-1)) and (self.rng.rand() < (obs_rate * dt)):
                lattice.shrink_left()
                # shrink all firms that have any value on the obsolete topic since lattice
                # is now empty on obsolescent site
                # firms of size one will be deleted by accounting for negative wealth next
                # firms that have left boundary pass right will be deleted next
                for i, f in enumerate(firms):
                    if f.sites[0] <= lattice.left:  # s.t. lattice of size 1 incurs loss!
                        f.sites = f.sites[0]+1, f.sites[1]

            # kill firms with rate proportional to amount of depressed sites
            removeix = []
            for i, f in enumerate(firms):
                if f.sites[1] == lattice.left: removeix.append(i)
                else: 
                    total_death_rate = self.rng.binomial(f.size(), self.depressed_frac)/f.size() * death_rate
                    if self.rng.rand() < total_death_rate * dt:
                        removeix.append(i)
            for count, ix in enumerate(removeix):
                f = firms.pop(ix - count)
                lattice.d_remove_range(f.sites[0], f.sites[1])
            lattice.push()
            # check_firms_occupancy(firms, lattice, iprint=True)

            # spawn new firms
            nNew = self.rng.poisson(G * dt)
            for i in range(nNew):
                firms.append(SimpleFirm(self.rng.randint(lattice.left+1, lattice.right+1),
                                        innov_rate,
                                        wealth=np.inf,
                                        lattice=lattice,
                                        rng=self.rng))
                lattice.d_add(firms[-1].sites[0])
            if nNew: lattice.push()
            # check_firms_occupancy(firms, lattice, iprint=True)
        
            # collect status
            if (counter%save_every)==0:
                firm_snapshot.append(snapshot_firms(firms))
                lattice_snapshot.append((lattice.left, lattice.right))  # lattice endpts
            
            # export results to file every few thousand steps
            if cache and (len(lattice_snapshot) > cache_every):
                if not save_key:
                    save_key = str(datetime.now())
                self.storage[save_key] = firm_snapshot, lattice_snapshot
                # this will save to file and clear the lists
                self.save(t - save_every * dt * (len(firm_snapshot) - 1), dt*save_every)
                firm_snapshot, lattice_snapshot = self.storage[save_key] 
                
                # save less frequently if less than 20% of RAM available
                # save more frequently if more than 50% of RAM available
                if (virtual_memory().available/virtual_memory().total) > .5:
                    cache_every *= 2
                    cache_every = min(cache_every, 8_000)
                elif (virtual_memory().available/virtual_memory().total) < .2:
                    cache_every /= 2
                    cache_every = max(cache_every, 500)
            
            # update time
            counter += 1
            t = counter * dt
        
        if cache and len(lattice_snapshot):
            # save any remnants of lists that were not saved to disk
            if not save_key:
                save_key = str(datetime.now())
            self.storage[save_key] = firm_snapshot, lattice_snapshot
            self.save(t - save_every * dt * len(firm_snapshot), dt*save_every)
            # this will save to file and clear the lists
            firm_snapshot, lattice_snapshot = self.storage[save_key] 
        
        return firm_snapshot, lattice_snapshot
        
    def parallel_simulate(self, n_samples, T, save_every,
                          iprint=True,
                          n_cpus=None):
        """Parallelize self.simulate() and save results into self.storage data member dict.
        
        Parameters
        ----------
        n_samples : int
            Number of random trajectories to sample.
        T : int
            Total simulation duration.
        save_every : int
            How often to snapshot simulation.
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
        thislock = Lock()

        assert not self.cache_dr is None
        if not os.path.isdir(self.cache_dr):
            try:
                os.makedirs(self.cache_dr)
            except FileExistsError:
                pass

        # save file with sim instance for info on the settings
        if not os.path.isfile(f'{self.cache_dr}/top.p'):
            # save class info without all of storage
            storage = self.storage
            self.storage = {}
            with open(f'{self.cache_dr}/top.p', 'wb') as f:
                pickle.dump({'simulator':self}, f)

            self.storage = storage

        def init(thislock):
            """Only for generating lock shared amongst processes for file writing."""
            global lock
            lock = thislock

        def loop_wrapper(args, self=self, T=T):
            firm_snapshot, lattice_snapshot = self.simulate(T, save_every, cache=True, reset_rng=True)
            assert len(firm_snapshot)==0 and len(lattice_snapshot)==0
            if iprint: print("Done with one rand trajectory.")
            #lock.acquire()  # only save one sim result at a time
            #self.save()
            #lock.release()
        
        if iprint: print("Starting simulations...")
        with threadpool_limits(user_api='blas', limits=1):
            with Pool(n_cpus, initializer=init, initargs=(thislock,)) as pool:
                pool.map(loop_wrapper, range(n_samples))

        if iprint: print(f"Runtime of {perf_counter()-t0} s.")
#end Simulator




class IterativeMFT():
    def __init__(self, G, ro, re, rd, I, alpha=1., Q=2):
        """Class for calculating discrete MFT quantities.

        Parameters
        ----------
        ro : float
        G : float
        re : float
        rd : float
        I : float
        alpha : float, 1.
            Cooperativity.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        
        assert alpha==1
        assert Q>=2

        self.ro = ro
        self.G = G
        self.re = re
        self.rd = rd
        self.I = I
        self.alpha = alpha
        self.Q = Q
        self.n0 = ro/re/I  # stationary density

        assert (2/(Q-1) * re - rd) * self.n0 - re*I*self.n0**2 <= 0, "Stationary criterion unmet."
        
        # MFT guess for L, which we will refine using tail convergence criteria
        try:
            self.L0 = G * re * I / (ro * (rd - 2*re/(Q-1) + ro))
        except ZeroDivisionError:
            self.L0 = np.inf
        
        # for handling infinite L
        if ~np.isfinite(self.L0) or self.L0<0:
            self.L0 = 100_000
            self.L = self.L0
            self.n = self.iterate_n()
        else:
            self.min_L(self.L0)

    def min_L(self, L0, mx_decimal=10):
        """Lower L slower and slower while keeping tail positive in order to
        find value of L that solves iterative solution.

        As a heuristic, we keep the numerically calculated value of the tail
        positive instead of the self-consistent iterative value, which seems to
        behave worse (probably because it depends on the estimate of n[-2],
        which itself can be erroneous).
        
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
        assert L0 > 2, L0

        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        Q = self.Q
        n0 = self.n0
        
        L = np.ceil(L0)
        decimal = 1

        nfun = self.iterate_n(L)
        # check that tail is positive; in case it is not, increase starting guess for L a few times
        counter = 0
        while nfun[-1] < 0: #(I * n0 * nfun[-2] + G/re/L / (I * n0 + rd/re)) < 0:
            L += 1
            nfun = self.iterate_n(L)
            counter += 1
            assert counter < 1e3
        
        while decimal <= mx_decimal:
            # ratchet down til the tail goes the wrong way
            while nfun[-1] > 0 and L>0: #(I * n0 * nfun[-2] + G/re/L / (I * n0 + rd/re)) > 0:
                L -= 10**-decimal
                nfun = self.iterate_n(L)
            L += 10**-decimal  # oops, go back up
            nfun = self.iterate_n(L)
            decimal += 1

        self.L, self.n = L, nfun
        return L, nfun

    def iterate_n(self, L=None, iprint=False):
        """Iterative solution to occupancy number. See NB II pg. 118."""
        
        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        n0 = self.n0
        L = L or self.L
        Q = self.Q

        #eps = I * n0**2 + (rd/re-2) * n0 - G/re/L

        n = np.zeros(int(L)+1)
        n[0] = n0
        if n.size>1:
            # rigid assumption about n[1]
            n[1] = (Q-1) * (I * n0**2 + rd * n0 / re - G / L / re)

            for i in range(2, n.size):
                n[i] = (Q-1) * (I * n0 * (n[i-1] - n[i-2]) + rd * n[i-1] / re - G / re / L)

        if iprint: print(eps)

        return n

    def estimate_L(self, x=2):
        """Invert stationary equation to solve for L.

        Parameters
        ----------
        x : int, 2
            Lattice point to use for estimating L. Too close to the right side
            boundary leads to large numerical errors.
        """

        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        Q = self.Q
        n0 = self.n0
        n = self.n
        
        return G/re / (I * n[0] * (n[x-1]-n[x-2]) + rd/re * n[x-1] - n[x]/(Q-1))
#end IterativeMFT



class DynamicalMFT():
    def __init__(self, G, ro, re, rd, I, dt,
                 correction=True,
                 alpha=1.,
                 Q=2):
        """Class for calculating discrete MFT quantities by running dynamics.

        Parameters
        ----------
        G : float
        re : float
        rd : float
        I : float
        dt : float
        correction : bool, True
            If True, use calculation for corrected L.
        alpha : float, 1.
            Cooperativity parameter.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        
        assert alpha>0
        assert Q>=2

        self.ro = ro
        self.G = G
        self.re = re
        self.rd = rd
        self.I = I
        self.dt = dt
        self.alpha = alpha * 1.
        self.Q = Q

        self.n0 = (ro/re/I)**(1/alpha)
        
        try:
            if correction:
                self.L = self.mft_L()
            else:
                self.L = G / (self.n0 * (rd - (1+1/(Q-1))*re + re*I*self.n0**self.alpha))
        except ZeroDivisionError:
            self.L = np.inf
        if self.L <= 0:
            self.L = np.inf
        else:
            self.L = min(self.L, 10_000)
    
    def update_n(self, L=None):
        """Update occupancy number for a small time step self.dt.

        Parameters
        ----------
        L : float, None
        """
        
        L = L or self.L

        # entrance
        dn = self.G/L 
        # death
        dn -= self.rd * self.n
        # shift
        dn -= self.re * self.I * self.n[0]**self.alpha * self.n
        dn[1:] += self.re * self.I * self.n[0]**self.alpha * self.n[:-1]
        # expansion
        dn[:-1] += self.re / (self.Q-1) * self.n[1:]
        
        self.n += dn * self.dt

    def solve_stationary(self,
                         tol=1e-5,
                         T=5e4,
                         L=None,
                         n0=None,
                         iprint=False):
        """Run til stationary state is met using convergence criterion.

        Parameters
        ----------
        tol : float, 1e-5
            Max absolute change permitted per lattice site per unit time.
        T : float, 5e4
        L : int, None
        n0 : ndarray, None
        iprint : bool, False

        Returns
        -------
        int
            Flag indicating if problem converged
                (0) to stationary solution with correct innov. density
                (1) to stationary soln with wrong innov. density
                (2) or did not converge
        float
            Maximum absolute difference between last two steps of simulation.
        """
        
        L = L or self.L
        # simply impose a large upper limit for infinite L
        if not np.isfinite(L):
            L = 10_000
        
        if n0 is None and 'n' in self.__dict__ and self.n.size==(L+1):
            n0 = self.n[:-1]
        elif n0 is None:
            n0 = np.ones(int(L))/2
        self.n = n0.copy()
        prev_n = np.zeros_like(n0)

        counter = 0
        while (self.dt*counter) < T and np.abs(prev_n-self.n).max()>(self.dt*tol):
            prev_n = self.n.copy()
            self.update_n(L)
            counter += 1
        
        if (self.dt*counter) >= T:
            flag = 2
        elif np.isclose(self.n[0], self.n0):
            flag = 0
        else:
            flag = 1
        
        mx_err = np.abs(prev_n-self.n).max()
        self.n = np.append(self.n, 0)
        return flag, mx_err

    def run(self, T, save_every, L=None, iprint=False):
        """
        Parameters
        ----------
        T : int
        save_every : int
        L : int, None
        iprint : bool, False

        Returns
        -------
        list of list
        """
        
        L = L or self.L
        # simply impose a large upper limit for infinite L
        if not np.isfinite(L):
            L = 10_000

        t = []
        n0 = np.ones(int(L))/2
        self.n = n0.copy()
        snapshot_n = []
        counter = 0
        while (self.dt*counter) < T:
            self.update_n(L)
            
            if np.isclose((self.dt*counter)%save_every, 0, atol=self.dt/10, rtol=0):
                if iprint: print(f"Recording {dt*counter}...")
                t.append(counter * self.dt)
                snapshot_n.append(self.n.copy())
            counter += 1

        return snapshot_n, t

    def solve_n0(self, L):
        """Quadratic equation solution for n0."""
            
        assert self.alpha==1, "This does not apply for alpha!=1."
        assert self.Q==2
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I

        return ((2-rd/re) + np.sqrt((2-rd/re)**2 + 4*I*G/re/L)) / 2 / I

    def corrected_n0(self):
        """Stil figuring out the logic of this.
        """
         
        assert self.alpha==1 and self.Q==2
        G = self.G
        I = self.I
        re = self.re
        rd = self.rd
        ro = self.ro
        n0 = self.n0
        
        n02 = -(re-rd) * n0 / ((re-rd)/2 - re*I*n0 + re)
        # this is our estimate of the correction to the slope, which gives corrections to the intercept
        z = - (re - rd) / ((re-rd)/2 + re*(1 - I*n0))

        return n0 + (n02 * z**-2. * (-1 + z + np.exp(-z)) if z!=0 else 0.)
    
    def mft_L(self, second_order=False): 
        """Calculate stationary lattice width accounting the first order correction
        (from Firms II pg. 140).
        
        Parameters
        ----------
        second_order : bool, False
            If True, include additional estimate of corrections to L. Usually, this
            is quite poor because we are guessing at the value of delta to order of
            magnitude.

        Returns
        -------
        float
            Estimated lattice width.
        """
        
        G = self.G
        I = self.I
        re = self.re
        rd = self.rd
        ro = self.ro
        n0 = self.n0
        a = self.alpha
        Q = self.Q

        z = (re/(Q-1)-rd) / re / (I*n0**a - 1/(Q-1))
        delta = - ro / (ro-re) * n0 * z/2  # effect of alpha not included...
        C = z**-1 * (np.exp(-z) - 1 + z)
        
        if not second_order:
            # correcting factor is correct; matches taylor expansion
            return -G / (n0 * (re * (1+1/(1-C)) / (Q-1) - rd - re*I*n0**a))
        
        assert a==1
        # second order correction for L
        # corrected equation for n'(0)
        # really, we don't know delta
        npp = (n0 - delta / z * C) / (1-C)
        # n0 * (1 + z/2) - delta/2  # taylor expansion of npp
        return -G * re * I / (ro * (re/(Q-1) - rd - ro + re*npp/n0/(Q-1)))
#end DynamicalMFT


class SimpleFirm(Firm):
    def grow(self):
        """Attempt to expand to a neighboring site. If successful, then it will be occupied.
        
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
        
        # consider tendency to innovate and the prob of exploiting it
        # innovation may be rewarding but not necessarily
        #assert self.sites[1]<=lattice.right
        if self.sites[1]==lattice.right:
            if self.rng.random() < self.innov:  # successful innovation?
                return 1, 1
            return 0, 0
        # if no innovation is necessary to grow
        self.sites = self.sites[0], self.sites[1] + 1
        return 1, 0



class Analytic():
    def __init__(self, G, ro, re, rd, I, L=None):
        """Class for analytic solution to MFT.

        Parameters
        ----------
        ro : float
        G : float
        re : float
        rd : float
        I : float
        L : float, None
        alpha : float, 1.
            Cooperativity.
        Q : int, 2
            Bethe lattice branching ratio.
        """
        
        #assert alpha==1
        #assert Q>=2

        self.ro = ro
        self.G = G
        self.re = re
        self.rd = rd
        self.I = I
        self.L = self.solve_L(L)
        #self.alpha = alpha
        #self.Q = Q
        self.n0 = ro/re/I  # stationary density
    
    def n(self, x):
        """Interpolated occupancy function.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """

        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        L = self.L

        a = -re**2 - 4*re*ro + ro**2 + 2*rd*(re+ro)
        sol = (G / ((np.exp(2*sqrt(a)/(re+ro))-1) * L * (re-rd)) *
               (1 - np.exp(2*sqrt(a)/(re+ro)) + np.exp(-(sqrt(a)*(x-1)+re*x-ro*(1+x)+re)/(re+ro)) -
                np.exp((-re*x+ro*(1+x)+sqrt(a)*(1+x)-re)/(re+ro)) +
                (L*(re-rd)*ro/(G*re*I)+1) * (np.exp((-re*x+ro*x+sqrt(a)*(2+x))/(re+ro)) -
                                             np.exp((-re*x+(ro-sqrt(a))*x)/(re+ro)))))
        return sol.real

    def solve_L(self, L0=None, full_output=False):
        """Solve for stationary value of L that matches self-consistency
        condition, i.e. analytic solution for L should be equal to the posited
        value of L.

        Parameters
        ----------
        L0 : float, None
            Initial guess.
        full_output : bool, False

        Returns
        -------
        float
        """

        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        L0 = L0 or max(IterativeMFT(G, ro, re, rd, I).L, 10)
        
        # analytic eq for L solved from continuum formulation in Mathematica
        a = -re**2 - 4*re*ro + ro**2 + 2*rd*(re+ro)
        num = lambda x:(np.exp(-(re-ro)/(re+ro)) * (np.exp(-sqrt(a)*(x-1)/(re+ro)) -
                                                    np.exp(sqrt(a)*(x+1)/(re+ro))) + 
                        np.exp((re-ro)*x/(re+ro)) * (1-np.exp(2*sqrt(a)/(re+ro))) -
                        np.exp(-sqrt(a)*x/(re+ro)) +
                        np.exp(sqrt(a)*(2+x)/(re+ro)))
        den = lambda x:(np.exp(-sqrt(a)*x/(re+ro)) -
                        np.exp(sqrt(a)*(2+x)/(re+ro))) * (re-rd)*ro / (G*re*I)

        statL = lambda x:num(x) / den(x)
        soln = minimize(lambda x:(statL(x).real - x)**2, L0)

        if full_output:
            return soln['x'][0], soln
        return soln['x'][0]
    
    def check_stat(self):
        """Violation of stationarity condition.

        Returns
        -------
        float
            n(1) - [n(1) from n(0)]
        """

        ro = self.ro
        G = self.G
        re = self.re
        rd = self.rd
        I = self.I
        L = self.L
        
        n0 = self.n(0) 
        return self.n(1) - I * n0**2 - rd*n0/re + G/re/L
#end Analytic

