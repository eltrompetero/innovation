# ====================================================================================== #
# Firm innovation models without wealth dynamics for simplified model.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from datetime import datetime
import string
from scipy.signal import fftconvolve
from time import perf_counter
import _pickle as cpickle
from time import sleep
from uuid import uuid4
from multiprocess import Lock
from psutil import virtual_memory

from workspace.utils import save_pickle
from .model_ext import TopicLattice, LiteFirm, snapshot_firms
from .organizer import SimLedger
from .utils import *
from . import sql
from .model import *



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
                 innov_rate=.5,
                 exploit_rate=.5,
                 death_rate=.2,
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
        g0 : float, .2
            Total new firm rate.
        obs_rate : float, .49
            Obsolescence rate (from left).
        expand_rate : float, .5
            Rate at which firms try to expand.
        innov_rate : float, .5
            Probability of successful innovation.
        exploit_rate : float, .5
            Probability of successfully exploiting innovated area.
        death_rate : float, .2
            Rate of death.
        dt : float, .1
            Time step size.
        rng : np.RandomState
        cache_dr : str, None
        """

        assert L0>0 and N0>=0 and g0>0 and dt>0
        
        self.L0 = L0
        self.N0 = N0
        self.g0 = g0
        self.obs_rate = obs_rate
        self.expand_rate = expand_rate
        self.innov_rate = innov_rate
        self.exploit_rate = exploit_rate
        self.death_rate = death_rate
        
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
        g0 = self.g0
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
            
            for f in firms:
                # attempt to grow (assuming wealth is positive)
                if f.rng.rand() < (expand_rate * dt):
                    out = f.grow(exploit_rate)
                    # if the firm grows and lattice does not
                    if out[0] and not out[1]:
                        lattice.d_add(f.sites[1])
                    # if the firm grows and the lattice grows
                    elif out[0] and out[1]:
                        new_occupancy += 1  # keep track of total frontier density
                        grow_lattice += 1
                    # if the firm does not grow and the lattice does
                    elif not out[0] and out[1]:
                        grow_lattice += 1

                f.age += dt
            lattice.push()
            # if any firm innovated, then grow the lattice
            if grow_lattice:
                lattice.extend_right()
                lattice.add(lattice.right, new_occupancy)

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
            nNew = self.rng.poisson(g0 * dt)
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
                            'g0':self.g0,
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
        
        print(f'new firm rate   =\t{self.g0}')
        print(f'innov rate      =\t{self.innov_rate}')
        print(f'grow frac cost  =\t{self.growf}')
        print(f'depressed frac  =\t{self.depressed_frac}')
        print(f'connection cost =\t{self.connect_cost}')
        print()

        print(f'This instance has {len(self.storage)} sims run on')
        for k in self.storage.keys():
            print(k)
        print()
#end Simulator
