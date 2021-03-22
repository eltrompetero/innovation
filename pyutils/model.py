# ======================================================================== #
# 1D firm innovation models
# Author : Eddie Lee, edlee@csh.ac.at
# ======================================================================== #
import numpy as np
from datetime import datetime
import string
from multiprocess import Pool, cpu_count
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
    
    Firms that don't survive into the next time step do not have a
    growth rate. They just disappear from the count.
    
    Parameters
    ----------
    fsnapshot : list of lists of Firm
    
    Returns
    -------
    list of list of floats
        Relative growth rate from previous time step, or (w_t - w_{t-1}) / w_{t-1},
        and the wealth levels used to calculate the ratio.
    """
    
    # extract firm ids
    ids = [[f.id for f in firms] for firms in fsnapshot]
    
    grate = []
    for i in range(1, len(ids)):
        grate.append([])
        for j, thisid in enumerate(ids[i]):
            if thisid in ids[i-1]:
                w0 = fsnapshot[i-1][ids[i-1].index(thisid)].wealth
                w1 = fsnapshot[i][j].wealth
                t = fsnapshot[i][j].age
                grate[-1].append(((w1-w0)/w0, w0, w1, t))
        if not grate[-1]:  # label empty elements with inf
            grate[-1].append((np.inf,np.inf,np.inf,np.inf))
    return grate
    
def find_wide_segments(leftright, mn_width=50, mn_len=50):
    """Find continuous segments with differences between left and right
    boundaries wider than some min width.
    
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
        Such that the interval can be extracted from the appropriate array
        by using the given left and right indices [left:right], i.e. the right
        index already contains the offset by 1.
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
    
    

# ======= #
# Classes #
# ======= #
class Simulator():
    def __init__(self,
                 L0=20,
                 N0=20,
                 g0=.2,
                 obsRate=.49,
                 expandRate=.5,
                 innovSuccessRate=.6,
                 depressionRate=.2,
                 innMutWidth=.05,
                 replicationP=.95,
                 growf=.9):
        """
        Parameters
        ----------
        L0 : int, 20
            Initial size of lattice.
        N0 : int, 20
            Initial population size.
        g0 : float, .2
            New firm rate per site.
        obsRate : float, .49
            Obsolescence rate (from left).
        expandRate : float, .5
            Rate at which firms try to expand.
        innovSuccessRate : float, .6
        depressionRate : float, .2
        innMutWidth : float, .05
            Width of the normal distribution determining mutant innovation.
        replicationP : float, .95
            Probability that new firm is a mutant replicate of an existing
            firm's innovation.
        growf : float, .9
            Growth cost fraction f.
        """
        
        self.L0 = L0
        self.N0 = N0
        self.g0 = g0
        self.obsRate = obsRate
        self.expandRate = expandRate
        self.innovSuccessRate = innovSuccessRate
        self.depressionRate = depressionRate
        self.innMutWidth = innMutWidth
        self.replicationP = replicationP
        self.growf = growf
        self.storage = {}  # store previous sim results

    def simulate(self, T, cache=True):
        """Run firm simulation for T time steps.
        
        Parameters
        ----------
        T : int
            Number of iterations starting with an initially empty lattice.
        cache : bool, True
            If True, save simulation result into self.storage dict.
            
        Returns
        -------
        list
            LiteFirm copies at each time step.
        list
            Lattice copies at each time step.
        """
        
        # settings
        L0 = self.L0
        N0 = self.N0
        g0 = self.g0
        obsRate = self.obsRate
        expandRate = self.expandRate
        innovSuccessRate = self.innovSuccessRate
        depressionRate = self.depressionRate
        innMutWidth = self.innMutWidth
        replicationP = self.replicationP
        growf = self.growf
        
        # variables for tracking history
        firmSnapshot = []
        latticeSnapshot = []
        
        # updated parameters
        firms = []
        lattice = TopicLattice()
        # initialize empty lattice
        lattice.extend_right(L0-1)

        # initialize
        for i in range(N0):
            # choose innovation tendency uniformly? how to impose selection?
            firms.append(Firm(np.random.randint(lattice.left, lattice.right+1),
                              np.random.rand(),
                              lattice=lattice))
            lattice.d_occupancy[firms[-1].sites[0]-lattice.left] += 1
        lattice.push()

        # run sim
        for t in range(T):
            # spawn new firms either by mutating from a sample of existing firms or by random sampling
            # from a uniform distribution
            nNew = np.random.poisson(g0)
            for i in range(nNew):
                if len(firms) and np.random.rand() <= replicationP:
                    # mutate an existing firm's innov
                    newInnov = np.clip(np.random.choice(firms).innov + np.random.normal(scale=innMutWidth),
                                       0, 1)
                    firms.append(Firm(np.random.randint(lattice.left, lattice.right+1),
                                      newInnov,
                                      lattice=lattice))
                    lattice.d_occupancy[firms[-1].sites[0]-lattice.left] += 1
                else:
                    firms.append(Firm(np.random.randint(lattice.left, lattice.right+1), np.random.rand(),
                                      lattice=lattice))
                    lattice.d_occupancy[firms[-1].sites[0]-lattice.left] += 1
            lattice.push()
        #     check_firms_occupancy(firms, lattice)

            # calculate firm income and growth
            growLattice = 0  # switch for if lattice needs to be grown
            newOccupancy = 0  # count new firms occupying innovated area
            if depressionRate:
                ix = np.random.rand(lattice.right-lattice.left+1) < depressionRate
                depressedSites = np.arange(lattice.left, lattice.right+1)[ix]
            else:
                depressionRate = []
            for f in firms:
                income = f.income(depressedSites)
                f.wealth += income
                f.age += 1

                growthcost = growf * f.wealth/f.size()
                eps = .01
                if f.wealth>(growthcost+eps) and f.rng.rand()<expandRate:
                    out = f.grow(innovSuccessRate, cost=growthcost)
                    if out[0] and not out[1]:
                        lattice.d_occupancy[f.sites[1]-lattice.left] += 1
                    elif out[0] and out[1]:
                        newOccupancy += 1
                        growLattice += out[1]
                    elif not out[0] and out[1]:
                        growLattice += 1
            lattice.push()
            # if any firm innovated, then grow the lattice
            if growLattice:
                lattice.extend_right()
                lattice.occupancy[-1] = newOccupancy
        #     check_firms_occupancy(firms, lattice)

            # shrink topic lattice
            if np.random.rand() < obsRate and lattice.left < lattice.right:
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

            # collect data on status
            firmSnapshot.append([f.copy() for f in firms])
            latticeSnapshot.append(lattice.copy())
        
        if cache:
            self.storage[str(datetime.now())] = firmSnapshot, latticeSnapshot
        return firmSnapshot, latticeSnapshot
        
    def parallel_simulate(self, nSamples, T):
        """Parallelize self.simulate() and save results into self.storage data member dict.
        
        Parameters
        ----------
        nSamples : int
        T : int
        
        Returns
        -------
        None
        """
        
        with Pool(cpu_count()) as pool:
            firmSnapshot, latticeSnapshot = list(zip(*pool.map(lambda args: self.simulate(T, cache=False),
                                                               range(nSamples))))
        for i in range(nSamples):
            self.storage[str(datetime.now())] = firmSnapshot[i], latticeSnapshot[i]
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
        self.age = 0.
        self.connectionCost = connection_cost
        
        if rng is None:
            self.rng = np.random.RandomState()
        self.id = id or ''.join(np.random.choice(LETTERS, size=20))
            
    def size(self):
        return self.sites[1]-self.sites[0]+1

    def income(self, depressed_sites=[]):
        """For each site, wealth grows by the amount
        (W/N) * di where di is the inverse site occupancy number. For depressed sites, wealth
        decreases by this amount.
        
        Parameters
        ----------
        depressed_sites : list, []
            Sites that see decreased income.
            
        Returns
        -------
        float
            Total change in wealth for this firm over one time step.
        """
        
        income = 0.
        for s in range(self.sites[0], self.sites[1]+1):
            if s in depressed_sites:
                income -= (self.wealth / (self.sites[1]-self.sites[0]+1)) / self.lattice.get_occ(s)
            else:
                income += (self.wealth / (self.sites[1]-self.sites[0]+1)) / self.lattice.get_occ(s)
        income -= self.wealth * self.connectionCost * (self.sites[1]-self.sites[0]+1)**.5
        return income
        
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
        
    def copy(self):
        return LiteFirm(self.sites,
                        self.innov,
                        self.connectionCost,
                        self.wealth,
                        self.age,
                        self.id)
#end Firm



class LiteFirm():
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
        self.age = age
        self.connectionCost = connection_cost
        self.id = id
#end LiteFirm


    
class TopicLattice():
    def __init__(self):
        """Start out one-dimensional lattice centered about 0."""
        
        self.left = 0
        self.right = 0
        self.occupancy = [0]
        self.d_occupancy = [0]
        
    # ============== #
    # Info functions
    # ============== #
    def get_occ(self, i):
        return self.occupancy[i-self.left]
        
    def len(self):
        return len(self.occupancy)
        
    # ============= #
    # Mod functions
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
