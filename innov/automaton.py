# JAX automaton implementation of innov/obs model on networks.
# Authors: Eddie Lee, edlee@csh.ac.at
#          Ernesto Ortega, ortega@csh.ac.at
import time
from jax import jit, vmap, config, random, device_put, devices
from jax.lax import fori_loop, cond, while_loop
import jax.numpy as jnp
from jax import debug
from jax.experimental.sparse import todense
import numpy as np


# ================ #
# Helper functions #
# ================ #
def inn_front_loc(inn_front, samples, el, K, return_max=False, pinned=False):
    """Returns mean location of innovation front by time step by replica.
    
    Parameters
    ----------
    inn_front : jnp.array
    samples : int
        Number of random replicas.
    el : int
        Total number of generations.
    K : int
        Number of branches.
    return_max : bool, False
        If True, return only the furthest location on each branch.
    """
    assert inn_front.ndim==3 and inn_front.shape[2]==el*K

    t = inn_front.shape[0]
    mean_loc = np.zeros((t, samples))
    x = inn_front.reshape(t, samples, el, K)
    
    if return_max:
        counter = np.arange(el)  # useful var for tracking index
        for t_ in range(t):
            for i in range(samples):
                m = []
                for j in range(K):
                    m.append(counter[x[t_,i,:,j]][-1])
                if pinned:
                    m = [max(m)-m_ for m_ in m]
                mean_loc[t_, i] = np.mean(m)
    else:
        # mean of innov front locations along each branch per replica
        counter = np.tile(np.arange(el), (K, 1)).T  # useful var for tracking index
        for t_ in range(t):
            for i in range(samples):
                if pinned:
                    mean_loc[t_, i] = (counter[x[t_,i,:,:]].max()-counter[x[t_,i,:,:]]).mean()
                else:
                    mean_loc[t_, i] = counter[x[t_,i,:,:]].mean()
    return mean_loc

def obs_front_loc(obs_front, samples, el, K, pinned=False):
    """Returns mean location of obsolescence front over all branches and then over replicas.

    The front is only the leading site along a given branch.

    Parameters
    ----------
    obs_front : jnp.array
    samples : int
        Number of random replicas.
    el : int
        Total number of generations.
    K : int
        Number of branches.

    Returns
    -------
    ndarray
        Avg location of obsolescence front per time point.
    """
    t = obs_front.shape[0]
    mean_loc = np.zeros((t, samples))
    x = obs_front.reshape(t, samples, el, K)
    counter = np.arange(el)[:,None]  # helper var for tracking index
    
    for t_ in range(t):
        for s in range(samples):
            if pinned:
                temp = [counter[ix].ravel()[-1] for ix in x[t_, s].T]
                mean_loc[t_, s] = np.mean([max(temp)-t_ for t_ in temp])
            else:
                mean_loc[t_, s] = np.mean([counter[ix].ravel()[-1] for ix in x[t_, s].T])
    return mean_loc

def leading_front_density(n, inn_front, el, K):
    """Given automaton output, return the mean density at the leading front averaged over branches and replicas.

    Parameters
    ----------
    n : jnp.array
    inn_front : jnp.array
    el : tuple
        (length of initial seeded branch, total length branches)   
    K : int

    Returns
    -------
    list
    """
    samples = inn_front.shape[1]
    T = inn_front.shape[0]

    inn_front_ = inn_front.reshape(T, samples, el[1], K)
    n_ = n.reshape(T, samples, el[1], K)

    n0 = []
    for t in range(T):
        n0.append([])
        for i in range(samples):
            for j in range(K):
                ix = inn_front_[t,i,:,j]
                if ix.any():
                    ix = np.where(ix)[0][-1]
                    n0[-1].append(n_[t,i,:,j][ix])
        n0[t] = np.mean(n0[t])
    return n0

def create_init_variables(el, K, n0):
    """Create a function to initialize variables for running the automaton.

    Parameters
    ----------
    el : tuple
        (length of initial seeded branch, total length branches)
    K : int
        Number of branches.
    n0 : float
        Seed density.
    """
    def init_variables(N, samples):
        """Define an initial condition. This assumes that the graph consists of
        a set of parallel branches and initializes the values on the beginning
        of each branch equally.

        Parameters
        ----------
        N : int
            Size of graph.
        samples : int
            Number of parallel jobs to run.

        Returns
        -------
        input variables for one_loop
        """
        inn = jnp.zeros((samples, N), dtype=jnp.bool_)
        obs_front = jnp.zeros((samples, N), dtype=jnp.bool_) 
        sub = jnp.zeros((samples, N), dtype=jnp.bool_)
        n = jnp.zeros((samples, N), dtype=jnp.float32)
        t = jnp.zeros(1, dtype=jnp.float32)

        # innovation front is a uniform line of sites
        inn = inn.at[:,el[0]*K:(el[0]+1)*K].set(True)
        # obsolescence front is a uniform line of sites at generation 0
        obs_front = obs_front.at[:,:K].set(True)
        # initial density is everything beyond the obs front up to and including innov front
        n = n.at[:,K:K+K*el[0]].set(n0)
        sub = sub.at[:,K:K+K*el[0]].set(True)
        return inn, obs_front, sub, n, t
    return init_variables


# ============== #
# Automaton code #
# ============== #
def setup_auto_sim(N, r, rd, I, r0, vo, samples, Ady,
                   init_fcn,
                   innov_front_mode='explorer',
                   obs_mode = 'random'):
    """Compile JAX functions necessary to run automaton simulation.

    Parameters
    ----------
    N : int
        Length of graph.
    r : float
    rd : float
    I : float
    r0 : float
    vo : float
    samples : int
    Ady : jax.numpy.ndarray
    init_fcn : function
        To set up the initial parameter values for running.
    innov_front_mode : str, 'explorer'
    """
    # convert rates to floats
    r = float(r)
    rd = float(rd)
    I = float(I)
    r0 = float(r0)
    vo = float(vo)
    assert vo>0

    # initialize graph properties
    sons = Ady.sum(1).todense()  # no. of children that a node has
    parents = Ady.sum(0).todense()  # no. of parents that a node has
    max_parents = parents.max()
    inverse_sons = Ady @ jnp.ones(N, dtype=jnp.int32)
    inverse_sons = inverse_sons.at[inverse_sons==0].set(1)
    inverse_sons = 1. / inverse_sons
    
    # define innovation front subroutine
    if innov_front_mode=='explorer':
        @jit
        def move_innov_front(urand_matrix, inn_front, in_sub_pop, n, dt):
            """Move innovation fronts stochastically. When progressing, move to
            occupy all children nodes.
            
            Parameters
            ----------
            urand_matrix : jnp.ndarray
                Matrix of random numbers from [0,1] interval.
            inn_front : boolean array
                Indicates sites that are innovation fronts using True.
            in_sub_pop : boolean array
                Indicates which sites are in the populated subgraph.
            n : jnp.ndarray
                Density values.
            
            Returns
            -------
            inn_front
            in_sub_pop
            """
            # randomly choose innovation fronts to move
            front_moved = jnp.logical_and(inn_front, urand_matrix < r*I*dt*n)
            
            # select new sites for innovation front, if not present in
            # subpopulated graph 
            new_front_ix = jnp.logical_and(front_moved @ Ady, ~in_sub_pop)
            #new_front_ix = front_moved @ Ady
            
            # add new nodes to the innovation front
            inn_front = jnp.logical_or(inn_front, new_front_ix)
           
            # now, add nodes in new innovation front to populated subgraph (must
            # come after removing parent nodes)
            in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)

            ## remove all nodes that moved
            #inn_front = jnp.logical_and(inn_front, ~front_moved)

            # remove parent innovation fronts only if all children are in populated subgraph
            # this can happen if neighboring sites move and occupy all children
            # must do this way (instead of removing parents who have children in innovation front)
            # because of colliding fronts
            # this also removes any front that has moved because all children are then
            # occupied
            inn_front = jnp.logical_and(inn_front, (in_sub_pop @ Ady.T)!=sons)

            return inn_front, in_sub_pop
    else:
        raise NotImplementedError("innov_front_mode not recognized.")

    # define obsolescence front subroutine
    if obs_mode =='random':
        @jit
        def move_obs_front(urand_matrix, in_sub_pop, inn_front, obs_front, n, dt):
            """Grow obsolescence subgraph stochastically.

            Parameters
            ----------
            urand_matrix : jnp.ndarray
                Matrix of random numbers from [0,1] interval.
            obs_front: boolean array
                Indicates sites that are in obsolescence subgraph.

            Returns
            -------
            obs_front
            in_sub_pop
            inn_front
            """
            # sample from obsolesence front sites to move
            front_moved = obs_front * (urand_matrix <= (vo*dt))

            # move into all children vertices
            obs_front = jnp.logical_or(obs_front, front_moved @ Ady)

            # remove new obsolescent sites from populated subgraph and zero the density
            in_sub_pop = in_sub_pop * ~obs_front
            n *= in_sub_pop
            inn_front = inn_front * ~obs_front
            return obs_front, in_sub_pop, inn_front, n
    else:
        raise NotImplementedError("obs_front_mode not recognized.")
    
    @jit
    def set_obs_as_smaller_dt(inn_dt, obs_dt):
        nloops_inn = jnp.array([1], dtype=jnp.int32)
        nloops_obs = jnp.array([inn_dt[0]//obs_dt[0] + 1], dtype=jnp.int32)
        obs_dt = inn_dt/nloops_obs
        return nloops_inn, nloops_obs, inn_dt, obs_dt
    
    @jit
    def set_inn_as_smaller_dt(inn_dt, obs_dt):
        nloops_obs = jnp.array([1], dtype=jnp.int32)
        nloops_inn = jnp.array([obs_dt[0]//inn_dt[0] + 1], dtype=jnp.int32)
        inn_dt = obs_dt/nloops_inn
        return nloops_inn, nloops_obs, inn_dt, obs_dt

    @jit
    def one_loop(i, val):
        """JAX routine to loop over a fixed number of time steps.
        
        Idea is to move fronts first because that determines dt. This is because
        the fronts can only move one lattice step at a time, where as the remaining
        density steps can be Poisson sampled.

        Also, the positive and negative changes to the density must be done in sequence
        to avoid negatives.
        """
        # read in values
        key = val[0]
        inn_front = val[1]
        obs_front = val[2]
        in_sub_pop = val[3]
        n = val[4]
        t = val[5]

        # compute adaptive time step using density at innovation front
        # in principle, the cap can be a large value, but it won't matter for the parameter
        # values we are using (i.e. large densities)
        # these choices set precision of the simulation
        thisdt = jnp.minimum(1/((n * inn_front).max() * r * I), 1/vo) / max_parents / 100
        thisdt = jnp.array([jnp.maximum(jnp.minimum(thisdt, 100), 1e-7)])
        t += thisdt

        key, subkey = random.split(key)
        urand_matrix = random.uniform(subkey, (samples, N))

        # move obsolescence front 
        obs_front, in_sub_pop, inn_front, n = move_obs_front(urand_matrix,
                                                                in_sub_pop,
                                                                inn_front,
                                                                obs_front,
                                                                n,
                                                                thisdt)

        # roll matrix of shared random numbers as a cheap way to get new random numbers
        #urand_matrix = jnp.roll(urand_matrix, 1, axis=0)
        key, subkey = random.split(key)
        urand_matrix = random.uniform(subkey, (samples, N))

        # move innovation front
        inn_front, in_sub_pop = move_innov_front(urand_matrix,
                                                inn_front,
                                                in_sub_pop,
                                                n,
                                                thisdt)

        # total rate at each site, includes replication (from all parents), influx, and death
        # keep n positive semi-definite
        total_rate = ((r * inverse_sons * n) @ Ady +
                      r0/in_sub_pop.sum(axis=1)[:,None] -
                      rd * n) * in_sub_pop
        total_rate_sign = jnp.sign(total_rate)

        key, subkey = random.split(key)
        dn = random.poisson(subkey, total_rate_sign * total_rate * thisdt)
        n += dn * total_rate_sign
        n = jnp.maximum(n, 0)
        
        return [key, inn_front, obs_front, in_sub_pop, n, t]

    init_vars = init_fcn(Ady.shape[0], samples)

    def run_save(key, out_vars, save_steps, max_steps, iprint=True):
        """
        Parameters
        ----------
        out_vars : list
            Initial state with which to start simulation.
        save_steps : float
            dt between saves.
        max_steps : float
            Simulation runtime is max_steps * dt.

        Returns
        -------
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        """
        assert save_steps<=max_steps

        # initialize variables for for loop
        out_vars = [key]+list(out_vars)

        # define output vars to copy GPU variables to CPU RAM
        key = np.zeros((max_steps, 2), dtype=np.uint32)
        inn_front = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        obs_front = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        in_sub_pop = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        n = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.float32)
        t = np.zeros(max_steps//save_steps+1, dtype=np.float32)

        # save initial variable values
        key[0] = out_vars[0]
        inn_front[0,:,:] = out_vars[1]
        obs_front[0,:,:] = out_vars[2]
        in_sub_pop[0,:,:] = out_vars[3]
        n[0,:,:] = out_vars[4]
        t[0] = out_vars[5][0]

        total_t = 0
        total_reading_t = 0
        t0 = time.time()
        for i in range(max_steps//save_steps):
            if iprint: print(i+1, (i+1)*save_steps, '/', max_steps, '...', end=' ', flush=True)
            loopt0 = time.time()
            out_vars = fori_loop(0, save_steps, one_loop, out_vars)
            if iprint: print(f'{time.time()-loopt0:.2f}', 's', '...', end=' ', flush=True)
        
            t0r = time.time()
            key[i+1] = out_vars[0]
            inn_front[i+1,:,:] = out_vars[1]
            obs_front[i+1,:,:] = out_vars[2]
            in_sub_pop[i+1,:,:] = out_vars[3]
            n[i+1,:,:] = out_vars[4]
            t[i+1] = out_vars[5][0]
            total_reading_t += time.time()-t0r

            if iprint: print("Done!", flush=True)
        total_t = time.time()-t0

        if iprint:
            print("Total time reading out vars", f'{total_reading_t:.2f}')
            print("Total time", f'{total_t:.2f}')
            print("Fraction reading", f'{total_reading_t/total_t:.2f}')

        return key, inn_front, obs_front, in_sub_pop, n, t
    
    def run(key, out_vars, t, iprint=True):
        """Run simulation until a certain duration (stop as soon as that
        duration is crossed). Only return output.

        Parameters
        ----------
        key : jax.random.PRNGKey
        out_vars : list
            Initial state with which to start simulation.
        t : float
            (approximate) duration of simulation.
        iprint : bool, True
            Print progress.

        Returns
        -------
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        """
        cond_fun = lambda out_vars: out_vars[-1][0]<t
        body_fun = lambda out_vars: one_loop(0, out_vars)

        # initialize variables for for loop
        out_vars = [key]+list(out_vars)

        # run while loop
        t0 = time.time()
        out_vars = while_loop(cond_fun, body_fun, out_vars)
        if iprint: print(f'Running simulation for dt={out_vars[-1][0]:.2f}', flush=True)
        total_t = time.time()-t0
        if iprint: print("Runtime", f'{total_t:.2f} s', flush=True)

        return out_vars

    def run_save_t(key, out_vars, save_dt, max_t, iprint=True):
        """run_save except using time.

        Note that this will only run til the maximum t that is an integer multiple of save_dt.

        Parameters
        ----------
        out_vars : list
            Initial state with which to start simulation.
        save_dt : float
            dt between saves.
        max_t : float
            Simulation runtime.

        Returns
        -------
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        ndarray
        """
        assert save_dt<=max_t
        save_steps = int(max_t//save_dt)

        # initialize variables for for loop
        out_vars = [key]+list(out_vars)

        # define output vars to copy GPU variables to CPU RAM
        key = np.zeros((save_steps+1, 2), dtype=np.uint32)
        inn_front = np.zeros((save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        obs_front = np.zeros((save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        in_sub_pop = np.zeros((save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        n = np.zeros((save_steps+1,samples,Ady.shape[0]), dtype=np.float32)
        t = np.zeros(save_steps+1, dtype=np.float32)

        # save initial variable values
        key[0] = out_vars[0]
        inn_front[0,:,:] = out_vars[1]
        obs_front[0,:,:] = out_vars[2]
        in_sub_pop[0,:,:] = out_vars[3]
        n[0,:,:] = out_vars[4]
        t[0] = out_vars[5][0]

        total_t = 0
        total_reading_t = 0
        t0 = time.time()
        for i in range(save_steps):
            if iprint: print(i+1, save_dt*(i+1), '/', save_steps*save_dt, '...', end=' ', flush=True)
            loopt0 = time.time()
            out_vars = run(key[i], out_vars[1:], (i+1)*save_dt-t[i], iprint=False)
            if iprint: print(f'{time.time()-loopt0:.2f}', 's', '...', end=' ', flush=True)
        
            t0r = time.time()
            key[i+1] = out_vars[0]
            inn_front[i+1,:,:] = out_vars[1]
            obs_front[i+1,:,:] = out_vars[2]
            in_sub_pop[i+1,:,:] = out_vars[3]
            n[i+1,:,:] = out_vars[4]
            t[i+1] = out_vars[5][0] + t[i]
            total_reading_t += time.time()-t0r

            # reset sim time
            out_vars[5] = out_vars[5].at[0].set(0)

            if iprint: print("Done!", flush=True)
        total_t = time.time()-t0

        if iprint:
            print("Total time reading out vars", f'{total_reading_t:.2f}')
            print("Total time", f'{total_t:.2f}')
            print("Fraction reading", f'{total_reading_t/total_t:.2f}')

        return key, inn_front, obs_front, in_sub_pop, n, t

    return init_vars, one_loop, run_save, run, run_save_t