# JAX automaton implementation of innov/obs model on networks.
# Authors: Eddie Lee, edlee@csh.ac.at
#          Ernesto Ortega, ortega@csh.ac.at
import time
from jax import jit, vmap, config, random, device_put, devices
from jax.lax import fori_loop, cond
import jax.numpy as jnp
from jax import debug
from jax.experimental.sparse import todense
import numpy as np


# ================ #
# Helper functions #
# ================ #
def body_fun(xi, mi):
    # Use lax.cond to perform the conditional logic
    return cond(mi,
                lambda _: False,  # If mi is True, return value
                lambda _: xi,     # If mi is False, return xi
                operand=None)
set_false = vmap(body_fun)

def body_fun(xi, mi):
    # Use lax.cond to perform the conditional logic
    return cond(mi,
                lambda _: True,  # If mi is True, return value
                lambda _: xi,     # If mi is False, return xi
                operand=None)
set_true = vmap(body_fun)

def body_fun(xi, mi):
    # Use lax.cond to perform the conditional logic
    return cond(mi,
                lambda _: 0,  # If mi is True, return value
                lambda _: xi,     # If mi is False, return xi
                operand=None)
set_zero = vmap(body_fun)

def compress_density(n):
    """Compress density into a memory efficient representation.

    Parameters
    ----------
    n : jnp.ndarray

    Returns
    -------
    jnp.ndarray
        Density values.
    jnp.ndarray
        Corresponding indices.
    """
    ix = jnp.where(n)[0]
    return n[ix], ix

def decompress_density(n, ix, ix0=0, ix1=None):
    """Decompress density from a memory efficient representation.

    Parameters
    ----------
    n : jnp.ndarray
        Density values.
    ix : jnp.ndarray
        Corresponding indices.
    ix0 : int, 0
        Starting index. If greater than the smallest value in ix, then the lower
        value will be chosen as the new 0.
    ix1 : int, None
        Last index of array. Total array size shall be ix1-ix0+1.

    Returns
    -------
    jnp.ndarray
        Density array.
    """
    filled_n = jnp.zeros(ix.max()-min(ix.min(), ix0)+1, dtype=jnp.int32)
    filled_n = filled_n.at[ix].set(n)

    if ix1 is None:
        return filled_n

    if filled_n.size==ix1-ix0+1:
        return filled_n
    if filled_n.size<ix1-ix0+1:
        return jnp.concatenate((filled_n, jnp.zeros(ix1-ix0+1-filled_n.size, dtype=jnp.int32)))
    return filled_n[:ix1-ix0+1]

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
        obs_sub = jnp.zeros((samples, N), dtype=jnp.bool_)  # must start with False
        adj_obs = jnp.zeros((samples, N), dtype=jnp.bool_)
        sub = jnp.zeros((samples, N), dtype=jnp.bool_)
        n = jnp.zeros((samples, N), dtype=jnp.float32)
        t = jnp.zeros(1, dtype=jnp.float32)

        # innovation front is a uniform line of sites
        inn = inn.at[:,el[0]*K:(el[0]+1)*K].set(True)
        # obs front is the first site in joint chain
        adj_obs = adj_obs.at[:,:K].set(True)
        # initial density is everything beyond the obs front up to and including innov front
        n = n.at[:,K:K+K*el[0]].set(n0)
        sub = sub.at[:,K:K+K*el[0]].set(True)
        return inn, obs_sub, sub, n, adj_obs, t
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
    # initialize graph properties
    n = jnp.zeros((samples, N), dtype=jnp.int32)

    # obsolescence sites must always appear the initial graph
    obs_sub = jnp.zeros((samples, N), dtype=jnp.bool_)
    adj_obs = jnp.zeros((samples, N), dtype=jnp.bool_)
    inn_front = jnp.zeros((samples, N), dtype=jnp.bool_)

    in_sub_pop = jnp.zeros((samples, N), dtype=jnp.bool_)
    sites = jnp.arange(N, dtype=jnp.int32)
    new_front = jnp.zeros((samples, N), dtype=jnp.bool_)

    sons = Ady.sum(1).todense()
    inverse_sons = Ady @ jnp.ones(N, dtype=jnp.int32)
    inverse_sons = inverse_sons.at[inverse_sons==0].set(1)
    inverse_sons = 1. / inverse_sons
    
    # define innovation front subroutine
    if innov_front_mode=='explorer':
        @jit
        def move_innov_front(urand_matrix, inn_front, in_sub_pop, obs_sub, n, dt):
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
            obs_sub : boolean array
                Indicates sites that are obsolescence fronts using True.
            n : jnp.ndarray
                Density values.
            
            Returns
            -------
            inn_front
            in_sub_pop
            """
            # randomly choose innovation fronts to move
            front_moved = jnp.logical_and(inn_front, urand_matrix > (1 - r*I*dt*n))
            
            # select new sites for innovation front, if not present in obsolescence or subpopulated graph 
            new_front_ix = jnp.logical_and(front_moved @ Ady, jnp.logical_and(~obs_sub, ~in_sub_pop))
            
            # add new nodes to the innovation front
            inn_front = jnp.logical_or(inn_front, new_front_ix)
           
            # now, add nodes in new innovation front to populated subgraph (must
            # come after removing parent nodes)
            in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)

            # remove parent innovation fronts only if all children are in populated subgraph
            # must do this way (instead of removing parents who have children in innovation front)
            # because of colliding fronts
            inn_front = jnp.logical_and(inn_front, (in_sub_pop @ Ady.T)!=sons)
            
            return inn_front, in_sub_pop

    elif innov_front_mode=='single_explorer':
        # ================ requires debugging ================ #
        raise NotImplementedError("To be checked.")
        @jit
        def move_innov_front(key, inn_front, in_sub_pop, obs_sub, n, dt):
            """Move innovation fronts stochastically to one child node. Parent node
            remains part of the front as long as at least one child is not occupied
            and leave as soon as all children nodes are occupied.
            
            Parameters
            ----------
            key : jax.random.PRNGKey
            inn_front : boolean array
                Indicates sites that are innovation fronts using True.
            in_sub_pop : boolean array
                Indicates which sites are in the populated subgraph.
            n : jnp.ndarray
                Density values.
            
            Returns
            -------
            key
            inn_front
            in_sub_pop
            """
            # randomly choose innovation fronts to move
            key, subkey = random.split(key)
            front_moved = inn_front * (random.uniform(subkey, (samples, N)) > (1 - r*I*dt*n))

            # randomly choose amongst children to move innovation front to
            key, subkey = random.split(key)
            new_front_ix = (Ady * random.uniform(subkey, (N,N))).todense().argmax(1)
            new_front = jnp.zeros((samples, N), dtype=jnp.bool_)
            # make sure the parent was one of the moving innov fronts
            new_front = new_front.at[(jnp.arange(N), new_front_ix)].set(True) & (front_moved @ Ady)

            # set children innovation fronts
            inn_front = jnp.logical_or(inn_front, new_front)
            inn_front = inn_front.at[:,0].set(False)  # bookkeeping

            # advance populated subgraph to innovation front
            in_sub_pop = jnp.logical_or(in_sub_pop, new_front)

            # remove parent innovation fronts only if all children are in populated subgraph
            inn_front = jnp.logical_and(inn_front, (in_sub_pop @ Ady.T)!=sons)

            return key, inn_front, in_sub_pop

    elif innov_front_mode=='ant':
        # ================ requires debugging ================ #
        raise NotImplementedError("To be checked.")
        @jit
        def move_innov_front(key, inn_front, in_sub_pop, obs_sub, n, dt):
            """Move innovation fronts stochastically to one child node. Parent node
            is no longer part of the front afterwards.
            
            Parameters
            ----------
            key : jax.random.PRNGKey
            inn_front : boolean array
                Indicates sites that are innovation fronts using True.
            in_sub_pop : boolean array
                Indicates which sites are in the populated subgraph.
            n : jnp.ndarray
                Density values.
            
            Returns
            -------
            key
            inn_front
            in_sub_pop
            """
            # randomly choose innovation fronts to move
            key, subkey = random.split(key)
            front_moved = in_sub_pop * (random.uniform(subkey, (samples,N)) > (1 - r*I*dt*n))
            
            # randomly choose amongst children to move innovation to
            key, subkey = random.split(key)
            new_front_ix = (Ady * random.uniform(subkey, (N,N))).todense().argmax(1)
            new_front = jnp.zeros((samples, N), dtype=jnp.bool_)
            # make sure the parent was one of the moving innov fronts
            new_front = new_front.at[(jnp.arange(N), new_front_ix)].set(True) & (front_moved @ Ady)

            # remove parent innovation fronts
            inn_front = jnp.logical_xor(inn_front, front_moved)

            # set children innovation fronts
            inn_front = jnp.logical_or(inn_front, new_front)
            inn_front = inn_front.at[:,0].set(False)  # bookkeeping

            # move populated subgraph
            in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)
            
            return key, inn_front, in_sub_pop
    else:
        raise NotImplementedError("innov_front_mode not recognized.")

    # define obsolescence front subroutine
    if obs_mode =='random':
        @jit
        def move_obs_front(urand_matrix, obs_sub, in_sub_pop, inn_front, adj_obs, n, dt):
            """Grow obsolescence subgraph stochastically.

            TODO: allow obs subgraph to expand to all children instead of choosing one
                  at a time (isn't this already done bleow?)

            Parameters
            ----------
            urand_matrix : jnp.ndarray
                Matrix of random numbers from [0,1] interval.
            obs_sub : boolean array
                Indicates sites that are obsolescence graph using True.

            Returns
            -------
            obs_sub
            in_sub_pop
            inn_front
            """
            # randomly choose obsolesence front sites to move
            front_moved = adj_obs * (urand_matrix < (vo*dt))

            # move into all children vertices if not in the front
            new_front_ix = front_moved @ Ady
            #new_front_ix = new_front_ix * ~inn_front

            # add new sites to obsolescence subgraph
            obs_sub = jnp.logical_or(obs_sub, front_moved)
            adj_obs = jnp.logical_or(adj_obs, new_front_ix)

            # remove new obsolescent sites from populated subgraph and zero the density
            in_sub_pop = in_sub_pop * ~obs_sub
            inn_front = inn_front * ~obs_sub
            adj_obs = adj_obs * ~obs_sub
            n *= in_sub_pop
            return obs_sub, in_sub_pop, inn_front, adj_obs, n
        
    else:
        raise NotImplementedError("obs_front_mode not recognized.")

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
        obs_sub = val[2]
        in_sub_pop = val[3]
        n = val[4]
        adj_obs = val[5]
        t = val[6]

        # compute adaptive time step using density at innovation front
        # in principle, the cap can be a large value, but it won't matter for the parameter
        # values we are using
        thisdt = jnp.minimum(1 / ((n * inn_front).max() * r * I) / 10, 1/vo/10)
        thisdt = jnp.minimum(thisdt, 10)
        t += thisdt
        
        # roll matrix of shared random numbers
        key, subkey = random.split(key)
        urand_matrix = random.uniform(subkey, (samples, N))

        # move obsolescence front 
        obs_sub, in_sub_pop, inn_front, adj_obs, n = move_obs_front(urand_matrix,
                                                                    obs_sub,
                                                                    in_sub_pop,
                                                                    inn_front,
                                                                    adj_obs,
                                                                    n,
                                                                    thisdt)

        # move innovation front
        # roll random matrix
        urand_matrix = jnp.roll(urand_matrix, 1, axis=0)
        inn_front, in_sub_pop = move_innov_front(urand_matrix,
                                                 inn_front,
                                                 in_sub_pop,
                                                 obs_sub,
                                                 n,
                                                 thisdt)

        # total rate includes replication, growth, and death
        total_rate = jnp.maximum((r * inverse_sons * n) @ Ady +
                                 r0/in_sub_pop.sum(axis=1)[:,None] -
                                 rd * n, 0) * in_sub_pop

        key, subkey = random.split(key)
        dn = random.poisson(subkey, total_rate * thisdt)
        n += dn
        
        return [key, inn_front, obs_sub, in_sub_pop, n, adj_obs, t]

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
        obs_sub = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        in_sub_pop = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        n = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.float32)
        adj_obs = np.zeros((max_steps//save_steps+1,samples,Ady.shape[0]), dtype=np.bool_)
        t = np.zeros(max_steps//save_steps+1, dtype=np.float32)

        # save initial variable values
        key[0] = out_vars[0]
        inn_front[0,:,:] = out_vars[1]
        obs_sub[0,:,:] = out_vars[2]
        in_sub_pop[0,:,:] = out_vars[3]
        n[0,:,:] = out_vars[4]
        adj_obs[0,:,:] = out_vars[5]
        t[0] = out_vars[6][0]

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
            obs_sub[i+1,:,:] = out_vars[2]
            in_sub_pop[i+1,:,:] = out_vars[3]
            n[i+1,:,:] = out_vars[4]
            adj_obs[i+1,:,:] = out_vars[5]
            t[i+1] = out_vars[6][0]
            total_reading_t += time.time()-t0r

            if iprint: print("Done!", flush=True)
        total_t = time.time()-t0

        if iprint:
            print("Total time reading out vars", f'{total_reading_t:.2f}')
            print("Total time", f'{total_t:.2f}')
            print("Fraction reading", f'{total_reading_t/total_t:.2f}')

        return key, inn_front, obs_sub, in_sub_pop, n, adj_obs, t
    return init_vars, one_loop, run_save
