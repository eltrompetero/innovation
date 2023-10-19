# JAX automaton implemenation of innov/obs model on network.
# Authors: Eddie Lee, edlee@csh.ac.at
#          Ernesto Ortega, ortega@csh.ac.at
from jax import jit, vmap, config, random, device_put, devices
from jax.lax import fori_loop, cond
import jax.numpy as jnp
from jax.experimental.sparse import todense
#import torch


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



# ======== #
# Sim code #
# ======== #
def setup_auto_sim(N, r, rd, I, G_in, dt, ro, key, samples, Ady,
                   init_fcn,
                   innov_front_mode='explorer'):
    """Compile JAX functions necessary to run automaton simulation.

    Parameters
    ----------
    N : int
        Length of graph.
    r : float
    rd : float
    I : float
    G_in : float
    dt : float
    ro : float
    key : int
    samples : int
    Ady : jax.numpy.ndarray
    init_fcn : function
        To set up the initial parameter values for running.
    innov_front_mode : str, 'explorer'
    """
    # initialize graph properties
    n = jnp.zeros((samples, N), dtype=jnp.int32)

    # obsolescence sites must always have a presence on the initial graph condition
    obs_sub = jnp.zeros((samples, N), dtype=jnp.bool_)
    inn_front = jnp.zeros((samples, N), dtype=jnp.bool_)

    in_sub_pop = jnp.zeros((samples, N), dtype=jnp.bool_)
    sites = jnp.arange(N, dtype=jnp.int32)
    new_front = jnp.zeros((samples, N), dtype=jnp.bool_)

    sons = Ady.sum(1)
    inverse_sons = Ady @ jnp.ones(N, dtype=jnp.int32)
    inverse_sons = inverse_sons.at[inverse_sons==0].set(1)
    inverse_sons = 1. / inverse_sons

    if innov_front_mode=='explorer':
        @jit
        def move_innov_front(key, inn_front, in_sub_pop, obs_sub, n):
            """Move innovation fronts stochastically. When progressing, move to
            occupy all children nodes.
            
            Parameters
            ----------
            key : jax.random.PRNGKey
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
            key
            inn_front
            in_sub_pop
            """
            # randomly choose innovation fronts to move
            key, subkey = random.split(key)
            front_moved = inn_front * (random.uniform(subkey, (samples, N)) > (1 - r*I*dt*n))
            
            # select new sites for innovation front if not present in obsolescence or subpopulated graph 
            new_front_ix = front_moved @ Ady * ~jnp.logical_xor(obs_sub, in_sub_pop)
            
            # add new nodes to the innovation front
            inn_front = jnp.logical_or(inn_front, new_front_ix)
            # no parents of the innovation front should also be in the innovation front b/c their
            # children are
            inn_front = jnp.logical_and(inn_front, ~jnp.logical_and(front_moved @ Ady.T, in_sub_pop))
            
            # now, add nodes in new innovation front to populated subgraph (must come after removing parent nodes)
            in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)
            
            return key, inn_front, in_sub_pop

    elif innov_front_mode=='single_explorer':
        @jit
        def move_innov_front(key, inn_front, in_sub_pop, obs_sub, n):
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
            inn_front = jnp.logical_and(inn_front, ~((in_sub_pop.astype(jnp.int16) @ Ady.T)==sons.todense()))

            return key, inn_front, in_sub_pop

    elif innov_front_mode=='ant':
        @jit
        def move_innov_front(key, inn_front, in_sub_pop, obs_sub, n):
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

    @jit
    def move_obs_front(key, obs_sub, in_sub_pop, inn_front):
        """Grow obsolescence subgraph stochastically.
        
        TODO: allow obs subgraph to expand to all children instead of choosing one
              at a time
        
        Parameters
        ----------
        key
        obs_sub : boolean array
            Indicates sites that are obsolescence graph using True.
        
        Returns
        -------
        key
        obs_sub
        in_sub_pop
        inn_front
        """
        # randomly choose obsolesence sites to move
        key, subkey = random.split(key)
        front_moved = obs_sub * (random.uniform(subkey, (samples, 1)) < ro*dt)
        
        # move into all children vertices if not in the innovation front
        key, subkey = random.split(key)
        new_front_ix = front_moved @ Ady
        new_front_ix = new_front_ix * ~inn_front
        
        # add new sites to obsolescence front
        obs_sub = jnp.logical_or(obs_sub, new_front_ix)
        
        # remove new sites from sub populated graph
        in_sub_pop = in_sub_pop * ~obs_sub
        
        return key, obs_sub, in_sub_pop, inn_front

    @jit
    def one_loop(i, val):
        # read in values
        key = val[0]
        inn_front = val[1]
        obs_sub = val[2]
        in_sub_pop = val[3]
        n = val[4]
        
        # move innovation front
        key, inn_front, in_sub_pop = move_innov_front(key, inn_front, in_sub_pop, obs_sub, n)
    #     debug.print("{x}", x=inn_front)
        
        # replicate
        key, subkey = random.split(key)
        to_replicate = random.poisson(subkey, ((r * inverse_sons * n * dt) @ Ady))
        n = n + to_replicate*in_sub_pop
    #     debug.print("DREP {x}", x=(Ady.T @ (r * inverse_sons * n * dt))[:10])
    #     debug.print("REP {x}", x=n[:10])
        
        # death
        key, subkey = random.split(key)
        to_die = random.poisson(subkey, rd * n * dt)
        n = n - jnp.minimum(n, to_die)

        # growth
        key, subkey = random.split(subkey)
        G_dt = random.poisson(subkey, G_in*dt/in_sub_pop.sum(axis=1), (N, samples))
        n = (n + G_dt.T) * in_sub_pop

        # obsolescence front 
        key, obs_sub, in_sub_pop, inn_front = move_obs_front(key, obs_sub, in_sub_pop, inn_front)
    #     debug.print("OBS {x}", x=n[:10])
 
        return [key, inn_front, obs_sub, in_sub_pop, n]

    init_vars = init_fcn(Ady.shape[0],
                         samples)

    def run_save(init_vars, save_dt, tmax):
        """
        Parameters
        ----------
        init_vars : list
            Initial state in which to start simulations.
        save_dt : float
            dt between saves.
        tmax : float
            Simulation runtime is tmax * dt.
        """
        inn_front_1, obs_sub_1, in_sub_pop_1, n_1 = init_vars[1:]
        
        n = [n_1]
        inn_front = [inn_front_1]
        obs_sub = [obs_sub_1]
        in_sub_pop = [in_sub_pop_1]
        t = [0]
        
        for i in range(int(tmax/save_dt)):
            if i==0:
                out_vars = fori_loop(0, save_dt, one_loop, init_vars)
            else:
                out_vars = fori_loop(0, save_dt, one_loop, out_vars)
            
            key, inn_front_1, obs_sub_1, in_sub_pop_1, n_1 = out_vars
            n.append(n_1)
            inn_front.append(inn_front_1)
            obs_sub.append(obs_sub_1)
            in_sub_pop.append(in_sub_pop_1)
            t.append((i+1)*save_dt)

            # move previous results into CPU mem
            n[-2] = [compress_density(n_) for n_ in n[-2]]
            n[-2] = device_put(n[-2], devices('cpu')[0])
            inn_front[-2] = device_put(inn_front[-2], devices('cpu')[0])
            obs_sub[-2] = device_put(obs_sub[-2], devices('cpu')[0])
            in_sub_pop[-2] = device_put(in_sub_pop[-2], devices('cpu')[0])
        
        # move previous results into CPU mem
        n[-1] = [compress_density(n_) for n_ in n[-1]]
        n[-1] = device_put(n[-1], devices('cpu')[0])
        inn_front[-1] = device_put(inn_front[-1], devices('cpu')[0])
        obs_sub[-1] = device_put(obs_sub[-1], devices('cpu')[0])
        in_sub_pop[-1] = device_put(in_sub_pop[-1], devices('cpu')[0])

        return key, t, n, inn_front, obs_sub, in_sub_pop
    return init_vars, one_loop, run_save
