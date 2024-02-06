# ====================================================================================== #
# Network innovation model numerical solutions.
# 
# Author : Ernesto Ortega, ortega@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
from scipy.special import gammaln, logsumexp
import warnings
from jax import jit, vmap, config, random, device_put, devices
from jax.lax import fori_loop, cond
import jax.numpy as jnp
from jax.experimental.sparse import todense
#import networkx as nx

from workspace.utils import save_pickle
from .utils import *

def compute_P_O_1(N, τo, t, init):
        logP_O = np.zeros(N)
        for i in range(N):
            logP_O[i] = np.log(τo)*i - τo*t + np.log(t)*i - gammaln(i+1)

        return np.exp(logP_O - logsumexp(logP_O))

def setup_auto_num_int(N, r, rd, I, G_in, Δt, ro, key, samples, Ady, init_fcn, innov_front_mode='explorer_random', propagate_mode='SDE', obs_mode = 'random'):
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
        key, subkey = random.split(key)
        # initialize graph properties
        n = jnp.zeros((samples, N), dtype=jnp.int32)

        # obsolescence sites must always have a presence on the initial graph condition
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
        print(inverse_sons)
        x_obs = jnp.zeros((samples, N), dtype=jnp.float32)
        x_inn = jnp.zeros((samples, N), dtype=jnp.float32)
        x_inn_1 = jnp.zeros((samples, N), dtype=jnp.float32)
        """
        Parameters
        ----------
        N : int
            Size of whole graph.
        sub_ix : list of indices or twople, None
            If list of indices, then fill in those values with n0. If a twople,
            fill entries between the two indices with n0.
        inn: list
            list of nodes in the innovation front
        obs: list
            list of nodes in the obsolescence front
        Ady: np.array
            adjacency matrix
        n0 : float, 0.
            Initial value of density.
        τo: float
            obsolecence decay
        rd: float
            death rate
        r: float
            expansion rate
        I: float
            innovation rate
        G_in: float
            growth rate
        tmax: float
            max computing time
        Δt: float
            time step for update
        λ: float
            density threshold
        method: str
            propagator method:
                1- 'density'
                2- 'frequency'
        """    
        @jit
        def one_loop(i, val):
            # read in values
            key = val[0]
            inn_front = val[1]
            obs_sub = val[2]
            in_sub_pop = val[3]
            n = val[4]
            adj_obs = val[5]
            x_inn = val[6]
            x_obs = val[7]
            
            # compute density
            key, n = propagate(key, obs_sub, in_sub_pop, inn_front, n)
            
            # move innov front
            key, inn_front, in_sub_pop, x_inn = move_inn_front(key, inn_front, in_sub_pop, obs_sub, n, x_inn)

            # move obs front
            key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs = move_obs_front(key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs)

            return [key, inn_front, obs_sub, in_sub_pop, n, adj_obs, x_inn, x_obs]

        if propagate_mode == 'ODE':
            @jit
            def propagate(key, obs_sub, in_sub_pop, inn_front, n):          # propagate density function by some amount of time

                """ Runge kutta order 4 update equations
                """
                k1 = (-rd*n  + (r*(n*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop + (in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)#+ np.random.choice([1, -1], len(n))* np.random.poisson(self.Δt*n, len(n))

                k2 = (-rd*(n+k1/2)  + (r*((n+k1/2)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop + (in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)#+ np.random.choice([1, -1], len(n))* np.random.poisson(self.Δt*n, len(n))

                k3 = (-rd*(n+k2/2) + (r*((n+k2/2)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop + (in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)#+ np.random.choice([1, -1], len(n))* np.random.poisson(self.Δt*n, len(n))

                k4= (-rd*(n+k3) + (r*((n+k3)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop +(in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)#+ np.random.choice([1, -1], len(n))* np.random.poisson(self.Δt*n, len(n))

                k = Δt*(k1 + 2*k2 + 2*k3 + k4)/6
                #k = (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)/6
                """ Update densities and time
                """
                n = n + k
                return key, n
        elif propagate_mode == 'SDE':
            @jit
            def propagate(key, obs_sub, in_sub_pop, inn_front, n):          # propagate density function by some amount of time

                """ Runge kutta order 4 update equations
                """
                
                beta = 0.0
                
                
                
                key, subkey = random.split(key)
                k1 = Δt*(-rd*n  + (r*(n*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop +(in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)
                key, subkey = random.split(key)
                k2 = Δt*(-rd*(n+Δt*k1/2)  + (r*((n+Δt*k1/2)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop + (in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)
                key, subkey = random.split(key)
                k3 = Δt*(-rd*(n+Δt*k2/2) + (r*((n+Δt*k2/2)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop +(in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)
                key, subkey = random.split(key)
                k4= Δt*(-rd*(n+Δt*k3) + (r*((n+Δt*k3)*inverse_sons*in_sub_pop) @ Ady)*in_sub_pop + (in_sub_pop.T* (G_in/in_sub_pop.sum(axis=1))).T)

                k = (k1 + 2*k2 + 2*k3 + k4)/6
                #k = (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)/6
                """ Update densities and time
                """
                n = n + k + beta*(jnp.sqrt(n*Δt)) * random.normal(subkey, (samples, N))
                n = n *(n>=0)
                return key, n
        else:
            raise NotImplementedError("propagate_mode not recognized.")
        
        if obs_mode=='average':
            @jit    
            def move_obs_front(key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs):
                front_moved = adj_obs * (x_obs>1)
                x_obs = x_obs*(x_obs<1)
                parents_in_obs = (obs_sub@Ady)
                parents_in_obs = parents_in_obs.at[:,0].set(1)
                x_obs+= ro*adj_obs*parents_in_obs*Δt
                new_front_ix = front_moved @ Ady
                obs_sub = jnp.logical_or(obs_sub, front_moved)
                adj_obs = jnp.logical_or(adj_obs, new_front_ix)

                # remove new sites from sub populated graph
                in_sub_pop = in_sub_pop * ~obs_sub
                inn_front = inn_front * ~obs_sub
                adj_obs = adj_obs * ~obs_sub
                return key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs
        elif obs_mode=='random':
            @jit    
            def move_obs_front(key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs):
                beta_1 = 0.2
                key, subkey = random.split(key)
                front_moved = adj_obs * (x_obs>1)
                x_obs = x_obs*(x_obs<1)
                parents_in_obs = (obs_sub@Ady)
                # move into all children vertices if not in the innovation front
                new_front_ix = front_moved @ Ady
                #new_front_ix = new_front_ix * ~inn_front

                # add new sites to obsolescence front
                obs_sub = jnp.logical_or(obs_sub, front_moved)
                adj_obs = jnp.logical_or(adj_obs, new_front_ix)

                # remove new sites from sub populated graph
                in_sub_pop = in_sub_pop * ~obs_sub
                inn_front = inn_front * ~obs_sub
                adj_obs = adj_obs * ~obs_sub
                
                x_obs+= ro*adj_obs*parents_in_obs*Δt+ adj_obs*beta_1*jnp.sqrt((ro*Δt)*(1-(ro*Δt)))* random.normal(subkey, (samples, 1))
                
                return key, obs_sub, in_sub_pop, inn_front, adj_obs, x_obs
        else:
            raise NotImplementedError("obs_mode not recognized.")
        
        if innov_front_mode=='explorer':
            @jit
            def move_inn_front(key, inn_front, in_sub_pop, obs_sub, n, x_inn):
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
                
                key, subkey = random.split(key)
                # randomly choose innovation fronts to move
                front_moved = jnp.logical_and(inn_front, (x_inn>=1.))
                x_inn += r*I*n*Δt*inn_front
                # select new sites for innovation front, if not present in obsolescence or subpopulated graph 
                new_front_ix = jnp.logical_and(front_moved @ Ady, jnp.logical_and(~obs_sub, ~in_sub_pop))

                # add new nodes to the innovation front
                inn_front = jnp.logical_or(inn_front, new_front_ix)

                # now, add nodes in new innovation front to populated subgraph (must come after removing parent nodes)
                in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)

                # remove parent innovation fronts only if all children are in populated subgraph
                # must do this way (instead of removing parents who have children in innovation front)
                # because of colliding fronts
                inn_front = jnp.logical_and(inn_front, (in_sub_pop @ Ady.T)!=sons)

                return key, inn_front, in_sub_pop, x_inn
        elif innov_front_mode=='explorer_random':
            @jit
            def move_inn_front(key, inn_front, in_sub_pop, obs_sub, n, x_inn):
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
                beta_2 = 0.2
                key, subkey = random.split(key)
                # randomly choose innovation fronts to move
                front_moved = jnp.logical_and(inn_front, (x_inn>=1.))
                #x_inn += r*I*n*Δt*inn_front + inn_front*jnp.sqrt((r*I*n*beta_2*Δt)*(1-(r*I*n*beta_2*Δt)))* random.normal(subkey, (samples, N))
                
                #x_inn += r*I*n*Δt*inn_front
                # select new sites for innovation front, if not present in obsolescence or subpopulated graph 
                new_front_ix = jnp.logical_and(front_moved @ Ady, jnp.logical_and(~obs_sub, ~in_sub_pop))

                # add new nodes to the innovation front
                inn_front = jnp.logical_or(inn_front, new_front_ix)

                # now, add nodes in new innovation front to populated subgraph (must come after removing parent nodes)
                in_sub_pop = jnp.logical_or(in_sub_pop, inn_front)

                # remove parent innovation fronts only if all children are in populated subgraph
                # must do this way (instead of removing parents who have children in innovation front)
                # because of colliding fronts
                inn_front = jnp.logical_and(inn_front, (in_sub_pop @ Ady.T)!=sons)
                
                x_inn += r*I*n*Δt*inn_front + beta_2*inn_front*jnp.sqrt((r*I*n*Δt)*(1-(r*I*n*Δt)))* random.normal(subkey, (samples, N))
                
                return key, inn_front, in_sub_pop, x_inn
        elif innov_front_mode=='single_explorer':
            @jit
            def move_inn_front(key, inn_front, in_sub_pop, obs_sub, n):
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
                front_moved = inn_front * (random.uniform(subkey, (1, N)) > (1 - r*I*Δt*n))

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
            @jit
            def move_inn_front(key, inn_front, in_sub_pop, obs_sub, n):
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
                front_moved = in_sub_pop * (random.uniform(key, (1,N)) > (1 - r*I*Δt*n))

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
        
        init_vars = init_fcn(Ady.shape[0], samples)
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
            key, inn_front_1, obs_sub_1, in_sub_pop_1, n_1, x_inn_1, x_obs_1 = init_vars

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

                key, inn_front_1, obs_sub_1, in_sub_pop_1, n_1, x_inn_1, x_obs_1 = out_vars
                n.append(n_1)
                inn_front.append(inn_front_1)
                obs_sub.append(obs_sub_1)
                in_sub_pop.append(in_sub_pop_1)
                t.append((i+1)*save_dt)
            """
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
            """
            return key, t, n, inn_front, obs_sub, in_sub_pop
        return init_vars, one_loop, run_save

    #end RKPropagator1D 
