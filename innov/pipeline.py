# Pipeline for manuscript
# Author: Eddie Lee, edlee@csh.ac.at
from itertools import product

from .automaton import *
from .tree_gen import *
from .network_model_1_SDE_jax_obs_front import * 
from .simple_calculations import *


def figure1(memfraction=.3, device=0):
    """Run and save the resuts from automaton.
    """
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{memfraction}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'

    if not os.path.isdir(f'cache/L_comparison'):
        os.makedirs(f'cache/L_comparison')

    key = random.PRNGKey(10)
    dt = .01      # time steps
    el = 10, 500  # size of binary trees: initial chain number of nodes, number of generations of K chains  
    samples = 100 # Automaton and SDE samples

    # Dynamical parameters
    I = .2  # innovativeness 
    r = .41 # replication rate

    # Structural parameters
    K = 2        # number of branches

    save_steps = 5_000       # dt's to save tepmoral variables 
    max_steps = 40_000  # number of temporal variables to save

    for r0, rd, vo, gamma in [(40, .4, .5, 0.), (160, .4, .5, .5), (640, .4, .5, 1.)]:
        fname = f'cache/L_comparison/{gamma=}_{K=}_{dt=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        if not os.path.isfile(fname):
            # define graph structure
            tree = KTree(*el, K, gamma)
            # transform Ady into a sparse matrix for JAX
            Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
            Ady.data = Ady.data.astype(jnp.int8)

            def init_variables(N, samples):
                """Define an initial condition.

                Parameters
                ----------
                N : int
                    Size of graph.
                samples : int
                    Number of parallel jobs to run.
                """
                inn = jnp.zeros((samples,N), dtype=jnp.bool_)
                obs = jnp.zeros((samples,N), dtype=jnp.bool_)
                adj_obs = jnp.zeros((samples,N), dtype=jnp.bool_)
                sub = jnp.zeros((samples,N), dtype=jnp.bool_)
                n = jnp.zeros((samples,N), dtype=jnp.float32)

                # this needs to be adjusted to account for changeable value of K
                # looks hard-coded for K=2
                x_inn = el[0] + K*20  # generation at which inn front is located
                inn = inn.at[:,[x_inn, x_inn+1]].set(True)
                adj_obs = adj_obs.at[:,el[0]-1].set(True)
                n = n.at[:,el[0]-1:x_inn+2].set(40)
                sub = sub.at[:,el[0]-1:x_inn+2].set(True)
                return inn, obs, sub, n, adj_obs

            # setup automaton simulations
            init_vars, one_loop, run_save = setup_auto_sim(N = Ady.shape[0],
                                                           r = r,
                                                           rd = rd,
                                                           I = I,
                                                           r0 = r0,
                                                           dt = dt,
                                                           vo = vo,
                                                           samples = samples,
                                                           Ady = Ady,
                                                           init_fcn = init_variables,
                                                           obs_mode = 'random',
                                                           innov_front_mode = 'explorer')
            output = run_save(key, init_vars, save_steps, max_steps)
            key_save, inn_front, obs_front, in_sub_pop, n, adj_obs, x_inn, x_obs = output

            with open(fname, 'wb') as f:
                pickle.dump({'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                             'samples':samples, 'key':key_save, 'inn_front':inn_front, 'obs_front':obs_front,
                             'in_sub_pop':in_sub_pop, 'n':n, 'adj_obs':adj_obs, 'x_inn':x_inn, 'x_obs':x_obs},
                            f)
            print(f"Done with {fname}.")

def figure2(memfraction=.3, device=0):
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{memfraction}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'

    grid_vars = [{'gamma':0.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8},
                 #{'gamma':0.5, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8},
                 {'gamma':1.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8},
                 {'gamma':0.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8},
                 {'gamma':0.5, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8},
                 {'gamma':1.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8},
                 {'gamma':0.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8},
                 {'gamma':0.5, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8},
                 {'gamma':1.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8},
                 {'gamma':0.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8},
                 {'gamma':0.5, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8},
                 {'gamma':1.0, 'K':2, 'dt':0.05, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8}]
    max_steps = 500_000  # dt's to save temporal variables 
    save_steps = 5_000  # time steps between saves
    el = 2, 20_000  # size of binary trees: initial chain number of nodes, number of generations of K chains  
    samples = 100 # Automaton and SDE samples

    for x in grid_vars:
        key = random.PRNGKey(10)
        gamma = x['gamma']
        K = x['K']
        dt = x['dt']
        r0 = x['r0']
        vo = x['vo']
        rd = x['rd']
        r = x['r']
        I = x['I']
        
        # define graph structure
        tree = KTree(*el, K, gamma)
        # transform Ady into a sparse matrix for JAX
        Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
        Ady.data = Ady.data.astype(jnp.int8)
        
        def init_variables(N, samples):
            """Define an initial condition.
        
            Parameters
            ----------
            N : int
                Size of graph.
            samples : int
                Number of parallel jobs to run.
            """
            inn = jnp.zeros((samples,N), dtype=jnp.bool_)
            obs = jnp.zeros((samples,N), dtype=jnp.bool_)
            adj_obs = jnp.zeros((samples,N), dtype=jnp.bool_)
            sub = jnp.zeros((samples,N), dtype=jnp.bool_)
            n = jnp.zeros((samples,N), dtype=jnp.float32)
            dt = jnp.zeros(samples, dtype=jnp.float32)
        
            x_inn = el[0] + K*20
            inn = inn.at[:,[x_inn, x_inn+1]].set(True)
            adj_obs = adj_obs.at[:,el[0]-1].set(True)
            n = n.at[:,el[0]-1:x_inn+2].set(10)
            sub = sub.at[:,el[0]-1:x_inn+2].set(True)
            return inn, obs, sub, n, adj_obs, dt
        
        # setup automaton simulations
        init_vars, one_loop, run_save = setup_auto_sim(N = Ady.shape[0],
                                                       r = r,
                                                       rd = rd,
                                                       I = I,
                                                       r0 = r0,
                                                       dt = dt,
                                                       vo = vo,
                                                       samples = samples,
                                                       Ady = Ady,
                                                       init_fcn = init_variables,
                                                       obs_mode = 'random',
                                                       innov_front_mode = 'explorer')
        key, inn_front, obs_front, in_sub_pop, n, adj_obs, x_inn, x_obs = run_save(key, init_vars, save_steps, max_steps)
        
        fname = f'cache/L_comparison/{gamma=}_{K=}_{dt=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        with open(fname, 'wb') as f:
            pickle.dump({'r0':r0, 'r':r, 'rd':rd, 'vo':vo, 'I':I, 'gamma':gamma,
                         'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                         'samples':samples, 'key':key, 'inn_front':inn_front, 'obs_front':obs_front,
                         'in_sub_pop':in_sub_pop, 'n':n, 'adj_obs':adj_obs, 'x_inn':x_inn, 'x_obs':x_obs},
                        f)

if __name__=='__main__':
    pass  
