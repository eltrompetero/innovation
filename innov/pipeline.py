# Pipeline for manuscript
# Author: Eddie Lee, edlee@csh.ac.at
from itertools import product

from .automaton import *
from .tree_gen import *
from .simple_calculations import *


def figure1(memfraction=.3, device=0):
    """Run and save the resuts from automaton.
    """
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{memfraction}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'

    if not os.path.isdir(f'cache/L_comparison'):
        os.makedirs(f'cache/L_comparison')

    key = random.PRNGKey(10)
    el = 20, 10_000  # size of binary trees: initial chain number of nodes, number of generations of K chains  
    samples = 100 # replica number

    # Dynamical parameters
    I = .2  # innovativeness 
    r = .41 # replication rate

    # Structural parameters
    K = 2        # number of branches

    save_steps = 5_000       # steps between saving
    max_steps = 40_000  # total run steps

    for r0, rd, vo, gamma in [(40, .4, .5, 0.), (160, .4, .5, .5), (640, .4, .5, 1.)]:
        fname = f'cache/L_comparison/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
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
            inn = jnp.zeros((samples, N), dtype=jnp.bool_)
            obs_sub = jnp.zeros((samples, N), dtype=jnp.bool_)
            adj_obs = jnp.zeros((samples, N), dtype=jnp.bool_)
            sub = jnp.zeros((samples, N), dtype=jnp.bool_)
            n = jnp.zeros((samples, N), dtype=jnp.float32)
            t = jnp.zeros(1, dtype=jnp.float32)

            # innovation front is the last site in the initial line
            inn = inn.at[:,el[0]-1].set(True)
            # obs front is the first site in joint chain
            # obs = obs.at[:,0].set(True)
            adj_obs = adj_obs.at[:,0].set(True)
            # initial density is everything beyond the obs front up to and including innov front
            n = n.at[:,1:el[0]].set(10)
            sub = sub.at[:,1:el[0]].set(True)
            return inn, obs_sub, sub, n, adj_obs, t

        # setup automaton simulations
        init_vars, one_loop, run_save = setup_auto_sim(N = Ady.shape[0],
                                                        r = r,
                                                        rd = rd,
                                                        I = I,
                                                        r0 = r0,
                                                        vo = vo,
                                                        samples = samples,
                                                        Ady = Ady,
                                                        init_fcn = init_variables,
                                                        obs_mode = 'random',
                                                        innov_front_mode = 'explorer')
        output = run_save(key, init_vars, save_steps, max_steps)
        key_save, inn_front, obs_front, in_sub_pop, n, adj_obs, t = output

        with open(fname, 'wb') as f:
            pickle.dump({'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                            'samples':samples, 'key':key_save, 'inn_front':inn_front, 'obs_front':obs_front,
                            'in_sub_pop':in_sub_pop, 'n':n, 'adj_obs':adj_obs, 't':t},
                        f)
        print(f"Done with {fname}.")

def figure2(memfraction=.4, device=0):
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{memfraction}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'

    grid_vars = [{'gamma':0.0, 'K':2, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':0.5, 'K':2, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':1.0, 'K':2, 'r0':30, 'vo':0.22, 'rd':0.22, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':0.0, 'K':2, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8,
                  'max_steps':10_000, 'save_steps':100},
                 {'gamma':0.5, 'K':2, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8,
                  'max_steps':10_000, 'save_steps':100},
                 {'gamma':1.0, 'K':2, 'r0':30, 'vo':0.8, 'rd':0.8, 'r':0.2, 'I':0.8,
                  'max_steps':10_000, 'save_steps':100},
                 {'gamma':0.0, 'K':2, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':0.5, 'K':2, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':1.0, 'K':2, 'r0':30, 'vo':0.1, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':0.0, 'K':2, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':0.5, 'K':2, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000},
                 {'gamma':1.0, 'K':2, 'r0':30, 'vo':0.35, 'rd':0.1, 'r':0.2, 'I':0.8,
                  'max_steps':60_000, 'save_steps':1_000}]
    el = 20, 30_000  # size of binary trees: length of initial chain, number of generations of K chains  
    samples = 100 # replica number
    key = random.PRNGKey(3)

    for x in grid_vars:
        gamma = x['gamma']
        K = x['K']
        r0 = x['r0']
        vo = x['vo']
        rd = x['rd']
        r = x['r']
        I = x['I']
        max_steps = x['max_steps']
        save_steps = x['save_steps']
        
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
            inn = jnp.zeros((samples, N), dtype=jnp.bool_)
            obs_sub = jnp.zeros((samples, N), dtype=jnp.bool_)
            adj_obs = jnp.zeros((samples, N), dtype=jnp.bool_)
            sub = jnp.zeros((samples, N), dtype=jnp.bool_)
            n = jnp.zeros((samples, N), dtype=jnp.float32)
            t = jnp.zeros(1, dtype=jnp.float32)

            # innovation front is the last site in the initial line
            inn = inn.at[:,el[0]-1].set(True)
            # obs front is the first site in joint chain
            # obs = obs.at[:,0].set(True)
            adj_obs = adj_obs.at[:,0].set(True)
            # initial density is everything beyond the obs front up to and including innov front
            n = n.at[:,1:el[0]].set(10)
            sub = sub.at[:,1:el[0]].set(True)
            return inn, obs_sub, sub, n, adj_obs, t
        
        # setup automaton simulations
        init_vars, one_loop, run_save = setup_auto_sim(N = Ady.shape[0],
                                                       r = r,
                                                       rd = rd,
                                                       I = I,
                                                       r0 = r0,
                                                       vo = vo,
                                                       samples = samples,
                                                       Ady = Ady,
                                                       init_fcn = init_variables,
                                                       obs_mode = 'random',
                                                       innov_front_mode = 'explorer')
        key_out, inn_front, obs_sub, in_sub_pop, n, adj_obs, t = run_save(key, init_vars, save_steps, max_steps)
        key = key_out[-1]
        
        fname = f'cache/L_comparison/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        with open(fname, 'wb') as f:
            pickle.dump({'r0':r0, 'r':r, 'rd':rd, 'vo':vo, 'I':I, 'gamma':gamma,
                         'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                         'samples':samples, 'key':key_out, 'inn_front':inn_front, 'obs_sub':obs_sub,
                         'in_sub_pop':in_sub_pop, 'n':n, 'adj_obs':adj_obs, 't':t},
                        f)

if __name__=='__main__':
    pass  
