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

    if not os.path.isdir(f'cache'):
        os.makedirs(f'cache')

    key = random.PRNGKey(10)
    el = 20, 10_000  # len of initial no. of nodes to seed, no. of generations in K branches
    samples = 100 # replica number
    n0 = 10  # initial density

    # Dynamical parameters
    I = .2  # innovativeness 
    r = .41 # replication rate

    # Structural parameters
    K = 2        # number of branches

    save_steps = 5_000       # steps between saving
    max_steps = 40_000  # total run steps

    for r0, rd, vo, gamma in [(10, .4, .5, 0.),
                              (320, .4, .5, .5),
                              (1280, .4, .5, 1.)]:
        fname = f'cache/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        # define graph structure
        tree = KTree(0, el[1], K, gamma)
        # transform Ady into a sparse matrix for JAX
        Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
        Ady.data = Ady.data.astype(jnp.int8)

        init_variables = create_init_variables(el, K, n0)

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
        
        init_variables = create_init_variables(el, K, 10)
        
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
        
        fname = f'cache/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        with open(fname, 'wb') as f:
            pickle.dump({'r0':r0, 'r':r, 'rd':rd, 'vo':vo, 'I':I, 'gamma':gamma,
                         'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                         'samples':samples, 'key':key_out, 'inn_front':inn_front, 'obs_sub':obs_sub,
                         'in_sub_pop':in_sub_pop, 'n':n, 'adj_obs':adj_obs, 't':t},
                        f)

if __name__=='__main__':
    pass  
