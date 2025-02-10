# Pipeline for manuscript
# Author: Eddie Lee, edlee@csh.ac.at
from itertools import product
from jax import clear_caches
from workspace.utils import save_pickle

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

    el = 20, 600  # tree dimensions
    samples = 100
    n0 = 20

    # Dynamical parameters
    r0 = 10
    I = 2
    r = .51
    rd = .5
    vo = .02

    # Structural parameters
    K = 50

    max_steps = 60_000  # dt's to save temporal variables 
    save_steps = max_steps//10  # time steps between saves

    with open('cache/fig1_base_params.p', 'wb') as f:
        pickle.dump({'el':el, 'samples':samples, 'n0':n0, 'I':I, 'r':r, 'K':K, 'r0':r0, 'rd':rd, 'vo':vo,
                     'save_steps':save_steps, 'max_steps':max_steps},
                    f)

    for gamma in [0, .5, 1]:
        fname = f'cache/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        # define graph structure
        tree = KTree(el[1], K, gamma)
        # transform Ady into a sparse matrix for JAX
        Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
        Ady.data = Ady.data.astype(jnp.int8)

        init_variables = create_init_variables(el, K, n0)

        # setup automaton simulations
        init_vars, one_loop, run_save, run = setup_auto_sim(N = Ady.shape[0],
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
        key_save, inn_front, obs_front, in_sub_pop, n, t = output

        with open(fname, 'wb') as f:
            pickle.dump({'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                         'samples':samples, 'key':key_save, 'inn_front':inn_front, 'obs_front':obs_front,
                         'in_sub_pop':in_sub_pop, 'n':n, 't':t},
                        f)
        print(f"Done with {fname}.")

def _figure2(memfraction=.4, device=0):
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
        key_out, inn_front, obs_sub, in_sub_pop, n, t = run_save(key, init_vars, save_steps, max_steps)
        key = key_out[-1]
        
        fname = f'cache/{gamma=}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        with open(fname, 'wb') as f:
            pickle.dump({'r0':r0, 'r':r, 'rd':rd, 'vo':vo, 'I':I, 'gamma':gamma,
                         'el':el, 'K':K, 'save_steps':save_steps, 'max_steps':max_steps,
                         'samples':samples, 'key':key_out, 'inn_front':inn_front, 'obs_sub':obs_sub,
                         'in_sub_pop':in_sub_pop, 'n':n, 't':t},
                        f)

def _front_test(key, r, rd, I, r0, vo, el, K, gamma, samples):
    """Helper function."""
    n0 = 20

    max_steps = 60_000  # dt's to save temporal variables 
    save_steps = max_steps//21  # time steps between saves

    # define graph structure
    tree = KTree(el[1], K, gamma)
    # transform Ady into a sparse matrix for JAX
    Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
    Ady.data = Ady.data.astype(jnp.int8)

    init_variables = create_init_variables(el, K, n0)

    # setup automaton simulations
    init_vars, one_loop, run_save, run = setup_auto_sim(N = Ady.shape[0],
                                                        r = r,
                                                        rd = rd,
                                                        I = I,
                                                        r0 = r0*K,
                                                        vo = vo,
                                                        samples = samples,
                                                        Ady = Ady,
                                                        init_fcn = init_variables,
                                                        obs_mode = 'random',
                                                        innov_front_mode = 'explorer')
    key, inn_front, obs_front, in_sub_pop, n, t = run_save(key, init_vars, save_steps, max_steps)
    key = key[-1]
    clear_caches()
    return key, inn_front, obs_front, in_sub_pop, n, t

def front_test(memfraction=.2, device=0):
    """Check mean-field analytic calculation of innovation front speed against automaton."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{memfraction}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'

    inn_lambda_auto = {}
    inn_lambda_anal = {}
    inn_vel_auto = {}
    inn_vel_anal = {}

    obs_lambda_auto = {}
    obs_lambda_anal = {}
    obs_vel_auto = {}
    obs_vel_anal = {}
    key = random.PRNGKey(6)

    # Structural parameters
    samples = 25
    r = .4
    I = 2.
    r0 = 50
    rd = .5
    vo = .2
    el = 20, 700  # size of binary trees: initial chain number of nodes, number of generations of K chains  
    K = 50        # number of branches

    for gamma in np.linspace(0, 1, 15):  # max gamma is close to collapse
        key, inn_front, obs_front, in_sub_pop, n, t = _front_test(key, r, rd, I, r0, vo, el, K, gamma, samples)

        # for solving for the correction to innovation front
        front_loc_max = inn_front_loc(inn_front, samples, el[1], K)
        n_i = np.array([n[i][inn_front[i]].mean() for i in range(t.size)])
        vi_tilde = n_i[-10:].mean() * r * I * I_tilde_coefficient(gamma, K)
        p1 = np.polyfit(t[-10:], front_loc_max.mean(1)[-10:], 1)

        inn_lambda_anal[gamma] = solve_inn_lambda(gamma)
        inn_lambda_auto[gamma] = inn_front_loc(inn_front, samples, el[1], K,
                                               return_max=True, pinned=True).mean(1)[-10:].mean()
        inn_vel_auto[gamma] = p1[0]
        inn_vel_anal[gamma] = vi_tilde
        
        # obsolescence front
        lam = solve_obs_lambda(gamma, 0.)
        vo_tilde = vo * vo_tilde_coefficient(gamma, K)

        p1 = np.polyfit(t, obs_front_loc(obs_front, samples, el[1], K).mean(1), 1)
        obs_lambda_auto[gamma] = obs_front_loc(obs_front, samples, el[1], K, pinned=True).mean(1)[-10:].mean()
        obs_lambda_anal[gamma] = lam
        obs_vel_auto[gamma] = p1[0]
        obs_vel_anal[gamma] = vo_tilde

        save_pickle(['r', 'I', 'K', 'n_i', 'inn_vel_anal', 'inn_vel_auto', 'obs_vel_anal', 'obs_vel_auto',
                     'inn_lambda_anal', 'inn_lambda_auto', 'obs_lambda_anal', 'obs_lambda_auto'],
                    'cache/front_vel_test.p', True)
        clear_caches()

if __name__=='__main__':
    figure1()
