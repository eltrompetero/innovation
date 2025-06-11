# For generating phase diagram along structural parameters gamma and K.
# Author: Eddie Lee, edlee@csh.ac.at
import os, sys
from jax import clear_caches
import jax.experimental.sparse as jsparse
from workspace.utils import save_pickle
from innov import *


K_range = np.unique(np.around(np.logspace(0, 2, 50))).astype(np.int32)
n0 = 10
el = 10, 300


def runaway_settings():
    # fixed simulation parameters
    r0 = 10
    I = 2.
    r = .52
    rd = .5
    vo = .4
    samples = 50  # number of independent replicas
    total_t = 20
    gamma_range = np.logspace(-3, 0, 50).astype(np.float32)
    return r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range

def runaway_micro_settings():
    # fixed simulation parameters
    r0 = 10
    I = 2.
    r = .52
    rd = .5
    vo = .4
    samples = 50  # number of independent replicas
    total_t = 20
    gamma_range = np.linspace(.01, 1, 6).astype(np.float32)
    return r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range

def stable_settings():
    # fixed simulation parameters
    r0 = 10
    I = 2.
    r = .4
    rd = .5
    vo = .4
    samples = 50
    total_t = 20
    gamma_range = np.logspace(-3, 0, 50).astype(np.float32)
    return r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range

def stable_half_settings():
    """For testing alternative simulation parameters."""
    r0 = 10
    I = 2.
    r = .4
    rd = .5
    vo = .2
    samples = 50
    total_t = 20
    gamma_range = np.logspace(-3, 0, 50).astype(np.float32)
    return r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range

def _create_init_variables(r, r0, I, rd, vo, gamma, K, samples):
    """Create a function to initialize variables for running the automaton based
    on mft solutions.

    Parameters
    ----------
    """
    model = CompartmentModel(r0/r, I, rd/r, vo/r, gamma, K)
    Nss, Lss, n0ss, nlss = model.stable_sol()
    if Nss.imag or Nss.real<0: Nss = 1e10
    if Lss.imag or Lss.real<0: Lss = 1e10

    Nss = Nss.real
    Lss = Lss.real
    n0ss = n0ss.real
    nlss = nlss.real
    # conditions under which we have collapse
    assert Nss > n0ss+nlss
    assert Lss > 2
    Lss = int(np.ceil(Lss))

    # for limiting sim requirements, we do not start with more than 20% of the total length of system
    # round up Lss
    Lss = min(Lss, el[1]//5)
    Nss = min(Nss, 1e4)
    n0ss = min(Nss, 1e4)
    nlss = min(Nss, 1e4)

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
        inn = inn.at[:,Lss*K:(Lss+1)*K].set(True)
        # obsolescence front is a uniform line of sites at generation 0
        obs_front = obs_front.at[:,:K].set(True)
        sub = sub.at[:,K:K+K*Lss].set(True)
        #initial density everywhere
        n = n.at[sub].set((N-n0ss-nlss)/(Lss-2))
        # initial density at fronts
        n = n.at[inn].set(n0ss)
        n = n.at[K:2*K].set(nlss)
        return inn, obs_front, sub, n, t
    return init_variables

def one_point(key, K, gamma, init_args, samples, total_t, iprint=True):
    # define graph structure
    tree = KTree(el[1], K, gamma)
    
    # transform Ady into a sparse matrix
    Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
    Ady.data = Ady.data.astype(jnp.int8)
    
    init_variables = _create_init_variables(r, r0, I, rd, vo, gamma, K, samples)
        
    init_vars, one_loop, run_save, run, run_save_t = setup_auto_sim(N = Ady.shape[0],
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
    key, inn_front, obs_sub, in_sub_pop, n, t = run(key, init_vars, total_t, iprint=iprint)

    # copy array to CPU memory
    n_ = np.zeros(n.shape, dtype=np.float32)
    n_[:] = n[:]
    in_sub_pop_ = np.zeros(in_sub_pop.shape, dtype=np.bool_)
    in_sub_pop_[:] = in_sub_pop

    return key, n_, in_sub_pop_

def check_pickle_name(fname, path='cache'):
    fname = path + '/' + fname
    base = fname.split('.')[0]
    counter = 1
    while os.path.exists(fname):
        fname = f'{base}_{counter}.p'
        counter += 1
    return fname

if __name__=='__main__':
    # read in parameters
    settings = sys.argv[1]
    if settings == 'runaway':
        r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range = runaway_settings()
        fname = 'structure_survival_grid_runaway.p'
    elif settings == 'runaway_micro':
        r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range = runaway_micro_settings()
        fname = 'structure_survival_grid_runaway.p'
    elif settings == 'stable':
        r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range = stable_settings()
        fname = 'structure_survival_grid_stable.p'
    elif settings == 'stable_half':
        r0, r, I, el, n0, rd, vo, samples, total_t, K_range, gamma_range = stable_half_settings()
        fname = 'structure_survival_grid_stable_half.p'
    else:
        raise ValueError("Invalid settings. Choose 'runaway', 'runaway_micro', 'stable_half', or 'stable'.")

    try: 
        device_id = sys.argv[2]
        assert device_id in ['0', '1']
    except IndexError:
        device_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    try:
        memfraction = sys.argv[3]
        assert 0 < float(memfraction) <= 1
    except IndexError:
        memfraction = '.5'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = memfraction

    if len(sys.argv)==6:
        # options for overwriting sample number and total runtime
        samples = int(sys.argv[4])
        total_t = float(sys.argv[5])

    fname = check_pickle_name(fname)
    key = random.PRNGKey(42**2)
    K_grid, gamma_grid = np.meshgrid(K_range, gamma_range)

    all_n = []
    all_in_sub_pop = []
    for K, gamma in zip(K_grid.ravel(), gamma_grid.ravel()):
        try:
            key, n, in_sub_pop = one_point(key, K, gamma,
                                           (r, r0, I, rd, vo, gamma, K),
                                           samples,
                                           total_t)
            all_n.append(n)
            all_in_sub_pop.append(in_sub_pop)
        except AssertionError:
            all_n.append(np.zeros(0, dtype=np.float32))
            all_in_sub_pop.append(np.zeros(0, dtype=np.bool_))
        clear_caches()  # clear compiled functions
        save_pickle(['r0', 'r', 'I', 'el', 'n0', 'rd', 'vo', 'samples', 'total_t',
                     'all_n', 'all_in_sub_pop', 'K_range', 'gamma_range'],
                    fname, True)
        print(f"Done with {K=}, {gamma=:.2f}.")

