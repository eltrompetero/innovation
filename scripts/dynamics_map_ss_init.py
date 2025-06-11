# For generating phase diagram along dynamical parameters rd and vo.
# Author: Eddie Lee, edlee@csh.ac.at
import os, sys
import jax.experimental.sparse as jsparse
from jax import clear_caches
from workspace.utils import save_pickle
from innov import *


def grid():
    # fixed simulation parameters
    r0 = 50
    I = 2.
    r = .4
    K = 50
    gamma = .5

    el = 300 

    rd_range = np.linspace(.01, 1.2, 10)
    vo_range = np.linspace(.01, 1.2, 30)
    return r0, I, r, K, gamma, el, rd_range, vo_range

def micro():
    # fixed simulation parameters
    r0 = 50
    I = 2.
    r = .4
    K = 50
    gamma = .5

    el = 300 

    rd_range = np.linspace(.01, 1.2, 5)
    vo_range = np.linspace(.01, 1.2, 20)
    return r0, I, r, K, gamma, el, rd_range, vo_range

def _create_init_variables(r, r0, I, rd, vo, gamma, K, samples):
    """Create a function to initialize variables for running the automaton based
    on mft solutions.

    Parameters
    ----------
    r : float
        Replication rate.
    r0 : float
        Growth rate.
    I : float
        Innovativeness.
    rd : float
        Death rate.
    vo : float
        Obsolescence rate.
    gamma : float
        Connectivity.
    K : int
        Branching number.
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
    Lss = min(Lss, el//5)
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

def one_point(key, rd, vo, samples, total_t, extra_params=(), iprint=True):
    r0, I, r, K, gamma, el = extra_params

    # define graph structure
    tree = KTree(el, K, gamma)
    
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
    n_ = np.zeros(n.shape, dtype=np.float32)
    L_ = np.zeros(n.shape[0], dtype=np.float32)
    in_sub_pop_ = np.zeros(n.shape, dtype=np.bool_)

    # copy array to CPU memory
    n_[:] = n[:]
    L_[:] = in_sub_pop.sum(1)[:]
    in_sub_pop_[:] = in_sub_pop[:]
    return key, n_, L_, in_sub_pop_


if __name__=='__main__':
    device_id = sys.argv[1]
    assert device_id in ['0', '1']
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    memfraction = sys.argv[2]
    assert 0 < float(memfraction) <= 1
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = memfraction

    settings = sys.argv[3]
    if settings=='grid':
        r0, I, r, K, gamma, el, rd_range, vo_range = grid()
    elif settings=='micro':
        r0, I, r, K, gamma, el, rd_range, vo_range = micro()
    else:
        raise NotImplementedError

    if len(sys.argv)==6:
        # options for overwriting sample number and total runtime
        samples = int(sys.argv[4])
        total_t = float(sys.argv[5])
    else:
        samples = 50
        total_t = 20

    fname = 'cache/dynamics_survival_grid.p'
    counter = 1
    while os.path.exists(fname):
        fname = f'cache/dynamics_survival_grid_{counter}.p'
        counter += 1

    key = random.PRNGKey(2)
    vo_grid, rd_grid = np.meshgrid(vo_range, rd_range)

    all_n = []
    all_L = []  # size of each replica
    all_in_sub_pop = []
    for rd, vo in zip(rd_grid.ravel(), vo_grid.ravel()):
        try:
            key, n, L, in_sub_pop = one_point(key, rd, vo, samples, total_t,
                                              extra_params=(r0, I, r, K, gamma, el),
                                              iprint=True)
            all_n.append(n)
            all_L.append(L)
            all_in_sub_pop.append(in_sub_pop)
        except AssertionError:
            all_n.append(np.zeros(0, dtype=np.float32))
            all_L.append(np.zeros(0, dtype=np.float32))
            all_in_sub_pop.append(np.zeros(0, dtype=np.bool_))

        clear_caches()  # clear compiled functions
        save_pickle(['r0', 'r', 'I', 'el', 'K', 'gamma', 'samples', 'total_t',
                     'all_n', 'all_L', 'all_in_sub_pop', 'rd_range', 'vo_range'],
                    fname, True)
        print(f"Done with {rd=:.2f}, {vo=:.2f}.", flush=True)
