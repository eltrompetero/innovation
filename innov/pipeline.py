# Pipeline for manuscript
# Author: Eddie Lee, edlee@csh.ac.at
import os
import pickle
from itertools import product
from jax import clear_caches
from workspace.utils import save_pickle

from .automaton import *
from .tree_gen import *
from .simple_calculations import *


def figure1(memfraction=0.3, device=0):
    """Run and save the results from automaton simulations for Figure 1."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memfraction)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    if not os.path.isdir('cache'):
        os.makedirs('cache')

    key = random.PRNGKey(10)

    # Tree and simulation parameters
    el = (10, 600)  # (initial chain length, total generations)
    samples = 100
    n0 = 20
    r0 = 10
    I = 2
    r = 0.51
    rd = 0.5
    vo = 0.02
    K = 50
    max_t = 60
    save_dt = 5.0

    # Save base parameters
    with open('cache/fig1_base_params.p', 'wb') as f:
        pickle.dump({'el': el, 'samples': samples, 'n0': n0, 'I': I, 'r': r, 'K': K, 'r0': r0, 'rd': rd, 'vo': vo,
                     'save_dt': save_dt, 'max_t': max_t}, f)

    for gamma in [0, 0.5, 1]:
        fname = f'cache/gamma={gamma}_K={K}_r0={r0}_vo={vo}_rd={rd}_r={r}_I={I}_automaton.p'
        tree = KTree(el[1], K, gamma)
        Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
        Ady.data = Ady.data.astype(jnp.int8)
        init_variables = create_init_variables(el, K, n0)

        # Setup automaton simulations
        init_vars, one_loop, run_save, run, run_save_t = setup_auto_sim(
            N=Ady.shape[0], r=r, rd=rd, I=I, r0=r0, vo=vo, samples=samples,
            Ady=Ady, init_fcn=init_variables, obs_mode='random', innov_front_mode='explorer')
        output = run_save_t(key, init_vars, save_dt, max_t)
        key_save, inn_front, obs_front, in_sub_pop, n, t = output

        with open(fname, 'wb') as f:
            pickle.dump({'el': el, 'K': K, 'save_dt': save_dt, 'max_t': max_t,
                         'samples': samples, 'key': key_save, 'inn_front': inn_front, 'obs_front': obs_front,
                         'in_sub_pop': in_sub_pop, 'n': n, 't': t}, f)
        print(f"Done with {fname}.")


def _figure2(memfraction=0.4, device=0):
    """Run and save automaton simulations for Figure 2 parameter grid."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memfraction)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    grid_vars = [
        {'gamma': g, 'K': 2, 'r0': 30, 'vo': vo, 'rd': rd, 'r': 0.2, 'I': 0.8,
         'max_steps': ms, 'save_steps': ss}
        for g in [0.0, 0.5, 1.0]
        for vo, rd, ms, ss in [
            (0.22, 0.22, 60000, 1000),
            (0.8, 0.8, 10000, 100),
            (0.1, 0.1, 60000, 1000),
            (0.35, 0.1, 60000, 1000)
        ]
    ]
    el = (20, 30000)
    samples = 100
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

        tree = KTree(*el, K, gamma)
        Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
        Ady.data = Ady.data.astype(jnp.int8)
        init_variables = create_init_variables(el, K, 10)

        init_vars, one_loop, run_save = setup_auto_sim(
            N=Ady.shape[0], r=r, rd=rd, I=I, r0=r0, vo=vo, samples=samples,
            Ady=Ady, init_fcn=init_variables, obs_mode='random', innov_front_mode='explorer')
        key_out, inn_front, obs_sub, in_sub_pop, n, t = run_save(key, init_vars, save_steps, max_steps)
        key = key_out[-1]

        fname = f'cache/gamma={gamma}_K={K}_r0={r0}_vo={vo}_rd={rd}_r={r}_I={I}_automaton.p'
        with open(fname, 'wb') as f:
            pickle.dump({'r0': r0, 'r': r, 'rd': rd, 'vo': vo, 'I': I, 'gamma': gamma,
                         'el': el, 'K': K, 'save_steps': save_steps, 'max_steps': max_steps,
                         'samples': samples, 'key': key_out, 'inn_front': inn_front, 'obs_sub': obs_sub,
                         'in_sub_pop': in_sub_pop, 'n': n, 't': t}, f)


def _front_test(key, r, rd, I, r0, vo, el, K, gamma, samples):
    """Helper function for front velocity tests."""
    n0 = 20
    max_t = 80
    save_dt = 2
    tree = KTree(el[1], K, gamma)
    Ady = jsparse.BCOO.from_scipy_sparse(tree.adj)
    Ady.data = Ady.data.astype(jnp.int8)
    init_variables = create_init_variables(el, K, n0)
    init_vars, one_loop, run_save, run, run_save_t = setup_auto_sim(
        N=Ady.shape[0], r=r, rd=rd, I=I, r0=r0*K, vo=vo, samples=samples,
        Ady=Ady, init_fcn=init_variables, obs_mode='random', innov_front_mode='explorer')
    key, inn_front, obs_front, in_sub_pop, n, t = run_save_t(key, init_vars, save_dt, max_t)
    key = key[-1]
    clear_caches()
    return key, inn_front, obs_front, in_sub_pop, n, t


def front_test(memfraction=0.2, device=0, sim_params=None,
               automaton_save_file='cache/front_vel_test_automata.p',
               comparison_save_file='cache/front_vel_test.p'):
    """Automate mean-field analytic calculation of innovation front speed against automaton across gamma."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memfraction)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    inn_lambda_auto, inn_vel_auto = {}, {}
    obs_lambda_auto, obs_vel_auto = {}, {}
    n_i = {}
    lambda_anal, inn_vel_anal, obs_vel_anal = {}, {}, {}
    key = random.PRNGKey(6)

    if sim_params is None:
        samples = 25
        r = 0.4
        I = 2.0
        r0 = 50
        rd = 0.5
        vo = 0.2
        el = (20, 500)
        K = 50
    else:
        samples = sim_params['samples']
        r = sim_params['r']
        I = sim_params['I']
        r0 = sim_params['r0']
        rd = sim_params['rd']
        vo = sim_params['vo']
        el = sim_params['el']
        K = sim_params['K']

    # Try loading existing simulation outputs
    if os.path.isfile(automaton_save_file):
        with open(automaton_save_file, 'rb') as f:
            data = pickle.load(f)
        inn_lambda_auto = data['inn_lambda_auto']
        inn_vel_auto = data['inn_vel_auto']
        obs_lambda_auto = data['obs_lambda_auto']
        obs_vel_auto = data['obs_vel_auto']
        n_i = data['n_i']
    gamma_range = np.linspace(0, 1, 15)

    # Automaton calculations
    try:
        if len(inn_lambda_auto) == 15:
            gamma_ix = 14
        else:
            for gamma_ix, gamma in enumerate(gamma_range[len(inn_lambda_auto):]):
                key, inn_front, obs_front, in_sub_pop, n, t = _front_test(
                    key, r, rd, I, r0, vo, el, K, gamma, samples)

                # Innovation front
                front_loc_max = inn_front_loc(inn_front, samples, el[1], K, return_max=True)
                n_i[gamma] = np.array(leading_front_density(n, inn_front, el, K))
                p1 = np.polyfit(t[-10:], front_loc_max.mean(1)[-10:], 1)
                inn_lambda_auto[gamma] = inn_front_loc(
                    inn_front, samples, el[1], K, return_max=True, pinned=True).mean(1)[-10:].mean()
                inn_vel_auto[gamma] = p1[0]

                # Obsolescence front
                p1 = np.polyfit(t, obs_front_loc(obs_front, samples, el[1], K).mean(1), 1)
                obs_lambda_auto[gamma] = obs_front_loc(
                    obs_front, samples, el[1], K, pinned=True).mean(1)[-10:].mean()
                obs_vel_auto[gamma] = p1[0]

                save_pickle([
                    'r', 'I', 'r0', 'rd', 'vo', 'el', 'K', 'n_i', 'inn_vel_auto',
                    'obs_vel_auto', 'inn_lambda_auto', 'obs_lambda_auto',
                    'inn_front', 'obs_front', 'n', 't', 'key'
                ], automaton_save_file, True)
                clear_caches()
            gamma_ix = np.where(gamma_range == gamma)[0][0]
    except IndexError:
        gamma_ix = np.where(gamma_range == gamma)[0][0] - 1
        clear_caches()

    # Mean-field calculations
    for gamma in gamma_range[:gamma_ix + 1]:
        lambda_anal[gamma] = solve_poisson_lambda(gamma)
        vi_tilde = n_i[gamma][-10:].mean() * r * I * I_tilde_coefficient(gamma, K)
        inn_vel_anal[gamma] = vi_tilde
        vo_tilde = vo * vo_tilde_coefficient(gamma, K)
        obs_vel_anal[gamma] = vo_tilde

        save_pickle([
            'r', 'I', 'K', 'inn_vel_anal', 'inn_vel_auto', 'obs_vel_anal', 'obs_vel_auto',
            'lambda_anal', 'inn_lambda_auto', 'obs_lambda_auto'
        ], comparison_save_file, True)


def figure2(memfraction=0.2, device=0):
    """Run front velocity tests for multiple K values for Figure 2."""
    for i, K in enumerate([25, 50, 100]):
        sim_params = {'samples': 50, 'r': 0.4, 'I': 2.0, 'r0': 50, 'rd': 0.5, 'vo': 0.2, 'el': (20, 500), 'K': K}
        front_test(
            memfraction=memfraction,
            device=device,
            sim_params=sim_params,
            automaton_save_file=f'cache/front_vel_test_automata_{i}.p',
            comparison_save_file=f'cache/front_vel_test_{i}.p'
        )
        clear_caches()


if __name__ == '__main__':
    figure1()
    figure2()