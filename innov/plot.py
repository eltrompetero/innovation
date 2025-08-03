# Module for plotting.
# Author: Eddie Lee, edlee@csh.ac.at
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, griddata
from scipy.integrate import odeint

from .simple_calculations import *
from .utils import *


# ================= #
# Helper functions. #
# ================= #
def density_snapshot(n, el, K, t, mean=False, replica_ix=0, **kwargs):
    """Plot density snapshots from automaton simulation. Show the first two
    branches separately and either one replica over mean over replicas.

    Parameters
    ----------
    n : np.ndarray
        Density.
    el : tuple
        (length of initial branch, length of later branches
    K : int
    t : list-like
        List of time-indices to show.
    mean : bool or str, False
        If False show only the first replica. Else 'replica' or 'branch+replica'
    **kwargs : dict
    """
    fig, ax = plt.subplots(figsize=(6,2))

    if mean=='replica': 
        for tix in t:
            ax.plot(n[tix,:,::K].mean(0))
            if K>1:  # show a second branch
                ax.plot(n[tix,:,1::K].mean(0))
    elif mean=='branch+replica':
        for tix in t:
            # mean over replicas then mean over branches
            ax.plot(n[tix].mean(0).reshape(el[1], K).mean(1))
    elif mean is False:
        for tix in t:
            ax.plot(n[tix,replica_ix,::K])
            if K>1:
                ax.plot(n[tix,replica_ix,1::K])
    else: raise NotImplementedError

    ax.set(**kwargs)
    return fig

def dynamics_low_density_boundary(vo_plot_range, r0, I, K, gamma):
    """Return boundary of low density region where N<L, rd as a function of vo.
    
    This has been fixed to work specifically for the values that are plotted in paper. 
    For other values, some fine-tuning in solving the boundary condition may be 
    necessary.

    Returns
    -------
    ndarray
        Solved rd corresponding to input range.
    """
    vo_range = np.linspace(.05, .6, 100)
    rd = np.zeros_like(vo_range)
    err = np.zeros_like(vo_range)
    model = CompartmentModel(r0, I=I, gamma=gamma, K=K)
    
    for i, vo in enumerate(vo_range):
        def cost(logrd):
            rd = np.exp(logrd)
            if rd>5 or rd<1: return 1e10
            return np.abs(model.L(vo=vo, rd=rd, quadratic_form=1)[0] - model.N(vo=vo, rd=rd, quadratic_form=1)[0])**2
        sol = minimize(cost, 1., method='powell', tol=1e-10)
        rd[i] = np.exp(sol['x'][0])
        err[i] = sol['fun']
    ix = (err<1e-5) & (~np.isnan(rd))
    x, y = vo_range[ix], rd[ix]
    spline = CubicSpline(x, y)
    y = spline(vo_plot_range)
    return y

@cache
def structure_low_density(vo):
    """For plots in paper, vo should be either .5 or 1."""
    assert vo==.5 or vo==1

    gamma_range = np.logspace(-3.5, 0, 60)
    r = .4

    r0 = 10/r
    I = 2
    rd = .5/r

    K = np.zeros_like(gamma_range)
    err = np.zeros_like(gamma_range)

    model = CompartmentModel(r0, I, rd, vo)
    for i, g in enumerate(gamma_range):
        def cost(logK):
            K = np.exp(logK[0]) + 1
            if g>.1 and K>25: return 1e20
            elif .01<=g<=.1 and K>1e2: return 1e20
            elif 1e-3<g<1e-2 and K>1e3: return 1e20
            elif g<=1e-3 and K>1e6: return 1e20
            return np.abs(model.L(gamma=g, K=K, quadratic_form=1)[0].real -
                          model.N(gamma=g, K=K, quadratic_form=1)[0].real)**2
        sol = minimize(cost, 0., tol=1e-10)
        K[i] = np.exp(sol['x'][0]) + 1
        err[i] = sol['fun']

    ix = (err<1e-4) & (~np.isnan(K))
    K, gamma_range = K[ix], gamma_range[ix]

    return K, gamma_range

@cache
def critical_K_line():
    """Critical K and gamma relation for phase diagram in Figure 2.

    For each gamma value in the range, find the critical K value. This is done by solving
    for where the real part L goes to 0.
    
    This returns the necessary values to plot the lines for the two values of vo shown."""
    r = .52
    r0 = 10/r
    I = 2
    rd = .5/r
    vo = 1.
    
    gamma_range = np.linspace(0, 1, 100)
    K_critical = np.zeros_like(gamma_range)

    model = CompartmentModel(r0=r0, I=I, rd=rd, vo=vo)
    K_critical = np.zeros_like(gamma_range)
    for i, g in enumerate(gamma_range):
        if g<.05:
            K_critical[i] = model.K_runaway(gamma=g, f_threshold=1e-5, K_max=1e3, K0=6.)
        else:
            K_critical[i] = model.K_runaway(gamma=g, f_threshold=1e-5, K_max=1e3)
    dcounter = 0
    while dcounter<25:
        counter = 0
        while counter<100 or (~np.isnan(K_critical)).all():
            for i in range(K_critical.size-1):
                if np.isnan(K_critical[i]) and not np.isnan(K_critical[i-1]):
                    K_critical[i] = model.K_runaway(gamma=gamma_range[i], f_threshold=1e-5, K_max=1e3,
                                                    K0=np.log(K_critical[i-1]) - dcounter*.1)
                if np.isnan(K_critical[i]) and not np.isnan(K_critical[i+1]):
                    K_critical[i] = model.K_runaway(gamma=gamma_range[i], f_threshold=1e-5, K_max=1e3,
                                                    K0=np.log(K_critical[i+1]) + dcounter*.1)
            counter += 1
        dcounter += 1
    return K_critical, gamma_range

def find_bistable_l1(r0, I, rd, vo, gamma, K, tmax,
                     initial_guess=100, tol=1e-6, dtol=1e-13):
    def y_end(l1):
        y0 = np.array([l1, l1, 1, 1])
        t = np.linspace(0, tmax, int(1e4)*2)
        y = odeint(pde_pseudogap, y0, t, args=(r0, I, rd, vo, gamma, K))
        return y[-1]

    l1 = initial_guess
    assert y_end(l1)[0]<2
    
    d = 1
    prev_val = np.inf
    val = y_end(l1)[0]
    while (val<2 or val>1e10 or abs(prev_val-val)>tol) and d>dtol:
        prev_val = val
        while val<2:  # keep incrementing til we diverge
            l1 += d
            val = y_end(l1)[0]
        # undo divergence step and shrink increment step size
        l1 -= d
        val = y_end(l1)[0]
        d /= 10

    return l1

def load_survival_grid(fname):
    with open(f'cache/{fname}', 'rb') as f:
        data = pickle.load(f)
    K_range = data['K_range']
    gamma_range = data['gamma_range']
    all_n = data['all_n']
    samples = data['samples']
    el = data['el']
    
    K_grid, gamma_grid = np.meshgrid(K_range, gamma_range)
    f_rep = np.zeros(K_grid.size)
    f_branch = np.zeros(K_grid.size)
    for i, (K, n) in enumerate(zip(K_grid.ravel(), all_n)):
        # fraction of surviving replicas
        if len(n)>0:
            f_rep[i] = (n.sum(1)>0).sum()/samples
            
            # avg fraction of empty branches in each replica
            f_branch[i] = np.mean([(n_.reshape(el[1], K).sum(0)>0).sum() for n_ in n])/K
    
    f_rep = f_rep.reshape(len(gamma_range), len(K_range))
    f_branch = f_branch.reshape(len(gamma_range), len(K_range))
    return K_grid, gamma_grid, f_rep, f_branch



# =================== #
# Plotting functions. #
# =================== #
def examples(axs, Lplot, t_ix):
    with open('cache/fig1_base_params.p', 'rb') as f:
        data = pickle.load(f)
    r0 = data['r0']
    I = data['I']
    rd = data['rd']
    vo = data['vo']
    K = data['K']
    r = data['r']

    for colix, gamma in enumerate([0, .5, 1]):
        fname = f'cache/gamma={gamma}_{K=}_{r0=}_{vo=}_{rd=}_{r=}_{I=}_automaton.p'
        gamma = float(gamma)
        with open(fname, 'rb') as f:
            out = pickle.load(f)

        # densities without pinning
        el = out['el']
        samples = out['samples']
        n = out['n'][t_ix]
        inn_front = out['inn_front'][t_ix]

        n_br = n.reshape(samples, el[1], K)  # density per branch
        n_br = np.swapaxes(n_br, 1, 2)

        # branch replicas pinned at innovation front (as identified per branch)
        inn_front = inn_front.reshape(samples, el[1], K)
        inn_front = np.swapaxes(inn_front, 1, 2)
        n_pinned = [[n_br[i,j,:np.where(inn_front[i, j])[0][0]+1][::-1] if inn_front[i,j].any()
                        else np.zeros(Lplot)+np.nan for j in range(K)]
                    for i in range(samples)]
        # standardize length of pseudogap snapshots to Lplot
        n_pinned = [[n_pinned[i][j][:Lplot] if Lplot<n_pinned[i][j].size
                        else np.concatenate((n_pinned[i][j],np.zeros(Lplot-n_pinned[i][j].size))) for j in range(K)]
                    for i in range(samples)]
        n_pinned = np.array(n_pinned)

        # plot ten random replicas
        randix = np.random.choice(range(samples), size=10, replace=False)

        for i in randix:
            # plot each branch in the replica separately
            for j in range(1):
                axs[0,colix].plot(n_br[i][j], '-', c=f'C{colix}', alpha=.12, lw=1)
        # plot average
        axs[0,colix].plot(np.nanmean(np.nanmean(n_br, 1), 0), '-', c=f'C{colix}', lw=2)

        for i in randix:
            # plot each branch in the replica separately
            for j in range(1):
                axs[1,colix].plot(n_pinned[i,j], '-', c=f'C{colix}', alpha=.12, lw=1)
        axs[1,colix].plot(np.nanmean(np.nanmean(n_pinned, 1), 0), '-', c=f'C{colix}', lw=2)

def front_test(ax, fname, c='C0'):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    inn_vel_auto = data['inn_vel_auto']
    inn_vel_anal = data['inn_vel_anal']
    obs_vel_auto = data['obs_vel_auto']
    obs_vel_anal = data['obs_vel_anal']
    inn_lambda_auto = data['inn_lambda_auto']
    obs_lambda_auto = data['obs_lambda_auto']
    lambda_anal = data['lambda_anal']

    h = []
    ax[0].plot(*zip(*inn_lambda_auto.items()), 'o', c=c)
    ax[0].plot(*zip(*obs_lambda_auto.items()), '^', c=c)
    ax[0].plot(*zip(*lambda_anal.items()), 's', mfc='none', mec=c)
    
    ax[1].plot(*zip(*inn_vel_auto.items()), 'o', c=c)
    ax[1].plot(*zip(*inn_vel_anal.items()), 's', mfc='none', mec=c)
        
    ax[2].plot(*zip(*obs_vel_auto.items()), 'o', c=c)
    ax[2].plot(*zip(*obs_vel_anal.items()), 's', mfc='none', mec=c)

def structure_phase_space(r0, I, rd, ax,
                          vo_range=[.5, 1],
                          runaway=False,
                          low_density=False):
    gamma_range_collapse = np.linspace(0, 1, 50)
    
    K_collapse = []
    for i, vo in enumerate(vo_range):
        model = CompartmentModel(r0, I, rd, vo)
        K_collapse.append(np.zeros_like(gamma_range_collapse))
        sol = []
        for j, gamma in enumerate(gamma_range_collapse):
            K = model.K_collapse(gamma=gamma)
            K_collapse[-1][j] = K
    
    if runaway:
        # runaway line
        K_runaway, gamma_range_runaway = critical_K_line()
        ax.plot(K_runaway, gamma_range_runaway, '-.', c='C3', alpha=.5)
        ax.plot(K_runaway, gamma_range_runaway, '-', c='C3')
        ax.fill_between(K_runaway, np.zeros(K_runaway.size), gamma_range_runaway, fc='#EAB2B0', lw=0)
        ax.fill_between([1, 10], [.05, .05], fc='#EAB2B0', lw=0)
        ax.fill_betweenx(gamma_range_runaway, K_runaway, np.zeros_like(K_runaway), fc='#EAB2B0', lw=0)
        # bifurcation
        ax.fill_between(K_runaway, np.ones(K_runaway.size), gamma_range_runaway,
                        hatch='x', fc='none', lw=1, alpha=.3)

    if low_density:
        # low density regime
        K, gamma = structure_low_density(.5)
        #ax.plot(K, gamma, '-.', color='k')
        K, gamma = structure_low_density(1)
        ax.plot(K, gamma, '-', color='k')
        ax.fill_between(K, np.zeros_like(gamma), gamma, hatch='//', fc='none')

    # collapse
    if len(K_collapse)>1:
        ax.plot(K_collapse[0], gamma_range_collapse, '-.', c='C0', alpha=.5)
        ax.plot(K_collapse[1], gamma_range_collapse, '-', c='C0')
        ax.fill_betweenx(gamma_range_collapse, K_collapse[1], np.zeros(K_collapse[0].size)+100,
                         fc='#C0D5E6', lw=0)
    else:
        ax.plot(K_collapse[0], gamma_range_collapse, '-', c='C0')
        ax.fill_betweenx(gamma_range_collapse, K_collapse[0], np.zeros(K_collapse[0].size)+100,
                         fc='#C0D5E6', lw=0)

    
    ax.set_ylim(0, 1)
    ax.set(yticks=(0,1), xticks=(1, 50, 100))
    ax.set(ylabel=r'connectivity                      ', xlim=(1, 100))

    ax.plot([], 'k-.', label=r'$v_o=1/2$')
    ax.plot([], 'k-', label=r'$v_o=1$')

def dynamical_phase_space(ax):
    r = .4
    r0 = 50/r
    I = 2.
    K = 50
    gamma = np.array([.25, .5])

    # bifurcation
    def define_vo_collapse_interp(vo_range=np.linspace(1.4, 1.8, 100)):
        rd_collapse = np.array([collapse_rd(vo, r0, I, gamma[1], K) for vo in vo_range])
        sortix = np.argsort(rd_collapse)
        rd_collapse = rd_collapse[sortix]
        vo_range = vo_range[sortix]
        return lambda rd, rd_collapse=rd_collapse, vo_range=vo_range: np.interp(rd, rd_collapse, vo_range)
    rd_range = np.linspace(0, 1, 40)
    ax.fill_betweenx(rd_range, np.zeros(rd_range.size), define_vo_collapse_interp()(rd_range),
                     hatch='x', fc='none', lw=1, alpha=.3)
    
    # runaway
    vo_range = np.linspace(0, 1.8, 100)
    rd = [runaway_rd(r0, I, gamma[0], K)(vo_range),
          runaway_rd(r0, I, gamma[1], K)(vo_range)]

    ax.plot(vo_range, rd[0], '-.', color = 'red', alpha=.5)
    ax.plot(vo_range, rd[1], '-', color = 'red', alpha=.5)
    ax.fill_betweenx(rd[1], vo_range, np.zeros_like(vo_range),
                     fc = '#EABFBF', lw=0)
    
    # collapsed
    vo_range = np.linspace(1, 2, 50)
    y_collapse = np.array([collapse_rd(vo, r0, I, gamma[1], K) for vo in vo_range])
    ax.plot(vo_range, [collapse_rd(vo, r0, I, gamma[0], K) for vo in vo_range], '-.', color='C0', alpha=.5)
    ax.plot(vo_range, y_collapse, '-', color='C0')
    ax.fill_betweenx([collapse_rd(vo, r0, I, gamma[1], K) for vo in vo_range], vo_range, np.ones_like(vo_range)*3,
                     color='C0', alpha=.3)

    # low density region
    vo_range = np.linspace(.1, .7, 100)[:-1]
    rd = dynamics_low_density_boundary(vo_range, r0, I, K, gamma[0])
    ix = (rd>1) | (vo_range>.8)
    ax.plot(vo_range[ix], rd[ix], 'k-.')
    # gamma=1/2
    vo_range = np.linspace(.07, .5, 200)[:-1]
    rd = dynamics_low_density_boundary(vo_range, r0, I, K, gamma[1])
    ix = (rd>1) | (vo_range>.8)
    ax.plot(vo_range[ix], rd[ix], 'k-')
    ax.fill_betweenx(rd[ix], np.zeros(ix.sum()), vo_range[ix], hatch='//', fc='none')
        
    ax.plot([], '-.', color ='black', label = r'$\gamma = \frac{1}{4}$')
    ax.plot([], '-', color ='black', label = r'$\gamma = \frac{1}{2}$')
    ax.set(xlim=(0, 3), ylim=(0, 3))
    ax.legend(loc=1, fontsize ='x-small', handlelength=1.14, framealpha=1)
    ax.set(title=f'', xlabel=r'exnovation velocity ${v}_o$', ylabel=r'death rate ${r}_d$')
    
def dynamics_density(fig, ax, colorbar=False,
                     r=.4, n_points=101):
    r0 = 50/r
    I = 2
    K = 50
    gamma = .5
    
    model = CompartmentModel(r0, I, gamma=gamma, K=K)
    
    vo_range = np.linspace(0, 3, n_points)
    rd_range = np.linspace(0, 3, n_points)
    vo_grid, rd_grid = np.meshgrid(vo_range, rd_range)
    
    L = np.zeros(vo_grid.size, dtype=np.complex64)
    N = np.zeros(vo_grid.size, dtype=np.complex64)
    for i, (vo_, rd_) in enumerate(zip(vo_grid.flatten(), rd_grid.flatten())):
        L[i] = model.L(rd=rd_, vo=vo_, quadratic_form=1)[0]
        N[i] = model.N(rd=rd_, vo=vo_, quadratic_form=1)[0]
    L = L.reshape(vo_grid.shape)
    N = N.reshape(vo_grid.shape)

    im = ax.imshow(N.real-L.real, origin='lower',
                   extent=[vo_range.min(), vo_range.max(), rd_range.min(), rd_range.max()],
                   aspect='auto',
                   cmap='seismic',
                   vmin=-100, vmax=100)

    # white-out runaway zone
    vo_range = np.linspace(0, 2, n_points)
    rd = runaway_rd(r0, I, gamma, K)(vo_range)
    ax.fill_betweenx(rd, vo_range, np.zeros_like(vo_range),
                     fc = 'gray', lw=0)    

    ax.set(xlim=(0,3), ylim=(0,3), xlabel=r'exnovation vel. $v_o$', ylabel=r'death rate $r_d$', xticks=[0,1,2,3])
    ax.set_xticklabels([0,1,2,3], fontsize='small')
    ax.set_yticklabels([0,1,2,3], fontsize='small')

    if colorbar:
        cb = fig.colorbar(im, label=r'$N-L$', ticks=[-100,0,100])
        
        cb.set_ticklabels([r'$\leq-10^2$',r'$0$',r'$\geq10^2$'], fontsize='small')
        cb.set_label(r'$N-L$', labelpad=-30)

def structure_density(fig, ax, cbax=None, r=.4, n_points=100, cbar_label=r'$N-L$'):
    r0 = 10/r
    I = 2
    rd = .5/r
    vo = 1
    
    model = CompartmentModel(r0=r0, I=I, rd=rd, vo=vo)
    
    K_range = np.linspace(1, 100, n_points)
    gamma_range = np.linspace(0, 1, n_points)
    K_grid, gamma_grid = np.meshgrid(K_range, gamma_range)
    
    L = np.zeros(K_grid.size, dtype=np.complex64)
    N = np.zeros(K_grid.size, dtype=np.complex64)
    for i, (K, gamma) in enumerate(zip(K_grid.flatten(), gamma_grid.flatten())):
        L[i] = model.L(K=K, gamma=gamma, quadratic_form=1)[0]
        N[i] = model.N(K=K, gamma=gamma, quadratic_form=1)[0]
    L = L.reshape(K_grid.shape)
    N = N.reshape(K_grid.shape)

    im = ax.imshow(N.real-L.real, origin='lower',
                   extent=[K_range.min(), K_range.max(), gamma_range.min(), gamma_range.max()],
                   aspect='auto',
                   cmap='seismic',
                   vmin=-100, vmax=100)
    
    if not cbax is None:
        cb = fig.colorbar(im, cax=cbax, ticks=[-100, 0, 100])
        cb.set_ticklabels([r'$\leq-10^2$',r'$0$',r'$\geq10^2$'], fontsize='small')
        cb.set_label(cbar_label, labelpad=-30)
    ax.set(xticks=[1, 50,100], yticks=[0,.5,1])
    ax.set_xticklabels([1,50,100], fontsize='small')
    ax.set_yticklabels([0,.5,1], fontsize='small')

def structure_phase_space_stable(ax):
    r0 = 10/.4
    I = 2.
    rd = .5/.4
    vo = 1.
    
    gamma_range_collapse = np.linspace(0, 1, 50)
    
    K_collapse = np.zeros_like(gamma_range_collapse)
    model = CompartmentModel(r0, I, rd, vo)
    sol = []
    for j, gamma in enumerate(gamma_range_collapse):
        K = model.K_collapse(gamma=gamma)
        K_collapse[j] = K
    
    # collapse line
    ax.plot(K_collapse, gamma_range_collapse, '-', c='C0')
    
def structure_phase_space_runaway(ax):
    r0 = 10/.52
    I = 2.
    rd = .5/.52
    vo = .4/.52
    
    gamma_range_collapse = np.linspace(0, 1, 50)
    
    K_collapse = np.zeros_like(gamma_range_collapse)
    model = CompartmentModel(r0, I, rd, vo)
    for j, gamma in enumerate(gamma_range_collapse):
        K = model.K_collapse(gamma=gamma)
        K_collapse[j] = K
    
    # collapse line
    ax.plot(K_collapse, gamma_range_collapse, '-', c='C0')

    # runaway line
    K_runaway, gamma_range_runaway = critical_K_line()
    ax.plot(K_runaway, gamma_range_runaway, '-', c='C3')

def dynamics_survival_phase_diagram(fname, ax, n_points=100):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    r0 = data['r0']
    r = data['r']
    I = data['I']
    el = data['el']
    K = data['K']
    gamma = data['gamma']
    samples = data['samples']
    total_t = data['total_t']
    all_n = data['all_n']
    all_L = data['all_L']
    rd_range = data['rd_range']
    vo_range = data['vo_range']

    r0 /= r
    
    vo_grid, rd_grid = np.meshgrid(vo_range, rd_range)
    f_rep = np.zeros(rd_grid.size)
    f_branch = np.zeros(rd_grid.size)
    for i, n in enumerate(all_n):
        if len(n):
            # fraction of surviving replicas
            f_rep[i] = (n.sum(1)>0).sum()/samples
            if f_rep[i]>1:
                print(samples, n.shape)

            # avg fraction of empty branches in each replica
            f_branch[i] = np.mean([(n_.reshape(el, K).sum(0)>0).sum() for n_ in n])/K
    
    f_rep = f_rep.reshape(len(rd_range), len(vo_range))
    f_branch = f_branch.reshape(len(rd_range), len(vo_range))
    
    N = np.zeros(vo_grid.size)  # avg over surviving replicas
    for i, n in enumerate(all_n):
        if len(n):
            N[i] = n.sum() / (np.nextafter(0, 1)+n.any(1).sum()) / K
    N = N.reshape(len(rd_range), len(vo_range))
    
    L = np.zeros(vo_grid.size)  # avg over surviving replicas
    for i, thisL in enumerate(all_L):
        if len(thisL):
            L[i] = thisL.sum() / (np.nextafter(0, 1)+(thisL>0).sum()) / K
    L = L.reshape(len(rd_range), len(vo_range))
    
    logscale = False
        
    # Interpolate results
    voi, rdi = np.meshgrid(np.linspace(0, 3, n_points),
                           np.linspace(0, 3, n_points))

    fi_complete = griddata((vo_grid.ravel()/r, rd_grid.ravel()/r), f_branch.flatten(),
                  (voi.ravel(), rdi.ravel()), method='nearest')
    fi = griddata((vo_grid.ravel()/r, rd_grid.ravel()/r), f_branch.flatten(),
                  (voi.ravel(), rdi.ravel()), method='linear', fill_value=np.nan)
    fi_complete = fi_complete.reshape(n_points, n_points)
    fi = fi.reshape(n_points, n_points)
    
    if logscale:
        cax = ax.imshow(np.log10(fi_complete), extent=(voi.min(), voi.max(), rdi.min(), rdi.max()),
                           origin='lower', aspect='auto', cmap='Reds', vmin=-3, vmax=0)
        cax = ax.imshow(np.log10(fi), extent=(voi.min(), voi.max(), rdi.min(), rdi.max()),
                           origin='lower', aspect='auto', cmap='Reds', vmin=-3, vmax=0)
    else:
        cax = ax.imshow(fi_complete, extent=(voi.min(), voi.max(), rdi.min(), rdi.max()),
                           origin='lower', aspect='auto', cmap='Reds', vmin=0, vmax=1)
        cax = ax.imshow(fi, extent=(voi.min(), voi.max(), rdi.min(), rdi.max()),
                           origin='lower', aspect='auto', cmap='Reds', vmin=0, vmax=1)

    # runaway boundary
    vo_range = np.linspace(0, 2, 100)
    rd = runaway_rd(r0, I, gamma, K)(vo_range)

    ax.plot(vo_range, rd, '-', color = 'red', alpha=.5)
    
    # collapsed boundary
    vo_range = np.linspace(1, 2, 50)
    y = np.array([collapse_rd(vo, r0, I, gamma, K) for vo in vo_range])
    ax.plot(vo_range, y, '-', color='C0')

    ax.set(xlabel=r'obs rate $v_o/r$', ylabel=r'death rate $r_d/r$', xlim=(0, 3), ylim=(0,3))

    return cax
