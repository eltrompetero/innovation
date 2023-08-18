# ====================================================================================== #
# Network innovation model numerical solutions probabilistic fronts.
# 
# Author : Ernesto Ortega, ortega@csh.ac.at
# ====================================================================================== #
from numba import njit, jit
from numba.typed import List
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
import warnings
import math
import jax.numpy as jnp
from jax import jit as jaxjit
from jax import config
from jax import random
from jax.lax import fori_loop
from jax import jit

config.update('jax_enable_x64', True)
from scipy.special import factorial
from scipy.special import gammaln, logsumexp
import jax.lax as lax
from functools import partial

from workspace.utils import save_pickle
from .utils import *

from scipy.sparse import bsr_array, eye
from scipy.integrate import solve_ivp
import numpy as np
from scipy.signal import medfilt
# Suggestions:

#@partial(jaxjit, static_argnums=(0,1,2,3))
def compute_P_O_1(N, τo, t, init):
        logP_O = np.zeros(N)
        for i in range(N):
            logP_O[i] = np.log(τo)*i - τo*t + np.log(t)*i - gammaln(i+1)

        return np.exp(logP_O - logsumexp(logP_O))
    
def fn(t, y, N, G_in, rd, Fix_R, Fix_P, Fix_O, Fix_a):
    n = y[0:N]
    #n_k = medfilt(n, 11)
    P_I = y[N:2*N]
    P_O = y[2*N:3*N]
    P_I_cum = P_I[::-1].cumsum()[::-1]
    P_O_cum = P_O.cumsum()
    P_L = P_I_cum * P_O_cum
    Fix_V = G_in * (P_L/np.sum(P_L))
    sites = np.arange(N)
    if (P_I*sites).sum() > (P_O*sites).sum():
        result = np.concatenate(( (-rd*n + (n * P_L) @ Fix_R + Fix_V), ((n * P_I) @ Fix_P), P_O @ Fix_O ))
    else:
        result = np.concatenate(( (-rd*n + (n * P_L) @ Fix_R + Fix_V), ((n * P_I) @ Fix_P), np.zeros(N) )) 
    return result

def infinit_growth(t, y, N, G_in, rd, Fix_R, Fix_P, Fix_O, Fix_a):
    parar=False
    P_I = y[N:2*N]
    P_O = y[2*N:3*N]
    if np.sum(y)>500000:
        parar=True
        print("infinite growth")
    sites = np.arange(N)
    if t > 30 and (P_I*sites).sum() < (P_O*sites).sum():
        print("collapse")
        parar = True
    #if t > 30 and (P_I*sites).sum() < (P_O*sites).sum():
        #print("collapse")
        #parar = True
    return 0 if parar else 1
    
# 1. object-oriented code
class RKPropagator1D():
    def __init__(self, N, sub_ix=None, n0=0, τo = 0.3, rd = 0.3, r = 0.4, I = 0.8, G_in= 4., tmax = 100, Δt = 0.1, λ = 1., method='density'):
        """
        Parameters
        ----------
        N : int
            Size of whole graph.
        sub_ix : list of indices or twople, None
            If list of indices, then fill in those values with n0. If a twople,
            fill entries between the two indices with n0.
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
        
        """ Initializing parametrers 
        """
        self.N = N
        self.τo = τo          
        self.rd = rd
        self.r = r
        self.I= I
        self.G_in= G_in
        self.tmax = tmax
        self.Δt = Δt
        self.n0 = n0 
        self.λ = λ
        self.t = 0
        self.count = 1
        self.switch = 1
        self.method = method
        """ Creating sites and densities
        """
        sites = np.arange(N)
        n = np.zeros(N)
        Fix_V = np.zeros(N)
        P_L = np.zeros(N)

        """ Initializing densities
        """
        n[sub_ix[0]:sub_ix[1]]= n0
        
        self.init = sub_ix[-1]
        
        """Init obsolescence and Innovation fronts
        """
        self.P_I = np.asarray([0. if i!=sub_ix[1]-1 else 1. for i in range(self.N)])
        self.P_O = np.asarray([0. if i!=sub_ix[0] else 1. for i in range(self.N)])       
        self.sites = sites
        self.n = n
        self.Fix_V = Fix_V
        self.P_L = P_L
        """ Getting adjacency matrix
        """
        
        self.Ady = np.eye(self.N, k=1)
        
        """ Computing inverse of offspring vector
        """
        inverse_sons = np.sum(self.Ady, axis=1)
        inverse_sons[inverse_sons==0.] = 1.
        self.inverse_sons = 1./inverse_sons
        
        """ Initialize Fix Matrix
        """
        self.Fix_M = (-rd) * np.eye(self.N)            # Set death    
        self.Fix_R = r * self.Ady
        self.Fix_P = r*I*(self.Ady - np.eye(self.N))
        self.Fix_O = τo*(self.Ady - np.eye(self.N))
        
        a = np.eye(N, k=0)
        for i in range(1, 3):
            a += np.eye(N, k=i)
            a += np.eye(N, k=-i)
        b = a.sum(axis=1)
        b = 1/b
        self.Fix_a = a *b
        
    def automaton_evolve(self):
        
        n = self.n
        
        
        
        
    def propagate(self):
        
        
        """Propagate density function by some amount of time.
        """
        infinit_growth.terminal = True
        nio = np.concatenate((self.n, self.P_I, self.P_O))
        
        sol = solve_ivp(fn, [0, self.tmax], nio, args=(self.N, self.G_in, self.rd, self.Fix_R, self.Fix_P, self.Fix_O, self.Fix_a), events=infinit_growth)
        self.t = sol.t
        self.n = sol.y[0:self.N]
        self.P_I = sol.y[self.N:2*self.N]
        self.P_O = sol.y[2*self.N:3*self.N]
    
    
        
    def propagate_jax(self):
        """Propagate density function by some amount of time."""
        self.n, self.P_I = jax_propagate(self.n, self.Fix_R, self.Fix_P, self.G_in,
                                         self.Δt, self.r, self.τo, self.rd, self.I,
                                         self.N, self.init, self.P_I, self.P_O)
        self.t += self.Δt
        self.P_O = compute_P_O_1(self.N, self.τo, self.t, self.init)
    
    
    
    def propagate_with_P_O_jax(self):
        """Propagate density function by some amount of time."""
        self.n, self.P_I, self.P_O = jax_propagate(self.n, self.Fix_R, self.Fix_P, self.Fix_O, self.G_in,
                                         self.Δt, self.r, self.τo, self.rd, self.I,
                                         self.N, self.init, self.P_I, self.P_O)
        self.t += self.Δt
        
    def run_and_save(self, save_dt=1., tmax=None, **kwargs):
        """Propagate over multiple time steps and save simulation steps.
        
        Parameters
        ----------
        save_dt : float, 1.
        tmax : float
            End simulation time.
        **kwargs
            To pass into self.propagate.
            
        Returns
        -------
        list
            Density at each time step.
        list
            System length at each time step.
        list
            Time.
        """
        
        n = [self.n.toarray()]
        t = [0]
        P_I = [self.P_I.toarray()]
        P_O = [self.P_O.toarray()]
        tmax = tmax if not tmax is None else self.tmax
        
        import time
        start = time.time()
        print(tmax)
        while self.t < tmax:
            if self.method == "frequency":
                self.propagate(**kwargs)
            else:
                self.propagate_with_P_O(**kwargs)

            #H_sub_in_time.append(self.In_H_sub[:])
            #Obs_in_time.append(self.In_Obs_Front[:])
            #Inn_in_time.append(self.In_Inn[:])
            
           # if np.isclose(self.t%save_dt, 0):
            n.append(self.n.toarray())
            P_I.append(self.P_I.toarray())
            P_O.append(self.P_O.toarray())
            t.append(self.t)
            #print(self.t)
            self.t+=self.Δt
            start = time.time()
        
        return n, P_I, P_O, t



    
class UnitSimulatonNetworks():
    def __init__(self, N, Ady, sub_ix= [0, 1], obs_ix = [0], inn_ix = [1], n0 = 10, τo = 0.3, rd = 0.3, r = 0.4, I = 0.8, G_in= 4., tmax = 100, dt = 0.1, method='ant'):
        """
            Parameters
            ----------
            N : int
                Size of whole graph.
            sub_ix : list of indices or twople, None
                If list of indices, then fill in those values with n0. If a twople,
                fill entries between the two indices with n0.
            obs_ix: list
                list of sites in the obsolescence front
            inn_ix: list
                list of sites in the innovation front
            Ady: np.array()
                Adyacency Matrix
            n0 : int or list
                If int Initial value of density in each site in sub_ix.
                If list, initial densities in the system
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
            dt: float
                time step for update
            method: str
                innovation method:
                    1- 'ant'
                    2- 'explorer'
        """

        """ Initializing parametrers 
        """

        self.N = N
        self.τo = τo          
        self.rd = rd
        self.r = r
        self.I= I
        self.G_in= G_in
        self.tmax = tmax
        self.dt = dt
        self.n0 = n0 
        self.t = 0
        self.count = 1
        self.switch = 1
        self.method = method
        """ Creating sites and densities
        """
        self.sites = np.arange(N)
        n = np.zeros(N)

        """ Initializing densities
        """
        n[sub_ix]= n0
        
        self.n = n
        """Init obsolescence and Innovation fronts
        """
        self.obs_front = np.array(obs_ix)
        self.inn_front = np.array(inn_ix)
        self.in_sub_pop = np.array(sub_ix)
        """Getting adjacency matrix
        """

        self.Ady = Ady

        """ Computing inverse of offspring vector
        """
        inverse_sons = np.sum(self.Ady, axis=1)
        inverse_sons[inverse_sons==0.] = 1.
        self.inverse_sons = 1./inverse_sons

    def move_innovation_ant(self):

            n = self.n
            obs_front = self.obs_front
            inn_front = self.inn_front

            to_innov = np.random.random(len(inn_front)) < self.r*self.I*self.dt*n[inn_front]
            new = [np.random.choice(self.sites[self.Ady[i]>0]) for i in inn_front[to_innov]]
            if new!=[]:
                #print(self.inn_front, new)
                self.inn_front = np.concatenate((self.inn_front, np.array(new)))
                self.inn_front = np.unique(self.inn_front)
                self.in_sub_pop = np.concatenate((self.in_sub_pop, np.array(new)))
                self.in_sub_pop = np.unique(self.in_sub_pop)
            self.inn_front = np.setdiff1d(self.inn_front, inn_front[to_innov])
    def move_innovation_explorer(self):

            n = self.n
            obs_front = self.obs_front
            inn_front = self.inn_front

            to_innov = np.random.random(len(inn_front)) < self.r*self.I*self.dt*n[inn_front]
            for i in inn_front[to_innov]:
                space_posible = self.sites[self.Ady[i]>0]
                space_posible = np.setdiff1d(space_posible, self.in_sub_pop)
                new = np.random.choice(space_posible)
                #print(new)
                self.inn_front = np.concatenate((self.inn_front, np.array([new])))
                self.inn_front = np.unique(self.inn_front)
                self.in_sub_pop = np.concatenate((self.in_sub_pop, np.array([new])))
                self.in_sub_pop = np.unique(self.in_sub_pop)
                if len(space_posible)==1:
                    self.inn_front = np.setdiff1d(self.inn_front, i)
            
    def automaton_evolve(self):

            n = self.n
            obs_front = self.obs_front
            inn_front = self.inn_front

            """
            Update the innovation front
            """
            if self.method =='ant':
                self.move_innovation_ant()
            else:
                self.move_innovation_explorer()

            """
            Replication 
            """
            to_replicate = np.random.poisson((self.r*self.inverse_sons* n * self.dt) @ self.Ady)
            #print(to_replicate)
            self.n[self.in_sub_pop] += to_replicate[self.in_sub_pop]

            """
            Death
            """
            to_died = np.random.poisson(self.rd * n *self.dt)
            self.n -= to_died                            
            self.n[self.n<0]=0

            """
            Growth
            """
            G_dt = np.random.poisson(self.G_in*self.dt)
            #print(G_dt)
            to_include = np.random.choice(self.in_sub_pop ,G_dt)
            to_include = np.unique(to_include, return_counts=True)
            self.n[to_include[0]]+= to_include[1]

            """
            Update obsolescence front 
            """
            obs_not_inn = np.setdiff1d(self.obs_front, self.inn_front)

            to_obsolete = np.random.random(len(obs_not_inn))<self.τo*self.dt
            #print(to_obsolete)
            to_obsolete = obs_not_inn[to_obsolete]
            #print(to_obsolete)
            self.obs_front =np.setdiff1d(self.obs_front, to_obsolete)
            self.in_sub_pop =np.setdiff1d(self.in_sub_pop, to_obsolete)

            to_add_obs = self.sites[np.sum(self.Ady[to_obsolete], axis=0)>0]
            to_add_obs = np.intersect1d(to_add_obs, self.in_sub_pop)
            self.obs_front = np.concatenate((self.obs_front, to_add_obs))                            
            self.obs_front = np.unique(self.obs_front)
     
    def run_and_save(self, tmax= None):
        """Propagate over multiple time steps and save simulation steps.
                Returns
        -------
        list
            Density at each time step.
        list
            System length at each time step.
        list
            Time.
        """
        
        n = [np.copy(self.n)]
        n1= np.copy(self.n)
        t = [0]
        P_I = list(self.inn_front[:])
        P_O = list(self.obs_front[:])
        tmax = tmax if not tmax is None else self.tmax
        
        import time
        start = time.time()
        #print(tmax)
        while self.t < tmax:
            self.automaton_evolve()
            #print(self.t, self.n[0:10], n1[0:10])
            n.append(np.copy(self.n))
            P_I.append(self.inn_front[:])
            P_O.append(self.obs_front[:])
            t.append(self.t)
            #print(self.t)
            self.t+=self.dt
            start = time.time()
        
        return n, P_I, P_O, t
    
class UnitSimulatonNetworksJax():
    def __init__(self, N, Ady, sub_ix= [0, 1], obs_ix = [0], inn_ix = [1], n0 = 10, τo = 0.3, rd = 0.3, r = 0.4, I = 0.8, G_in= 4., tmax = 100, dt = 0.1, method='ant', seed = None):
        """
            Parameters
            ----------
            N : int
                Size of whole graph.
            sub_ix : list of indices or twople, None
                If list of indices, then fill in those values with n0. If a twople,
                fill entries between the two indices with n0.
            obs_ix: list
                list of sites in the obsolescence front
            inn_ix: list
                list of sites in the innovation front
            Ady: np.array()
                Adyacency Matrix
            n0 : int or list
                If int Initial value of density in each site in sub_ix.
                If list, initial densities in the system
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
            dt: float
                time step for update
            method: str
                innovation method:
                    1- 'ant'
                    2- 'explorer'
        """

        """ Initializing parametrers 
        """
        if not seed:
            seed = np.random.randint(100000)
        #print(seed)
        self.key = random.PRNGKey(seed)
        self.N = N
        self.τo = τo          
        self.rd = rd
        self.r = r
        self.I= I
        self.G_in= G_in
        self.tmax = tmax
        self.dt = dt
        self.n0 = n0 
        self.t = 0
        self.count = 1
        self.switch = 1
        self.method = method
        """ Creating sites and densities
        """
        self.sites = jnp.arange(N, dtype=jnp.int64)
        self.n = jnp.zeros(N, dtype=jnp.int64)

        """ Initializing densities
        """
        self.n = self.n.at[0].set(10)
        self.n = self.n.at[1].set(10)
        
        """Init obsolescence and Innovation fronts
        """
        self.obs_front = jnp.zeros(1, dtype=jnp.int64)
        self.inn_front = jnp.ones(1, dtype=jnp.int64)
        self.in_sub_pop = jnp.array([0,1])
        """Getting adjacency matrix
        """

        self.Ady = jnp.eye(N, k=1, dtype=jnp.bool_)

        """ Computing inverse of offspring vector
        """
        inverse_sons = Ady.sum(1)
        inverse_sons = inverse_sons.at[~inverse_sons].set(1)
        self.inverse_sons = 1 / inverse_sons

    def move_innovation_ant(self):
        self.key, subkey = random.split(self.key)
        #print(self.key)
        inn_front = self.inn_front
        to_innov = random.uniform(subkey, (inn_front.size,)) < self.r*self.I*self.dt*self.n[inn_front]
        n_growing_sites = to_innov.sum()

        if n_growing_sites:
            new = jnp.zeros(n_growing_sites, dtype=jnp.int64)
            self.key, *subkeys = random.split(self.key, n_growing_sites+1)
            for i, ix in enumerate(inn_front[to_innov]):
                new = new.at[i].set(random.choice(subkeys[i], self.sites[self.Ady[ix]]))

            self.inn_front = jnp.concatenate((inn_front, new))
            self.inn_front = jnp.unique(self.inn_front)
            self.in_sub_pop = jnp.concatenate((self.in_sub_pop, new))
            self.in_sub_pop = jnp.unique(self.in_sub_pop)
            #print(self.inn_front, inn_front[to_innov], new, jnp.concatenate((inn_front, new)))
        self.inn_front = jnp.setdiff1d(self.inn_front, inn_front[to_innov])  # is it unnecessary?

    def move_innovation_explorer(self):

            n = self.n
            obs_front = self.obs_front
            inn_front = self.inn_front

            to_innov = np.random.random(len(inn_front)) < self.r*self.I*self.dt*n[inn_front]
            for i in inn_front[to_innov]:
                space_posible = self.sites[self.Ady[i]>0]
                space_posible = np.setdiff1d(space_posible, self.in_sub_pop)
                new = np.random.choice(space_posible)
                #print(new)
                self.inn_front = np.concatenate((self.inn_front, np.array([new])))
                self.inn_front = np.unique(self.inn_front)
                self.in_sub_pop = np.concatenate((self.in_sub_pop, np.array([new])))
                self.in_sub_pop = np.unique(self.in_sub_pop)
                if len(space_posible)==1:
                    self.inn_front = np.setdiff1d(self.inn_front, i)
            
    def automaton_evolve(self, oq, initwqw):

            n = self.n
            obs_front = self.obs_front
            inn_front = self.inn_front

            """
            Update the innovation front
            """
            self.move_innovation_ant()
            
            """
            Replication 
            """
            self.key, subkey = random.split(self.key)
            to_replicate = random.poisson(subkey, (self.r * self.inverse_sons * n * self.dt) @ self.Ady)
            self.n = self.n.at[self.in_sub_pop].set(n[self.in_sub_pop] + to_replicate[self.in_sub_pop])

            """
            Death
            """
            self.key, subkey = random.split(self.key)
            to_died = random.poisson(subkey, self.rd * n * self.dt)
            self.n = self.n - to_died                            
            self.n = self.n.at[self.n<0].set(0)

            """
            Growth
            """
            key, subkey = random.split(self.key)
            G_dt = random.poisson(subkey, self.G_in*self.dt/self.in_sub_pop.size, (self.in_sub_pop.size,))
            self.n = self.n.at[self.in_sub_pop].set(n[self.in_sub_pop] + G_dt)

            """
            Update obsolescence front 
            """
            obs_not_inn = jnp.setdiff1d(self.obs_front, self.inn_front)

            self.key, subkey = random.split(self.key)
            to_obsolete = random.uniform(subkey, (obs_not_inn.size,)) < self.τo*self.dt
            to_obsolete = obs_not_inn[to_obsolete]
            self.obs_front = jnp.setdiff1d(self.obs_front, to_obsolete)
            self.in_sub_pop = jnp.setdiff1d(self.in_sub_pop, to_obsolete)

            to_add_obs = self.sites[self.Ady[to_obsolete].sum(0)>0]
            to_add_obs = jnp.intersect1d(to_add_obs, self.in_sub_pop)
            self.obs_front = jnp.concatenate((self.obs_front, to_add_obs))                            
            self.obs_front = jnp.unique(self.obs_front)
     
    def run_and_save(self, tmax= None):
        """Propagate over multiple time steps and save simulation steps.
                Returns
        -------
        list
            Density at each time step.
        list
            System length at each time step.
        list
            Time.
        """
        
        n = [np.copy(self.n)]
        n1= np.copy(self.n)
        t = [0]
        P_I = list(self.inn_front[:])
        P_O = list(self.obs_front[:])
        tmax = tmax if not tmax is None else self.tmax
        
        import time
        start = time.time()
        #print(tmax)
        while self.t < tmax:
            self.automaton_evolve()
            #print(self.t, self.n[0:10], n1[0:10])
            n.append(np.copy(self.n))
            P_I.append(self.inn_front[:])
            P_O.append(self.obs_front[:])
            t.append(self.t)
            #print(self.t)
            self.t+=self.dt
            start = time.time()
        
        return n, P_I, P_O, t                                                                     
       
@jaxjit
def jax_propagate(n, Fix_R, Fix_P, G_in, Δt, r, τo, rd, I, N, init, P_I, P_O):
    """Propagate density function by some amount of time.
    """
    P_L = jnp.cumsum(P_O) * jnp.cumsum(P_I[::-1])[::-1]
    
    # Create growth vector
    Fix_V = G_in * P_L/(jnp.nextafter(0, 1) + jnp.sum(P_L))

    # Runge kutta order 4 update equations
    k1 = Δt*(-rd*n + (n*P_L) @ Fix_R + Fix_V)
    q1 = Δt*(((n*P_I)) @ Fix_P)
    #q1 = (P_I) @ Fix_P
    
    k2 = Δt*(-rd*(n+k1/2) +((n+k1/2)*P_L) @ Fix_R + Fix_V)
    q2 = Δt*(((n+k1/2)*(P_I+q1/2)) @ Fix_P)
    #q2 = ((P_I+q1/2)) @ Fix_P
    
    k3 = Δt*(-rd*(n+k2/2) +((n+k2/2)*P_L) @ Fix_R + Fix_V)
    q3 = Δt*(((n+k2/2)*(P_I+q2/2)) @ Fix_P)
    #q3 = ((P_I+q2/2)) @ Fix_P
    
    k4 = Δt*(-rd*(n+k3) +((n+k3)*P_L) @ Fix_R + Fix_V)
    q4 = Δt*(((n+k2/2)*(P_I+q3)) @ Fix_P)
    #q4 = ((P_I+q3)) @ Fix_P
    
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    q = (q1 + 2*q2 + 2*q3 + q4)/6
  
    return n + k, P_I + q

@jaxjit
def jax_propagate_with_P_O(n, Fix_R, Fix_P, Fix_O, G_in, Δt, r, τo, rd, I, N, init, P_I, P_O):
    """Propagate density function by some amount of time.
    """
    P_L = jnp.cumsum(P_O) * jnp.cumsum(P_I[::-1])[::-1]
    
    # Create growth vector
    Fix_V = G_in * P_L/(jnp.nextafter(0, 1) + jnp.sum(P_L))

    # Runge kutta order 4 update equations
    k1 = Δt*(-rd*n + (n*P_L) @ Fix_R + Fix_V)
    q1 = Δt*(((n*P_I)) @ Fix_P)
    w1 = (P_O) @ Fix_O
    
    k2 = Δt*(-rd*(n+k1/2) +((n+k1/2)*P_L) @ Fix_R + Fix_V)
    q2 = Δt*(((n+k1/2)*(P_I+q1/2)) @ Fix_P)
    w2 = ((P_O+w1/2)) @ Fix_O
    
    k3 = Δt*(-rd*(n+k2/2) +((n+k2/2)*P_L) @ Fix_R + Fix_V)
    q3 = Δt*(((n+k2/2)*(P_I+q2/2)) @ Fix_P)
    w3 = ((P_O+w2/2)) @ Fix_O
    
    k4 = Δt*(-rd*(n+k3) +((n+k3)*P_L) @ Fix_R + Fix_V)
    q4 = Δt*(((n+k2/2)*(P_I+q3)) @ Fix_P)
    w4 = ((P_O+w3)) @ Fix_O
    
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    q = (q1 + 2*q2 + 2*q3 + q4)/6
    w = (w1 + 2*w2 + 2*w3 + w4)/6
  
  
    return n + k, P_I + q, P_O + w


