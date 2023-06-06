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
from scipy.special import factorial, gammaln, logsumexp


from workspace.utils import save_pickle
from .utils import *

# Suggestions:

@jaxjit
def compute_logP_O_1(P_O, τo, t, init):
        """ Read parameters, variavles and vectors
        """
        logP_O = P_O
        for i in range(logP_O.size):
            logP_O = logP_O.at[i].set(jnp.log(τo)*i - τo*t + jnp.log(t)*i - gammaln(i+1))
        return logP_O

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
        self.sites = jnp.arange(N)
        self.n = jnp.zeros(N)
        self.Fix_V = jnp.zeros(N)
        self.P_L = jnp.zeros(N)

        """ Initializing densities
        """
        self.n = self.n.at[sub_ix[0]:sub_ix[1]].set(n0)
        
        self.init = sub_ix[-1]
        
        """Init obsolescence and Innovation fronts
        """
        self.P_I = jnp.asarray([0. if i!=sub_ix[-1] else 1. for i in range(self.N)])
        self.P_O = jnp.asarray([0. if i!=sub_ix[0] else 1. for i in range(self.N)])       
        #self.cum_P_O[sub_ix[0]] = 1
        #self.P_O, self.cum_P_O = compute_P_O(self, sub_ix[0],  *args, **kwargs)
        """ Getting adjacency matrix
        """
        
        self.Ady = jnp.eye(self.N, k=1)
        
        """ Computing inverse of offspring vector
        """
        self.inverse_sons = jnp.sum(self.Ady, axis=1)
        self.inverse_sons = self.inverse_sons.at[self.inverse_sons==0].set(1)
        self.inverse_sons = 1./self.inverse_sons
        
        """ Initialize Fix Matrix
        """
        self.Fix_M = (-rd) * jnp.eye(self.N)            # Set death    
        self.Fix_R = r * self.Ady
        self.Fix_P = r*I*(self.Ady-jnp.eye(self.N))
        self.Fix_O = τo*(self.Ady-jnp.eye(self.N))
    
    def propagate(self):
        """Propagate density function by fixed amount of time.
        """
        self.n, self.P_I = jax_propagate(self.n, self.Fix_R, self.Fix_P, self.G_in,
                                         self.Δt, self.r, self.τo, self.rd, self.I,
                                         self.N, self.init, self.P_I, self.P_O)
        if self.t > 0:
            logP_O = compute_logP_O_1(self.P_O, self.τo, self.t, self.init)
            self.P_O = jnp.exp(logP_O - logsumexp(logP_O))

        self.t += self.Δt

    def propagate_P_O(self, *args, **kwargs):          # propagate density function by some amount of time
        
        """ Read parameters, variavles and vectors
        """
        n = self.n
        Fix_M = self.Fix_M
        Fix_R = self.Fix_R
        Fix_P = self.Fix_P
        Fix_O = self.Fix_O
        G_in =self.G_in
        Δt=self.Δt
        λ = self.λ
        Ady = self.Ady
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        init = self.init
        P_I = self.P_I
        P_O = self.P_O
        P_L = compute_P_L(P_I, P_O, *args, **kwargs)
        self.P_L = P_L
        
        """ Creat vector of Growth
        """
        Fix_V = G_in * P_L/sum(P_L)
        self.Fix_V = Fix_V 
        """ Runge kutta order 4 update equations
        """
        
        k1 = Δt*(-rd*n + (n*P_L) @ Fix_R + Fix_V)
        q1 = Δt*(((n*P_I)) @ Fix_P)
        w1 = Δt* P_O @ Fix_O
        #q1 = (P_I) @ Fix_P
        
        k2 = Δt*(-rd*(n+k1/2) +((n+k1/2)*P_L) @ Fix_R + Fix_V)
        q2 = Δt*(((n+k1/2)*(P_I+q1/2)) @ Fix_P)
        w2 = Δt*((P_O+w1/2) @ Fix_O)
        #q2 = ((P_I+q1/2)) @ Fix_P
        
        k3 = Δt*(-rd*(n+k2/2) +((n+k2/2)*P_L) @ Fix_R + Fix_V)
        q3 = Δt*(((n+k2/2)*(P_I+q2/2)) @ Fix_P)
        w3 = Δt*((P_O+w2/2) @ Fix_O)
        #q3 = ((P_I+q2/2)) @ Fix_P
        
        k4 = Δt*(-rd*(n+k3) +((n+k3)*P_L) @ Fix_R + Fix_V)
        q4 = Δt*(((n+k2/2)*(P_I+q3)) @ Fix_P)
        w4 = Δt*((P_O+w3) @ Fix_O)
        #q4 = ((P_I+q3)) @ Fix_P
        
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        q = (q1 + 2*q2 + 2*q3 + q4)/6
        w = (w1 + 2*w2 + 2*w3 + w4)/6
        #print(q)
        #print(q1)
        #print(q2)
        #print(q3)
        #print(q4)
        #print(k)
        #print(k1)
        #print(k2)
        #print(k3)
        #print(k4)
        
        """ Update densities and time
        """
        self.n = n + k
        self.t += Δt
        self.P_I = P_I + q
        self.P_O = P_O + w
        
    def run_and_save(self, *args, **kwargs):
        """Propagate over multiple time steps and save simulation steps.
        
        Parameters
        ----------
        tmax : float
            End simulation time.
        *args
            To pass into self.propagate.
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
        n = [self.n[:]]
        time = [0]
        P_I = [self.P_I]

        while self.t < self.tmax:
            if self.method == "frequency":
                self.propagate(*args, **kwargs)
            else:
                self.propagate_P_O(*args, **kwargs)

            #H_sub_in_time.append(self.In_H_sub[:])
            #Obs_in_time.append(self.In_Obs_Front[:])
            #Inn_in_time.append(self.In_Inn[:])
            
            if jnp.isnan(self.P_I[-1]).any():
                print("nan values")
                break

            n.append(self.n[:])
            P_I.append(self.P_I[:])
            time.append(self.t)
            
        return n, P_I, time

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

def diffusion_simulation_deterministic(lattice_size, num_steps, num_particles, ro, r, rd, I , inn_pos, G):
    lattice = [0] * lattice_size
    obs_front_position = 0
    inn_front_position = inn_pos
    for i in range(inn_pos):
        lattice[i] = int(num_particles*1.0/inn_pos)
    
    for t in range(num_steps):
        lattice_copy = lattice[:]
        #print(t)
        if random.random() < r*I*lattice[inn_front_position] and inn_front_position < lattice_size - 1:
            inn_front_position += 1
        for i in range(G):
            p = random.randint(0, (inn_front_position-obs_front_position))
            lattice[obs_front_position+p]+=1
        #print(t)
        for q in range(lattice_size):
            for j in range(lattice_copy[q]):
                #print(t, q, j, lattice_copy[q])
                if random.random() < rd and lattice[q]>0:
                    lattice[q] -= 1  # Decrease the particle count at the current position
            
                # Move the particle to the right with probability r only if it is on the right side of the front
                if obs_front_position <= q < inn_front_position:
                    #print(t, x, lattice_copy[x])
                    if random.random() < r:
                        lattice[q+1] += 1  # Increase the particle count at the new position
            
                # Move the front to the right with probability ro
        #print(t)
        if random.random() < ro and obs_front_position < lattice_size - 1:
            obs_front_position += 1
        
    return lattice, obs_front_position, inn_front_position

import random

def diffusion_simulation_stochastic(lattice_size, num_steps, num_particles, ro, r, rd, I , inn_pos, G):
    lattice = [0] * lattice_size
    obs_front_position = 0
    inn_front_position = inn_pos
    for i in range(inn_pos):
        lattice[i] = int(num_particles*1.0/inn_pos)
    
    for t in range(num_steps):
        lattice_copy = lattice[:]
        if random.random() < r*I*lattice[inn_front_position] and inn_front_position < lattice_size - 1:
            inn_front_position += 1
        for i in range(G):
            p = random.randint(0, (inn_front_position-obs_front_position))
            lattice[obs_front_position+p]+=1
        for q in range(lattice_size):
            for j in range(lattice_copy[q]):
                #print(t, q, j, lattice_copy[q])
                if random.random() < rd and lattice[q]>0:
                    lattice[q] -= 1  # Decrease the particle count at the current position
            
                # Move the particle to the right with probability r only if it is on the right side of the front
                if obs_front_position <= q < inn_front_position:
                    #print(t, x, lattice_copy[x])
                    if random.random() < r:
                        lattice[q+1] += 1  # Increase the particle count at the new position
            
                # Move the front to the right with probability ro
        if random.random() < ro and obs_front_position < lattice_size - 1:
            obs_front_position += 1
        
    return lattice, obs_front_position, inn_front_position
