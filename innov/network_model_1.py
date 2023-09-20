# ====================================================================================== #
# Network innovation model numerical solutions.
# 
# Author : Ernesto Ortega, ortega@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
import warnings
#import networkx as nx

from workspace.utils import save_pickle
from .utils import *

# Suggestions:
# 1. object-oriented code
class RKPropagator1D():
    def __init__(self, N, sub_ix = [1], inn = [1], obs = [0], Ady = None, n0=0, τo = 0.3, rd = 0.3, r = 0.4, I = 0.8, G_in= 4., tmax = 100, Δt = 0.1, λ = 1., method='density'):
        """
        Parameters
        ----------
        N : int
            Size of whole graph.
        sub_ix : list of indices or twople, None
            If list of indices, then fill in those values with n0. If a twople,
            fill entries between the two indices with n0.
        inn: list
            list of nodes in the innovation front
        obs: list
            list of nodes in the obsolescence front
        Ady: np.array
            adjacency matrix
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
        
        """ Graph generator
        """
        self.Ady = Ady
        
        """ Initializing parametrers 
        """
        self.N = len(Ady)  # Total size of the system    
        self.τo = τo          
        self.rd = rd
        self.r = r
        self.I= I
        self.G_in= G_in
        self.tmax = tmax
        self.Δt = Δt
        self.n0 = n0 
        self.λ = 1
        self.t = 0
        self.count = 1
        self.switch = 1
        
        """ Creating sites and densities
        """
        self.sites = np.arange(N)
        self.n = np.zeros(N)
        
        """ Initializing densities
        """
        
        #if isinstance(sub_ix, list):
        #    pass
        #elif isinstance(sub_ix, tuple):
        #sub_ix = list(range(*sub_ix))
        #else:
        #    raise NotImplementedError
        self.n[[sub_ix]] = n0
        
        
        """Init obsolescence method
        """
        self.method = method
        
        """Init populated subgraph
        """
        self.In_H_sub = sub_ix
        
        """Init obsolescence and Innovation fronts
        """
        self.In_Obs_Front = obs
        self.In_Inn = inn
                
        """ Init distance to space of posible
        """
        
        self.x = {}
        for i in self.In_Inn:
            self.x[i]=0
        
        """ Init distance to obsolescence
        """
        self.x_obs = {}
        for i in self.In_Obs_Front:
            self.x_obs[i]= λ
        
        """ Computing inverse of offspring vector
        """
        inverse_sons = np.sum(self.Ady, axis=1)
        inverse_sons[inverse_sons==0]=1
        self.inverse_sons = 1./inverse_sons
        
        """ Initialize Fix Matrix
        """
        self.Fix_M = (-rd) * np.eye(self.N)            # Set death    
        self.Fix_R = r * self.Ady
        self.Fix_P = r*I*(self.Ady - np.eye(self.N))
        self.Fix_O = τo*(self.Ady - np.eye(self.N))
            
                
    def propagate(self, *args, **kwargs):          # propagate density function by some amount of time
        
        """ Read parameters, variavles and vectors
        """
        n = self.n
        Fix_M = self.Fix_M
        G_in =self.G_in
        In_H_sub = self.In_H_sub
        Δt=self.Δt
        λ = self.λ
        Ady = self.Ady
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        In_Obs_Front = self.In_Obs_Front
        In_Inn = self.In_Inn
        x = self.x
        space_possible = self.sites[np.sum(Ady[In_Inn], axis=0)>0]
        """ Creat vector of Growth
        """
        In_H_sp = np.concatenate((In_H_sub, space_possible))
        #print(In_H_sp, space_possible, In_H_sub)
        n_in = np.zeros(self.N)
        n_in[In_H_sub] = self.n[In_H_sub] 
        self.Fix_V = (n_in>0) * self.G_in/(len(In_H_sub))
        
        """ Runge kutta order 4 update equations
        """
        
        k1 = self.Δt*(-self.rd*n + (n_in*self.inverse_sons) @ self.Fix_R + self.Fix_V)
        
        k11 = np.zeros(self.N)
        k11[In_H_sub] = k1[In_H_sub] 
        
        k2 = self.Δt*(-self.rd*(n+k1/2) + ((n_in +k11/2)*self.inverse_sons)@ self.Fix_R + self.Fix_V)
        
        
        k22 = np.zeros(self.N)
        k22[In_H_sub] = k1[In_H_sub]
        
        k3 = self.Δt*(-self.rd*(n+k2/2) + ((n_in +k22/2)*self.inverse_sons)@ self.Fix_R + self.Fix_V)
        
        k33 = np.zeros(self.N)
        k33[In_H_sub] = k1[In_H_sub] 
        
        k4 = self.Δt*(-self.rd*(n+k3) + ((n_in +k33)*self.inverse_sons)@ self.Fix_R + self.Fix_V)
        
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        
        """ Update densities and time
        """
        self.n = n + k
        self.t += Δt
        
        """ Update Obsolescence front
        """
        self.update_H_sub_and_Obs(*args, **kwargs)
        
        """ Update Innovation front
        """
        self.update_H_sub_and_Innov(*args, **kwargs)
        
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
        L = [len(self.In_H_sub)]
        H_sub_in_time = [self.In_H_sub[:]]
        Obs_in_time = [self.In_Obs_Front[:]]
        Inn_in_time = [self.In_Inn[:]]
        time = [0]
        dt = self.Δt
        t = 0
        time_to_check_stability=int(100./dt)
        err =1./time_to_check_stability
        while t < self.tmax:
            self.propagate(*args, **kwargs)
            n.append(self.n[:])
            L.append(len(self.In_H_sub))
            t += dt
            time.append(t)
            H_sub_in_time.append(self.In_H_sub[:])
            Obs_in_time.append(self.In_Obs_Front[:])
            Inn_in_time.append(self.In_Inn[:])
            if len(L)>=2*time_to_check_stability:
                L_old = np.average(L[-2*time_to_check_stability :len(L)-time_to_check_stability])
                L_ave = np.average(L[-time_to_check_stability:len(L)])
                #print(t, abs(L_ave-L_old), err)
                if abs(L_ave-L_old)<=err:
                    break
                L_old = L_ave
        return n, L, time, H_sub_in_time, Obs_in_time, Inn_in_time
    
    def update_H_sub_and_Obs(self):
        
        """ Read parameters, variavles and vectors
        """
        n = self.n
        Fix_M = self.Fix_M
        G_in =self.G_in
        In_H_sub = self.In_H_sub
        Δt=self.Δt
        λ = self.λ
        Ady = self.Ady
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        In_Obs_Front = self.In_Obs_Front
        In_Inn = self.In_Inn
        x = self.x
        x_obs = self.x_obs
        t = self.t
        #print(x)
        if self.method=="frequency":
            if t>=self.count*(1/τo):
                self.count+=1
                to_remove = In_Obs_Front.copy()
                """ Remove nodes from obsolescence front and the populated subgraph if not in Innovation front
                """
                space_obs = self.sites[np.sum(Ady[to_remove], axis=0)>0]
                space_obs = list(np.unique(space_obs))
                to_obs = np.setdiff1d(space_obs, to_remove)
                In_Obs_Front = to_obs
                In_H_sub = np.setdiff1d(In_H_sub, to_remove)
        else:
            In_H_sub = [i for i in In_H_sub]
            In_Obs_Front_1 = In_Obs_Front[:]
            for i in In_Obs_Front_1:
                if i not in In_Inn:
                    x_obs[i]-= n[i]*τo*Δt                             #reduce distance to obsolescence
                if x_obs[i]<=0:
                    In_H_sub.pop(In_H_sub.index(i))                   #remove from populated subgraph
                    In_Obs_Front.pop(In_Obs_Front.index(i))           #remove from Obsolescence front
                    x_obs.pop(i)                                      #remove from Obsolescence distances 
                    Fix_M[i]-= r*self.Ady[i]*self.inverse_sons        #remove replication
                    for j in self.H.neighbors(i):                
                        """ Check if any of the parents of node is still in obs front
                        """
                        obs_count=0
                        for q in self.H.predecessors(j):
                            if q!=i:
                                if q in In_Obs_Front:
                                    obs_count+=1
                        """ Add to Obs front if no parent in Obs front
                        """
                        if obs_count==0:
                            In_Obs_Front.append(j)            
                            x_obs[j] = λ
        
        """ update vectors and Matrices
        """
        self.In_Obs_Front = In_Obs_Front
        self.In_H_sub = In_H_sub
        
    def update_H_sub_and_Innov(self):
        
        """ Read parameters, variavles and vectors
        """
        n = self.n
        G_in =self.G_in
        In_H_sub = self.In_H_sub
        Δt=self.Δt
        λ = self.λ
        Ady = self.Ady
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        In_Obs_Front = self.In_Obs_Front
        In_Inn = self.In_Inn
        x = self.x
        t = self.t
        
        In_Inn_1 = In_Inn.copy()
        for i in In_Inn_1:
            #print(i, x, In_Inn_1)
            x[i]+= n[i]*r*I*Δt
            if x[i]>=λ:
                """ Remove node from Inn
                """
               
                In_Inn.pop(In_Inn.index(i))
                
                """ Remove all the parents from Innovation front
                """
                #for j in self.H.predecessors(i):
                #    if j in In_Inn_1:
                #        In_Inn.pop(In_Inn.index(j))
                #        x.pop(j)
                
                """ Add all the offspring to Innovation front if not already in 
                """
                space_poss = self.sites[self.sites[Ady[i]>0]]
                #print(space_poss, In_Inn, In_Inn + list(space_poss))
                In_Inn += list(space_poss)
                In_H_sub = list(In_H_sub)+ list(space_poss)
                In_Inn = list(np.unique(In_Inn))
                In_H_sub = list(np.unique(In_H_sub))
                #x.fromkeys(In_Inn, 0)
                for j in space_poss:
                    if j not in In_Inn_1:
                        x[j]= 0
                x.pop(i)    
                #print(x, In_Inn, type(In_Inn), type(list(space_poss)))
                    
        """ update vectors and Matrices
        """
        self.In_Inn = In_Inn
        self.In_H_sub = In_H_sub
        self.x = x
        
