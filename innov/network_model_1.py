# ====================================================================================== #
# Network innovation model numerical solutions.
# 
# Author : Ernesto Ortega, ortega@csh.ac.at
# ====================================================================================== #
from numba import njit, jit
from numba.typed import List
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
import warnings
import networkx as nx

from workspace.utils import save_pickle
from .utils import *

# Suggestions:
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
        
        """ Graph generator
        """
        G = nx.circulant_graph(N, [1])
        G.remove_edge(0, N-1)      
        self.H = nx.DiGraph()
        self.H.add_edges_from(G.edges())
        
        """ Initializing parametrers 
        """
        self.N = len(self.H)  # Total size of the system    
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
        
        """ Creating sites and densities
        """
        self.sites = np.arange(N)
        self.n = np.zeros(N)
        
        """ Initializing densities
        """
        
        if isinstance(sub_ix, list):
            pass
        elif isinstance(sub_ix, tuple):
            sub_ix = list(range(*sub_ix))
        #else:
        #    raise NotImplementedError
        self.n[sub_ix] = n0
        
        
        """Init obsolescence method
        """
        self.method = method
        
        """Init populated subgraph
        """
        self.H_sub = self.H.subgraph(sub_ix)
        self.In_H_sub = np.array([i for i in self.H_sub])
        
        """Init obsolescence and Innovation fronts
        """
        self.In_Obs_Front = []
        self.In_Inn = []
        for i in self.H_sub.nodes():
            if list(self.H_sub.predecessors(i))==[]:
                self.In_Obs_Front.append(i)                #Obs front
            if list(self.H_sub.neighbors(i))==[]:
                self.In_Inn.append(i)                      #Innov front
                
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
        
        """ Getting adyacency matrix
        """
        
        A = nx.adjacency_matrix(self.H, weight=None)
        A = A.toarray()
        nodes_sub = np.array(list(self.H.nodes()))
        perm = nodes_sub.argsort()
        self.Ady =np.transpose(np.transpose(A[:, perm])[:, perm])
        
        """ Computing inverse of offspring vector
        """
        inverse_sons = np.sum(self.Ady, axis=1)
        inverse_sons[inverse_sons==0]=1
        self.inverse_sons = 1./inverse_sons
        
        """ Initialize Fix Matrix
        """
        self.Fix_M = (-rd)*np.eye(self.N, self.N)             # Set death
        
        for i in self.In_H_sub:                               # Set replication
            self.Fix_M[i]+=r*self.Ady[i]*self.inverse_sons
            if i in self.In_Inn:
                self.Fix_M[i]-=r*self.Ady[i]*self.inverse_sons
            
                
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
        
        """ Creat vector of Growth
        """
        Fix_V = np.zeros(self.N)
        if In_H_sub!=[]:
            Fix_V[In_H_sub]= G_in/len(In_H_sub)
        
        """ Runge kutta order 4 update equations
        """
        
        k1 = n @ Fix_M + Fix_V
        k2 = (n+k1/2) @ Fix_M + Fix_V
        k3 = (n+k2/2) @ Fix_M + Fix_V
        k4 = (n+k3) @ Fix_M + Fix_V
        k = (k1 + 2*k2 + 2*k3 + k4) * Δt/6
        
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
        
        if self.method=="frequency":
            if t>=self.count*(1/τo):
                self.count+=1
                to_remove = In_Obs_Front[:]
                """ Remove nodes from obsolescence front and the populated subgraph if not in Innovation front
                """
                for i in to_remove:
                    if i not in In_Inn:
                        In_Obs_Front.pop(In_Obs_Front.index(i))           #remove from Obsolescence front
                        In_H_sub.pop(In_H_sub.index(i))                   #remove from populated subgraph
                        """ Check if any of the parents is in obs front
                        """
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
                                In_Obs_Front.append(j)                     #add to Obsolescence front
                        Fix_M[i]-= r*self.Ady[i]*self.inverse_sons         #remove replication
        
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
        self.Fix_M = Fix_M
        self.In_Obs_Front = In_Obs_Front
        self.In_Inn = In_Inn
        self.In_H_sub = In_H_sub
        self.x = x
        self.x_obs = x_obs
        
    def update_H_sub_and_Innov(self):
        
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
        t = self.t
        
        In_H_sub = [i for i in In_H_sub]
        #In_H_sub = In_H_sub+In_Obs_Front
        space_poss = []
        In_Inn_1 = In_Inn[:]
        for i in In_Inn_1:
            x[i]+= n[i]*r*I*Δt
            if x[i]>=λ:
                """ Remove node from Inn
                """
                x.pop(i)
                In_Inn.pop(In_Inn.index(i))
                
                """ Remove all the parents from Innovation front
                """
                
                for j in self.H.predecessors(i):
                    if j in In_Inn_1:
                        In_Inn.pop(In_Inn.index(j))
                        x.pop(j)
                
                """ Add all the offspring to Innovation front if not already in 
                """
                for j in self.H.neighbors(i):
                    if j not in In_Inn:
                        x[j]=0
                        In_Inn.append(j)
                        In_H_sub.append(j)
                        Fix_M[i][j]+=r/(self.inverse_sons[i])
        
        """ update vectors and Matrices
        """
        self.Fix_M = Fix_M
        self.In_Obs_Front = In_Obs_Front
        self.In_Inn = In_Inn
        self.In_H_sub = In_H_sub
        self.x = x
        