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
#import networkx as nx
from graph_tool.all import *
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
        H = Graph()
        H.add_vertex(N)
        for i in range(N-1):
            H.add_edge(H.vertex(i), H.vertex(i+1))
        self.H = H
        
        """ Initializing parametrers 
        """
        self.N = self.H.num_vertices() # Total size of the system    
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
        
        """ Creating sites and densities and prescence
        """
        self.density = self.H.new_vertex_property("double")
        
        self.presence = self.H.new_vertex_property("int")
        
        """ Initializing densities
        """
        
        if isinstance(sub_ix, list):
            pass
        elif isinstance(sub_ix, tuple):
            sub_ix = list(range(*sub_ix))
        #else:
        #    raise NotImplementedError
        self.density.a[list(range(N))] = 0
        self.density.a[sub_ix[0:-1]] = n0
        
        
        """prescence define the position in the graph
           prescence: int
               0:  not in populated subgraph
               1:  in the populated subgraph
               2:  in obsolescence front
               3:  in inovation front
               4:  obsolete
        """
        self.presence.a[list(range(N))] = 0
        self.presence.a[sub_ix] = 1
        self.presence.a[sub_ix[0]] = 2
        self.presence.a[sub_ix[-1]] = 3
        
        # add a vertex property called "change" to each node and initialize it based on "presence"
        self.k1 = self.H.new_vertex_property("float")  # create the property
        self.k2 = self.H.new_vertex_property("float")  # create the property
        self.k3 = self.H.new_vertex_property("float")  # create the property
        self.k4 = self.H.new_vertex_property("float")  # create the property
        for v in self.H.vertices():
            self.update_k1(v)
            self.update_k2(v)
            self.update_k3(v)
            self.update_k4(v)
        
        self.method = method
    
    def update_k1(self, v):
        if self.presence[v] == 0 or self.presence[v] == 4:
            self.k1[v] = -self.rd*(self.density[v])
        elif self.presence[v] == 1:
            self.k1[v] = -self.rd*(self.density[v])+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
            for v1 in v.in_neighbors():
                self.k1[v] += self.r*(self.density[v1])/v1.out_degree() 
        elif self.presence[v] == 2:
            self.k1[v] = -self.rd*(self.density[v])
            for v1 in v.in_neighbors():
                self.k1[v] += self.r*self.I*(self.density[v1])/v1.out_degree() 
        else:
            self.k1[v] = -self.rd*(self.density[v])+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
    def update_k2(self, v):
        if self.presence[v] == 0 or self.presence[v] == 4:
            self.k2[v] = -self.rd*(self.density[v]+self.k1[v]/2)
        elif self.presence[v] == 1:
            self.k2[v] = -self.rd*(self.density[v]+self.k1[v]/2)+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
            for v1 in v.in_neighbors():
                self.k2[v] += self.r*(self.density[v1]+self.k1[v1]/2)/v1.out_degree() 
        elif self.presence[v] == 2:
            self.k2[v] = -self.rd*(self.density[v]+self.k1[v]/2)
            for v1 in v.in_neighbors():
                self.k2[v] += self.r*self.I*(self.density[v1]+self.k1[v1]/2)/v1.out_degree() 
        elif self.presence[v] == 3:
            self.k2[v] = -self.rd*(self.density[v]+self.k1[v]/2)+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
    def update_k3(self, v):
        if self.presence[v] == 0 or self.presence[v] == 4:
            self.k3[v] = -self.rd*(self.density[v]+self.k2[v]/2)
        elif self.presence[v] == 1:
            self.k3[v] = -self.rd*(self.density[v]+self.k2[v]/2)+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
            for v1 in v.in_neighbors():
                self.k3[v] += self.r*(self.density[v1]+self.k2[v1]/2)/v1.out_degree()
        elif self.presence[v] == 2:
            self.k3[v] = -self.rd*(self.density[v]+self.k2[v]/2)
            for v1 in v.in_neighbors():
                self.k3[v] += self.r*self.I*(self.density[v1]+self.k2[v1]/2)/v1.out_degree()
        elif self.presence[v] == 3:
            self.k3[v] = -self.rd*(self.density[v]+self.k2[v]/2)+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
    def update_k4(self, v):
        if self.presence[v] == 0 or self.presence[v] == 4:
            self.k4[v] = -self.rd*(self.density[v]+self.k3[v])
        elif self.presence[v] == 1:
            self.k4[v] = -self.rd*(self.density[v]+self.k3[v])+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
            for v1 in v.in_neighbors():
                self.k4[v] += self.r*(self.density[v1]+self.k3[v1])/v1.out_degree() 
        elif self.presence[v] == 2:
            self.k4[v] = -self.rd*(self.density[v]+self.k3[v])
            for v1 in v.in_neighbors():
                self.k4[v] += self.r*self.I*(self.density[v1]+self.k3[v1])/v1.out_degree() 
        elif self.presence[v] == 3:
            self.k4[v] = -self.rd*(self.density[v]+self.k3[v])+self.G_in/len(find_vertex_range(self.H, self.presence, [1, 2]))
    def propagate(self, *args, **kwargs):          # propagate density function by some amount of time
        
        """ Read parameters, variavles and vectors
        """
        G_in =self.G_in
        Δt=self.Δt
        λ = self.λ
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        
        """ Creat vector of Growth
        """
        
        """ Runge kutta order 4 update equations
        """
        In_Hsub = find_vertex(self.H, self.presence, 1)
        In_Hsub += find_vertex(self.H, self.presence, 2)
        In_Hsub += find_vertex(self.H, self.presence, 3)
        for v in In_Hsub:
            self.update_k1(v)
            self.update_k2(v)
            self.update_k3(v)
            self.update_k4(v)
        for v in In_Hsub:
            k = (self.k1[v] + 2*self.k2[v] + 2*self.k3[v] + self.k4[v]) * Δt/6
            self.density[v] += k
        """ Update densities in obsolescence and time
        """
        self.update_densities_obs(*args, **kwargs)
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
        n = [self.density.a]
        L = [len(find_vertex_range(self.H, self.presence, [1, 2]))]
        H_sub_in_time = [find_vertex_range(self.H, self.presence, [1, 2])]
        Obs_in_time = [find_vertex(self.H, self.presence,  2)]
        Inn_in_time = [find_vertex(self.H, self.presence,  3)]
        time = [0]
        dt = self.Δt
        t = 0
        time_to_check_stability=int(100./dt)
        err =1./time_to_check_stability
        while t < self.tmax:
            self.propagate(*args, **kwargs)
            n.append(self.density.a)
            L.append(len(find_vertex_range(self.H, self.presence, [1, 2])))
            t += dt
            time.append(t)
            H_sub_in_time.append(find_vertex_range(self.H, self.presence, [1, 2]))
            Obs_in_time.append(find_vertex(self.H, self.presence,  2))
            Inn_in_time.append(find_vertex(self.H, self.presence,  3))
            if len(L)>=2*time_to_check_stability:
                L_old = np.average(L[-2*time_to_check_stability :len(L)-time_to_check_stability])
                L_ave = np.average(L[-time_to_check_stability:len(L)])
                #print(t, abs(L_ave-L_old), err)
                if abs(L_ave-L_old)<=err:
                    break
                L_old = L_ave
        return n, L, time, H_sub_in_time, Obs_in_time, Inn_in_time
    
    def update_H_sub_and_Obs(self, *args, **kwargs):
        
        """ Read parameters, variavles and vectors
        """
        G_in =self.G_in
        Δt=self.Δt
        λ = self.λ
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        t = self.t
        
        if self.method=="frequency":
            #print("update_H_sub_and_Obs", t>=self.count*(1/τo), t, self.count*(1/τo))
            if t>=self.count*(1/τo):
                self.count+=1
                to_remove = find_vertex(self.H, self.presence, 2)
                """ Remove nodes from obsolescence front and the populated subgraph if not in Innovation front
                """
                for v in to_remove:
                    self.presence[v]= 4           #remove from Obsolescence front
                    """ Check if any of the parents is in obs front
                    """
                    #print(v, v.out_neighbors)
                    for v1 in v.out_neighbors():                
                        """ Check if any of the parents of node is still in obs front
                        """
                        if self.presence[v1]!=3:
                            obs_count=0
                            for v2 in v1.in_neighbors():
                                if v2!=v:
                                    
                                    if self.presence[v2]==2:
                                        obs_count+=1
                            """ Add to Obs front if no parent in Obs front
                            """
                            #print(v, v1, obs_count)
                            if obs_count==0:
                                #print("entree")
                                self.presence[v1]=2                    #add to Obsolescence front
                                #print(self.presence[v1], v1)
        else:
            to_remove = find_vertex(self.H, self.presence, 2)
            for v in to_remove:
                if self.density[v]<λ:
                    self.prescence[v]= 4           #remove from Obsolescence front
                    """ Check if any of the parents is in obs front
                    """
                    for v1 in v.out_neighbors():                
                        """ Check if any of the parents of node is still in obs front
                        """
                        if self.presence[v1]!=3:
                            obs_count=0
                            for v2 in v1.in_neighbors():
                                if v2!=v:
                                    if self.prescence[v2]==2:
                                        obs_count+=1
                            """ Add to Obs front if no parent in Obs front
                            """
                            if obs_count==0:
                                self.presence[v1]=2                     #add to Obsolescence front
                                
        
        
    def update_H_sub_and_Innov(self, *args, **kwargs):
        
        """ Read parameters, variavles and vectors
        """
        G_in =self.G_in
        Δt=self.Δt
        λ = self.λ
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        t = self.t
        
        In_Inn_1 = find_vertex(self.H, self.presence, 3)
        for v in In_Inn_1:
            if self.density[v]>λ:
                """ Remove node from Inn
                """
                self.presence[v]=1
                """ Remove all the parents from Innovation front
                """
                for v1 in v.in_neighbors():
                    if self.presence[v1]==1:
                        for v2 in v1.in_neighbors():
                            if self.presence[v2]==3:
                                   self.presence[v2]=0
                """ Add all the offspring to Innovation front if not already in 
                """
                for v1 in v.out_neighbors():
                    if self.presence[v1]==0:
                        self.presence[v1]=3
        
    def update_densities_obs(self, *args, **kwargs):
        
        """ Read parameters, variavles and vectors
        """
        #print("entree")
        G_in =self.G_in
        Δt=self.Δt
        λ = self.λ
        r = self.r
        τo = self.τo
        rd = self.rd
        I = self.I
        N= self.N
        t = self.t
        
        In_Obs = [int(v) for v in find_vertex(self.H, self.presence, 4)]
        self.density.a[In_Obs]*=np.exp(-self.rd*Δt)    