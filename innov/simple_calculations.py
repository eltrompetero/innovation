# ====================================================================================== #
# Minimal innovation model implementations and solutions.
# 
# Author : Eddie Lee, edlee@csh.ac.at
#          Ernesto Ortega, ernesto.ortega.dias.25@gmail.com
# ====================================================================================== #
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from cmath import sqrt
import warnings
from scipy.special import logsumexp
from scipy.integrate import odeint
from .utils import *


def pde_pseudogap(y0, t, G_in, I, r, rd, ro, gamma, K):
    """Lattice length given compartment approximation.
    
    Input
    -----
    Initial values of: 
    y0 = [N      : float
         n(L-1) : float
         n(0)   : float
         L      : float]
    t : float
    
    Parameters
    ----------
    G : float
    ro : float
    rd : float
    I : float
    gamma : float
    k : int
    
    Returns
    -------
    Derivatives of variables
    dydt : floats
    """
    N, nel, n0, l = y0
    if l<=2 or N<=0:
        dN = - N
        dnel = - nel
        dn0 = - n0
        dl = - l
    elif l>1e10 and N>1e15:
        dN = 0
        dnel = 0
        dn0 = 0
        dl = 0
    elif N>1e15:
        dN = 0
        dnel = 0
        dn0 = 0
        dl = r*I * n0*(1+gamma*(K-1)) - ro*(1+gamma*(K-1))
    elif l>1e10:
        dN = (G_in + (r-rd)*N -r*n0 - ro*(1+gamma*(K-1))*nel)
        dnel = (G_in/l -rd *nel -ro *(1+gamma*(K-1))* nel + r*I * n0*(1+gamma*(K-1)) * ((N- n0 - nel)/(l-2.000001)))
        dn0 = (G_in/l + r*(N- n0 - nel)/(l-2.000001) -rd*n0 - r*I*(1+gamma*(K-1))*n0**2)
        dl = 0
    else:
        dN = (G_in + (r-rd)*N -r*n0 - ro*(1+gamma*(K-1))*nel)
        dnel = (G_in/l -rd *nel -ro *(1+gamma*(K-1)) * nel + r*I * n0*(1+gamma*(K-1)) * ((N- n0 - nel)/(l-2.000001)))
        dn0 = (G_in/l +r*(N- n0 - nel)/(l-2.000001) -rd*n0 - r*I*(1+gamma*(K-1))*n0**2)
        dl = r * I * (1+gamma*(K-1)) * n0 - ro *(1+gamma*(K-1))

    return np.array([dN, dnel, dn0, dl])

def pde_pseudogap_large_L(y0, t, G_in, I, r, rd, ro, gamma, K):
    """Lattice length given compartment approximation and large L.
    
    Input
    -----
    Initial values of: 
    y0 = [N      : float
         n(L-1) : float
         n(0)   : float
         L      : float]
    t : float
    
    
    Parameters
    ----------
    G : float
    ro : float
    rd : float
    I : float
    gamma : float
    k : int
    
    Returns
    -------
    Derivatives of variables
    dydt : floats
    """
    
    N, nel, n0, l = y0
    dN = (G_in + (r-rd)*N -r*K*n0 - ro*(1+gamma*(K-1))*K*nel)
    dnel = (G_in/l -rd *nel -ro *(1+gamma*(K-1))* nel + r*I * n0*(1+gamma*(K-1)) * ((N)/(l)))
    dn0 = (G_in/l +r*(N)/(l) -rd*n0 - r*I*(1+gamma*(K-1))*n0**2)
    dl = K*r * I* (1+gamma*(K-1)) * n0 - ro*K *(1+gamma*(K-1))

    return np.array([dN, dnel, dn0, dl])

def Equilibrium_compartment_model_equations(y, G, I, r, rd, ro, gamma, k):
            N, nel, n0, l = y
            return ((G + (r-rd)*N -r*n0 - ro*(1+gamma*(k-1))*nel), (G/l -rd *nel -ro *(1+gamma*(k-1))* nel + r*I*(1+gamma*(k-1)) * n0 * ((N-n0-nel)/(l-2))), (G/l + r*((N-n0-nel)/(l-2)) -rd*n0 - r*I*(1+gamma*(k-1))*n0**2), r*I * n0*(1+gamma*(k-1)) - ro*(1+gamma*(k-1)))


def rd_critic(G , I, ro, r, rd, gamma, k):
    """Lattice length given linear pseudogap approximation.

    Parameters
    ----------
    G : float
    ro : float
    rd : float
    I : float
    gamma : float
    k : int
    
    Returns
    -------
    float
        critic death rate.
    """
    assert k>=1
    #assert z(re, rd, ro)>0
    def cost(lk):
        rd = lk
        #print(ro, rd)
        nl_star = (ro * (rd+ ro*(1+gamma*(k-1))-2*r) )/(r*I*(rd+(1+gamma*(k-1))*ro))
        #print(ro)
        return  ((rd - r + ro*ro * (1+gamma*(k-1)) * nl_star /G))**2
    sol = minimize(cost, 0.3, tol=1e-10, bounds=[(0,np.inf)])
    return sol['x']

def rd_crit_quadratic(r0 , I, vo, r, gamma, k, initial_guess=-1):
    """Critical death rate rd as a function of other parameters by solving
    the discriminant.

    Parameters
    ----------
    r0 : float
    vo : float
    I : float
    gamma : float
    k : int
    initial_guess : float, -1
        Log of rd.
    
    Returns
    -------
    float
        critic death rate.
    """
    assert k>=1
    def cost(log_rd):
        rd = np.exp(log_rd)
        a = (rd/r -1)*I
        b = (rd/r -1)*I*r0/r +vo**3*(1+gamma*(k-1))**2/r**3 -(r0/r - vo/r/I)*I
        c = (vo/r/I - r0/r)*I*r0/r + vo**2*(1+gamma*(k-1))/r**2*r0/r 
        return  (b**2-4*a*c)**2
    sol = minimize(cost, initial_guess, tol=1e-10)
    if sol['x'][0]==initial_guess:
        return np.nan
    return np.exp(sol['x'][0])

def K_critical_quadratic(r0 , I, vo, r, rd, gamma, initial_guess=300):
    """Critical branching number K as a function of other parameters by solving 
    the discriminant.

    Parameters
    ----------
    r0 : float
    vo : float
    rd : float
    I : float
    gamma : float
    initial_guess : float, 300
        Starting guess for K.
    
    Returns
    -------
    float
        Critical gamma. Return np.nan if no solution is found.
    """
    def cost(k):
        a = (rd/r -1)*I
        b = (rd/r -1)*I*r0/r +vo**3*(1+gamma*(k-1))**2/r**3 -(r0/r - vo/r/I)*I
        c = (vo/r/I - r0/r)*I*r0/r + vo**2*(1+gamma*(k-1))/r**2*r0/r 
        return  (b**2-4*a*c)**2
    sol = minimize(cost, initial_guess, tol=1e-10, bounds=[(1, np.inf)])

    if sol['success']==False or sol['x'][0]==initial_guess:
        return np.nan
    return sol['x'][0]