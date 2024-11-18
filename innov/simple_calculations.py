# ====================================================================================== #
# Minimal innovation model implementations and solutions.
# 
# Author : Eddie Lee, edlee@csh.ac.at
#          Ernesto Ortega, ernesto.ortega.dias.25@gmail.com
# ====================================================================================== #
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from scipy.integrate import odeint
import warnings
from .utils import *


def pde_pseudogap(y0, t, r0, I, r, rd, vo, gamma, K):
    """Compartment approximation model.
    
    Parameters
    ----------
    y0 = [N      : float
          L      : float
          n(0)   : float
          n(L-1) : float]
    t : float
    G : float
    vo : float
    rd : float
    I : float
    gamma : float
    K : int
        Branching number.
     
    Returns
    -------
    dydt : floats
        Derivatives of variables.
    """
    N, L, n0, nl = y0
    k = 1 + gamma * (K - 1)

    if L<=2 or N<=0:
        dN = -N
        dL = -L
        dn0 = -n0
        dnl = -nl
    elif L>1e10 and N>1e15:
        dN = 0
        dL = 0
        dn0 = 0
        dnl = 0
    elif N>1e15:
        dN = 0
        dnl = 0
        dn0 = 0
        dL = r*I * n0*k - vo*k
    elif L>1e10:
        dN = r0 + (r-rd)*N -r*n0 - vo*k*nl
        dnl = r0/L - rd*nl - vo*k*nl + r*I*n0*k*((N - n0 - nl)/(L-2))
        dn0 = r0/L + r*(N - n0 - nl)/(L-2) - rd*n0 - r*I*k*n0**2
        dL = 0
    else:
        dN = r0 + (r-rd)*N -r*n0 - vo*k*nl
        dnl = r0/L - rd*nl - vo*k*nl + r*I*n0*k * ((N - n0 - nl)/(L-2))
        dn0 = r0/L + r*(N - n0 - nl)/(L-2) -rd*n0 - r*I*k*n0**2
        dL = r*I*k*n0 - vo*k

    return np.array([dN, dL, dn0, dnl])

def pde_pseudogap_large_L(y0, t, r0, I, r, rd, vo, gamma, K):
    """Lattice length given compartment approximation and large L.
    
    Parameters
    ----------
    Initial values of: 
    y0 = [N      : float
         n(L-1) : float
         n(0)   : float
         L      : float]
    t : float
    r0 : float
    vo : float
    rd : float
    I : float
    gamma : float
    k : int
    
    Returns
    -------
    Derivatives of variables
    dydt : floats
    """
    N, nel, n0, L = y0
    k  = 1+gamma*(K-1)
    dN = (r0 + (r-rd)*N -r*K*n0 - vo*k*K*nel)
    dnl = (r0/L -rd *nel -vo *k* nel + r*I * n0*k * (N/L))
    dn0 = (r0/L +r*N/L -rd*n0 - r*I*k*n0**2)
    dL = K*r*I*k*n0 - vo*K*k

    return np.array([dN, dnl, dn0, dL])

def Equilibrium_compartment_model_equations(y, G, I, r, rd, vo, gamma, k):
            N, nel, n0, l = y
            return ((G + (r-rd)*N -r*n0 - vo*(1+gamma*(k-1))*nel), (G/l -rd *nel -vo *(1+gamma*(k-1))* nel + r*I*(1+gamma*(k-1)) * n0 * ((N-n0-nel)/(l-2))), (G/l + r*((N-n0-nel)/(l-2)) -rd*n0 - r*I*(1+gamma*(k-1))*n0**2), r*I * n0*(1+gamma*(k-1)) - vo*(1+gamma*(k-1)))


def rd_critic(G , I, vo, r, rd, gamma, k):
    """Lattice length given linear pseudogap approximation.

    Parameters
    ----------
    G : float
    vo : float
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
    #assert z(re, rd, vo)>0
    def cost(lk):
        rd = lk
        #print(vo, rd)
        nl_star = (vo * (rd+ vo*(1+gamma*(k-1))-2*r) )/(r*I*(rd+(1+gamma*(k-1))*vo))
        #print(vo)
        return  ((rd - r + vo*vo * (1+gamma*(k-1)) * nl_star /G))**2
    sol = minimize(cost, 0.3, tol=1e-10, bounds=[(0,np.inf)])
    return sol['x']

def rd_crit_quadratic(r0 , I, vo, r, gamma, k, initial_guess=-1):
    """Critical death rate rd as a function of other parameters by solving
    the discriminant. (Given compartment model?)

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

def _K_critical_quadratic(r0, I, vo, r, rd, gamma, initial_guess=300, full_output=False):
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
    if r < rd: return np.nan

    def cost(K):
        k = 1 + gamma*(K-1)
        a = (rd/r - 1)*I
        b = (rd/r - 1)*I*r0/r + (vo / r)**3 * k**2 - (r0/r * I - vo/r)
        c = (vo/r - r0/r * I) * r0/r + (vo / r)**2 * k * r0/r 
        return  (b**2 - 4*a*c)**2
    sol = minimize(cost, initial_guess, tol=1e-10, bounds=[(1, np.inf)])

    if full_output:
        return sol
    elif sol['x'][0]==initial_guess or sol['fun']>1e-6:
        return np.nan
    return sol['x'][0]

def K_critical_quadratic(r0, I, vo, r, rd, gamma, initial_guess=300, full_output=False):
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
    if r < rd: return np.nan
    r0 /= r
    vo /= r
    rd /= r

    def cost(K):
        k = 1 + gamma*(K-1)
        a = (rd - 1)*I
        b = r0*rd*I - 2*r0*I + vo + vo**4*k**2
        c = r0*rd*I - r0*I - r0**2*I + r0*vo + r0*vo**3*k
        return  (b**2 - 4*a*c)**2
    sol = minimize(cost, initial_guess, tol=1e-10, bounds=[(1, np.inf)])

    if full_output:
        return sol
    elif sol['x'][0]==initial_guess or sol['fun']>1e-6:
        return np.nan
    return sol['x'][0]