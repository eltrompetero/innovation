# ====================================================================================== #
# Minimal innovation model implementations and solutions.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from cmath import sqrt
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
    r0 : float
    I : float
    r : float
    rd : float
    vo : float
    gamma : float
    K : int
        Branching number.
     
    Returns
    -------
    dydt : floats
        Derivatives of variables.
    """
    N, L, n0, nl = y0
    k = 1 + gamma * (K - 1) + gamma * K * solve_lambda(gamma)

    # assume system collapses fully if it falls below these limits
    if L<=2 or N<=0:
        dN = -N
        dL = -L
        dn0 = -n0
        dnl = -nl
    else:
        dN = r0 - rd*N + r*(N-n0) - vo*k*nl
        dL = r*I*k*n0 - vo*k
        dn0 = r0/L - rd*n0 + r*(N-n0-nl)/(L-2) - r*I*k*n0*n0
        dnl = r0/L - rd*nl - vo*k*(nl - (N-n0-nl)/(L-2))

    return np.array([dN, dL, dn0, dnl])

def critical_rd(r0, I, gamma, K):
    """Define function for returning curve of critical rd as a function of vo.

    See Mathematica notebook 20241129_derivation.nb for derivation.

    Parameters
    ----------
    r0 : float
        Rescaled birth rate.
    I : float
    gamma : float
    K : int
        Branching number.

    Returns
    -------
    function
        Takes vo as input and returns critical rd.
    """
    # Define function for root finding
    lam = solve_lambda(gamma)
    k = 1 + gamma*(K-1) + gamma*K*lam
    naive_rd = lambda vo: (I*r0 + 3*vo - k*vo**2 - np.sqrt(((-I*r0 - 3*vo + k*vo**2)**2 -
                           8*vo*(-k*vo**2 - k**2*vo**3 + 2*np.sqrt(I*r0*vo + I*k**2*r0*vo**3)))))/(4*vo)
    vo_star = minimize(lambda vo: (naive_rd(vo) - 1)**2, .01, tol=1e-10)['x'][0]

    def f(vo):
        if vo < vo_star:
            return 1.
        return naive_rd(vo)
    return np.vectorize(f)

class CompartmentModel:
    def __init__(self, r0=None, I=None, rd=None, vo=None, gamma=None, K=None):
        """Steady-state properties of compartment model including state
        variables and phase boundaries.
        """
        self.r0 = r0
        self.I = I
        self.rd = rd
        self.vo = vo
        self.gamma = gamma
        self.K = K

    def N(self, r0=None, I=None, rd=None, vo=None, gamma=None, K=None):
        """Steady state solution for N, total number of agents per branch for compartment model."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        gamma = gamma if gamma is not None else self.gamma
        K = K if K is not None else self.K
        k = 1 + gamma*(K-1) + gamma*K*solve_lambda(gamma)
        
        A = I * (-1 + rd) * (rd + k * vo) * (-1 + 2 * rd**2 - k**2 * vo**2 + rd * (-1 + k * vo))
        B = (-2 * I * r0 * rd - 2 * I * r0 * rd**2 + 4 * I * r0 * rd**3 - 2 * I * k * r0 * vo +
             2 * rd * vo - 2 * I * k * r0 * rd * vo + 2 * rd**2 * vo + 5 * I * k * r0 * rd**2 * vo -
             4 * rd**3 * vo + 2 * k * vo**2 + 2 * k * rd * vo**2 - I * k**2 * r0 * rd * vo**2 -
             3 * k * rd**2 * vo**2 + I * k**2 * r0 * rd**2 * vo**2 - 2 * k * rd**3 * vo**2 -
             2 * I * k**3 * r0 * vo**3 + 6 * k**2 * rd * vo**3 + I * k**3 * r0 * rd * vo**3 -
             4 * k**2 * rd**2 * vo**3 - 2 * k**2 * rd**3 * vo**3 + 5 * k**3 * vo**4 - 2 * k**3 * rd * vo**4 -
             3 * k**3 * rd**2 * vo**4 + k**5 * vo**6)
        C = ((rd + k * vo)**2 * (4 * I * r0 * (-1 + rd) * vo * (1 + rd - 2 * rd**2 - k * rd * vo + k**2 * vo**2) +
                                 (I * r0 * rd - vo * (-2 * rd**2 + rd * (3 - k * vo) + k * vo * (1 + k * vo)))**2))
        quadform = (B + k * vo * sqrt(C) - k**2 * vo**2 * sqrt(C)) / (2 * A)
        return quadform, A, B, C

    def L(self, r0=None, I=None, rd=None, vo=None, gamma=None, K=None):
        """Steady state solution for length of lattice along each branch for compartment model."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        gamma = gamma if gamma is not None else self.gamma
        K = K if K is not None else self.K
        k = 1 + gamma*(K-1) + gamma*K*solve_lambda(gamma)

        A = (-1 + rd) * vo * (rd + k * vo)**2
        B = (I * r0 * rd**2 + I * k * r0 * rd * vo - 3 * rd**2 * vo + 2 * rd**3 * vo -
            4 * k * rd * vo**2 + 3 * k * rd**2 * vo**2 - k**2 * vo**3 - k**3 * vo**4)
        C = (rd + k * vo)**2 * (4 * I * r0 * (-1 + rd) * vo * (1 + rd - 2 * rd**2 - k * rd * vo + k**2 * vo**2) +
                                (I * r0 * rd - vo * (-2 * rd**2 + rd * (3 - k * vo) + k * vo * (1 + k * vo)))**2) 
        quadform = (B + sqrt(C)) / (2 * A)
        return quadform, A, B, C

    @classmethod
    def N_as_fun(cls, r0, I, rd, vo, gamma, K):
        """Steady state solution for N, total number of agents per branch for
        compartment model."""
        k = 1 + gamma*(K-1) + gamma*K*solve_lambda(gamma)
        
        A = I * (-1 + rd) * (rd + k * vo) * (-1 + 2 * rd**2 - k**2 * vo**2 + rd * (-1 + k * vo))
        B = (-2 * I * r0 * rd - 2 * I * r0 * rd**2 + 4 * I * r0 * rd**3 - 2 * I * k * r0 * vo +
             2 * rd * vo - 2 * I * k * r0 * rd * vo + 2 * rd**2 * vo + 5 * I * k * r0 * rd**2 * vo -
             4 * rd**3 * vo + 2 * k * vo**2 + 2 * k * rd * vo**2 - I * k**2 * r0 * rd * vo**2 -
             3 * k * rd**2 * vo**2 + I * k**2 * r0 * rd**2 * vo**2 - 2 * k * rd**3 * vo**2 -
             2 * I * k**3 * r0 * vo**3 + 6 * k**2 * rd * vo**3 + I * k**3 * r0 * rd * vo**3 -
             4 * k**2 * rd**2 * vo**3 - 2 * k**2 * rd**3 * vo**3 + 5 * k**3 * vo**4 - 2 * k**3 * rd * vo**4 -
             3 * k**3 * rd**2 * vo**4 + k**5 * vo**6)
        C = ((rd + k * vo)**2 * (4 * I * r0 * (-1 + rd) * vo * (1 + rd - 2 * rd**2 - k * rd * vo + k**2 * vo**2) +
                                 (I * r0 * rd - vo * (-2 * rd**2 + rd * (3 - k * vo) + k * vo * (1 + k * vo)))**2))
        quadform = (B + k * vo * sqrt(C) - k**2 * vo**2 * sqrt(C)) / (2 * A)
        return quadform, A, B, C

    @classmethod
    def L_as_fun(cls, r0, I, rd, vo, gamma, K):
        """Steady state solution for length of lattice along each branch for compartment model."""
        k = 1 + gamma*(K-1) + gamma*K*solve_lambda(gamma)

        A = (-1 + rd) * vo * (rd + k * vo)**2
        B = (I * r0 * rd**2 + I * k * r0 * rd * vo - 3 * rd**2 * vo + 2 * rd**3 * vo -
            4 * k * rd * vo**2 + 3 * k * rd**2 * vo**2 - k**2 * vo**3 - k**3 * vo**4)
        C = (rd + k * vo)**2 * (4 * I * r0 * (-1 + rd) * vo * (1 + rd - 2 * rd**2 - k * rd * vo + k**2 * vo**2) +
                                (I * r0 * rd - vo * (-2 * rd**2 + rd * (3 - k * vo) + k * vo * (1 + k * vo)))**2) 
        quadform = (B + sqrt(C)) / (2 * A)
        return quadform, A, B, C
    
    def gamma_runaway(self, r0=None, I=None, rd=None, vo=None, K=None):
        """Critical gamma delineating runaway boundary."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        K = K if K is not None else self.K

        def cost(loggamma):
            if loggamma>0:
                return 1e10
            gamma = np.exp(loggamma)[0]
            lam = solve_lambda(gamma)
            #return ((0.5 * (-K * vo * (-rd + 2 * vo -rd * lam + 2 * vo * lam)
            #               -np.sqrt(K**2 * vo**2 * (-rd + 2 * vo -rd * lam + 2 * vo * lam)**2 -
            #                        4 * K**2 * vo**2 * (1 + rd - 2 * rd**2 -rd * vo + vo**2) * (1 + 2 * lam + lam**2)))) /
            #                        (K**2 * vo**2 * (1 + 2 * lam + lam**2)) - gamma)**2
            return ((.5 * (-K * vo * (-rd + 2 * vo -rd * lam + 2 * vo * lam) +
                           np.sqrt(K**2 * vo**2 * (-rd + 2 * vo -rd * lam + 2 * vo * lam)**2 -
                                   4 * K**2 * vo**2 * (1 + rd - 2 * rd**2 -rd * vo + vo**2) * (1 + 2 * lam + lam**2)))) /
                                   (K**2 * vo**2 * (1 + 2 * lam + lam**2)) - gamma)**2
        sol = minimize(cost, -1)
        if sol['fun']>1e-5:
            return np.nan
        return np.exp(sol['x'])[0]
        return ((2 - 2 * K - rd / vo + (K * rd) / vo + ((-1 + K) *
                sqrt(-4 - 4 * rd + 9 * rd**2)) / vo) / (2 * (1 - 2 * K + K**2)))
    
    def gamma_collapse(self, r0=None, I=None, rd=None, vo=None, K=None):
        """Solve for critical gamma delineating collapse boundary."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        K = K if K is not None else self.K

        def cost(loggamma):
            gamma = np.exp(loggamma)[0]
            return np.abs(self.L(r0, I, rd, vo, gamma, K)[0] - 2)**2
        return np.exp(minimize(cost, 0.)['x'])[0]
    
def solve_lambda(gamma, g0=-1.):
    if hasattr(gamma, '__len__'):
        lam = np.zeros_like(gamma)
        for i, gamma_ in enumerate(gamma):
            def cost(loglam):
                lam = np.exp(loglam)
                p0 = np.exp(-lam)
                p1 = lam*np.exp(-lam)
                return (np.exp(-lam)/(1+lam)-gamma_)**2

            sol = minimize(cost, g0)
            lam[i] = np.exp(sol['x'])
        return lam

    def cost(loglam):
        lam = np.exp(loglam)
        p0 = np.exp(-lam)
        p1 = lam*np.exp(-lam)
        return (np.exp(-lam)/(1+lam)-gamma)**2
    
    sol = minimize(cost, g0)
    lam = np.exp(sol['x'])[0]
    return lam
