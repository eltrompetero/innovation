# ====================================================================================== #
# Minimal innovation model implementations and solutions.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from cmath import sqrt
from functools import cache
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
    vo *= vo_tilde_coefficient(gamma, K)
    I *= I_tilde_coefficient(gamma, K)

    # assume system collapses fully if it falls below these limits
    if L<=2 or N<=0:
        dN = -N
        dL = -L
        dn0 = -n0
        dnl = -nl
    else:
        dN = r0 - rd*N + r*(N-n0) - vo*nl
        dL = r*I*n0 - vo
        dn0 = r0/L - rd*n0 + r*(N-n0-nl)/(L-2) - r*I*n0*n0
        dnl = r0/L - rd*nl - vo*(nl - (N-n0-nl)/(L-2))

    return np.array([dN, dL, dn0, dnl])

def collapse_rd(vo, r0, I, gamma, K):
    """Define function for returning curve of collapse rd as a function of vo.
    """
    vo *= vo_tilde_coefficient(gamma, K)
    I *= I_tilde_coefficient(gamma, K)
    raise Exception("there is some sign error here. Check mathematica notebook carefully.")
    return ((-4 * vo**3 + I * r0 * (1 + vo) + np.sqrt(16 * vo**4 + 8 * I * r0 * vo *
            (1 + vo) + (I**2) * r0**2 * (1 + vo)**2)) / (4 * vo * (1 + vo)))
    
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
    lam = solve_obs_lambda(gamma)
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

        # corrections
        vo *= vo_tilde_coefficient(gamma, K) 
        I *= I_tilde_coefficient(gamma, K)

        A = I * (-1 + rd) * (rd + vo) * (-1 + 2 * rd**2 + rd * (-1 + vo) - vo**2)
        B = I * r0 * (4 * rd**3 + rd**2 * (-2 + 5 * vo + vo**2) - 2 * (vo + vo**3) + rd * (-2 - 2 * vo - vo**2 + vo**3)) + vo * (2 * vo + 5 * vo**3 + vo**5 - 2 * rd**3 * (2 + vo + vo**2) + rd * (2 + 2 * vo + 6 * vo**2 - 2 * vo**3) - rd**2 * (-2 + 3 * vo + 4 * vo**2 + 3 * vo**3))
        C = (rd + vo)**2 * (I**2 * r0**2 * rd**2 + vo**2 * (3 * rd - 2 * rd**2 + vo - rd * vo + vo**2)**2 + 2 * I * r0 * vo * (-2 * rd**3 - rd**2 * (-3 + vo) + rd * vo * (1 + vo) - 2 * (1 + vo**2)))
        D = vo * (1 - vo) 
        quadform = (B + D * sqrt(C)) / (2 * A)
        return quadform, A, B, C

    def L(self, r0=None, I=None, rd=None, vo=None, gamma=None, K=None):
        """Steady state solution for length of lattice along each branch for compartment model."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        gamma = gamma if gamma is not None else self.gamma
        K = K if K is not None else self.K

        # corrections
        vo *= vo_tilde_coefficient(gamma, K) 
        I *= I_tilde_coefficient(gamma, K)

        A = (-1 + rd) * vo * (rd + vo)**2
        B = -3 * rd**2 * vo + 2 * rd**3 * vo - 4 * rd * vo**2 + 3 * rd**2 * vo**2 - vo**3 - vo**4 + I * r0 * rd * (rd + vo)
        C = (rd + vo)**2 * (I**2 * r0**2 * rd**2 + vo**2 * (3 * rd - 2 * rd**2 + vo - rd * vo + vo**2)**2 + 2 * I * r0 * vo * (-2 * rd**3 - rd**2 * (-3 + vo) + rd * vo * (1 + vo) - 2 * (1 + vo**2))) 
        quadform = (B + sqrt(C)) / (2 * A)
        return quadform, A, B, C

    def gamma_runaway(self, r0=None, I=None, rd=None, vo=None, K=None):
        """Critical gamma delineating runaway boundary."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        K = K if K is not None else self.K

        vo *= vo_tilde_coefficient(1, K)
        I *= I_tilde_coefficient(1, K)

        def cost(loggamma):
            gamma = np.exp(loggamma)[0]
            if gamma>1: return 1e10
            return np.abs(self.L(r0, I, rd, vo, gamma, K)[0])**2
        return np.exp(minimize(cost, -1.)['x'])[0]

        def cost(loggamma):
            gamma = np.exp(loggamma)[0]
            if gamma>1: return 1e10
            lam = solve_obs_lambda(gamma)
            return ((-2 * K + (K * rd) / vo - 2 * K * lam + (K * rd * lam) / vo + (K * np.sqrt(-4
                    - 4 * rd + 9 * rd**2) * (1 + lam)) / vo) / (2 * (K**2 + 2 * K**2 * lam + K**2 *
                    lam**2)) - gamma)**2
        sol = minimize(cost, -1)
        if sol['fun']>1e-5:
            return np.nan
        return np.exp(sol['x'])[0]
    
    def gamma_collapse(self, r0=None, I=None, rd=None, vo=None, K=None):
        """Solve for critical gamma delineating collapse boundary."""
        r0 = r0 if r0 is not None else self.r0
        I = I if I is not None else self.I
        rd = rd if rd is not None else self.rd
        vo = vo if vo is not None else self.vo
        K = K if K is not None else self.K

        vo *= vo_tilde_coefficient(1, K)
        I *= I_tilde_coefficient(1, K)

        def cost(loggamma):
            gamma = np.exp(loggamma)[0]
            if gamma>1: return 1e10
            return np.abs(self.L(r0, I, rd, vo, gamma, K)[0] - 2)**2
        return np.exp(minimize(cost, -1.)['x'])[0]

@cache
def solve_obs_lambda(gamma, g0=-1.):
    """Solve for average distance of obsolescence front from leading one.
    
    Parameters
    ----------
    gamma : float
        Connectivity.
    g0 : float, -1.
        Initial guess for optimization.
    
    Returns
    -------
    float
        Average distance of obsolescence front from leading one.
    """
    if hasattr(gamma, '__len__'):
        assert np.all(0<=gamma) and np.all(gamma<=1)
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

    assert 0<=gamma<=1
    def cost(loglam):
        lam = np.exp(loglam)
        p0 = np.exp(-lam)
        p1 = lam*np.exp(-lam)
        return (np.exp(-lam)/(1+lam)-gamma)**2
    
    sol = minimize(cost, g0)
    lam = np.exp(sol['x'])[0]
    return lam

@cache
def solve_inn_lambda(gamma, use_x1=False):
    """Solve for average distance of innovation front from leading one.

    Parameters
    ----------
    gamma : float
        Connectivity.
    use_x1 : bool, True
        Whether to use x=1 condition as the leading innovation front.

    Returns
    -------
    float
        Average distance of innovation front from leading one.
    """
    assert 0<=gamma<=1

    def cost(loglam):
        lam = np.exp(loglam)
        # this are inconsistent conditions, naturally b/c poisson is an assumption!
        # but x=1 is slightly better when compared to samples
        if use_x1:
            term = (1-gamma) + lam**2/2 * (1-np.exp(-lam))*(1-np.exp(-lam)*lam)*gamma - lam
        else:
            term = gamma + gamma*lam*(1-np.exp(-lam)) - np.exp(-lam)
        return term**2
    
    sol = minimize(cost, 0.)
    lam = np.exp(sol['x'])[0]
    return lam

@cache
def I_tilde_coefficient(gamma, K, mx_x=100):
    """For solving for the correction to innovation front velocity.

    Parameters
    ----------
    gamma : float
        Connectivity.
    K : int
        Branching number.
    mx_x : int, 100
        Max (inclusive) value of x to which to calculate Poisson distribution.

    Returns
    -------
    float
        Correction factor to innovativeness. Multiply this to I to obtain Itilde in paper.
    """
    assert 0<=gamma<=1 and K>=1 and mx_x>=10

    mx_x += 1
    if gamma==0:
        return 1.
    if gamma==1:
        return K
        
    lam = solve_inn_lambda(gamma)
    p = poisson(np.arange(mx_x), lam)

    term1 = 0.
    term2 = 0.
    for x in range(mx_x):
        term1 += p[x]**2 * np.prod([1-p[xp] for xp in range(x)])
    for x in range(mx_x):
        term2 += p[x] * np.prod([1-p[xp] for xp in range(x)]) * sum([p[xpp] * (xpp-x) for xpp in range(x+1, mx_x)])

    return term1 * (1 + gamma * (K-1)) + term2 * gamma * K

@cache
def vo_tilde_coefficient(gamma, K, mx_x=100):
    """For solving for the correction to obsolescence front velocity.
    """
    assert 0<=gamma<=1 and mx_x>=10
    mx_x += 1
    i_range = np.arange(mx_x)

    # correction to obsolescence
    lam = solve_obs_lambda(gamma)
    pk = poisson(i_range, lam)
    correction = sum([pk[i]*(pk[:i+1]*(i-np.arange(i+1))).sum() for i in i_range]) * K * gamma
    return (1+gamma*(K-1)) + correction
