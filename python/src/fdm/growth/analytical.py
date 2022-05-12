import constants as c 
from fdm.util import b_f, mbH0
from special_functions import jv, jvp, jv_taylor, k_norm, test_jv
from numba import njit
import numpy as np 

#Renormalised growth factor without unphysical divergences
@njit
def D(k, eta, m):
    p1 = np.sqrt(c.ETA_IN/eta)
    bk = b_f(k, m)
    p2 = jv(-2.5, bk/eta)
    #Using taylor expansion in denominator eliminates divergencies and fast oscillations
    #This is a physical approximation eq. (27) in Lagüe 2020
    p3 = jv_taylor(2.5, bk/c.ETA_IN)
    return p1 * p2 / p3

#Derivative of renormalised growth factor
@njit
def dD(k, eta, m):
    bk = b_f(k, m)
    p1 = np.sqrt(c.ETA_IN/eta)
    # Contribution from chain rule
    p2 = (-bk)/eta**2 
    #Derivative of Bessel function
    p3 = jvp(-2.5, bk/eta) 
    p4 = jv_taylor(2.5, bk/c.ETA_IN)
    p5 = jv(-2.5, bk/eta)
    return p1 * p2 * p3 / p4 - 1/2 * p1 / eta * p5 / p4
    

#Linear FDM time evolution
#according to analytical solution of ODE governing time evolution
#exhibits non-physical divergences at roots of p3
def D_plus(k, a, a_in, m):
    eta    = c.eta_from_a(a)
    eta_in = c.eta_from_a(a_in)
    p1     = np.sqrt(eta_in/eta)
    bk     = b_f(k, m)
    p2     = jv(-2.5, bk/eta)
    p3     = jv(-2.5, bk/eta_in)
    return p1 * p2 / p3



#Derivative of D with respect to eta
def dD_plus(k, a, a_in, m):
    eta    = c.eta_from_a(a)
    eta_in = c.eta_from_a(a_in)
    bk = b_f(k, m)
    p1 = np.sqrt(eta_in/eta)
    p2 = (-bk)/eta**2 # Contribution from chain rule
    p3 = jvp(-2.5, bk/eta) #Derivative of Bessel function
    p4 = jv(-2.5, bk/eta_in)
    p5 = jv(-2.5, bk/eta)
    return 1/np.sqrt(a)*(p1 * p2 * p3 / p4 - 1/2 * p1 / eta * p5 / p4)


def ddD_plus(k, a, a_in, m):
    da = 1e-8 
    return (dD_plus(k, a + da, a_in, m) - dD_plus(k, a - da, a_in, m))/(2*da)

def D_minus(k, a, a_in, m):
    eta    = c.eta_from_a(a)
    eta_in = c.eta_from_a(a_in)
    p1 = np.sqrt(eta_in/eta)
    bk = b_f(k, m)
    p2 = jv(2.5, bk/eta)
    p3 = jv(2.5, bk/eta_in)
    return p1 * p2 / p3

#Derivative of D with respect to eta
def dD_minus(k, a, a_in, m):
    eta    = c.eta_from_a(a)
    eta_in = c.eta_from_a(a_in)
    bk = b_f(k, m)
    p1 = np.sqrt(eta_in/eta)
    p2 = (-bk)/eta**2 # Contribution from chain rule
    p3 = jvp(2.5, bk/eta) #Derivative of Bessel function
    p4 = jv(2.5, bk/eta_in)
    p5 = jv(2.5, bk/eta)
    return 1/np.sqrt(a)*(p1 * p2 * p3 / p4 - 1/2 * p1 / eta * p5 / p4)

def ddD_minus(k, a, a_in, m):
    da = 1e-8 
    return (dD_minus(k, a + da, a_in, m) - dD_minus(k, a - da, a_in, m))/(2*da)

##Growth factor is given in Lague for comparison
def D_lague(k, a, a_in, m):
    return (a_in/a)**.25 * test_jv(k**2/mbH0(m) / np.sqrt(a))/test_jv(k**2/mbH0(m) / np.sqrt(a_in))


