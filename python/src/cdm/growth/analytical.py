import constants as c
from numba import njit

#CDM growth factor for matter dominated universe with Omega_m = 1
#corresponds to D = a/a_in
@njit
def D(eta):
    return (eta/c.ETA_IN)**2

@njit
def dD(eta):
    return 2*eta/c.ETA_IN**2

@njit
def D_plus(k, a, a_in, m = 0):
    return a/a_in
    
def D_plus_nnjit(k, a, a_in, m = 0):
    return a/a_in

@njit
def dD_plus(k, a, a_in, m = 0):
    return 1/a_in
  
@njit  
def ddD_plus(k, a, a_in, m = 0):
    return 0
   
@njit 
def D_minus(k, a, a_in, m = 0):
    return a**(-3/2) / (a_in)**(-3/2)
    
def D_minus_nnjit(k, a, a_in, m = 0):
    return a**(-3/2) / (a_in)**(-3/2)

@njit
def dD_minus(k, a, a_in, m = 0):
    return a**(-5/2) * (-3/2) / (a_in)**(-3/2)
    
@njit
def ddD_minus(k, a, a_in, m = 0):
    return a**(-7/2) * (-3/2) * (-5/2) / (a_in)**(-3/2)