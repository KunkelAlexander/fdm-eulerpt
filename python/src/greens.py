import numpy as np
from numba import njit 

from constants import a_from_eta 

#All terms depending on a_in cancel in propagators and f1, f2
a_in = 1

#This code assumes that Dp = D_plus(a), dDp = d/da D(a), Dm = D_minus(a), dDm = d/da dD_minus(a)
@njit
def propagator(k, s, eta, m, D_plus, dD_plus, D_minus, dD_minus):
  a = a_from_eta(s)
  b = a_from_eta(eta)
  
  Dp_a, dDp_a  = D_plus (k, a, a_in, m), dD_plus (k, a, a_in, m) * np.sqrt(a)
  Dm_a, dDm_a  = D_minus(k, a, a_in, m), dD_minus(k, a, a_in, m) * np.sqrt(a)
  Dp_b, dDp_b  = D_plus (k, b, a_in, m), dD_plus (k, b, a_in, m) * np.sqrt(b)
  Dm_b, dDm_b  = D_minus(k, b, a_in, m), dD_minus(k, b, a_in, m) * np.sqrt(b)

  wronskian =  Dm_a * dDp_a - dDm_a * Dp_a
  result    = (Dm_a *  Dp_b -  Dm_b * Dp_a) / wronskian
  return result

#Apply chain rule d^2f / d^2eta = d/deta (eta/2 * df/da) = 1/2 df/a + a * d^2f/da^2
@njit
def d_s_propagator(k, s, eta, m, D_plus, dD_plus, ddD_plus, D_minus, dD_minus, ddD_minus):
  a = a_from_eta(s)
  b = a_from_eta(eta)
  
  Dp_a, dDp_a    = D_plus (k, a, a_in, m), dD_plus (k, a, a_in, m) * np.sqrt(a)
  Dm_a, dDm_a    = D_minus(k, a, a_in, m), dD_minus(k, a, a_in, m) * np.sqrt(a)
  Dp_b, dDp_b    = D_plus (k, b, a_in, m), dD_plus (k, b, a_in, m) * np.sqrt(b)
  Dm_b, dDm_b    = D_minus(k, b, a_in, m), dD_minus(k, b, a_in, m) * np.sqrt(b)
  ddDp_a         = 0.5 * dD_plus (k, a, a_in, m)  + a * ddD_plus (k, a, a_in, m)
  ddDm_a         = 0.5 * dD_minus(k, a, a_in, m)  + a * ddD_minus(k, a, a_in, m)
  wronskian      =   Dm_a * dDp_a - dDm_a *  Dp_a
  result         = (dDm_a *  Dp_b -  Dm_b * dDp_a) / wronskian - (Dm_a * Dp_b - Dm_b * Dp_a) / (wronskian * wronskian) * (Dm_a * ddDp_a - ddDm_a * Dp_a)
  return result


@njit
def d_eta_propagator(k, s, eta, m, D_plus, dD_plus, D_minus, dD_minus):
  a = a_from_eta(s)
  b = a_from_eta(eta)
  
  Dp_a, dDp_a  = D_plus (k, a, a_in, m) , dD_plus (k, a, a_in, m) * np.sqrt(a)
  Dm_a, dDm_a  = D_minus(k, a, a_in, m), dD_minus (k, a, a_in, m) * np.sqrt(a)
  Dp_b, dDp_b  = D_plus (k, b, a_in, m) , dD_plus (k, b, a_in, m) * np.sqrt(b)
  Dm_b, dDm_b  = D_minus(k, b, a_in, m), dD_minus (k, b, a_in, m) * np.sqrt(b)

  wronskian =  Dm_a * dDp_a - dDm_a * Dp_a
  result    = (Dm_a * dDp_b - dDm_b * Dp_a) / wronskian
  return result

@njit
def d_s_eta_propagator(k, s, eta, m, D_plus, dD_plus, ddD_plus, D_minus, dD_minus, ddD_minus):
  a = a_from_eta(s)
  b = a_from_eta(eta)
  
  Dp_a, dDp_a    = D_plus (k, a, a_in, m), dD_plus (k, a, a_in, m) * np.sqrt(a)
  Dm_a, dDm_a    = D_minus(k, a, a_in, m), dD_minus(k, a, a_in, m) * np.sqrt(a)
  Dp_b, dDp_b    = D_plus (k, b, a_in, m), dD_plus (k, b, a_in, m) * np.sqrt(b)
  Dm_b, dDm_b    = D_minus(k, b, a_in, m), dD_minus(k, b, a_in, m) * np.sqrt(b)
  ddDp_a         = 0.5 * dD_plus (k, a, a_in, m) + a * ddD_plus (k, a, a_in, m)
  ddDm_a         = 0.5 * dD_minus(k, a, a_in, m) + a * ddD_minus(k, a, a_in, m)

  wronskian =   Dm_a * dDp_a - dDm_a *  Dp_a
  result    = (dDm_a * dDp_b - dDm_b * dDp_a) / wronskian - (Dm_a * dDp_b - dDm_b * Dp_a) / (wronskian * wronskian) * (Dm_a * ddDp_a - ddDm_a * Dp_a)
  return result 

@njit
def f1_coupling(k, s, eta, m, D_plus):
    a = a_from_eta(s)
    b = a_from_eta(eta)
    return  D_plus(k, a, a_in, m) / D_plus(k, b, a_in, m)

@njit
def f2_coupling(k, s, eta, m, D_plus, dD_plus):
    a = a_from_eta(s)
    b = a_from_eta(eta)
    return  - dD_plus(k, a, a_in, m) / D_plus(k, b, a_in, m) * np.sqrt(a)

### From numba source
#Dispatching with arguments that are functions has extra overhead.
#If this matters for your application, you can also use a factory function 
#to capture the function argument in a closure:
def make_greens_utilities(D_plus, dD_plus, ddD_plus, D_minus, dD_minus, ddD_minus):
    @njit
    def G(k, s, eta, m):
        return propagator        (k = k, s = s, eta = eta, m = m, D_plus = D_plus, dD_plus = dD_plus, D_minus = D_minus, dD_minus = dD_minus)

    @njit
    def ds_G(k, s, eta, m):
        return d_s_propagator    (k = k, s = s, eta = eta, m = m, D_plus = D_plus, dD_plus = dD_plus, ddD_plus = ddD_plus, D_minus = D_minus, dD_minus = dD_minus, ddD_minus = ddD_minus)

    @njit
    def deta_G(k, s, eta, m):
        return d_eta_propagator  (k = k, s = s, eta = eta, m = m, D_plus = D_plus, dD_plus = dD_plus, D_minus = D_minus, dD_minus = dD_minus)

    @njit
    def dseta_G(k, s, eta, m):
        return d_s_eta_propagator(k = k, s = s, eta = eta, m = m, D_plus = D_plus, dD_plus = dD_plus, ddD_plus = ddD_plus, D_minus = D_minus, dD_minus = dD_minus, ddD_minus = ddD_minus)

    @njit
    def f1(k, s, eta, m):
        return f1_coupling(k = k, s = s, eta = eta, m = m, D_plus = D_plus)

    @njit
    def f2(k, s, eta, m):
        return f2_coupling(k = k, s = s, eta = eta, m = m, D_plus = D_plus, dD_plus = dD_plus)

    return G, ds_G, deta_G, dseta_G, f1, f2 

def wronskian_determinant(k, eta, eta_in, m, D_plus, dD_plus, D_minus, dD_minus):
    a    =  a_from_eta(eta)
    a_in =  a_from_eta(eta_in)
    Dp   =  D_plus (k, a, a_in, m)
    dDp  = dD_plus (k, a, a_in, m)
    dDm  = dD_minus(k, a, a_in, m)
    Dm   =  D_minus(k, a, a_in, m)
    #print("Ana ", "Dp ", D_plus, " dDp ", dD_plus, " Dm ", D_minus, " dDm ", dD_minus)
    return Dm * dDp - Dp * dDm
