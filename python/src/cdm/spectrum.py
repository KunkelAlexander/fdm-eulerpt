from numba import njit 
import constants as c 
import special_functions as sf 
import fdm.util
from initial_spectrum import P_0
from cdm.growth.analytical import D

#Linear CDM time evolution
@njit
def P_CDM_IC(k, *args):
  eta = args[0]
  k = sf.k_norm(k)
  return  P_0(k) * D(eta)**2


#Linear CDM time evolution
@njit
def P_FDM_IC(k, *args):
  eta = args[0]
  k = sf.k_norm(k)
  return  P_0(k) * fdm.util.FDM_transfer(k, c.FDM_M)**2 * D(eta)**2

#Define scale-free initial spectrum for CDM time evolution
@njit
def P_sc(k, *args):
  eta = args[0]
  n   = -2.

  kn = sf.k_norm(k)
  if kn < c.EPS:
    return 0
  return kn**n * D(eta)**2

#Define scale-free initial spectrum for CDM time evolution
@njit
def P_n(k, *args):
  eta, n = args

  kn = sf.k_norm(k)
  if kn < c.EPS:
    return 0

  return kn**n * D(eta)**2