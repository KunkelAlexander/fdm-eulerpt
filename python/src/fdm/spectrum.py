
from numba import njit, objmode
import numpy as np

import constants
from special_functions import k_norm
from initial_spectrum import P_0
import fdm.util 

import scipy.interpolate as interpolate

if constants.LINEAR_GROWTH_MODE is constants.LinearGrowthModes.ANALYTICAL_FDM:
    from fdm.growth.analytical import D, dD
    import cdm.growth.analytical as cdg
elif constants.LINEAR_GROWTH_MODE is constants.LinearGrowthModes.SUPPRESSED_CDM:
    from fdm.growth.suppressed_cdm import D, dD
    import cdm.growth.analytical as cdg
elif constants.LINEAR_GROWTH_MODE is constants.LinearGrowthModes.NUMERICAL_FIT:
    from fdm.growth.suppressed_cdm import D, dD
    import cdm.growth.analytical as cdg
elif constants.LINEAR_GROWTH_MODE is constants.LinearGrowthModes.NUMERICAL_FIT:
    from fdm.growth.suppressed_cdm import D, dD
    import cdm.growth.analytical as cdg
else:
    raise ValueError(f"Unsupported linear growth mode {constants.LINEAR_GROWTH_MODE}.")

#Define scale-free initial spectrum for FDM time evolution
@njit
def P_sc(k, *args):
  eta, m = args
  n   = -2.
  kn = k_norm(k)
  if kn < constants.EPS:
    return 0

  return kn**n * D(k, eta, m)**2

#Define scale-free initial spectrum for FDM time evolution
@njit
def P_n(k, *args):
  eta, m, n = args

  kn = k_norm(k)
  if kn < constants.EPS:
    return 0

  return kn**n * D(k, eta, m)**2
  

#P_FDM(k) = FDM_transfer(k)**2* P_CDM(k)
#Linear FDM time evolution
@njit
def P_CDM_IC(k, *args):
  eta, m = args
  k      = k_norm(k)
  return  P_0(k) * D(k, eta, m)**2


#P_FDM(k) = FDM_transfer(k)**2* P_CDM(k)
#Linear FDM time evolution
@njit
def P_FDM_IC(k, *args):
  eta, m = args
  k      = k_norm(k)
  return  P_0(k) * fdm.util.FDM_transfer(k, m)**2 * D(k, eta, m)**2



#P_FDM(k) = FDM_transfer(k)**2* P_CDM(k)
#Linear FDM time evolution
@njit
def P_CDM_FDM_IC(k, *args):
  eta, m = args
  k      = k_norm(k)
  return  P_0(k) * fdm.util.FDM_transfer(k, m)**2 * cdg.D(k, eta)**2



k_vec, P_vec = np.loadtxt(constants.CAMB_SPECTRUM_PATH, unpack=True)

P0x   = interpolate.interp1d(k_vec,  P_vec, kind='cubic', fill_value="extrapolate")
minkh  = k_vec[0]
maxkh  = k_vec[-1]
dk     = 1e-8
alphal = (np.log(P0x(minkh + dk)) - np.log(P_vec[0]))/(np.log(minkh + dk) - np.log(minkh))
coeffl = P_vec[0] / minkh**alphal
alphah = -5
coeffh = P_vec[-1] / maxkh**alphah

@njit
def P_axionCAMB_0(k):
  k      = k_norm(k)

  if (k > minkh) and (k < maxkh):
    with objmode(y='float64'):
      y = P0x(k)
    return y

  elif (k < minkh):
    return coeffl*k**alphal

  elif (k > maxkh):
    return coeffh*k**alphah

  else:
    raise ValueError("Error in P_spline") 

@njit
def P_axionCAMB(k, *args):
  eta, m = args
  k      = k_norm(k)
  return P_axionCAMB_0(k) * D(k, eta, m)**2