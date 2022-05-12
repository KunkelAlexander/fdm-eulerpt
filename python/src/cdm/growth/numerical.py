import numpy as np

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import constants as c 
from numba import njit, objmode

import scipy.interpolate as interpolate 

def get_cdm_ic(a_in, hubble = c.hubble, dlogHubble = c.dloghubble, omega_m = c.OMEGA_M_GROWTH):
  def dD_da(a, y):
    D, dD = y
    return [dD, -(3/a + dlogHubble(a))*dD + 3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5)*D]
  def jacobian(a, y):
    D, dD = y
    return [[0, 1], [3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5), -(3/a + dlogHubble(a))]]

  print(f"Integrate CDM IC for O_m = {c.OMEGA_M_GROWTH} and H_0 = {c.H_HUBBLE} at a_in = {a_in}")

  #Initial guess
  a0 = 1e-6
  a1 = 1
  D0 = [a0, 1]

  a_s    = np.logspace(-6, 0, 1000)
  temp1  = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= "DOP853", jac = jacobian, t_eval=a_s, dense_output=True)
  dD0_p  = temp1.sol(a_in)[1]/temp1.sol(a_in)[0]

  #Initial guess
  a0 = 10
  a1 = 1e-6
  D0 = [10, -3/2]
  a_s = np.logspace(1, -6, 1000)
  #temp, info = odeint(dD_da, D0, a_s, args=(k,), Dfun=jacobian, full_output=True)
  temp2 = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= "DOP853", jac = jacobian, t_eval=a_s, dense_output=True)
  D0_m, dD0_m  = np.array(temp2.sol(1))/temp2.sol(a_in)[0]

  return dD0_p, D0_m, dD0_m

def numerical_CDM_D(a_in = c.A_IN, debug = False, hubble = c.hubble, dlogHubble = c.dloghubble, omega_m = c.OMEGA_M_GROWTH):
  def dD_da(a, y):
    D, dD = y
    return [dD, -(3/a + dlogHubble(a))*dD + 3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5)*D]
  def jacobian(a, y):
    D, dD = y
    return [[0, 1], [3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5), -(3/a + dlogHubble(a))]]

  ### COMPUTATION OF GROWING MODE ###
  print(f"Integrate CDM D+ for O_m = {c.OMEGA_M_GROWTH} and H_0 = {c.H_HUBBLE} at a_in = {a_in}")

  a_s = c.SPLINE_SCALE_FACTORS.copy()
  a0  = c.A0_INTEGRATION
  a1  = 1 
  dDp0, Dm0, dDm0 = get_cdm_ic(a0, hubble, dlogHubble, omega_m)

  D0 = [1, dDp0]
  temp1 = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= "DOP853", jac = jacobian, t_eval=a_s, dense_output=True)

  if debug:
    print("Initial conditions D", D0[0], "dD", D0[1])
    print("nfev ", temp1.nfev, "njev", temp1.njev, "status", temp1.status)

  D1  = temp1.y[0, :] #Normalise such that D(a=1) = 1
  dD1 = temp1.y[1, :]
  
  ### COMPUTATION OF DECAYING MODE ###
  print(f"Integrate CDM D- for O_m = {c.OMEGA_M_GROWTH} and H_0 = {c.H_HUBBLE} at a_in = {a_in}")

  a0 = 1
  a1 = 1e-5
  D0 = [Dm0, dDm0]
  a_s = np.flip(a_s)
  temp2 = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= "DOP853", jac = jacobian, t_eval=a_s, dense_output=True)

  if debug:
    print("Initial conditions D", D0[0], "dD", D0[1])
    print("nfev ", temp2.nfev, "njev", temp2.njev, "status", temp2.status)
  D2 =  np.flip(temp2.y[0, :])#/temp2.sol(a_in)[0] #Normalise such that D(a=1) = 1
  dD2 = np.flip(temp2.y[1, :])#/temp2.sol(a_in)[0]
  a_s = np.flip(a_s)

  return a_s, D1, dD1, D2, dD2, temp1, temp2

a_s, D1, dD1, D2, dD2, temp1, temp2 = numerical_CDM_D()
D_plus_spline   = interpolate.interp1d(c.SPLINE_SCALE_FACTORS,  D1, kind='cubic', fill_value="extrapolate")
D_minus_spline  = interpolate.interp1d(c.SPLINE_SCALE_FACTORS,  D2, kind='cubic', fill_value="extrapolate")
dD_plus_spline  = interpolate.interp1d(c.SPLINE_SCALE_FACTORS, dD1, kind='cubic', fill_value="extrapolate")
dD_minus_spline = interpolate.interp1d(c.SPLINE_SCALE_FACTORS, dD2, kind='cubic', fill_value="extrapolate")

da = 1e-8

@njit
def check_range(a, a_in):
  if (5e-6 > a_in) or (5e-6 > a):
    raise ValueError("Spline inaccurate below a = 1e-5")

  if (1.1 < a_in) or (1.1 < a):
    raise ValueError("Spline inaccurate above a = 1")

@njit
def D_plus  (k, a, a_in, m = 0):
    check_range(a, a_in)
    with objmode(y='float64'):
      y = D_plus_spline(a)/D_plus_spline(a_in)
    return y

def D_plus_nnjit  (k, a, a_in, m = 0):
    with objmode(y='float64'):
      y = D_plus_spline(a)/D_plus_spline(a_in)
    return y

@njit
def dD_plus (k, a, a_in, m = 0):
    check_range(a, a_in)

    with objmode(y='float64'):
      y = dD_plus_spline(a)/D_plus_spline(a_in)
    return y

@njit
def ddD_plus (k, a, a_in, m = 0):
    check_range(a, a_in)

    with objmode(y='float64'):
      ddD = (dD_plus_spline(a + da) - dD_plus_spline(a - da))/(2*da)
      y = ddD/D_plus_spline(a_in)
    return y

@njit
def D_minus (k, a, a_in, m = 0):
    check_range(a, a_in)

    with objmode(y='float64'):
      y = D_minus_spline(a)/D_minus_spline(a_in)
    return y

def D_minus_nnjit (k, a, a_in, m = 0):
    with objmode(y='float64'):
      y = D_minus_spline(a)/D_minus_spline(a_in)
    return y

@njit
def dD_minus(k, a, a_in, m = 0):
    check_range(a, a_in)

    with objmode(y='float64'):
      y = dD_minus_spline(a)/D_minus_spline(a_in)

    return y


@njit
def ddD_minus (k, a, a_in, m = 0):

    with objmode(y='float64'):
      ddD = (dD_minus_spline(a + da) - dD_minus_spline(a - da))/(2*da)
      y = ddD/D_minus_spline(a_in)
    return y
