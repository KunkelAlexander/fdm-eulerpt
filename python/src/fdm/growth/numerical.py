import numpy as np

import fdm.util 
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from numba import njit, objmode
from scipy import interpolate
from special_functions import k_norm 
import cdm.growth.numerical

import constants as c 

def c_sqs(k, a, m):
  return (c.HBAR/m)**2*(c.SPEED_OF_LIGHT*k/c.MEGAPARSEC)**2/(4*a**2)

def c_sq(k, a, m):
  return c_sqs(k, a, m)/(1 + c_sqs(k, a, m))

integration_method = "Radau"

dDp0, Dm0, dDm0 = cdm.growth.numerical.get_cdm_ic(c.A_IN)

def numerical_D(k, a_in, m, debug = False, hubble = c.hubble, dlogHubble = c.dloghubble, omega_m = c.OMEGA_M_GROWTH):

  def dD_da(a, y):
    D, dD = y
    return [dD, (-(3/a + dlogHubble(a))*dD-(1/(hubble(a)**2*a**2)*c_sq(k, a, m)*(k/c.MEGAPARSEC*c.SPEED_OF_LIGHT)**2/a**2 - 3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5))*D)]

  def jacobian(a, y):
    D, dD = y
    return [[0, -1], [ -(1/(hubble(a)**2*a**2)*c_sq(k, a, m)*(k/c.MEGAPARSEC*c.SPEED_OF_LIGHT)**2/a**2 - 3/2*omega_m*c.H0_SI_HUBBLE**2/hubble(a)**2*a**(-5)),  -(3/a + dlogHubble(a)) ]]



  ### COMPUTATION OF DECAYING MODE ###
  print(f"Integrate FDM D- for O_m = {c.OMEGA_M_GROWTH} and H_0 = {c.H_HUBBLE} at a_in = {a_in}")

  ### Reverse integration from a0 = 1 to a1 = 1e-6 enables us to obtain decaying solution
  ### of ODE with high accuracy
  a0 = 1000
  a1 = c.A0_INTEGRATION
  a_s = np.flip(c.SPLINE_SCALE_FACTORS)
  D0 = [Dm0, dDm0]

  print(f"IC for FDM D-: ({D0[0]}, {D0[1]}) at a_in = {a0}")
  
  temp2 = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= "RK45", jac = jacobian, t_eval=a_s, dense_output=True)
  if debug:
    print("D-: nfev ", temp2.nfev, "njev", temp2.njev, "status", temp2.status)

  #Flip arrays because we integrated in reverse
  norm = temp2.sol(a_in)[0]

  D2 =  np.flip(temp2.y[0, :])
  dD2 = np.flip(temp2.y[1, :])


  ### COMPUTATION OF GROWING MODE ###
  print(f"Integrate FDM D+ for O_m = {c.OMEGA_M_GROWTH} and H_0 = {c.H_HUBBLE} at a_in = {a_in}")

  a0   = c.A0_INTEGRATION
  a1   = 1
  D0   = [1, dDp0]#
  print(f"IC for FDM D+: ({D0[0]}, {D0[1]}) at a_in = {a0}")

  a_s  = c.SPLINE_SCALE_FACTORS
  temp1 = solve_ivp(fun = dD_da, t_span = [a0, a1], y0 = D0, method= integration_method, jac = jacobian, t_eval=a_s, dense_output=True)

  D1  = temp1.y[0, :]
  dD1 = temp1.y[1, :]

    
  return D1, dD1, D2, dD2, temp1, temp2

def create_splines(m, a_in):
  Dp_spline_data  = []
  Dm_spline_data  = []
  dDp_spline_data = []
  dDm_spline_data = []

  if c.LOAD_SPLINE_FROM_FILE:
    try:
      with np.load(c.SPLINE_FILE) as data:
        Dp_spline_data  = data['Dp']
        dDp_spline_data = data['dDp']
        Dm_spline_data  = data['Dm']
        dDm_spline_data = data['dDm']
    except OSError:
      print("Fit new spline because no splines are stored.")
      for k in c.SPLINE_MOMENTA:
        D1, dD1, D2, dD2, temp1, temp2 = numerical_D(k = k, a_in=a_in, m=m, debug=False, dlogHubble=c.dloghubble)
        Dp_spline_data.append(D1)
        Dm_spline_data.append(D2)
        dDp_spline_data.append(dD1)
        dDm_spline_data.append(dD2)
  else:
    print("Fit new spline.")
    for k in c.SPLINE_MOMENTA:
        D1, dD1, D2, dD2, temp1, temp2 = numerical_D(k = k, a_in=a_in, m=m, debug=False, dlogHubble=c.dloghubble)
        Dp_spline_data.append(D1)
        Dm_spline_data.append(D2)
        dDp_spline_data.append(dD1)
        dDm_spline_data.append(dD2)

  if c.SAVE_SPLINE_TO_FILE:
    np.savez(c.SPLINE_FILE, Dp = Dp_spline_data, Dm = Dm_spline_data, dDp = dDp_spline_data, dDm = dDm_spline_data)
  
  log_k = np.log(c.SPLINE_MOMENTA)
  log_a = np.log(c.SPLINE_SCALE_FACTORS)

  Dp_spline  = interpolate.RectBivariateSpline(log_k, log_a, np.array(Dp_spline_data) )
  dDp_spline = interpolate.RectBivariateSpline(log_k, log_a, np.array(dDp_spline_data))
  Dm_spline  = interpolate.RectBivariateSpline(log_k, log_a, np.array(Dm_spline_data) )
  dDm_spline = interpolate.RectBivariateSpline(log_k, log_a, np.array(dDm_spline_data))

  return Dp_spline, dDp_spline, Dm_spline, dDm_spline

@njit
def check_range(k, a, a_in, m):
  if (5e-6 > a_in) or (5e-6 > a):
    raise ValueError("Spline inaccurate below a = 1e-5")

  if (1.1 < a_in) or (1.1 < a):
    raise ValueError("Spline inaccurate above a = 1")

spline_buffer_mass = c.FDM_M
Dp_spline, dDp_spline, Dm_spline, dDm_spline = create_splines(c.FDM_M, c.A_IN)

@njit 
def get_logk(k):
  return np.log(k_norm(k))

@njit
def D(k, eta, m):
  a = c.a_from_eta(eta)
  logk = get_logk(k)

  with objmode(y = "float64"):
    y = Dp_spline(logk, np.log(a))
  return y

@njit
def dD(k, eta, m):
    a = c.a_from_eta(eta)
    logk = get_logk(k)

    with objmode(y = "float64"):
      y = dDp_spline(logk, np.log(a))
    return y * np.sqrt(a)

@njit
def D_plus(k, a, a_in, m):
    logk = get_logk(k)

    with objmode(y = "float64"):
      y  = Dp_spline(logk, np.log(a))/Dp_spline(logk, np.log(a_in))
    return y
  
def D_plus_nnjit(k, a, a_in, m):
    logk = get_logk(k)

    with objmode(y = "float64"):
      y  = Dp_spline(logk, np.log(a))/Dp_spline(logk, np.log(a_in))
    return y

@njit
def dD_plus(k, a, a_in, m):
    logk = get_logk(k)

    with objmode(y = "float64"):
      y = dDp_spline(logk, np.log(a))/Dp_spline(logk, np.log(a_in))
    return y

@njit
def ddD_plus(k, a, a_in, m):
    #Compute second derivative wrt. a
    dloga = 1e-3

    logk = get_logk(k)
    loga = np.log(a)

    with objmode(y = "float64"):
      y = (dDp_spline(logk, loga + dloga) - dDp_spline(logk, loga - dloga))/(2*dloga)
      #Norm
      y /= Dp_spline(logk, np.log(a_in))
    return y

@njit
def D_minus(k, a, a_in, m):   
    logk = get_logk(k)   

    with objmode(y = "float64"):
      y = Dm_spline(logk, np.log(a)) / Dm_spline(logk, np.log(a_in))
    return y
    
    
def D_minus_nnjit(k, a, a_in, m):   
    logk = get_logk(k)   

    with objmode(y = "float64"):
      y = Dm_spline(logk, np.log(a)) / Dm_spline(logk, np.log(a_in))
    return y

@njit
def dD_minus(k, a, a_in, m):
    logk = get_logk(k)

    with objmode(y = "float64"):
      y = dDm_spline(logk, np.log(a)) / Dm_spline(logk, np.log(a_in))

    return y


@njit
def ddD_minus(k, a, a_in, m):
    #Compute second derivative wrt. a
    dloga = 1e-3
    logk = get_logk(k)
    loga = np.log(a)

    with objmode(y = "float64"):
      y = (dDm_spline(logk, loga + dloga) - dDm_spline(logk, loga - dloga))/(2*dloga)

      #Norm
      y /= Dm_spline(logk, np.log(a_in))

    return y

