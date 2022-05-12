from numba import njit
import numpy as np
import vegas

from uncertainties import ufloat
import constants 

#args has following structure
#cosine theta, length of k_1, length of k
#s1, s2 and m in the case of FDM
#eta in both cases

@njit
def P1L_vegas_integrand(P, F2, F3, *args):
  #Distinguish between CDM and FDM cases
  sargs = ()
  F21_args = ()
  F22_args = ()
  F3_args  = ()
  
  #FDM
  if len(args) == 7:
    #Input arguments
    ctheta = args[0]
    k1n    = args[1]
    s1     = args[2]
    s2     = args[3]
    kn     = args[4]
    eta    = args[5]
    m      = args[6]

    sargs    = (eta, m)
    F3_args  = (s1, s2) + sargs
    F21_args = (s1, )   + sargs
    F22_args = (s2, )   + sargs

  #CDM
  elif len(args) == 5:
    ctheta = args[0]
    k1n    = args[1]
    kn     = args[2]
    eta    = args[3]

    sargs    = (eta, )
    F3_args  = sargs
    F21_args = sargs
    F22_args = sargs

  else:
    raise ValueError("P1L vegas integrand takes either four of seven arguments for CDM or FDM. ")

  #Vectors
  k  = kn    * np.array([np.sqrt(1 - ctheta**2), 0, ctheta])
  k1 = k1n   * np.array([0, 0, 1])
  k2 = k - k1
  k3 = k + k1

  #Lengths of remaining vectors
  k2n = np.linalg.norm(k2)
  k3n = np.linalg.norm(k3)
  
  res =    6 * P(kn, *sargs)  * P(k1n, *sargs) * F3(k, k1, -k1, *F3_args)

  if k1n < k2n:
    res += 2 * P(k1n, *sargs) * P(k2n, *sargs) * F2( k1, k2, *F21_args) * F2(k1, k2, *F22_args)
  if k1n < k3n:
    res += 2 * P(k1n, *sargs) * P(k3n, *sargs) * F2(-k1, k3, *F21_args) * F2(-k1, k3, *F22_args)

  #Jacobi-determinant, Fourier convention and phi-integration
  res *= k1n**2 * 1/(2*np.pi)**2
  return res


#This class manages the vegas integrator object along with the function template for a given initial spectrum
class Vegas_integrator:
  neval       = constants.VEGAS_NEVAL
  nitn        = constants.VEGAS_NITN
  restart_max = constants.VEGAS_RESTART_MAX

  #Creates integrand, sets integration boundaries, initialises integrator and runs integrator s.t. it adapts to function
  def __init__(self, f, integration_boundaries):
    self.f  = f
    self.boundaries = integration_boundaries
    self.integrator = vegas.Integrator(integration_boundaries)
    #Upon first call to get_result, vegas integrator does dry run to adapt to integrand
    #Therefore the first call to get_result will take more time than subsequent calls
    self.isFirstCall = True

  #Reruns integrator s.t. it adapts to function
  def reset(self, *args):
    self.integrator = vegas.Integrator(self.boundaries)
    self.integrator(lambda x: self.f(x, *args), nitn=self.nitn, neval=self.neval)

  def get_result(self, *args):
    if self.isFirstCall:
      self.integrator(lambda x: self.f(x, *args), nitn=self.nitn, neval=self.neval)
      self.isFirstCall = False

    integrand = lambda x: self.f(x, *args)

    result    = self.integrator(integrand, nitn= self.nitn, neval=self.neval)
    counter = 0
    while counter < self.restart_max and (result.sdev > np.abs(result.mean) or result.Q < 0.1):
      #Let integrator adapt to integrand
      self.reset(*args)
      #Compute result again
      result  = self.integrator(integrand, nitn= self.nitn, neval=self.neval)
      counter += 1
      if counter == self.restart_max:
        print("Vegas Integrator Warning: Integration result for args unreliable", *args, " Mean: ", result.mean, " SDEV: ", result.sdev, " Q: ", result.Q)
    return result

  def debug_result(result):
    print("Summary of the integration run:")
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

  def __call__(self, *args):
    result = self.get_result(*args)
    return ufloat(result.mean, result.sdev)

