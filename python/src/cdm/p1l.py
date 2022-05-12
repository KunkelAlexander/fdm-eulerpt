import numpy as np
from numba import njit 

import scipy.integrate as integrate
from special_functions import * 

import cdm.spectrum
import cdm.coupling
import constants 
import montecarlo as mc 

from cdm.coupling import F2s, F3s

from uncertainties import ufloat 

@njit
def get_cartesian_P1L(ctheta, k1, k):
    k1_vec = k1 * np.array([0, 0, 1])
    k_vec  = k * np.array([np.sqrt(1 - ctheta**2), 0, ctheta])
    k2_vec = k_vec - k1_vec
    k2     = np.linalg.norm(k2_vec)
    return k1_vec, k2_vec, k2, k_vec

@njit
def P22i(ctheta, k1, k, eta, P):
    k1_vec, k2_vec, k2, k_vec = get_cartesian_P1L(ctheta, k1, k)
    return 2/(2*np.pi)**2*k1**2*F2s(k1_vec, k2_vec, eta)**2*P(k1, eta)*P(k2, eta)

@njit 
def P31i(ctheta, k1, k, eta, P):
    k1_vec, k2_vec, k2, k_vec = get_cartesian_P1L(ctheta, k1, k)
    return 6/(2*np.pi)**2*k1**2*F3s(k_vec, k1_vec, -k1_vec, eta)*P(k1, eta)*P(k, eta) 

#Compute integrals in a numerically well-behaved way
def P1(k, eta, integrand):
  def r_lim(k, eta):
    return [constants.LOWER_RADIAL_CUTOFF, constants.UPPER_RADIAL_CUTOFF]

  def ctheta_lim(r, k, eta):
    limit = k/(2*r)
    return [-1, 1] if limit > 1 else [-1, limit]

  options=constants.NQUAD_OPTIONS
  mean, std = integrate.nquad(integrand, [ctheta_lim, r_lim] , opts=[options,options], args = (k, eta, ))

  return 2 * ufloat(mean, std)

def P2(k, eta, integrand):

  def r_lim(k, eta):
    return [constants.LOWER_RADIAL_CUTOFF, constants.UPPER_RADIAL_CUTOFF]

  def ctheta_lim(r, k, eta):
    return [-1,1]

  options=constants.NQUAD_OPTIONS
  mean, std = integrate.nquad(integrand, [ctheta_lim, r_lim] , opts=[options,options], args = (k, eta, ))

  return ufloat(mean, std)

#Factory function to circumvent numba's slow function dispatch 
#Speed up of factor 10 :)
#Debugging with timeit helped to get a speed up of factor 300 :)
def make_nquad_template(P):
    @jit_2D
    def integrand1(ctheta, k1, args):
      return P22i(ctheta, k1, args[0], args[1], P)
    @jit_2D
    def integrand2(ctheta, k1, args):
      return P31i(ctheta, k1, args[0], args[1], P)

    # Note: a new f() is created each time make_f() is called!
    def f(k, eta):
      return P1(k, eta, integrand1) + P2(k, eta, integrand2)
    return f


def make_vegas_template(P):
  @njit
  def template(x, *args):
    ctheta = x[0]
    #Variable transformation from [0, 1] to [0, infinity]
    r      = x[1]/(1-np.abs(x[1]))
    det    = 1   /(1-np.abs(x[1]))**2

    return det*mc.P1L_vegas_integrand(P, cdm.coupling.F2s, cdm.coupling.F3s, ctheta, r, *args)

  return template


#Create two 1-loop corrections for different initial spectra 

linear_spectra = {
  "CDM IC":      cdm.spectrum.P_CDM_IC,
  "FDM IC":      cdm.spectrum.P_FDM_IC,
  "Scale-free":  cdm.spectrum.P_sc
}

nquad_integrators = {
  "CDM IC":      make_nquad_template(cdm.spectrum.P_CDM_IC),
  "FDM IC":      make_nquad_template(cdm.spectrum.P_FDM_IC),
  "Scale-free":  make_nquad_template(cdm.spectrum.P_sc)
}

vegas_integrators = {
  "CDM IC":      mc.Vegas_integrator(make_vegas_template(cdm.spectrum.P_CDM_IC), constants.CDM_P1L_integration_boundaries),
  "FDM IC":      mc.Vegas_integrator(make_vegas_template(cdm.spectrum.P_FDM_IC), constants.CDM_P1L_integration_boundaries),
  "Scale-free":  mc.Vegas_integrator(make_vegas_template(cdm.spectrum.P_sc), constants.CDM_P1L_sc_integration_boundaries)
}