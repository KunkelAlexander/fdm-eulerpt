import numpy as np
from numba import njit 
from special_functions import *
import scipy.integrate as integrate
import fdm.spectrum
import cdm.spectrum
import fdm.coupling
import constants
import montecarlo as mc
from uncertainties import ufloat 

@njit
def get_cartesian_P1L(ctheta, k1, k):
    k1_vec = k1 * np.array([0, 0, 1])
    k_vec  = k * np.array([np.sqrt(1 - ctheta**2), 0, ctheta])
    k2_vec = k_vec - k1_vec
    k2     = np.linalg.norm(k2_vec)
    return k1_vec, k2_vec, k2, k_vec

@njit
def P22_ic(ctheta, s1, s2, k1, k, eta, m, P):
    k1_vec, k2_vec, k2, k_vec = get_cartesian_P1L(ctheta, k1, k)
    return 4/(2*np.pi)**2 * k1**2 * P(k1, eta, m) * P(k2, eta, m) * fdm.coupling.C21si(k1_vec, k2_vec, s1, eta, m)* fdm.coupling.C21si(k1_vec, k2_vec, s2, eta, m)

@njit
def P311_ic(ctheta, s, k1, k, eta, m, P):
    k1_vec, k2_vec, k2, k_vec = get_cartesian_P1L(ctheta, k1, k)
    return 2/(2*np.pi)**2 * k1**2 * P(k1, eta, m) * P(k, eta, m) * fdm.coupling.H3i_c_Li(k_vec, k1_vec,  s, eta, m)

@njit
def P312_ic(ctheta, s2, s, k1, k, eta, m, P):
    k1_vec, k2_vec, k2, k_vec = get_cartesian_P1L(ctheta, k1, k)
    return 4/(2*np.pi)**2 * k1**2 * P(k1, eta, m) * P(k, eta, m) * fdm.coupling.C31i(k1_vec, -k1_vec, k_vec, s, s2, eta, m)


#Full integration of r, ctheta and both s's
def P22_c(k, eta, m, integrand):

  def ctheta_lim(s1, s2, r, k, eta, m):
    limit = k/(2*r)
    return [-1, 1] if limit > 1 else [-1, limit]

  def s1_lim(s2, r, k, eta, m):
    return [constants.ETA_IN, eta]

  def s2_lim(r, k, eta, m):
    return [constants.ETA_IN, eta]

  def r_lim(k, eta, m):
    return [constants.LOWER_RADIAL_CUTOFF, constants.UPPER_RADIAL_CUTOFF]

  options=constants.NQUAD_OPTIONS
  mean, std = integrate.nquad(integrand, [ctheta_lim, s1_lim, s2_lim, r_lim] , opts=[options,options,options,options], args = (k, eta, m, ))
  
  return ufloat(mean, std)


#Full integration of r, ctheta and both s's
def P311_c(k, eta, m, integrand):

  def ctheta_lim(s, r, k, eta, m):
    return [-1, 1]

  def s_lim(r, k, eta, m):
    return [constants.ETA_IN, eta]

  def r_lim(k, eta, m):
    return [constants.LOWER_RADIAL_CUTOFF, constants.UPPER_RADIAL_CUTOFF]

  options=constants.NQUAD_OPTIONS
  mean, std = integrate.nquad(integrand, [ctheta_lim, s_lim, r_lim] , opts=[options,options, options], args = (k, eta, m, ))
  return ufloat(mean, std)


#Full integration of r, ctheta and both s's
def P312_c(k, eta, m, integrand):

  def ctheta_lim(s2, s, r, k, eta, m):
    return [-1, 1]

  def s2_lim(s, r, k, eta, m):
    return [constants.ETA_IN, s]

  def s_lim(r, k, eta, m):
    return [constants.ETA_IN, eta]

  def r_lim( k, eta, m):
    return [constants.LOWER_RADIAL_CUTOFF, constants.UPPER_RADIAL_CUTOFF]

  options=constants.NQUAD_OPTIONS
  mean, std = integrate.nquad(integrand, [ctheta_lim, s2_lim, s_lim, r_lim] , opts=[options,options, options, options], args = (k, eta, m, ))
  return ufloat(mean, std)


#Factory function to circumvent numba's slow function dispatch 
def make_nquad_template(P):
    @jit_4D
    def integrand1(ctheta, s1, s2, k1, args):
      return P22_ic(ctheta, s1, s2, k1, args[0], args[1], args[2], P = P)
    @jit_3D
    def integrand2(ctheta, s1, k1, args):
      return P311_ic(ctheta, s1, k1, args[0], args[1], args[2], P = P)
    @jit_4D
    def integrand3(ctheta, s2, s1, k1, args):
      return P312_ic(ctheta, s2, s1, k1, args[0], args[1], args[2], P = P)

    # Note: a new f() is created each time make_f() is called!
    def func(k, eta, m):
      r1 = P22_c(k, eta, m, integrand1)
      print("F2: ", r1)
      r2 = P311_c(k, eta, m, integrand2)
      print("H3: ", r2)
      r3 = P312_c(k, eta, m, integrand3)
      print("F3: ", r3)
      return r1 + r2 + r3
    return func


def make_vegas_template(P):
  @njit
  def template(x, *args):
    ctheta = x[0]
    #Variable transformation from [0, 1] to [0, infinity]
    k1     = x[1]/(1-np.abs(x[1]))
    det    = 1   /(1-np.abs(x[1]))**2
    s      = x[2]
    s2     = x[3]

    return det * mc.P1L_vegas_integrand(P, fdm.coupling.F2si, fdm.coupling.F3si, ctheta, k1, s, s2, *args)

  return template

#Create two 1-loop corrections for different initial spectra 
 
linear_spectra = {
  "CDM IC":      fdm.spectrum.P_CDM_IC,
  "FDM IC":      fdm.spectrum.P_FDM_IC,
  "Scale-free":  fdm.spectrum.P_sc,
}

nquad_integrators = {
  "CDM IC":      make_nquad_template(fdm.spectrum.P_CDM_IC),
  "FDM IC":      make_nquad_template(fdm.spectrum.P_FDM_IC),
  "Scale-free":  make_nquad_template(fdm.spectrum.P_sc),
}

vegas_integrators = {
  "CDM IC":      mc.Vegas_integrator(make_vegas_template(fdm.spectrum.P_CDM_IC), constants.FDM_P1L_integration_boundaries),
  "FDM IC":      mc.Vegas_integrator(make_vegas_template(fdm.spectrum.P_FDM_IC), constants.FDM_P1L_integration_boundaries),
  "Scale-free":  mc.Vegas_integrator(make_vegas_template(fdm.spectrum.P_sc    ), constants.FDM_P1L_sc_integration_boundaries),
}
