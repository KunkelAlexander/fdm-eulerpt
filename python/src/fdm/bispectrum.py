import bispectrum
import fdm.coupling as fc
import fdm.spectrum 
import fdm.p1l 
import constants 
import montecarlo as mc 
from numba import njit 
import numpy as np 

@njit
def B222(q, k1, k2, k3, s, s2, s3, eta, m, P):
  return bispectrum.B222 (P, fc.F2si,          q, k1, k2, k3, s, s2, s3, eta, m)

@njit
def B3211(q, k1, k2, k3, s, s2, s3, eta, m, P):
  return bispectrum.B3211(P, fc.F2si, fc.F3si, q, k1, k2, k3, s, s2, s3, eta, m)

@njit
def B3212(q, k1, k2, k3, s, s2, s3, eta, m, P):
  return bispectrum.B3212(P, fc.F2si, fc.F3si, q, k1, k2, k3, s, s2, s3, eta, m)

@njit
def B411(q, k1, k2, k3, s, s2, s3, eta, m, P):
  return bispectrum.B411 (P, fc.F4si,          q, k1, k2, k3, s, s2, s3, eta, m)

@njit
def B1L_integrand_template(phi, ct, q, s, s2, s3, k1, k2, k3, eta, m, P):
  q_vec = bispectrum.spherical2cartesian(q, ct, phi)

  determinant = q**2/(2*np.pi)**3
  result  = B222 (q_vec, k1, k2, k3, s, s2, s3, eta, m, P)
  result += B3211(q_vec, k1, k2, k3, s, s2, s3, eta, m, P)
  result += B3212(q_vec, k1, k2, k3, s, s2, s3, eta, m, P)
  result += B411 (q_vec, k1, k2, k3, s, s2, s3, eta, m, P)

  #Return integrand including Jacobi-determinant
  return result * determinant


def make_B1L_integrand(P):
  @njit
  def integrand(phi, ct, q, s, s2, s3, k1, k2, k3, eta, m):
    return B1L_integrand_template(phi, ct, q, s, s2, s3, k1, k2, k3, eta, m, P)
  return integrand


#Full IR-safe CDM bispectrum 1-loop corrections
def make_B1L_vegas_template(P): 
  @njit 
  def integrand(x, *args):
    phi           = x[0]
    ctheta        = x[1]
    #Variable transformation from [0, 1] to [0, infinity]
    q             = x[2]/(1-np.abs(x[2]))
    determinant   = 1/(1-np.abs(x[2]))**2
    s1            = x[3]
    s2            = x[4]
    s3            = x[5]

    return determinant * B1L_integrand_template(phi, ctheta, q, s1, s2, s3, *args, P)

  return integrand


vegas_integrators = {
  "CDM IC":      mc.Vegas_integrator(make_B1L_vegas_template(fdm.spectrum.P_CDM_IC) , constants.FDM_B1L_integration_boundaries),
  "FDM IC":      mc.Vegas_integrator(make_B1L_vegas_template(fdm.spectrum.P_FDM_IC) , constants.FDM_B1L_integration_boundaries),
  "Scale-free":  mc.Vegas_integrator(make_B1L_vegas_template(fdm.spectrum.P_sc    ) , constants.FDM_B1L_sc_integration_boundaries)
}

utils = {
  "CDM IC":      bispectrum.make_tree_bispectrum_utilities(fdm.spectrum.P_CDM_IC, fdm.coupling.F2s, fdm.p1l.vegas_integrators["CDM IC"]),
  "FDM IC":      bispectrum.make_tree_bispectrum_utilities(fdm.spectrum.P_FDM_IC, fdm.coupling.F2s, fdm.p1l.vegas_integrators["FDM IC"]),
  "Scale-free":  bispectrum.make_tree_bispectrum_utilities(fdm.spectrum.P_sc,     fdm.coupling.F2s, fdm.p1l.vegas_integrators["Scale-free"])
}