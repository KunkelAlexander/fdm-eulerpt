import bispectrum
import cdm.coupling as cc
import cdm.spectrum 
import cdm.p1l
import constants 
import montecarlo as mc 
import numpy as np 
from numba import njit 

@njit
def B222(q, k1, k2, k3, eta, P):
  return bispectrum.B222 (P, cc.F2s_bispectrum,                     q, k1, k2, k3, s = .0, s2 = .0, s3 = .0, eta = eta, m = 0.)
  
@njit
def B3211(q, k1, k2, k3, eta, P):
  return bispectrum.B3211(P, cc.F2s_bispectrum, cc.F3s_bispectrum,  q, k1, k2, k3, s = .0, s2 = .0, s3 = .0, eta = eta, m = 0.)

@njit
def B3212(q, k1, k2, k3, eta, P):
  return bispectrum.B3212 (P, cc.F2s_bispectrum, cc.F3s_bispectrum, q, k1, k2, k3, s = .0, s2 = .0, s3 = .0, eta = eta, m = 0.)

@njit
def B411(q, k1, k2, k3, eta, P):
  return bispectrum.B411 (P, cc.F4s_bispectrum,                     q, k1, k2, k3, s = .0, s2 = .0, s3 = .0, eta = eta, m = 0.)

@njit
def B1L_integrand_template(phi, ct, q, k1, k2, k3, eta, m, P):
  q_vec = bispectrum.spherical2cartesian(q, ct, phi)

  determinant = q**2/(2*np.pi)**3
  result  = B222 (q_vec, k1, k2, k3, eta, P)
  result += B3211(q_vec, k1, k2, k3, eta, P)
  result += B3212(q_vec, k1, k2, k3, eta, P)
  result += B411 (q_vec, k1, k2, k3, eta, P)

  #Return integrand including Jacobi-determinant
  return result * determinant

def make_B1L_integrand(P):
  @njit
  def integrand(phi, ct, q, k1, k2, k3, eta, m):
    return B1L_integrand_template(phi, ct, q, k1, k2, k3, eta, m, P)
  return integrand


def make_B1L_vegas_template(P):
  @njit
  def template(x, *args):
    phi            = x[0]
    ctheta         = x[1]
    #Variable transformation from [0, 1] to [0, infinity]
    q              = x[2]/(1-np.abs(x[2]))
    determinant    = 1/(1-np.abs(x[2]))**2
    return determinant * B1L_integrand_template(phi, ctheta, q, *args, P)

  return template

vegas_integrands = {
  "CDM IC":      make_B1L_integrand(cdm.spectrum.P_CDM_IC),
  "FDM IC":      make_B1L_integrand(cdm.spectrum.P_FDM_IC),
  "Scale-free":  make_B1L_integrand(cdm.spectrum.P_sc)
}

vegas_integrators = {
  "CDM IC":      mc.Vegas_integrator(make_B1L_vegas_template(cdm.spectrum.P_CDM_IC   ), constants.CDM_B1L_integration_boundaries),
  "FDM IC":      mc.Vegas_integrator(make_B1L_vegas_template(cdm.spectrum.P_FDM_IC   ), constants.CDM_B1L_integration_boundaries),
  "Scale-free":  mc.Vegas_integrator(make_B1L_vegas_template(cdm.spectrum.P_sc),        constants.CDM_B1L_sc_integration_boundaries)
}

utils = {
  "CDM IC":      bispectrum.make_tree_bispectrum_utilities(cdm.spectrum.P_CDM_IC,    cdm.coupling.F2s, cdm.p1l.vegas_integrators["CDM IC"]),
  "FDM IC":      bispectrum.make_tree_bispectrum_utilities(cdm.spectrum.P_FDM_IC,    cdm.coupling.F2s, cdm.p1l.vegas_integrators["FDM IC"]),
  "Scale-free":  bispectrum.make_tree_bispectrum_utilities(cdm.spectrum.P_sc,        cdm.coupling.F2s, cdm.p1l.vegas_integrators["Scale-free"])
}