from numba import njit 
import numpy as np 
import special_functions as sf
import constants         as c

if not c.USE_CAMB_FIT:
  from scipy import optimize
  import camb
  from camb import model, initialpower

  #Now get matter power spectra and sigma8 at redshift 0 and 0.8
  pars = camb.CAMBparams()

  #ombh2=0.02230, omch2=0.1188 are Planck parameters
  #Omega_b * h**2, Omega_c * h**2 with h=0.68
  pars.set_cosmology(H0=c.H0_HUBBLE, ombh2=c.OMBH2, omch2=c.OMCH2)

  pars.InitPower.set_params(ns=c.NS_SPECTRUM)
  #Note non-linear corrections couples to smaller scales than you want
  pars.set_matter_power(redshifts=[c.Z_FIN, c.Z_IN], kmax=c.MAX_K)

  #Linear spectra
  pars.NonLinear = model.NonLinear_none
  results = camb.get_results(pars)
  KH_LINEAR, REDSHIFTS_LINEAR, PK_LINEAR = results.get_matter_power_spectrum(minkh=c.MIN_K, maxkh=c.MAX_K, npoints = c.NUMBER_K_POINTS)
  SIGMA8 = np.array(results.get_sigma8()) 

  #Non-Linear spectra (Halofit)
  pars.NonLinear = model.NonLinear_both
  results.calc_power_spectra(pars)
  KH_NONLINEAR, Z_NONLINEAR, PK_NONLINEAR = results.get_matter_power_spectrum(minkh=c.MIN_K, maxkh=c.MAX_K, npoints = c.NUMBER_K_POINTS)

  def P0_fit(k, A, B, C, D, E, F):
    return A*k**c.NS_SPECTRUM/(1 + (k/B)**C + (k**E + k**F)/D)

  popt, pcov = optimize.curve_fit(P0_fit, KH_LINEAR, PK_LINEAR[1, :], p0=[4.3e2, 4.7e-2, 3.4, 4.4e-4, 2.3, 1.9], maxfev=10000)

  @njit
  def P0_interpolation(k_):
    k = sf.k_norm(k_)

    maxkh = KH_LINEAR[-1]
    minkh = KH_LINEAR[0]
    dk    = np.log(KH_LINEAR[1]) - np.log(KH_LINEAR[0])

    #Distance between neighbouring points in log scaling
    i_low_0 = 0
    i_low_1 = 1
    i_high_0 = -1
    i_high_1 = -2
    
    #Model spectrum as pure power spectrum below minkh
    if k < minkh:
      alpha = (np.log(PK_LINEAR[1, i_low_1]) - np.log(PK_LINEAR[1, i_low_0]))/(np.log(KH_LINEAR[i_low_1]) - np.log(KH_LINEAR[i_low_0]))
      a     = PK_LINEAR[1, i_low_0] / minkh**alpha
      return a*k**alpha

    #Model spectrum as pure power spectrum above maxkh
    if k >= maxkh:
      alpha = (np.log(PK_LINEAR[1, i_high_1]) - np.log(PK_LINEAR[1, i_high_0]))/(np.log(KH_LINEAR[i_high_1]) - np.log(KH_LINEAR[i_high_0]))
      a     = PK_LINEAR[1, i_high_0] / maxkh**alpha
      return a*k**alpha

    index = int(np.floor((np.log(k/minkh))/ dk))
    delta = np.log(k / minkh)/dk - index
    
    y = PK_LINEAR[1, index] + (PK_LINEAR[1, index + 1] - PK_LINEAR[1, index])*np.exp(delta)/np.exp(dk)
    return y
else:
  #Define initial spectrum with fit parameters
  print("Do not compute initial spectrum using CAMB")
  popt = [4.35034371e+02, 4.77528415e-02, 3.42036680e+00, 4.40507786e-04, 2.38535303e+00, 1.90447152e+00]

A_PARAM, B_PARAM, C_PARAM, D_PARAM, E_PARAM, F_PARAM = popt

@njit
def P0_fit(k):
  return A_PARAM*k**c.NS_SPECTRUM/(1 + (k/B_PARAM)**C_PARAM + (k**E_PARAM + k**F_PARAM)/D_PARAM)

#Transfer function from primordial power spectrum 
#Power spectrum at radiation-matter-equivalence
def P0_naive(k_):
  k_0 = 0.01

  k = sf.k_norm(k_)
  return A_PARAM * k * 1/(1 + (k/k_0)**4)

#Define P_0 used for FDM_P and CDM_P
P_0 = P0_fit