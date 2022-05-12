import numpy as np
from scipy.optimize import curve_fit
import constants 

import fdm.growth.numerical as flg
import cdm.growth.numerical as clg

import fdm.util 

from numba import njit 
from special_functions import k_norm 



def get_fit_D(D_CDM, m, a_in):
    def D(M, alpha, beta):
        k, a = M
        kj = fdm.util.lague_kj(a_in, m)
        return 1/(1 + alpha * (k/kj)**beta) * D_CDM(k, a, a_in, m)
        #return (1 - (1 + np.exp(-2 * alpha * (k - k0 * fdm.util.lague_kj(a, m))))**(-8)) * D_CDM(k, a, a_in, m)
    return D

def get_model_D(alpha, beta, D_FDM, D_CDM):
    @njit
    def D(k, a, a_in, m):
        kn = k_norm(k)
        kj = fdm.util.lague_kj(a_in, m)
        return 1/(1 + alpha * (kn/kj)**beta)  * D_CDM(kn, a, a_in, m)
        #return (1 - (1 + np.exp(-2 * alpha * (kn - beta * kj)))**(-8)) * D_CDM(k, a, a_in, m) + 10*constants.EPS
    
    N_rms = 2000
    k_sample = np.random.choice(constants.SPLINE_MOMENTA,       N_rms)
    a_sample = np.random.choice(constants.SPLINE_SCALE_FACTORS, N_rms)
    a_in = constants.A_IN 
    m    = constants.FDM_M 

    fit   = np.zeros(N_rms)
    ref   = np.zeros(N_rms)

    for i in range(N_rms):
        fit[i] = D    (k_sample[i], a_sample[i], a_in, m)
        ref[i] = D_FDM(k_sample[i], a_sample[i], a_in, m)

    rms = np.sqrt(np.mean((ref - fit)**2))
    print('RMS residual =', rms)
    
    return D 

def fit_model_D(D_FDM, D_CDM, D_CDM_wjit, a_in = constants.A_IN, m = constants.FDM_M):
    x = constants.SPLINE_MOMENTA
    y = constants.SPLINE_SCALE_FACTORS

    # The two-dimensional domain of the fit.
    X, Y = np.meshgrid(x, y)

    # The function to be fit is Z.
    Z = np.zeros(X.shape)
    for i, k in enumerate(x):
        for j, a in enumerate(y):
            Z[j, i] = D_FDM(k, a, a_in = a_in, m = m)

    D_fit = get_fit_D(D_CDM, m, a_in)

    # Initial guesses to the fit parameters.
    p0 = [1, 1]

    xdata      = np.vstack((X.ravel(), Y.ravel()))
    popt, pcov = curve_fit(D_fit, xdata, Z.ravel(), p0)

    print('Fitted parameters in num fit:')
    print(popt, pcov)

    return popt

if constants.FIT_TO_SPLINE:
    popt_p  = fit_model_D(flg.D_plus, clg.D_plus_nnjit,  clg.D_plus)
    popt_m = fit_model_D(flg.D_minus, clg.D_minus_nnjit, clg.D_minus)
else:
    popt_p = constants.NUMFIT_DPLUS_PARAMETERS
    popt_m = constants.NUMFIT_DMINUS_PARAMETERS

print("popt_p", popt_p, "popt_m", popt_m)
    
D_plus  = get_model_D(popt_p[0], popt_p[1], flg.D_plus,  clg.D_plus)
D_minus = get_model_D(popt_m[0], popt_m[1], flg.D_minus, clg.D_minus)

@njit
def D(k, eta, m):
    a = c.a_from_eta(eta)
    return D_plus(k, a, constants.A_IN, m)

@njit
def dD(k, eta, m):
    a = c.a_from_eta(eta)
    da = 1e-8 

    result = (D_plus(k, a + da, constants.A_IN, m) - D_plus(k, a - da, constants.A_IN, m))/(2*da)
    return result * np.sqrt(a)

@njit
def dD_plus(k, a, a_in, m):
    da = 1e-8
    result = (D_plus(k, a + da, a_in, m) - D_plus(k, a - da, a_in, m))/(2*da)
    return result

@njit
def ddD_plus(k, a, a_in, m):
    da = 1e-8 
    #Compute second derivative wrt. a
    result = (D_plus(k, a + da, a_in, m) - 2 * D_plus(k, a, a_in, m) + D_plus(k, a - da, a_in, m))/(da**2)
    return result


@njit
def dD_minus(k, a, a_in, m):
    da = 1e-8
    result = (D_minus(k, a + da, a_in, m) - D_minus(k, a - da, a_in, m))/(2*da)
    return result


@njit
def ddD_minus(k, a, a_in, m):
    da = 1e-8 
    #Compute second derivative wrt. a
    result = (D_minus(k, a + da, a_in, m) - 2 * D_minus(k, a, a_in, m) + D_minus(k, a - da, a_in, m))/(da**2)
    return result
