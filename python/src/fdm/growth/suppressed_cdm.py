import constants as c
from special_functions import k_norm
from numba import njit
from fdm.util import li_kj, lague_kj

@njit
def suppression(k, a, m, n = 9):
    kj = lague_kj(a, m)
    x  = k_norm(k)/kj
    return 1/(1 + x**n)

@njit
def dsuppression(k, a, m, n = 9):
    da = 1e-8 
    return  (suppression(k, a + da, m, n) -  suppression(k, a - da, m, n))/(2*da)

@njit
def ddsuppression(k, a, m, n = 9):
    da = 1e-8 
    return (dsuppression(k, a + da, m, n) - dsuppression(k, a - da, m, n))/(2*da)


@njit
def D(k, eta, m):
    a    = c.a_from_eta(eta)
    return a/c.A_IN * suppression(k, a, m)

@njit
def dD(k, eta, m):
    a    = c.a_from_eta(eta)
    p1   = 1 * suppression(k, a, m)
    p2   = a * dsuppression(k, a, m)
    return np.sqrt(a) / c.A_IN * (p1 + p2)


@njit
def D_plus(k, a, a_in, m):
    return a/a_in* suppression(k, a, m)

@njit
def dD_plus(k, a, a_in, m):
    p1 = 1 *  suppression(k, a, m)
    p2 = a * dsuppression(k, a, m)
    return 1/a_in * (p1 + p2)

@njit
def ddD_plus(k, a, a_in, m):
    p1 = 2 * dsuppression(k, a, m)
    p2 = a * ddsuppression(k, a, m)
    return 1/a_in * (p1 + p2)

@njit
def D_minus(k, a, a_in, m):
    return (a/a_in)**(-3/2) * suppression(k, a, m)

@njit
def dD_minus(k, a, a_in, m):
    p1 = -3/2 * a**(-5/2) * suppression(k, a, m)
    p2 = a**(-3/2) * dsuppression(k, a, m)
    return (p1 + p2)/a_in**(-3/2)

@njit
def ddD_minus(k, a, a_in, m):
    p1 = -3/2 * -5/2 * a**(-7/2) * suppression(k, a, m)
    p2 = -3/2 * a**(-5/2) * dsuppression(k, a, m) * 2
    p3 = a**(-3/2) * ddsuppression(k, a, m)
    return (p1 + p2 + p3)/a_in**(-3/2)