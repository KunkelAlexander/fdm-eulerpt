from numba import njit 
import numpy as np
import constants as c

from special_functions import jv, jvp, jv_taylor, k_norm

from fdm.util import b_f
from fdm.growth.analytical import D, dD 

#Propagator for inh. time evolutoin
@njit
def G(k, s, eta, m):
  bk = b_f(k, m)
  p1 = np.pi * s**2 / 2 * s**(-1/2) * eta**(-1/2)

  j1 = jv( 2.5, bk/s)
  j2 = jv(-2.5, bk/eta)
  j3 = jv( 2.5, bk/eta)
  j4 = jv(-2.5, bk/s)

  y  = p1 * (j1*j2-j3*j4)

  return y

#s-derivative of propagator
@njit
def ds_G(k, s, eta, m):
  bk = b_f(k, m)
  p1 = 3/4*np.pi * s**(1/2) * eta**(-1/2)
  p2 = np.pi/2 * bk * s**(-1/2) * eta**(-1/2)

  j1 = jv(  2.5, bk/s)
  j2 = jv( -2.5, bk/eta)
  j3 = jv(  2.5, bk/eta)
  j4 = jv( -2.5, bk/s)
  j5 = jvp( 2.5, bk/s)
  j6 = jvp(-2.5, bk/eta)
  j7 = jvp( 2.5, bk/eta)
  j8 = jvp(-2.5, bk/s)
  

  y  = p1*(j1*j2-j3*j4)+p2*(j3*j8-j5*j2)

  return y

#eta-derivative of propagator
@njit
def deta_G(k, s, eta, m):
  bk = b_f(k, m)
  p1 = -np.pi/4 *      s**(3/2) * eta**(-3/2)
  p2 =  np.pi/2 * bk * s**(3/2) * eta**(-5/2)

  j1 = jv(  2.5, bk/s)
  j2 = jv( -2.5, bk/eta)
  j3 = jv(  2.5, bk/eta)
  j4 = jv( -2.5, bk/s)
  j5 = jvp( 2.5, bk/s)
  j6 = jvp(-2.5, bk/eta)
  j7 = jvp( 2.5, bk/eta)
  j8 = jvp(-2.5, bk/s)
  

  y  = p1*(j1*j2-j3*j4)+p2*(j4*j7-j6*j1)
  return y

#s- and eta- second derivative of propagator
@njit
def dseta_G(k, s, eta, m):
  bk = b_f(k, m)
  p1 = -3*np.pi/8 * s**( 1/2) * eta**(-3/2)
  p2 =    np.pi/4 * s**(-1/2) * eta**(-3/2) * bk

  p3 = - 3/4 * np.pi* s**( 1/2) * eta**(-5/2) * bk
  p4 =   1/2 * np.pi* s**(-1/2) * eta**(-5/2) * bk**2

  j1 = jv(  2.5, bk/s)
  j2 = jv( -2.5, bk/eta)
  j3 = jv(  2.5, bk/eta)
  j4 = jv( -2.5, bk/s)
  j5 = jvp( 2.5, bk/s)
  j6 = jvp(-2.5, bk/eta)
  j7 = jvp( 2.5, bk/eta)
  j8 = jvp(-2.5, bk/s)
  

  y  = p1*(j1*j2-j3*j4)+p2*(j5*j2-j3*j8)+p3*(j6*j1-j4*j7)+p4*(j5*j6-j7*j8)
  return y

#Helper functions used for FDM coupling
@njit
def f1(k, s, eta, m):
  return D(k, s, m)/D(k, eta, m)

@njit
def f2(k, s, eta, m):
  return -dD(k, s, m)/D(k, eta, m)