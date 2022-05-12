import numpy as np
import scipy.special as sc
from scipy import integrate, LowLevelCallable, constants
import numba
from numba import cfunc, carray, jit, vectorize, njit, types
from numba import types
from numba.types import intc, CPointer, float64
from numba.extending import get_cython_function_address, overload

import numba_scipy
import ctypes
import scipy.special.cython_special as cysp    
import math

#import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation as R
#from multiprocessing import Pool, cpu_count

def jit_2D(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)),error_model="numpy",fastmath=True)
    def wrapped(n, xx):
        ar = carray(xx, n)
        return jitted_function(ar[0], ar[1], ar[2:])
    return LowLevelCallable(wrapped.ctypes)

def jit_3D(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        ar = carray(xx, n)
        return jitted_function(ar[0], ar[1], ar[2], ar[3:])
    return LowLevelCallable(wrapped.ctypes)

def jit_4D(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        ar = carray(xx, n)
        return jitted_function(ar[0], ar[1], ar[2], ar[3], ar[4:])
    return LowLevelCallable(wrapped.ctypes)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1jv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
jv_fn = functype(addr)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0gamma")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gamma_fn = functype(addr)

#Bessel function
@njit('float64(float64, float64)')
def jv(v, z):
    return jv_fn(v, z)

#Gamma function
@njit('float64(float64)')
def my_gamma(z):
    return sc.gamma(z)#gamma_fn(z)

#Copied from scipy source code
@njit
def bessel_diff_formula(v, z, n, L, phase):
    # from AMS55.
    # L(v,z) = J(v,z), Y(v,z), H1(v,z), H2(v,z), phase = -1
    # L(v,z) = I(v,z) or exp(v*pi*i)K(v,z), phase = 1
    # For K, you can pull out the exp((v-k)*pi*i) into the caller
    p = 1.0
    s = L(v-n, z)
    for i in range(1, n+1):
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        s += p*L(v-n + i*2, z)
    return s / (2.**n)

#Derivative of Bessel function
@njit
def jvp(v,z,n=1):
    """Return the nth derivative of Jv(z) with respect to z.
    """
    if n == 0:
        return jv(v,z)
    else:
        return bessel_diff_formula(v, z, n, jv, -1)

#Implement numba norm function that works with scalars and vectors
def k_norm(k):
    return np.linalg.norm(k)
    

@overload(k_norm)
def implement_k_norm(k):
    if isinstance(k, types.Float) or isinstance(k, types.Integer):
        def impl(k):
            return np.sqrt(k**2)

    elif isinstance(k, types.npytypes.Array):
        def impl(k):
            return np.linalg.norm(k)
        
    else:
        def impl(k):
            print("no valid type for k")
            raise(ValueError)

    return impl

#First three positive terms of Taylor series of Bessel function with negative n
@njit
def jv_taylor(n, x):
  y = 1-n
  p =  my_gamma(y)
  #First three terms in Taylor expansion
  t1 = 2**n
  t2 = 2**(n-2.)*x**2/(n-1.)
  t3 = 2**(n-5.)*x**4/(n-2.)/(n-1.)
  return x**(-n)*(t1 + t2 + t3)/p

@njit
#Check whether implementations work correctly
def test_jv(x):
  return np.sqrt(2/np.pi/x)*(3*np.cos(x)/x**2 + 3*np.sin(x)/x - np.cos(x))
