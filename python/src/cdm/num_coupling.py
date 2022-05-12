import montecarlo as mc
from numba import njit
import numpy as np
import constants
import scipy.integrate as integrate
from uncertainties import ufloat

from cdm.util import b_f
from cdm.couplings.numerical import f1, f2, G, ds_G, dseta_G, deta_G

@njit
def f(k, s, eta, m):
    return np.array([f1(k, s, eta, m), f2(k, s, eta, m)], dtype=np.float64)
    
@njit
def gamma(k, k1, k2, eta, m, a, b, c):
    # Gamma_0
    if a == 0:
        if b == 0 and c == 1:
            k2s = np.dot(k2, k2)
            if k2s < constants.EPS:
                return 0
            return -np.dot(k, k2)/(2*k2s)

        if b == 1 and c == 0:
            k1s = np.dot(k1, k1)
            if k1s < constants.EPS:
                return 0
            return -np.dot(k, k1)/(2*k1s)

        if b == 0 and c == 0:
            return 0

        if b == 1 and c == 1:
            return 0

    # Gamma 1
    if a == 1:
        if b == 0 and c == 0:
            bk = b_f(k, m)
            p1 = bk**2/(4*eta**4)
            p2 = 1 + (np.dot(k1, k1) + np.dot(k2, k2))/np.dot(k, k)
            return -p1 * p2

        if b == 1 and c == 1:
            k1s = np.dot(k1, k1)
            k2s = np.dot(k2, k2)
            if k1s < constants.EPS or k2s < constants.EPS:
                return 0
            return -np.dot(k, k) * np.dot(k1, k2) / (2*k1s*k2s)

        if b == 0 and c == 1:
            return 0

        if b == 1 and c == 0:
            return 0

    print("Value Error in gamma, a,b,c = ", a, b, c)
    raise ValueError


@njit
def theta2111(k, k1, k2, k3, s, eta, m):
    bk = b_f(k, m)
    return bk**2/(8*eta**4)*(
        1 +
        (np.dot(k1, k1) + np.dot(k2, k2) + np.dot(k3, k3))/np.dot(k, k) +
        (np.dot(k1 + k2, k1 + k2) + np.dot(k2 + k3, k2 + k3) +
         np.dot(k1 + k3, k1 + k3))/(3*np.dot(k, k))
    )


@njit
def xi2111(k, k1, k2, k3, k4, s, eta, m):
    bk = b_f(k, m)
    return -bk**2*3/(32*eta**4)*(
        1 +
        2/3*(np.dot(k1, k1) + np.dot(k2, k2) + np.dot(k3, k3) + np.dot(k4, k4))/np.dot(k, k) +
        1/3*(np.dot(k1 + k2, k1 + k2) + np.dot(k1 + k3, k1 + k3) + np.dot(k1 + k4, k1 + k4) +
             np.dot(k2 + k3, k2 + k3) + np.dot(k2 + k4, k2 + k4) + np.dot(k3 + k4, k3 + k4))/np.dot(k, k)
    )


@njit
def W_kernel(k, k1, k2, s, eta, m, a):
    y = np.zeros((2, 2), dtype=np.float64)

    # When deriving W_c, we assumed k1, k2 and k to be != 0
    # If they are zero, we need to explicitly return zero or will obtain divisions by zero
    if np.linalg.norm(k) < constants.EPS or np.linalg.norm(k1) < constants.EPS or np.linalg.norm(k2) < constants.EPS:
        return y

    if a == 0:
        dg = ds_G(k, s, eta, m)
        g = G(k, s, eta, m)
        for b in range(2):
            for c in range(2):
                ga1 = gamma(k, k1, k2, s, m, 0, b, c)
                ga2 = gamma(k, k1, k2, s, m, 1, b, c)
                y[b, c] = (-ga1*dg - ga2*g + 2/s*ga1*g)
        return y

    if a == 1:
        dg = deta_G(k, s, eta, m)
        ddg = dseta_G(k, s, eta, m)

        y = np.zeros((2, 2), dtype=np.float64)
        for b in range(2):
            for c in range(2):
                ga1 = gamma(k, k1, k2, s, m, 0, b, c)
                ga2 = gamma(k, k1, k2, s, m, 1, b, c)
                y[b, c] = (ga1*ddg + ga2*dg - 2/s*ga1*dg)
        return y
    else:
        raise ValueError("Wrong mode in W kernel")


@njit
def U_kernel(k, k1, k2, k3, s, eta, m, a):
    if a == 0:
        return -theta2111(k, k1, k2, k3, s, eta, m)*G(k, s, eta, m)

    if a == 1:
        return theta2111(k, k1, k2, k3, s, eta, m)*deta_G(k, s, eta, m)

    raise ValueError


@njit
def V_kernel(k, k1, k2, k3, k4, s, eta, m, a):
    if a == 0:
        return -xi2111(k, k1, k2, k3, k4, s, eta, m)*G(k, s, eta, m)

    if a == 1:
        return xi2111(k, k1, k2, k3, k4, s, eta, m)*deta_G(k, s, eta, m)

    raise ValueError

# Cartesian helper function, basically C21 with variable a


@njit
def fwaf(k1, k2, s, s1, eta, m, a):
    return f(k1, s1, eta, m).T@W_kernel(k1 + k2, k1, k2, s1, s, m, a)@f(k2, s1, eta, m)

# Cartesian C21 integrand


@njit
def C21si(k1, k2, s, eta, m):
    res = f(k1, s, eta, m).T@W_kernel(k1 + k2, k1, k2, s, eta, m, 0)@f(k2, s, eta, m)
    return res 



# Cartesian C21 after integration
def C21s(k1, k2, eta, m):
    def integrand(s):
        return C21si(k1, k2, s, eta, m)
    return integrate.quad(integrand, constants.ETA_IN, eta, limit=1000, epsrel = 1e-4, epsabs=1e-4)

# Cartesian 31
@njit
def C31i(k1, k2, k3, s, s1, eta, m):
    k = k1 + k2 + k3

    # Compute g_b = W_bb'c' f_b' f_c'
    g = np.array([
        fwaf(k2, k3, s, s1, eta, m, 0),
        fwaf(k2, k3, s, s1, eta, m, 1)
    ])

    sum = 0

    vector1 = f(k1, s, eta, m)
    matrix1 = W_kernel(k, k1, k2 + k3, s, eta, m, 0)
    vector2 = g
    for b in [0, 1]:
        for c in [0, 1]:
            sum += matrix1[b, c] * vector1[b] * vector2[c]

    return 2*sum

# Symmetrised F3 integrand


@njit
def C31si(k1, k2, k3, s, s1, eta, m):
    return 1/3 * (
        C31i(k1, k2, k3, s, s1, eta, m) +
        C31i(k2, k1, k3, s, s1, eta, m) +
        C31i(k3, k2, k1, s, s1, eta, m)
    )

# Symmetrised F3


def C31s(k1, k2, k3, eta, m):
    def integrand(s1, s):
        return C31si(k1, k2, k3, s, s1, eta, m)
    return integrate.dblquad(integrand, constants.ETA_IN, eta, lambda s: constants.ETA_IN, lambda s: s)

#Regular F3 as in Li 2018
def F3_Li(k, k1, eta, m):
 def integrand(s1, s):
   return F3i(k1, -k1, k, s, s1, eta, m)
 return integrate.dblquad(integrand, constants.ETA_IN, eta, lambda s: constants.ETA_IN, lambda s: s)

#Cartesian H3 as in Li 2018
@njit
def H3i_c_Li(k, k2, s, eta, m):
 #return (U_kernel(k, k2, -k2, k, s, eta, 0) + U_kernel(k, k2, k, -k2, s, eta, 0) + U_kernel(k,k,k2, -k2, s,eta, 0))*f1(k, s, eta)*f1(k2, s, eta)**2*f1(k, eta, eta) + (U_kernel(k, k2, -k2, k, s, eta, 1) + U_kernel(k, k2, k, -k2, s, eta, 1) + U_kernel(k,k,k2, -k2, s,eta, 1))*f1(k, s, eta)*f1(k2, s, eta)**2*f2(k, eta, eta)
 return (U_kernel(k, k2, -k2, k, s, eta, m, 0) + U_kernel(k, k2, k, -k2, s, eta, m, 0) + U_kernel(k,k,k2, -k2, s, eta, m, 0))*f1(k, s, eta, m)*f1(k2, s, eta, m)**2

def H3_c_Li(k, k2, eta, m):
 def integrand(s):
   return H3i_c_Li(k, k2, s, eta, m)
 return integrate.quad(integrand, constants.ETA_IN, eta)


# Cartesian C32 (Implementation of H3 where arguments are not yet substituted)
@njit
def C32i(k1, k2, k3, s, eta, m):
    k = k1 + k2 + k3
    return U_kernel(k, k1, k2, k3, s, eta, m, 0)*f1(k1, s, eta, m)*f1(k2, s, eta, m)*f1(k3, s, eta, m)

# C32_i already is symmetric


@njit
def C32si(k1, k2, k3, s, eta, m):
    return C32i(k1, k2, k3, s, eta, m)

# Symmetric C32


def C32s(k1, k2, k3, eta, m):
    def integrand(s):
        return C32si(k1, k2, k3, s, eta, m)
    return integrate.quad(integrand, constants.ETA_IN, eta)


# C41 integrand
@njit
def C41i(k1, k2, k3, k4, s, eta, m):
    k = k1 + k2 + k3 + k4
    return V_kernel(k, k1, k2, k3, k4, s, eta, m, a=0)*f1(k1, s, eta, m)*f1(k2, s, eta, m)*f1(k3, s, eta, m)*f1(k4, s, eta, m)

# C41 already is symmetric


@njit
def C41si(k1, k2, k3, k4, s, eta, m):
    return C41i(k1, k2, k3, k4, s, eta, m)

# Integrate out s-dependencies


def C41s(k1, k2, k3, k4, eta, m):
    def integrand(s):
        return C41si(k1, k2, k3, k4, s, eta, m)

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    return integrate.nquad(integrand, [s_lim], opts=[options])

# C42


@njit
def C42i(k1, k2, k3, k4, s, s1, eta, m):
    k = k1 + k2 + k3 + k4
    return 3*U_kernel(k, k1, k2, k3 + k4, s, eta, m, a=0)*f1(k1, s, eta, m)*f1(k2, s, eta, m)*fwaf(k3, k4, s, s1, eta, m, a=0)

# Symmetrised C42


@njit
def C42si(k1, k2, k3, k4, s, s1, eta, m):
    return 1/6 * (
        C42i(k1, k2, k3, k4, s, s1, eta, m) +
        C42i(k1, k3, k2, k4, s, s1, eta, m) +
        C42i(k1, k4, k3, k2, s, s1, eta, m) +
        C42i(k2, k3, k1, k4, s, s1, eta, m) +
        C42i(k2, k4, k1, k3, s, s1, eta, m) +
        C42i(k3, k4, k1, k2, s, s1, eta, m)
    )


# C42
def C42i_c_correct(k1, k2, k3, k4, s, s1, eta, m):
    k = k1 + k2 + k3 + k4
    return (U_kernel(k, k2, k3, k1 + k4, s, eta, m, a=0)*f1(k3, s, eta, m)*f1(k2, s, eta, m)*fwaf(k1, k4, s, s1, eta, m, a=0) +
            U_kernel(k, k1, k3, k2 + k4, s, eta, m, a=0)*f1(k1, s, eta, m)*f1(k3, s, eta, m)*fwaf(k2, k4, s, s1, eta, m, a=0) +
            U_kernel(k, k1, k2, k3 + k4, s, eta, m, a=0)*f1(k1, s, eta, m)*f1(k2, s, eta, m)*fwaf(k3, k4, s, s1, eta, m, a=0))


# Integrate out s-dependencies
def C42s(k1, k2, k3, k4, eta, m):
    def integrand(s1, s):
        return C42si(k1, k2, k3, k4, s, s1, eta, m)

    def s1_lim(s):
        return [constants.ETA_IN, s]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    return integrate.nquad(integrand, [s1_lim, s_lim], opts=[options, options])

# C43


@njit
def C43i(k1, k2, k3, k4, s, s1, s2, eta, m):
    res = 0
    k = k1 + k2 + k3 + k4
    W = W_kernel(k, k1 + k2, k3 + k4, s, eta, m, a=0)

    for b in [0, 1]:
        for c in [0, 1]:
            res += W[b, c]*fwaf(k1, k2, s, s1, eta, m, b) * \
                fwaf(k3, k4, s, s2, eta, m, c)

    return res

# Symmetrised C43


@njit
def C43si(k1, k2, k3, k4, s, s1, s2, eta, m):
    return 1/3 * (
        C43i(k1, k2, k3, k4, s, s1, s2, eta, m) +
        C43i(k1, k3, k2, k4, s, s1, s2, eta, m) +
        C43i(k1, k4, k3, k2, s, s1, s2, eta, m)
    )

# Integrate out s-dependencies


def C43s(k1, k2, k3, k4, eta, m):
    def integrand(s1, s2, s):
        return C43si(k1, k2, k3, k4, s, s1, s2, eta, m)

    def s1_lim(s1, s):
        return [constants.ETA_IN, s]

    def s2_lim(s):
        return [constants.ETA_IN, s]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    return integrate.nquad(integrand, [s1_lim, s2_lim, s_lim], opts=[options, options, options])


@njit
def C44i(k1, k2, k3, k4, s, s1, eta, m):
    res = 0
    k = k1 + k2 + k3 + k4

    # Compute W_1bc f_b
    fW1 = f(k1, s, eta, m)@W_kernel(k, k1, k2 + k3 + k4, s, eta, m, a=0)

    # Compute W_1bc f_b (U_cedf f_e f_d f_f)
    for c in [0, 1]:
        res += fW1[c] * U_kernel(k2 + k3 + k4, k2, k3, k4, s1, s, m, c) * \
            f1(k2, s1, eta, m) * f1(k3, s1, eta, m) * f1(k4, s1, eta, m)

    # Factor two since psi^(1) psi^(3) = psi^(3) psi^(1)
    return 2 * res


@njit
def C44si(k1, k2, k3, k4, s, s1, eta, m):
    return 1/4 * (
        C44i(k1, k2, k3, k4, s, s1, eta, m) +
        C44i(k2, k1, k3, k4, s, s1, eta, m) +
        C44i(k3, k2, k1, k4, s, s1, eta, m) +
        C44i(k4, k2, k3, k1, s, s1, eta, m)
    )

# Integrate out s-dependencies
def C44s(k1, k2, k3, k4, eta, m):
    def integrand(s1, s):
        return C44si(k1, k2, k3, k4, s, s1, eta, m)

    def s1_lim(s):
        return [constants.ETA_IN, s]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    return integrate.nquad(integrand, [s1_lim, s_lim], opts=[options, options])

@njit
def C45i(k1, k2, k3, k4, s, s1, s2, eta, m):
    res = 0
    k = k1 + k2 + k3 + k4

    # Compute W_1bc f_b
    fW1 = f(k1, s, eta, m)@W_kernel(k, k1, k2 + k3 + k4, s, eta, m, a=0)

    # Compute W_ekl f_k f_l
    g = np.array([fwaf(k3, k4, s1, s2, eta, m, 0),
                 fwaf(k3, k4, s1, s2, eta, m, 1)])

    # Compute 2 W_cde f_d (W_ekl f_k f_l)
    x = 2 * np.array([f(k2, s1, eta, m)@W_kernel(k2 + k3 + k4, k2, k3 + k4, s1, s, m, a=0)@g,
                      f(k2, s1, eta, m)@W_kernel(k2 + k3 + k4, k2, k3 + k4, s1, s, m, a=1)@g])

    # Compute W_1bc f_b (2 W_cde f_d (W_ekl f_k f_l))
    res += fW1@x

    return 2 * res

@njit
def C45si(k1, k2, k3, k4, s, s1, s2, eta, m):
    return 1/12 * (
        C45i(k1, k2, k3, k4, s, s1, s2, eta, m) +
        C45i(k1, k3, k2, k4, s, s1, s2, eta, m) +
        C45i(k1, k4, k3, k2, s, s1, s2, eta, m) +
        C45i(k2, k3, k1, k4, s, s1, s2, eta, m) +
        C45i(k2, k4, k1, k3, s, s1, s2, eta, m) +
        C45i(k3, k4, k1, k2, s, s1, s2, eta, m) +
        C45i(k2, k1, k3, k4, s, s1, s2, eta, m) +
        C45i(k3, k1, k2, k4, s, s1, s2, eta, m) +
        C45i(k4, k1, k3, k2, s, s1, s2, eta, m) +
        C45i(k3, k2, k1, k4, s, s1, s2, eta, m) +
        C45i(k4, k2, k1, k3, s, s1, s2, eta, m) +
        C45i(k4, k3, k1, k2, s, s1, s2, eta, m)
    )


def C45s(k1, k2, k3, k4, eta, m):
    def integrand(s2, s1, s):
        return C45si(k1, k2, k3, k4, s, s1, s2, eta, m)

    def s2_lim(s1, s):
        return [constants.ETA_IN, s1]

    def s1_lim(s):
        return [constants.ETA_IN,  s]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    return integrate.nquad(integrand, [s2_lim, s1_lim, s_lim], opts=[options, options, options, options])

@njit
def F2si(k1, k2, *args):
    s, eta, m = args
    if (s >= eta):
        return 0

    return C21si(k1, k2, s, eta, m)

def F2s(k1, k2, *args):
    eta, m = args
    mean, std = C21s(k1, k2, eta, m)
    return ufloat(mean, std)

# F3 as in papers on mode coupling
@njit
def F3si(k1, k2, k3, *args):
    s, s1, eta, m = args
    res = 0

    if (s >= eta) or (s1 >= eta):
        return 0

    if s1 < s:
        res += C31si(k1, k2, k3, s, s1, eta, m)

    # divide by eta - constants.ETA_IN since C32 is independent of s1
    res += C32si(k1, k2, k3, s, eta, m) / (eta - constants.ETA_IN)
    return res


def F3s(k1, k2, k3, *args):
    eta, m = args
    res1, sdev1 = C31s(k1, k2, k3, eta, m)
    r1 = ufloat(res1, sdev1)
    res2, sdev2 = C32s(k1, k2, k3, eta, m)
    r2 = ufloat(res2, sdev2)
    return  r1 + r2


def F3s_nquad(k1, k2, k3, eta, m):
    def integrand(s1, s):
        return F3si(k1, k2, k3, s, s1, eta, m)
    
    def s1_lim(s):
        return [constants.ETA_IN, eta]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    mean, std =  integrate.nquad(integrand, [s1_lim, s_lim], opts=[options, options])
    res = ufloat(mean, std)
    return res



@njit
def F4si(k1, k2, k3, k4, *args):
    s, s1, s2, eta, m = args

    if (s >= eta) or (s1 >= eta) or (s2 >= eta):
        return 0

    res = 0

    res += C41si(k1, k2, k3, k4, s, eta, m) / (eta - constants.ETA_IN)**2

    if s1 < s:
        res += C42si(k1, k2, k3, k4, s, s1, eta, m) / (eta - constants.ETA_IN)  # s1 from constants.ETA_IN to s

    if s1 < s and s2 < s:
        # s1 and s2 from constants.ETA_IN to s
        res += C43si(k1, k2, k3, k4, s, s1, s2, eta, m)

    if s1 < s and s2 < s1:
        # s1 from constants.ETA_IN to s, s2 from constants.ETA_IN to s1
        res += C45si(k1, k2, k3, k4, s, s1, s2, eta, m)

    if s1 < s:
        res += C44si(k1, k2, k3, k4, s, s1, eta, m) / (eta - constants.ETA_IN)

    return res


def F4s(k1, k2, k3, k4, *args):
    eta, m = args

    res1, sdev1 = C41s(k1, k2, k3, k4, eta, m)
    res2, sdev2 = C42s(k1, k2, k3, k4, eta, m)
    res3, sdev3 = C43s(k1, k2, k3, k4, eta, m)
    res4, sdev4 = C45s(k1, k2, k3, k4, eta, m)
    res5, sdev5 = C44s(k1, k2, k3, k4, eta, m)
    r1 = ufloat(res1, sdev1)
    r2 = ufloat(res2, sdev2)
    r3 = ufloat(res3, sdev3)
    r4 = ufloat(res4, sdev4)
    r5 = ufloat(res5, sdev5)
    return r1 + r2 + r3 + r4 + r5


def F4s_nquad(k1, k2, k3, k4, eta, m):
    def integrand(s2, s1, s):
        return F4si(k1, k2, k3, k4, s, s1, s2, eta, m)

    def s2_lim(s1, s):
        return [constants.ETA_IN, eta]

    def s1_lim(s):
        return [constants.ETA_IN, eta]

    def s_lim():
        return [constants.ETA_IN, eta]

    options = constants.NQUAD_OPTIONS

    mean, std = integrate.nquad(integrand, [s2_lim, s1_lim, s_lim], opts=[options, options, options, options])
    res = ufloat(mean, std)
    return res 

@njit 
def F2si_vegas(x, *args):
    s = x[0]
    k1, k2, eta, m = args
    return F2si(k1, k2, s, eta, m)

@njit
def F3si_vegas(x, *args):
    s, s1 = x
    k1, k2, k3, eta, m = args
    return F3si(k1, k2, k3, s, s1, eta, m)

@njit
def F4si_vegas(x, *args):
    s, s1, s2 = x
    k1, k2, k3, k4, eta, m = args
    return F4si(k1, k2, k3, k4, s, s1, s2, eta, m)


F2_integration_boundaries = [[constants.ETA_IN, constants.ETA_FIN]]
F3_integration_boundaries = [[constants.ETA_IN, constants.ETA_FIN], [constants.ETA_IN, constants.ETA_FIN]]
F4_integration_boundaries = [[constants.ETA_IN, constants.ETA_FIN], [constants.ETA_IN, constants.ETA_FIN], [constants.ETA_IN, constants.ETA_FIN]]
F2s_vegas = mc.Vegas_integrator(F2si_vegas, F2_integration_boundaries)
F3s_vegas = mc.Vegas_integrator(F3si_vegas, F3_integration_boundaries)
F4s_vegas = mc.Vegas_integrator(F4si_vegas, F4_integration_boundaries)
