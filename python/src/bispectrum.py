import numpy as np
from scipy.spatial.transform import Rotation as R
from numba import njit

@njit
def generate_vectors(phi, ct, q, ct12, r1, r2):
  q  = q  * np.array([np.sqrt(1 - ct**2)*np.cos(phi), np.sqrt(1 - ct**2)*np.sin(phi), ct])
  k1 = r1 * np.array([0                             , 0                             , 1.])
  k2 = r2 * np.array([np.sqrt(1 - ct12**2)          , 0                             , ct12])
  k3 = -k1 - k2
  return q, k1, k2, k3

@njit 
def spherical2cartesian(radius, cosine_theta, phi):
  #How does this mapping work?
  #phi = polar angle, theta = azimuthal angle
  #We choose k1 to point in the direction of the z-axis in a spherical coordinate system
  #The z- and x-axis are chosen to span the plane spanned by k1 and k2 with the x-axis pointing in the same direction as k2 
  #k2 therefore has a vanishing y-component and its position can be characterised fully by the cosine between k1 and k2
  #The coordinates of the vector q are then given in this coordinate system
  #Again giving the cosine of the angle between q and k1 is sufficient
  #This is because the arcus cosine is bijective on [-1, 1]
  #k3 is calculated s.t. k1 + k2 + k3 = 0

  x  = radius * np.array([np.sqrt(1 - cosine_theta**2)*np.cos(phi), np.sqrt(1 - cosine_theta**2)*np.sin(phi), cosine_theta])
  return x

def generate_triangle_numpy(theta = 2*np.pi/3, r1 = 1.0, r2 = 1.0, degrees = False):
  if degrees:
    theta = theta/(360.)*2*np.pi
  rotation = R.from_euler('z', theta)
  k1 = np.array([r1, 0, 0], dtype=np.float)
  k2 = rotation.apply(k1)/r1*r2
  k3 = - k1 - k2
  return k1, k2, k3

#Default arguments generate equilateral triangle with unit length
@njit
def generate_triangle(theta = 2*np.pi/3, r1 = 1., r2 =1.):
  ct12 = np.cos(theta)
  k1   = r1 * np.array([0                    , 0 , 1.])
  k2   = r2 * np.array([np.sqrt(1 - ct12**2) , 0 , ct12])
  k3   = -k1 - k2
  return k1, k2, k3


#Cartesian bispectrum at tree level
def generic_B_tree(F2s, P, k1, k2, k3, *args):
  p1 = 2*F2s(k1, k2, *args) * P(k1, *args) * P(k2, *args)
  p2 = 2*F2s(k2, k3, *args) * P(k2, *args) * P(k3, *args)
  p3 = 2*F2s(k3, k1, *args) * P(k3, *args) * P(k1, *args)
  return p1 + p2 + p3

#From above eq. 177 in Bernardeau 2002
def generic_Sigma_tree(P, k1, k2, k3, *args):
  p1 = P(k1, *args) * P(k2, *args)
  p2 = P(k2, *args) * P(k3, *args)
  p3 = P(k3, *args) * P(k1, *args)
  return p1 + p2 + p3


#From above eq. 177 in Bernardeau 2002
#P1L is nonlinear power spectrum with one-loop corrections
def generic_Sigma_1L(P, P1L, k1, k2, k3, *args):
  x1 = P1L(k1, *args)
  x2 = P1L(k2, *args)
  x3 = P1L(k3, *args)
  p1 = x1 * P(k2, *args)
  p2 = x2 * P(k3, *args)
  p3 = x3 * P(k1, *args)
  p4 = P(k1, *args) * x2
  p5 = P(k2, *args) * x3
  p6 = P(k3, *args) * x1
  return p1 + p2 + p3 + p4 + p5 + p6

#Reduced cartesian bispectrum
def generic_B_tree_red(F2s, P, k1, k2, k3, *args):
  return generic_B_tree(F2s, P, k1, k2, k3, *args)/generic_Sigma_tree(P, k1, k2, k3, *args)


### Cartesian mode-coupling

#We need three independent s integrations from eta_in to eta for the three factors F2s
@njit
def B222_00(P, F2s, q, k1, k2, k3, s, s2, s3, eta, m):
  return 8*P(q, eta, m)*P(q + k1, eta, m)*P(q-k2, eta, m)*F2s(-q, q + k1, s, eta, m)*F2s(-q-k1, q-k2, s2, eta, m)*F2s(k2 -q, q, s3, eta, m)


#IR-safe integrand implementation following Baldauf 2014
@njit
def B222_01(P, F2s, q, k1, k2, k3, s, s2, s3, eta, m):
  res = 0.

  if (np.linalg.norm(k1 + q) > np.linalg.norm(q)) and (np.linalg.norm(k2 - q) > np.linalg.norm(q)):
    res += B222_00(P, F2s,  q, k1, k2, k3, s, s2, s3, eta, m)

  if (np.linalg.norm(k1 - q) > np.linalg.norm(q)) and (np.linalg.norm(k2 + q) > np.linalg.norm(q)):
    res += B222_00(P, F2s, -q, k1, k2, k3, s, s2, s3, eta, m)

  res *= 0.5

  return res


@njit
def B222(P, F2s, q, k1, k2, k3, s, s2, s3, eta, m):
  return \
  B222_01(P, F2s, q, k1, k2, k3, s, s2, s3, eta, m) + \
  B222_01(P, F2s, q, k1, k3, k2, s, s2, s3, eta, m) + \
  B222_01(P, F2s, q, k3, k2, k1, s, s2, s3, eta, m) 


#We need three independent s-integrations, 2 for F3s and one for s2s
@njit
def B3211p_0(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m):
  return 6*P(k3, eta, m)*P(q, eta, m)*F3s(-q, q-k2, -k3, s, s2, eta, m)*P(q-k2, eta, m)*F2s(q, k2-q, s3, eta, m)


#IR-safe integrand implementation following Baldauf 2014
@njit
def B3211p(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m):
  res = 0.
  if np.linalg.norm(k2 - q) > np.linalg.norm(q):
      res += B3211p_0(P, F2s, F3s,  q, k1, k2, k3, s, s2, s3, eta, m)
  if np.linalg.norm(k2 + q) > np.linalg.norm(q):
      res += B3211p_0(P, F2s, F3s, -q, k1, k2, k3, s, s2, s3, eta, m)
  return res

#6 full permutations
@njit
def B3211(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m):
  return \
  B3211p(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m) + \
  B3211p(P, F2s, F3s, q, k1, k3, k2, s, s2, s3, eta, m) + \
  B3211p(P, F2s, F3s, q, k3, k2, k1, s, s2, s3, eta, m) + \
  B3211p(P, F2s, F3s, q, k3, k1, k2, s, s2, s3, eta, m) + \
  B3211p(P, F2s, F3s, q, k2, k1, k3, s, s2, s3, eta, m) + \
  B3211p(P, F2s, F3s, q, k2, k3, k1, s, s2, s3, eta, m)

@njit
def B3212p(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m):
  return 6*P(k2, eta, m)*P(k3, eta, m)*F2s(k2, k3, s, eta, m)*P(q, eta, m)*F3s(k3, q, -q, s2, s3, eta, m)
  
#6 full permutations
@njit
def B3212(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m):
  return \
  B3212p(P, F2s, F3s, q, k1, k2, k3, s, s2, s3, eta, m) + \
  B3212p(P, F2s, F3s, q, k1, k3, k2, s, s2, s3, eta, m) + \
  B3212p(P, F2s, F3s, q, k3, k2, k1, s, s2, s3, eta, m) + \
  B3212p(P, F2s, F3s, q, k3, k1, k2, s, s2, s3, eta, m) + \
  B3212p(P, F2s, F3s, q, k2, k1, k3, s, s2, s3, eta, m) + \
  B3212p(P, F2s, F3s, q, k2, k3, k1, s, s2, s3, eta, m)

@njit
def B411p(P, F4s, q, k1, k2, k3, s, s2, s3, eta, m):
  return 12*P(k2, eta, m)*P(k3, eta, m)*P(q, eta, m)*F4s(q, -q, -k2, -k3, s, s2, s3, eta, m)

#3 cyclic permutations
@njit
def B411(P, F4s, q, k1, k2, k3, s, s2, s3, eta, m):
  return \
  B411p(P, F4s, q, k1, k2, k3, s, s2, s3, eta, m) + \
  B411p(P, F4s, q, k3, k1, k2, s, s2, s3, eta, m) + \
  B411p(P, F4s, q, k2, k3, k1, s, s2, s3, eta, m)

def make_tree_bispectrum_utilities(P, F2s, P1L):
    #Reduced scale-free cartesian CDM-bispectrum as function of spectral index
    def B_tree(k1, k2, k3, *args):
        return generic_B_tree_red(F2s, P, k1, k2, k3, *args)

    def Sigma_tree(k1, k2, k3, *args):
        return generic_Sigma_tree(P, k1, k2, k3, *args)

    def Sigma_1L(k1, k2, k3, *args):
        return generic_Sigma_1L(P, P1L, k1, k2, k3, *args)

    #Reduced scale-free cartesian CDM-bispectrum as function of spectral index
    def B_tree_red(k1, k2, k3, *args):
        return generic_B_tree_red(F2s, P, k1, k2, k3, *args)
    
    bispectrum_utilities = {
        "bispectrum tree":      B_tree,
        "sigma tree":           Sigma_tree,
        "sigma loop":           Sigma_1L,
        "red. bispectrum tree": B_tree_red
    }
    return bispectrum_utilities

#From above eq. 177 in Bernardeau 2002
#P1L is nonlinear power spectrum with one-loop corrections
def B1L_red(b, b1L, sigma, sigma1L):
  return (b1L   - sigma1L     * b)/sigma

def B_full_red(b, b1L, sigma, sigma1L):
  return (b + b1L)/(sigma + sigma1L)

def compute_bispectrum(x, k1, k2, k3, eta, m, bispectrum_util, bispectrum_integrator, debug=False):
    
    if debug:
        print("computing sigma tree:")
    s0      = bispectrum_util["sigma tree"](k1, k2, k3, eta, m)
    if debug:
        print(s0)
        print("computing sigma loop:")
    s1  = bispectrum_util["sigma loop"](k1, k2, k3, eta, m)
    if debug:
        print(s1)
        print("computing reduced bispectrum tree:")
    br0     = bispectrum_util["red. bispectrum tree"](k1, k2, k3, eta, m)
    if debug:
        print(br0)
        print("computing bispectrum loop:")
    b1  = bispectrum_integrator(k1, k2, k3, eta, m)
    if debug:
        print(b1)
        print("computing reduced bispectrum")
    br1     = B1L_red(br0, b1, s0, s1)
    if debug:
        print(br1)

    output = {
        "x":                        x,
        "sigma tree":               s0,
        "sigma loop":               s1,
        "red. bispectrum tree":     br0,
        "bispectrum loop":          b1,
        "red. bispectrum":          br1,
    }
    
    return output 

def create_compute_bispectrum(bispectrum_util, bispectrum_integrator, debug = False):
  def f(x, k1, k2, k3, eta, m):
    return compute_bispectrum(x, k1, k2, k3, eta, m, bispectrum_util, bispectrum_integrator, debug)
  return f