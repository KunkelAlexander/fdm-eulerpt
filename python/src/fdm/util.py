import constants as c
import fdm.util 
from special_functions import k_norm 
from numba import njit 
import numpy as np 

@njit
def m_bar(m):
  return m * c.ELEMENTARY_CHARGE_OVER_HBAR #1/s

@njit
def mbH0(m, omega_m = c.OMEGA_M):
  # 2*k^2 / (H0*Omega_m^1/2 mb) * c^2 = 2*k^2 1/257.3 Mpc^2
  return (m_bar(m) * c.H0_SI_HUBBLE * omega_m**(0.5))/c.SPEED_OF_LIGHT**2*c.MEGAPARSEC**2 #1/m**2

#Mass and momentum dependent FDM-scale
@njit
def b_f(k, m):
  return k_norm(k)**2*2/mbH0(m)



#Quantum jeans scale as function of scale factor and mass
#Equation (18) from Li 2020
@njit
def li_kj(a, m, omega_m = c.OMEGA_M):
  return 44.7*(6*a*omega_m/0.3)**(0.25) * (100/70*m/(1e-22))**0.5  #Mpc^-1

@njit
def li_aj(k, m, omega_m = c.OMEGA_M_GROWTH):
  return 1/(6*omega_m/0.3) * (k * (c.H0_HUBBLE/70*m/(1e-22))**(-0.5)/44.7)**4  #Mpc^-1


#Quantum jeans scale as function of scale factor and mass
#Equation (18) from Lagüe 2020 agrees with Li
@njit
def lague_kj(a, m, omega_dm = c.OMEGA_M_GROWTH):
  return 66.5*a**(0.25) * (m/(1e-22))**0.5 * (omega_dm*c.H_HUBBLE**2/0.12)**0.25  #Mpc^-1

#The following two functions are from the fit of the growth factor in Lagüe 2020
#Equation (28) from Lagüe 2020
@njit
def lague_k0(a, m):
  return 0.0334*(m/(1e-24))**(-0.00485)*(c.OMEGA_M/c.OMEGA_DM)**0.527 * lague_kj(a, m, omega_dm = c.OMEGA_M_GROWTH)

#Equation (29) from Lagüe 2020
@njit
def lague_alpha(m):
  return 0.194*(m/(1e-24))**(-0.501)*(c.OMEGA_M/c.OMEGA_DM)**0.0829


#Use empirical transfer function
#from Hu et al. 2000
#Valid at matter radiation equality z = 1100 - 1300
@njit 
def FDM_transfer(k, m):
  kJQ = 9*(m/1e-22)**0.5 # where k in 1/Mpc and m in eV
  x   = 1.61*(m/1e-22)**(1/18)*k/kJQ
  return np.cos(x**3)/(1 + x**8)


#This should roughly correspond to D
@njit 
def D_empirical(k, eta, m):
  return FDM_transfer(k, m)*(eta/c.ETA_IN)**2