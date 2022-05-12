import numpy as np
from enum import Enum, auto
from numba import njit 

####
#Constants
####

ELEMENTARY_CHARGE           = 1.602176634e-19 #C
ELEMENTARY_CHARGE_OVER_HBAR = 1519267447000000.0 #A J^-1
SPEED_OF_LIGHT              = 299792458.0 #m s^-1
MEGAPARSEC                  = 3.0857e22 #m
HBAR                        = 6.582119569e-16 #eV⋅s
EPS                         = np.finfo(float).eps
G_NEWTON                    = 6.674e-11 # m^3/kg/s^2

####
#Input parameters defining dark energy cosmology
####
class CosmologyModes(Enum):
    MATTER_DOMINATED = 1 #for Einstein-de Sitter universe with OMEGA_M = 1
    DARK_ENERGY      = 2 #for dark energy universe defined above
    LAMBDA           = 3


COSMOLOGY_MODE    = CosmologyModes.DARK_ENERGY

if COSMOLOGY_MODE == CosmologyModes.MATTER_DOMINATED:
  OMEGA_M         = 1.0
  OMEGA_DM        = OMEGA_M
  FDM_M           = 1e-23 #Mass of FDM in units of eV
  H_HUBBLE        = 0.68
  OMEGA_M_GROWTH  = 1.0
elif COSMOLOGY_MODE == CosmologyModes.DARK_ENERGY:
  FDM_M           = 1e-23
  OMEGA_M         = 0.3
  OMEGA_M_GROWTH  = 0.3
  OMEGA_B         = 0.04
  H_HUBBLE        = 0.68
  OMEGA_DM        = OMEGA_M - OMEGA_B
  OMEGA_Q         = 1 - OMEGA_M
  W_DARK_ENERGY   = -0.9 
  W_DARK_ENERGY_PRIME = 0.0
elif COSMOLOGY_MODE == CosmologyModes.LAMBDA:
  FDM_M           = 8e-23
  OMEGA_M         = 0.315903
  OMEGA_M_GROWTH  = 0.315903
  H_HUBBLE        = 0.67321170
  OMEGA_B         = 0.022383/H_HUBBLE**2
  OMEGA_DM        = OMEGA_M - OMEGA_B
  OMEGA_Q         = 1 - OMEGA_M
else:
  raise ValueError("Cosmology Mode not defined")


H0_HUBBLE       = 100*H_HUBBLE #km/s/Mpc
H0_SI_HUBBLE    = H0_HUBBLE*1000/MEGAPARSEC #1/s
RHO_BACKGROUND  = OMEGA_M_GROWTH*1.8788e-26*H_HUBBLE #kg/m^3


if COSMOLOGY_MODE == CosmologyModes.MATTER_DOMINATED:

  def hubble(a):
    return H0_SI_HUBBLE * a**(-1.5)

  def dloghubble(a):
    return (-1.5)/a

  NUMFIT_DPLUS_PARAMETERS   = np.array([0.6097572,  6.45948689])#[1.12771078 7.18362996]
  NUMFIT_DMINUS_PARAMETERS   = np.array([0.04547614, 10.3235837 ])#[1.12771078 7.18362996]
  #NUMFIT_DMINUS_PARAMETERS  = np.array([0.09028865, 1.89129119])
  #Fitted parameters:
  #[0.04398693 2.33157639]
  #RMS residual = 7.546628901122081

elif COSMOLOGY_MODE == CosmologyModes.DARK_ENERGY:
 
  def hubble(a):
    result = OMEGA_M / a**3
    aux = -(1.0 + W_DARK_ENERGY + W_DARK_ENERGY_PRIME) * np.log(a) + W_DARK_ENERGY_PRIME * (a - 1.0)
    result += OMEGA_Q * np.exp(3.0 * aux)

    result = H0_SI_HUBBLE * np.sqrt(result)
    return result

  def dloghubble(a):
    p1      = OMEGA_M / a**3
    aux     = -(1.0 + W_DARK_ENERGY + W_DARK_ENERGY_PRIME) * np.log(a) + W_DARK_ENERGY_PRIME * (a - 1.0)
    p1     += OMEGA_Q * np.exp(3.0 * aux)
    p2      = -3 * OMEGA_M / a**4
    p2     += OMEGA_Q * np.exp(3.0 * aux) * (3 * (-(1.0 + W_DARK_ENERGY + W_DARK_ENERGY_PRIME)/a + W_DARK_ENERGY_PRIME))
    result  = 0.5 * p2/p1
    return result

  NUMFIT_DPLUS_PARAMETERS  = np.array([1.11665851, 7.20473245])
  #Fitted parameters:
  #[3.51790335 8.54257481]
  #RMS residual = 0.34346466397437064
  NUMFIT_DMINUS_PARAMETERS = np.array([2.23506882e-12, 1.64744511e+01])
  #Fitted parameters:
  #[1.32871261e-05 3.09892968e+01]
  #RMS residual = 6.565750696503666

elif COSMOLOGY_MODE == CosmologyModes.LAMBDA:
  def hubble(a):
    result  = OMEGA_M / a**3 + OMEGA_Q

    result = H0_SI_HUBBLE * np.sqrt(result)
    return result

  def dloghubble(a):
    p1      = OMEGA_M / a**3 + OMEGA_Q
    result  = 0.5 * (-3) * OMEGA_M * a**(-4) / p1
    return result

  NUMFIT_DPLUS_PARAMETERS  = np.array([1.13971596, 7.19942633])
  NUMFIT_DMINUS_PARAMETERS = np.array([2.23506882e-12, 1.64744511e+01])

class LinearGrowthModes(Enum):
    ANALYTICAL_FDM = auto() #for Einstein-de Sitter universe with OMEGA_M = 1
    NUMERICAL_FDM  = auto() #for dark energy universe defined above
    SUPPRESSED_CDM = auto() #for dark energy universe defined above with Jeans scale cutoff
    NUMERICAL_FIT  = auto() #for dark energy universe defined above with Jeans scale cutoff

class NonlinearGrowthModes(Enum):
    ANALYTICAL_FDM = auto() #for Einstein-de Sitter universe with OMEGA_M = 1
    NUMERICAL_FDM  = auto() #for dark energy universe defined above
    SUPPRESSED_CDM = auto() #for dark energy universe defined above with Jeans scale cutoff
    ANALYTICAL_CDM = auto() #For FDM mode couplings
    NUMERICAL_FIT  = auto() #for dark energy universe defined above with Jeans scale cutoff


LINEAR_GROWTH_MODE    = LinearGrowthModes.NUMERICAL_FIT
NONLINEAR_GROWTH_MODE = NonlinearGrowthModes.ANALYTICAL_FDM

#We use CAMB to compute initial linear power spectrum at Z_IN
#The nonlinear corrections are computed for the evolution from Z_IN to Z_FIN

#Helper functions to convert between time variables
@njit
def a_from_z(z):
  return 1/(z+1)

@njit
def z_from_a(a):
  return 1./a - 1.

@njit
def eta_from_z(z):
  return 2*np.sqrt(a_from_z(z))

@njit
def eta_from_a(a):
  return 2*np.sqrt(a)

@njit
def a_from_eta(eta):
  return eta**2/4.

@njit
def z_from_eta(eta):
  a = a_from_eta(eta)
  return 1./a - 1.


#A_IN       = EPS
#A_FIN      = 1
#Z_IN       = z_from_a(A_IN)
#Z_FIN      = z_from_a(A_FIN)
#ETA_IN     = eta_from_a(A_IN)
#ETA_FIN    = eta_from_a(A_FIN)

A_IN       = a_from_z(100)
A_FIN      = a_from_z(0)
Z_IN       = z_from_a(A_IN)
Z_FIN      = z_from_a(A_FIN)
ETA_IN     = eta_from_a(A_IN)
ETA_FIN    = eta_from_a(A_FIN)

A_TODAY    = 1
Z_TODAY    = 0
ETA_TODAY  = 2

#Radial integration limits for Monte-Carlo integration
#The IR cutoff is justified because we do not want to integrate over scales much larger than the Hubble radius
#Large scales are physically suppressed in FDM which is why we do not need a UV cutoff
LOWER_RADIAL_CUTOFF = 1e-4
UPPER_RADIAL_CUTOFF = np.inf

NQUAD_OPTIONS = {'limit':1000, 'epsrel':10e-4, 'epsabs':10e-6}

####
# CUBA INTEGRATOR OPTIONS for 1-loop power spectrum 
####

VEGAS_NEVAL         = 1000
VEGAS_NITN          = 10 
VEGAS_RESTART_MAX   = 10

IR = LOWER_RADIAL_CUTOFF / (1 + LOWER_RADIAL_CUTOFF)
UV = 1
BOX_IR = 1/5 
BOX_UV = 16./17
CDM_P1L_integration_boundaries     =  [              [-1, 1], [IR,         UV]]
CDM_P1L_sc_integration_boundaries  =  [              [-1, 1], [BOX_IR, BOX_UV]]
FDM_P1L_integration_boundaries     =  [              [-1, 1], [IR,         UV], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY]]
FDM_P1L_sc_integration_boundaries  =  [              [-1, 1], [BOX_IR, BOX_UV], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY]]
CDM_B1L_integration_boundaries     =  [[0, 2*np.pi], [-1, 1], [IR,         UV]]
CDM_B1L_sc_integration_boundaries  =  [[0, 2*np.pi], [-1, 1], [BOX_IR, BOX_UV]]
FDM_B1L_integration_boundaries     =  [[0, 2*np.pi], [-1, 1], [IR,         UV], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY]]
FDM_B1L_sc_integration_boundaries  =  [[0, 2*np.pi], [-1, 1], [BOX_IR, BOX_UV], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY], [ETA_IN, ETA_TODAY]]



####
# CONFIGURATION OF CAMB INITIAL SPECTRUM
####
USE_CAMB_FIT     = True #Do not compute initial spectrum using CAMB but use fit instead
OMBH2            = 0.02230
OMCH2            = 0.1188
NS_SPECTRUM      = 0.965

MIN_K              = 1e-4
MAX_K              = 200
NUMBER_K_POINTS    = 400

CAMB_SPECTRUM_PATH = "../../C/IC/gamer_powerspectrum/planck_2018_axion_matterpower.dat"

####
# OPTIONS FOR SPLINES FOR GROWTH AND DECAY FUNCTIONS D+ and D-
####
LOAD_SPLINE_FROM_FILE  = False
SAVE_SPLINE_TO_FILE    = False
N_SPLINE_MOMENTA       = 1000
N_SPLINE_SCALE_FACTORS = 500
A0_INTEGRATION         = A_IN
SPLINE_MOMENTA         = np.logspace(-4, 2, N_SPLINE_MOMENTA)
SPLINE_SCALE_FACTORS   = np.insert(np.logspace(-2, 0, N_SPLINE_SCALE_FACTORS - 1), 0, A_IN)
SPLINE_FILE            = f"splines/fdm_cos_{COSMOLOGY_MODE}_om_{OMEGA_M_GROWTH}_m_{FDM_M}_Nk_{N_SPLINE_MOMENTA}_Na_{N_SPLINE_SCALE_FACTORS}.npz"

FIT_TO_SPLINE  = False