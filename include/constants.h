
#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <string>
#include <math.h>
#include <memory>
#include <gsl/gsl_const_num.h>
#include <gsl/gsl_const_mksa.h>


/***
 * This header file defines all physical constants as well as the cosmology required by the perturbation code. 
 * Further, it defines hyperparameters required for the CUBA integrations (further hyperparameters can be found at the top of spectrum.cpp)
 * Also, it contains hard-coded references to all the spline files on the hard disk. 
 */


/***
 * PHYSICAL CONSTANTS IN SI UNITS
 * **/

#define const_electron_charge GSL_CONST_MKSA_ELECTRON_CHARGE
#define const_hbar GSL_CONST_MKSA_PLANCKS_CONSTANT_HBAR
#define const_clight GSL_CONST_MKSA_SPEED_OF_LIGHT
#define const_parsec GSL_CONST_MKSA_PARSEC

#define const_kparsec (1.0e3 * const_parsec)
#define const_mparsec (1.0e6 * const_parsec)
#define const_gparsec (1.0e9 * const_parsec)
#define const_hubble (1.0e5 / const_mparsec)
#define const_dhubble (const_clight / const_hubble / const_mparsec)
#define const_thubble (1.0 / const_hubble / 3600.0 / 24.0 / 365.25 / 1e9)
#define const_tcmb 2.726
#define const_hbar_in_ev (const_hbar/GSL_CONST_MKSA_ELECTRON_VOLT)

#define const_a_in     0.01
#define const_eta_in   0.2
#define const_a_fin    1
#define const_eta_fin  2

const std::string cosmo_string  = "de_z99";
const std::string cdm_camb_path = "IC/cdm.dat";

const double fdm_masses[3] = {1e-21, 1e-22, 1e-23};
const std::string fdm_mass_strings[3] = {"m21", "m22", "m23"};
const std::string fdm_camb_paths[3] = {"IC/m21.dat", "IC/m22.dat", "IC/m23.dat"};

//const std::string cdm_loop_spectrum_path        = "";//"splines/cdm.dat";
//const std::string fdm_loop_spectrum_path        = "";//"splines/"+fdm_string+".dat";
//const std::string cdm_loop_lensing_path         = "";//"data/lensing_spectrum/loop/cdm_new.dat";
//const std::string fdm_loop_lensing_path         = "";//"data/lensing_spectrum/loop/"+fdm_string+".dat";
//const std::string loop_difference_lensing_path  = "";//"data/lensing_spectrum/loop/"+fdm_string+"_diff.dat";


#define SAVE_D_SPLINES 1
#define LOAD_D_SPLINES 0

/***
 * COSMOLOGY
 * **/
#define const_omega_m  0.315903
#define const_omega_b  0.04938726593811469
#define const_h_hubble 0.67321170

#define const_omega_dm (const_omega_m - const_omega_b)
#define const_omega_q  (1.0 - const_omega_m)
#define const_w_dark_energy -0.9
#define const_w_dark_energy_prime 0.0
#define const_sigma8  0.8
#define const_rsmooth 8.0   

const double const_hubble_bar  = const_hubble * pow(const_omega_m, 0.5);

//Spectral index in Bardeen spectrum
const double n_s = 1.0;

//Lower cutoff for radial momentum integrations in loop spectra, translates to k_min = IR_CUTOFF/(1 - IR_CUTOFF)
#define VEGAS_IR_CUTOFF 1e-4
//Lower cutoff for radial momentum integrations in loop spectra, translates to k_max = infinity
#define VEGAS_UV_CUTOFF 1

//Number of time steps (scale factors) used in loop spectrum interpolation
const size_t p1l_spline_na = 20;
//Number of momentum steps used in loop spectrum interpolation
const size_t p1l_spline_nk = 100;

#if 0
//Number of time steps (scale factors) used in loop bispectrum interpolation
const size_t b1l_spline_na = 5;
//Number of momentum steps used in loop bispectrum interpolation
const size_t b1l_spline_nk = 15;
#endif 

const double TOMO_SPECTRUM_ABSERR         = 0;
const double TOMO_SPECTRUM_RELERR         = 1e-5;
const double TOMO_CUBA_SPECTRUM_ABSERR    = 0;
const double TOMO_CUBA_SPECTRUM_RELERR    = 1e-4;
const double TOMO_BISPECTRUM_ABSERR       = 0;
const double TOMO_BISPECTRUM_RELERR       = 1e-3;
const double TOMO_LOOP_BISPECTRUM_ABSERR  = 0;
const double TOMO_LOOP_BISPECTRUM_RELERR  = 5e-2;
const double TOMO_TRISPECTRUM_ABSERR      = 0;
const double TOMO_TRISPECTRUM_RELERR      = 1e-1;

const double SPECTRUM_S2N_ABSERR          = 0;
const double SPECTRUM_S2N_RELERR          = 1e-2;
const double SPECTRUM_CHISQ_ABSERR        = 0;
const double SPECTRUM_CHISQ_RELERR        = 1e-2;
const double BISPECTRUM_S2N_ABSERR        = 0;
const double BISPECTRUM_S2N_RELERR        = 1e-2;
const double BISPECTRUM_CHISQ_ABSERR      = 0;
const double BISPECTRUM_CHISQ_RELERR      = 1e-1;
const double TRISPECTRUM_CHISQ_ABSERR     = 0;
const double TRISPECTRUM_CHISQ_RELERR     = 1e-1;
const double TRISPECTRUM_S2N_ABSERR       = 0;
const double TRISPECTRUM_S2N_RELERR       = 1e-2;

const size_t INTEGRATION_WORKSPACE_SIZE = 10000;
const double F2_INTEGRATION_ABSERR      = 1e-20;
const double F2_INTEGRATION_RELERR      = 1e-4;
const size_t F2_SUBDIVISIONS            = 10000;
const double F3_INTEGRATION_ABSERR      = 5e-2;
const double F3_INTEGRATION_RELERR      = 5e-2;
const size_t F3_SUBDIVISIONS            = 10000;
const double T0_INTEGRATION_ABSERR      = 1e-20;
const double T0_INTEGRATION_RELERR      = 1e-5;
const size_t T0_SUBDIVISIONS            = 10000;

const double CUBA_ABSERR = 1e-30;
const double CUBA_RELERR = 1e-1;
const double CUHRE_RELERR = 1e-2;


#endif
