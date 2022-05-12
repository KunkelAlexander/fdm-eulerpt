#ifndef __TOMO_H__
#define __TOMO_H__

// weak lensing spectrum, bispectrum and trispectrum tomography
// intrinsic ellipticity spectra for spiral and elliptical galaxies
// cross-correlations between intrinsic alignments and weak lensing
//
// by Bjoern Malte Schaefer, GSFP/Heidelberg, bjoern.malte.schaefer@uni-heidelberg.de

// gcc-mp-4.8 -o tomo tomo.c -L. -L/sw/lib -I/sw/include -lgsl -lgslcblas -lcuba -lm -fopenmp
// CUBA parallelisation by setting environment variable CUBACORES
// OpenMP parallelisation of intensive l- and bin-looping

// --- includes ---
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <memory>
#include <math.h>
#include "cuba/cuba.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_const_num.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_statistics.h>

const int on = 0;
const int off = 1;

#define ASTEP 200 // increase to 200
#define AMIN 0.01
#define AMAX 1.0
#define CHIMIN 3e1
#define CHIMAX 5e3
#define CHISTEP 100
#define LMAX 10000 // ultimately: l=1e4 for integrations, and l=3e3 for intrinsic alignments
#define LMIN 10
#define LSTEP 25

#define DCHI 10.0 // overlap of the two dchi-integrations for the Limber-equations
#define CMIN 0.01
#define CMAX 1.0e3
#define CSTEP 100
#define TMIN (0.1 * spirou_min2rad)
#define TMAX (1.0e3 * spirou_min2rad)
#define TSTEP 100
#define index_pp 0 // flag for <e+e+> and <exex>
#define index_cc 1
#define index_sp 2
#define index_ss 3

#define NBIN 1			 // 1: no tomography, n: n-bin tomography

#define NBIN2 (2 * NBIN) // twice the bin number, for large covariance matrices in cross-measurements
#define mode_lensing 0
#define mode_ebmode 1
#define mode_inflation 2
#define mode_suyama 3
#define mode_ellipticity 4
#define mode_tomography 5
#define mode_fisher 6
#define mode_bias 7
#define mode_isw 8
#define mode_spectrum_s2n 9
#define mode_spectrum_lensing_chi_squared 10
#define mode_bispectrum_lensing_chi_squared 11
#define mode_trispectrum_lensing_chi_squared 12
#define mode_trispectrum_s2n 13
#define mode_old_spectrum_lensing_chi_squared 14


enum class spectrum_type {
type_local,
type_equil,
type_ortho,
type_cdm  ,
type_fdm  ,
type_cdm_fdm_ic,
};

#define inside 0
#define outside 1

#define NFISHER 5

// --- structures ---
struct SCONFIG
{
	int a, b;

	double l, k;
	double theta;
};

/************************************************
 *  Bispectrum configuration
 ***********************************************/
struct BCONFIG
{
	int a, b, c;

	gsl_vector *l1, *l2, *l3;
	gsl_vector *k1, *k2, *k3;
};

/************************************************
 *  Trispectrum configuration
 ***********************************************/
struct TCONFIG
{
	int a, b, c, d;

	gsl_vector *l1, *l2, *l3, *l4;
	gsl_vector *k1, *k2, *k3, *k4;
};

struct ZETAPARAM
{
	int n;
	double r;
};

struct LPARAM
{
	double chi, chiprime;
	double theta;
	int index;
};

struct CosmoUtil;
struct Spectrum;
struct Bispectrum;
struct Trispectrum;
struct LensingSpectrum;

struct COSMOLOGY
{
	double omega_m; // cosmological parameters
	double omega_q;
	double omega_b;
	double h;
	double w, wprime;
	double n_s;
	double sigma8;
	double rsmooth;
	double rscale;

	double fnl; // non-Gaussianity parameters
	double gnl;
	double tnl;

	double zmean, beta; // observation: EUCLID
	double f_sky, nmean;
	double ellipticity;

	gsl_interp_accel *acc_z2; // moments
	gsl_spline *spline_z2;
	gsl_interp_accel *acc_z3;
	gsl_spline *spline_z3;
	gsl_interp_accel *acc_z4;
	gsl_spline *spline_z4;

	gsl_interp_accel *acc_pp; // ellipticity correlation functions
	gsl_spline *spline_pp;
	gsl_interp_accel *acc_cc;
	gsl_spline *spline_cc;

	gsl_interp_accel *acc_ss;
	gsl_spline *spline_ss;
	gsl_interp_accel *acc_sp;
	gsl_spline *spline_sp;

	gsl_interp_accel *acc_dplus; // accelerators for growth
	gsl_spline *spline_dplus;
	gsl_interp_accel *acc_a; // a(chi) inversion
	gsl_spline *spline_a;

	gsl_interp_accel *acc_ksigma; // nonlinear CDM spectrum
	gsl_spline *spline_ksigma;
	gsl_interp_accel *acc_nslope;
	gsl_spline *spline_nslope;
	gsl_interp_accel *acc_curv;
	gsl_spline *spline_curv;

	double lmax;

	std::shared_ptr<CosmoUtil>   fdm_cu;
	std::shared_ptr<CosmoUtil>   cdm_cu;
	std::shared_ptr<Spectrum>    fdm_spectrum;
	std::shared_ptr<Spectrum>    cdm_spectrum;
	std::shared_ptr<Bispectrum>  fdm_bispectrum;
	std::shared_ptr<Bispectrum>  cdm_bispectrum;
	std::shared_ptr<Bispectrum>  cdm_fdm_ic_bispectrum;
	std::shared_ptr<Trispectrum> cdm_trispectrum;
	std::shared_ptr<Trispectrum> fdm_trispectrum;
	std::shared_ptr<Trispectrum> cdm_fdm_ic_trispectrum;

	std::shared_ptr<LensingSpectrum> cdm_loop_lensing_spectrum;
	std::shared_ptr<LensingSpectrum> fdm_loop_lensing_spectrum;
};

struct DATA
{
	struct COSMOLOGY *cosmology;

	double dp[ASTEP]; // tables for growth function and a(chi)
	double a[ASTEP];
	double chi[ASTEP];

	double ksigma[ASTEP]; // nonlinear CDM spectrum
	double nslope[ASTEP];
	double curv[ASTEP];

	double c_noise[LMAX][NBIN][NBIN]; // cached noisy spectra for the covariances

	double b_s2n[LSTEP]; // s2n-ratio of the convergence bispectrum, local
	double t_s2n[LSTEP]; // s2n-ratio of the convergence trispectrum, local

	double e_s2n[LSTEP]; // s2n-ratio of the convergence bispectrum, equil and ortho
	double o_s2n[LSTEP];

	double fdm_s2n[LSTEP];// s2n-ratio of the convergence bispectrum for FDM
	double cdm_s2n[LSTEP];// s2n-ratio of the convergence bispectrum for FDM

	double s2n_cuba[LSTEP];// s2n-ratio of the convergence spectrum calculated using cuba
	double diff_chi2[LSTEP]; //Chi squared values for distinguishing two bispectra

	double ck_s2n[LMAX]; // s2n-ratio of the convergence spectrum
	double ce_s2n[LMAX]; // s2n-ratios of the ellipticity E- and B-mode spectra
	double cb_s2n[LMAX];
	double cs_s2n[LMAX];   // <ss> alone
	double cc_s2n[LMAX];   // <cc> alone
	double cse_s2n[LMAX];  // combination of <ee> and <ss>
	double csec_s2n[LMAX]; // combination of <ee>, <ss> and <cc>

	double epp[CSTEP], ecc[CSTEP]; // 3d-correlation functions and moments
	double r[CSTEP];
	double zeta2[CSTEP], zeta3[CSTEP], zeta4[CSTEP];

	double theta[TSTEP];
	double epsilon_pp[TSTEP], epsilon_cc[TSTEP]; // angular correlation functions
	double epsilon_sp[TSTEP], epsilon_ss[TSTEP];

	double gamma[TSTEP][NBIN][NBIN]; // lensing correlation function

	double ce[LMAX][NBIN]; // ellipticity spectra
	double cb[LMAX][NBIN];
	double cs[LMAX][NBIN];
	double cc[LMAX][NBIN];

	double cv[LMAX][NBIN][NBIN]; // fiducial spectrum

	double cu[LMAX][NBIN][NBIN][NFISHER]; // derivative construction and Fisher matrix
	double cl[LMAX][NBIN][NBIN][NFISHER];
	double dc[LMAX][NBIN][NBIN][NFISHER];
	gsl_matrix *fish;

	double cuu[LMAX][NBIN][NBIN][NFISHER][NFISHER]; // construction of second derivatives, matrices for bias
	double cul[LMAX][NBIN][NBIN][NFISHER][NFISHER];
	double cll[LMAX][NBIN][NBIN][NFISHER][NFISHER];
	double d2c[LMAX][NBIN][NBIN][NFISHER][NFISHER];

	gsl_matrix *gext; // extended Fisher matrix, cumulative in i
	gsl_vector *avec; // a-vector
	gsl_vector *bias; // parameter estimation bias

	double ccoefficient[LMAX][NBIN][NBIN]; // correlation coefficient between weak lensing bins

};

// --- prototypes ---
int COMPUTE_LENSING_SPECTRA(struct DATA *data);
int COMPUTE_NOISY_LENSING_SPECTRA(struct DATA *data);
int COMPUTE_LENSING_CORRELATION(struct DATA *data);
int COMPUTE_LENSING_CCOEFFICIENTS(struct DATA *data);
int SAVE_LENSING_SPECTRA2DISK(struct DATA *data, const char* suffix = NULL);
int SAVE_LENSING_CORRELATION2DISK(struct DATA *data);
int SAVE_LENSING_CCOEFFICIENTS2DISK(struct DATA *data);

// --- splitting up galaxy sample ---
int SPLIT_UP_GALAXY_SAMPLE(struct DATA *data);
double p_galaxy_inverse(double frac);
double aux_p_galaxy_inverse(double z, void *params);
double p_galaxy_cumulative(double z);

// --- s2n computation ---
int COMPUTE_S2N_SPECTRUM(struct DATA *data);
int COMPUTE_UNIFIED_S2N_SPECTRUM(struct DATA *data);
int COMPUTE_S2N_SPECTRUM_CUBA(struct DATA *data);
int COMPUTE_S2N_ELLIPTICITY(struct DATA *data);
int COMPUTE_S2N_CROSS(struct DATA *data);
int COMPUTE_S2N_BISPECTRUM(struct DATA *data);
int COMPUTE_UNIFIED_S2N_BISPECTRUM(struct DATA *data);
int COMPUTE_S2N_TRISPECTRUM(struct DATA *data);
int COMPUTE_UNIFIED_S2N_TRISPECTRUM(struct DATA *data);
int SAVE_SIGNIFICANCE2DISK(struct DATA *data, const char* prefix = NULL);

// --- chi squared computation ---
int COMPUTE_SPECTRUM_DIFFERENCE_CHISQUARED  (struct DATA *data);
int COMPUTE_SPECTRUM_DIFFERENCE_CHISQUARED_CUBA  (struct DATA *data);
int COMPUTE_UNIFIED_SPECTRUM_DIFFERENCE_CHISQUARED  (struct DATA *data);
int COMPUTE_BISPECTRUM_DIFFERENCE_CHISQUARED(struct DATA *data);
int COMPUTE_UNIFIED_BISPECTRUM_DIFFERENCE_CHISQUARED(struct DATA *data);
int COMPUTE_TRISPECTRUM_DIFFERENCE_CHISQUARED(struct DATA *data);
int COMPUTE_UNIFIED_TRISPECTRUM_DIFFERENCE_CHISQUARED(struct DATA *data);

// --- main functions ---
int INIT_COSMOLOGY(struct DATA *data, int fdm_mass_id);
int PREPARE_NEW_COSMOLOGY(struct DATA *data);
int COMPUTE_EVOLUTION(struct DATA *data);
int COMPUTE_COMOVING_DISTANCE(struct DATA *data);
int ALLOCATE_INTERPOLATION_MEMORY(struct DATA *data);
int FREE_INTERPOLATION_MEMORY(struct DATA *data);
int SET_FIDUCIAL_COSMOLOGY(struct DATA *data);

// --- stepping ---
double astep(int i);
double lstep(int i);
double tstep(int i);
double chistep(int i);

// --- primary functions ---
double map_multipole(double xx, double lmax);
double map_polar_angle(double xx);

// --- cosmometry ---
double a2z(double a);
double z2a(double z);
double a2com(double a);
double aux_dcom(double a, void *params);
//double hubble(double a);

// --- growth ---
int d_plus_function(double t, const double y[], double f[], void *params);
double aux_d_plus(double a, double *result_d_plus, double *result_d_plus_prime);
double d_plus(double a);
double d_plus_prime(double a);
double fdm_d_plus(double k, double a);
double x_plus(double a);
double aux_r(double a);
double aux_dr(double a);
double aux_q(double a);
double aux_dq(double a);

// --- dark energy model ---
double w_de(double a);
double p_galaxy(double z);

// --- cdm-spectrum ---

double spectrum(double k, double a = 1.0);
double cdm_spectrum_aux(double k);
double cdm_transfer(double k);
/**
 * Calculate sigma8
 * */
double sigma8(double a);
double aux_dsigma8(double k, void *params);
double grav_potential_spectrum(double k, double a = 1.0);
double spectrum_smooth(double k, double a = 1.0);

// nonlinear CDM-spectrum
int COMPUTE_CDM_PARAMETERS(struct DATA *data);
double spectrum_slope(double k, double a = 1.0);
double nonlinear_spectrum(double k, double a);
double nonlinear_transfer(double k, double a);
double rscale(double a);
double aux_rscale(double rscale, void *params);
double sigma2(double rscale);
double aux_sigma2(double k, void *params);
double dsigma2(double rscale);
double aux_dsigma2(double k, void *params);
double ddsigma2(double rscale);
double aux_ddsigma2(double k, void *params);

// --- spectra ---
double tomo_spectrum(struct SCONFIG *sconfig);
double tomo_spectrum_gsl(struct SCONFIG *sconfig);
double aux_tomo_spectrum_gsl(double chi, void *param);
double tomo_spectrum_cuba(struct SCONFIG *sconfig);
int    aux_tomo_spectrum_cuba(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_tomo_spectrum_s2n_cuba_integrand(struct SCONFIG *sconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);

double tomo_noise(double l, int a, int b);
double tomo_kappa(double l, int a, int b);
int spectrum_s2n(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);

double spectrum_covariance(struct SCONFIG *sconfig);

// --- bispectra ---
int load_config_bispectrum(struct BCONFIG *bconfig, double l1, double l3, double phi);
double tomo_bispectrum(struct BCONFIG *bconfig);
double tomo_bispectrum_cuba(struct BCONFIG *bconfig);
double aux_tomo_bispectrum(double chi, void *param);
int aux_tomo_bispectrum_cuba(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
int bispectrum_s2n(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
//Only works for NBIN = 0, that means no tomography
int unified_bispectrum_s2n(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_tomo_bispectrum_s2n_cuba_integrand(struct BCONFIG *bconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_fdm_tomo_bispectrum_tree_integrand(struct BCONFIG *bconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_fdm_tomo_bispectrum_loop_integrand(struct BCONFIG *bconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_cdm_tomo_bispectrum_tree_integrand(struct BCONFIG *bconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_cdm_tomo_bispectrum_loop_integrand(struct BCONFIG *bconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);

double bispectrum_covariance(struct BCONFIG *bconfig1, struct BCONFIG *bconfig2);
/**
 * Modification w.r.t. Björn's version
 * In CDM case, bispectrum is evolved in time by multiplying it with d_plus
 * In FDM case, even the tree level bispectrum depends on the integration of F2 and is therefore explicitly time-dependent
 ***/
double potential_bispectrum(struct BCONFIG *bconfig, double a = 1.0);
double bispectrum(struct BCONFIG *bconfig, double scale_factor);
double fdm_potential_bispectrum(struct BCONFIG *bconfig, double a, double s1);
double cdm_bispectrum_aux(struct BCONFIG *bconfig, double a);
double fdm_bispectrum_aux(struct BCONFIG *bconfig, double a);
int free_config_bispectrum(struct BCONFIG *bconfig);
int check_birange(struct BCONFIG *bconfig);
int check_biwedge(struct BCONFIG *bconfig);

// --- trispectra ---
int load_config_trispectrum(struct TCONFIG *tconfig, double l1, double l2, double l4, double phi, double psi);
double tomo_trispectrum(struct TCONFIG *tconfig);
double tomo_trispectrum_cuba(struct TCONFIG *tconfig);
double cdm_tomo_trispectrum(struct TCONFIG *tconfig);
double fdm_tomo_trispectrum(struct TCONFIG *tconfig);
double potential_trispectrum(struct TCONFIG *tconfig, double a = 1.0);
double fdm_potential_trispectrum(struct TCONFIG *tconfig, double a, double s1, double s2);
double aux_tomo_trispectrum(double chi, void *param);
int aux_tomo_trispectrum_cuba(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
int trispectrum_s2n(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
int trispectrum_difference_chisquared(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
int unified_trispectrum_s2n(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);
double unified_tomo_trispectrum_s2n_cuba_integrand(struct TCONFIG *tconfig, const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata);

double trispectrum_covariance(struct TCONFIG *tconfig1, struct TCONFIG *tconfig2);
double cdm_trispectrum(struct TCONFIG *tconfig);
int free_config_trispectrum(struct TCONFIG *tconfig);
int check_trirange(struct TCONFIG *tconfig);
int check_triwedge(struct TCONFIG *tconfig);
double spirou_delta(gsl_vector *x, gsl_vector *y);

// --- weak lensing ---
double tomo_efficiency(double chi, double a, int i);
double tomo_weighting(double chi, int i);
double aux_weighting(double chiprime, void *params);
double kronecker(int a, int b);

// --- intrinsic alignments ---
double ellipticity_emode(int l, int i);
double ellipticity_bmode(int l, int i);
double aux_demode(double theta, void *param);
double aux_dbmode(double theta, void *param);

double ellipticity_smode(int l, int i);
double ellipticity_cmode(int l, int i);
double aux_dsmode(double theta, void *param);
double aux_dcmode(double theta, void *param);

// ellipticity correlations
double correlation_pp(double theta, int i);
double correlation_cc(double theta, int i);
double ellipticity_pp(double r, double alpha);
double ellipticity_cc(double r, double alpha);

double correlation_sp(double theta, int i);
double correlation_ss(double theta, int i);
double ellipticity_sp(double r, double alpha);
double ellipticity_ss(double r, double alpha);

double aux_correlation_dchiprime(double chiprime, void *param);
double aux_correlation_dchi(double chi, void *param);

// zeta-functions
double spirou_zeta(double r, int n);
double aux_spirou_zeta(double k, void *param);
double spirou_zeta_norm();
double aux_spirou_zeta_norm(double k, void *param);

// lensing correlation function
double gamma_lensing(double theta, int a, int b);
double aux_dgamma(double l, void *param);

// caching for ellipticities
int COMPUTE_ZETA_COEFFICIENTS(struct DATA *data); // ok
int COMPUTE_ANGULAR_CORRELATIONS(struct DATA *data, int i);
int WRITE_ANGULAR_ELLIPTICITY2DISK(struct DATA *data, int i);

// compute ellipticity spectra
int COMPUTE_ELLIPTICITY_SPECTRA(struct DATA *data); // ok
int COMPUTE_ELLIPTICITY_CMODE(struct DATA *data);
int SAVE_ELLIPTICITY_SPECTRA2DISK(struct DATA *data); // ok
int SAVE_ELLIPTICITY_CMODE2DISK(struct DATA *data);

// auxiliary functions
double spirou_trace(gsl_matrix *matrix);
int view_matrix(gsl_matrix *matrix);
int comp(const void *a, const void *b);

// lensing fisher matrix
int COMPUTE_FIDUCIAL_SPECTRUM(struct DATA *data);
int COMPUTE_SPECTRUM_P(struct DATA *data);
int COMPUTE_SPECTRUM_M(struct DATA *data);
int COMPUTE_SPECTRUM_DERIVATIVE(struct DATA *data);
int CONSTRUCT_FISHER_MATRIX(struct DATA *data);
int SAVE_FISHER_MATRIX2DISK(struct DATA *data);

// lensing bias formalism
int COMPUTE_SPECTRUM_PP(struct DATA *data);
int COMPUTE_SPECTRUM_PM(struct DATA *data);
int COMPUTE_SPECTRUM_MM(struct DATA *data);
int COMPUTE_HESSIAN_DERIVATIVE(struct DATA *data);
int CONSTRUCT_GMATRIX(struct DATA *data);
int CONSTRUCT_AVECTOR(struct DATA *data);
int COMPUTE_ESTIMATION_BIAS(struct DATA *data);
int SAVE_BIAS2DISK(struct DATA *data);

// --- global variables ---
extern struct COSMOLOGY *gcosmo;
extern struct DATA *gdata;
extern double anorm;
extern double tomo[NBIN + 1], tomo_z[NBIN + 1];
extern int flag_mode;
extern bool flag_enable_loop;
extern bool use_lensing_spectrum_splines;
extern spectrum_type flag_type;
extern spectrum_type flag_type_reference; //Reference spectrum for chi^2 calculation
extern bool flag_compute_spectrum_difference;
extern bool unify_cuba_integrations;
#endif
