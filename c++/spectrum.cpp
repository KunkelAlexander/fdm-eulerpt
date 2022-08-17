#include "../include/spectrum.h"
#include "../include/constants.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <list>
//#include <boost/timer/timer.hpp>
#include <fstream>
#include <vector>
#if 0
#include <datatable.h>
#include <bsplinebuilder.h>
#endif

// SETTINGS FOR CUBA INTEGRATOR
#define NCOMP 1
#define NVEC 1
#define LAST 4
#define SEED 0
#define MINEVAL 0
#define MAXEVAL 10000000

#define NSTART 1000
#define NINCREASE 500
#define NBATCH 1000
#define GRIDNO 0
#define STATEFILE NULL
#define SPIN NULL

#define NNEW 1000
#define NMIN 2
#define FLATNESS 25.

#define KEY1 47
#define KEY2 1
#define KEY3 1
#define MAXPASS 5
#define BORDER 1e-3
#define MAXCHISQ 10.
#define MINDEVIATION .25
#define NGIVEN 0
#define NEXTRA 0

#define KEY 0

const int NEVAL = 1000;
const double smalleps = 1e-6;
const double epsrel = smalleps;
const double verysmalleps = 1e-20;

// Intervals for splines used to represent growth factor and greens' functions
#define const_a_min -2 // on logarithmic scale with base 10
#define const_a_max 0  // on logarithmic scale with base 10
const size_t const_a_res = 1000;
#define const_k_min -6 // on logarithmic scale with base 10
#define const_k_max 3  // on logarithmic scale with base 10
const size_t const_k_res = 2000;

/***
 * MODE COUPLING
 * ***/

// Operator overloads for wrapper for GSL vector
vec operator+(const vec &v1, const vec &v2)
{
  vec answer(v1);
  for (size_t i = 0; i < answer.size(); ++i)
  {
    answer[i] += v2[i];
  }
  return answer;
}

vec operator-(const vec &v1, const vec &v2)
{
  vec answer(v1);
  for (size_t i = 0; i < answer.size(); ++i)
  {
    answer[i] -= v2[i];
  }
  return answer;
}

vec operator-(const vec &v)
{
  vec answer(v);
  for (size_t i = 0; i < answer.size(); ++i)
  {
    answer[i] *= -1;
  }
  return answer;
}

vec operator*(float c, const vec &v)
{
  vec answer(v);
  for (size_t i = 0; i < answer.size(); ++i)
  {
    answer[i] *= c;
  }
  return answer;
}

vec operator*(const vec &v, float c)
{
  return c * v;
}

// Dot product d = sum over i : k1_i * k2_i
double dot(const vec &k1, const vec &k2)
{
  double sum = 0;
  for (size_t i = 0; i < k1.size(); ++i)
  {
    sum += k1[i] * k2[i];
  }
  return sum;
}

// Squared sum of two vectors
double ssum(const vec &k1, const vec &k2)
{
  double sum = 0;
  for (size_t i = 0; i < k1.size(); ++i)
  {
    sum += pow(k1[i] + k2[i], 2);
  }
  return sum;
}

// Time conversion
double a_from_z(double z)
{
  return 1 / (z + 1);
}

double eta_from_z(double z)
{
  return 2 * sqrt(a_from_z(z));
}

double eta_from_a(double a)
{
  return 2 * sqrt(a);
}

double a_from_eta(double eta)
{
  return pow(eta, 2) / 4.;
}

double z_from_eta(double eta)
{
  double a = a_from_eta(eta);
  return (1. / a - 1.);
}

Spline1D::Spline1D(const double *x, const double *y, size_t steps) : steps_(steps)
{
  acc_spline_ = gsl_interp_accel_alloc();
  spline_ = gsl_spline_alloc(gsl_interp_cspline, steps_);
  gsl_spline_init(spline_, x, y, steps_);
  #ifdef SPLINE_RANGE_CHECK
  xmin_ = x[0] + verysmalleps;
  xmax_ = x[steps-1] - verysmalleps;
  #endif
}

Spline1D::~Spline1D()
{
  gsl_interp_accel_free(acc_spline_);
  gsl_spline_free(spline_);
}

double Spline1D::operator()(double x)
{

  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
    throw std::runtime_error("Called 1D spline with argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));
  #endif

  return gsl_spline_eval(spline_, x, acc_spline_);
}

double Spline1D::dx(double x)
{
  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
    throw std::runtime_error("Called 1D spline derivative with argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));
  #endif
  return gsl_spline_eval_deriv(spline_, x, acc_spline_);
}

Spline2D::Spline2D(const double *x, const double *y, const double *z, size_t xsize, size_t ysize) : xsize_(xsize), ysize_(ysize)
{
  xacc_spline_ = gsl_interp_accel_alloc();
  yacc_spline_ = gsl_interp_accel_alloc();
  spline_ = gsl_spline2d_alloc(gsl_interp2d_bilinear, xsize_, ysize_);
  gsl_spline2d_init(spline_, x, y, z, xsize_, ysize_);


  #ifdef SPLINE_RANGE_CHECK
  xmin_ = x[0] + verysmalleps;
  xmax_ = x[xsize-1] - verysmalleps;
  ymin_ = y[0] + verysmalleps;
  ymax_ = y[ysize-1] - verysmalleps;
  #endif
}

Spline2D::~Spline2D()
{
  gsl_interp_accel_free(xacc_spline_);
  gsl_interp_accel_free(yacc_spline_);
  gsl_spline2d_free(spline_);
}

double Spline2D::operator()(double x, double y)
{

  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
      throw std::runtime_error("Called 2D spline derivative with first argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));

  if (y < ymin_ || y > ymax_)
      throw std::runtime_error("Called 2D spline derivative with second argument out of interpolation range: ymin = "+std::to_string(ymin_) + " ymax = " + std::to_string(ymax_) + " at x = " + std::to_string(y));
  #endif

  return gsl_spline2d_eval(spline_, x, y, xacc_spline_, yacc_spline_);
}

double Spline2D::dx(double x, double y)
{

  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
      throw std::runtime_error("Called 2D spline derivative with first argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));

  if (y < ymin_ || y > ymax_)
      throw std::runtime_error("Called 2D spline derivative with second argument out of interpolation range: ymin = "+std::to_string(ymin_) + " ymax = " + std::to_string(ymax_) + " at x = " + std::to_string(y));
  #endif
  
  return gsl_spline2d_eval_deriv_x(spline_, x, y, xacc_spline_, yacc_spline_);
}

double Spline2D::d2x(double x, double y)
{

  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
      throw std::runtime_error("Called 2D spline derivative with first argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));

  if (y < ymin_ || y > ymax_)
      throw std::runtime_error("Called 2D spline derivative with second argument out of interpolation range: ymin = "+std::to_string(ymin_) + " ymax = " + std::to_string(ymax_) + " at x = " + std::to_string(y));
  #endif
  
  return gsl_spline2d_eval_deriv_xx(spline_, x, y, xacc_spline_, yacc_spline_);
}

double Spline2D::dy(double x, double y)
{


  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
      throw std::runtime_error("Called 2D spline derivative with first argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));

  if (y < ymin_ || y > ymax_)
      throw std::runtime_error("Called 2D spline derivative with second argument out of interpolation range: ymin = "+std::to_string(ymin_) + " ymax = " + std::to_string(ymax_) + " at x = " + std::to_string(y));
  #endif
  

  return gsl_spline2d_eval_deriv_y(spline_, x, y, xacc_spline_, yacc_spline_);
}

double Spline2D::d2y(double x, double y)
{


  #ifdef SPLINE_RANGE_CHECK
  if (x < xmin_ || x > xmax_)
      throw std::runtime_error("Called 2D spline derivative with first argument out of interpolation range: xmin = "+std::to_string(xmin_) + " xmax = " + std::to_string(xmax_) + " at x = " + std::to_string(x));

  if (y < ymin_ || y > ymax_)
      throw std::runtime_error("Called 2D spline derivative with second argument out of interpolation range: ymin = "+std::to_string(ymin_) + " ymax = " + std::to_string(ymax_) + " at x = " + std::to_string(y));
  #endif
  

  return gsl_spline2d_eval_deriv_yy(spline_, x, y, xacc_spline_, yacc_spline_);
}

// Hubble function H / H0, do not include hubble constant or everything will fail
double hubble(double a)
{
  double result;

#ifdef FDM_TEST_COSMOLOGY
  result = pow(a, -1.5);
#elif defined(GAMER_COSMOLOGY)
  result = sqrt(const_omega_m * pow(a, -3.) + const_omega_q);
#else
  double aux;
  // Define cosmology with dark energy
  /* radiation, matter and curvature */
  result = const_omega_m / pow(a, 3);

  /* dark energy , parameterised with eos w(a) = w0 + (1-a) * w' */
  aux = -(1.0 + const_w_dark_energy + const_w_dark_energy_prime) * log(a) + const_w_dark_energy_prime * (a - 1.0);
  result += const_omega_q * exp(3.0 * aux);

  result = sqrt(result);
#endif
  return (result);
}

/* --- function of logarithm of hubble function w.r.t. scale factor --- */
double dloghubble(double a)
{
  double result;

#ifdef FDM_TEST_COSMOLOGY
  result = -1.5 / a;
#elif defined(GAMER_COSMOLOGY)
  result = 1 / (const_omega_m * pow(a, -3) + const_omega_q) * 0.5 * (-3) * pow(a, -4) * const_omega_m;
#else
  double da = 1e-8;
  result = (log(hubble(a + da)) - log(hubble(a - da))) / (2 * da);
#endif
  return (result);
}

/* --- function of logarithm of hubble function w.r.t. scale factor --- */
double d2loghubble(double a)
{
  double result;
  double da = 1e-8;
  result = (dloghubble(a + da) - dloghubble(a - da)) / (2 * da);
  return (result);
}


SplineSpectrum::SplineSpectrum(const std::string &filename)
{
  std::ifstream inFile;
  inFile.open(filename);
  if (!inFile)
  {
    std::cerr << "Unable to open spline file in spline spectrum, ";
    std::cerr << "Filename: " << filename;
    throw std::runtime_error("Unable to open spline file"); // call system to stop
  }
  size_t nk = p1l_spline_nk, neta = p1l_spline_na;

  std::vector<double> times(neta), momenta(nk), spectra(neta * nk);

  for (size_t i = 0; i < neta; ++i)
  {
    for (size_t j = 0; j < nk; ++j)
    {
      double l, eta, k, tree, vegas, error;
      inFile >> l >> eta >> k >> tree >> vegas >> error;

      // std::cout<<"l "<<l<<" eta "<<eta<<" k "<<k<<" tree + vegas " << tree + vegas <<"\n";
      if (i == 0)
      {
        momenta[j] = k;
      }
      if (j == 0)
      {
        times[i] = eta;
      }

      spectra[j * neta + i] = tree + vegas;
    }
  }

  std::cout << "Momenta\n";

  for (double k : momenta)
    std::cout << k << " ";

  std::cout << "\n\n Times \n";

  for (double eta : times)
    std::cout << eta << " ";

  std::cout << "\n\n P at eta = 2 \n";

  for (size_t i = 0; i < nk; ++i)
    std::cout << spectra[neta * i + neta - 1] << " ";

  std::cout << "\n\n";

  s = std::shared_ptr<Spline2D>(new Spline2D(times.data(), momenta.data(), spectra.data(), neta, nk));

  tmin = times.front();
  tmax = times.back();
  kmin = momenta.front();
  kmax = momenta.back();
}

double SplineSpectrum::operator()(double k, const CosmoUtil &cu) const
{
  if ((cu.eta < tmin) || (cu.eta) > tmax) {
    std::cerr << "Spline spectrum called with time out of bounds tmin "<<tmin<< " tmax " << tmax << " t "<<cu.eta<<"\n";
    throw std::runtime_error("Spline spectrum called with time out of bounds tmin ");

  }
  if ((k > kmin) && (k < kmax))
    return (*s)(cu.eta, k);

  if (k <= kmin)
  {
    double dk = 1e-5;
    double pmin = (*s)(cu.eta, kmin);
    double alphal = (log((*s)(cu.eta, kmin + dk)) - log(pmin)) / (log(kmin + dk) - log(kmin));

#ifdef DEBUG_LOG
    std::cout << "Called spline spectrum with k = " << k << " < kmin = " << kmin << ", extrapolating with exponent alpha = " << alphal << "\n";
#endif
    return pmin * pow(k / kmin, alphal);
  }
  if (k >= kmax)
  {
    double dk = 10;
    double pmax = (*s)(cu.eta, kmax);
    double alphah = (log(pmax) - log((*s)(cu.eta, kmax - dk))) / (log(kmax) - log(kmax - dk));
#ifdef DEBUG_LOG
    std::cout << "Called spline spectrum with k = " << k << " > kmax = " << kmax << ", pmax = " << pmax << " extrapolating with exponent alpha = " << alphah << "\n";
#endif
    return pmax * pow(k / kmax, alphah);
  }
  return 0;
}

LensingSpectrum::LensingSpectrum(const std::string &filename)
{
  std::ifstream inFile;
  inFile.open(filename);
  if (!inFile)
  {
    std::cerr << "Unable to open spline file in for lensing spectrum";
    throw std::runtime_error("Unable to open spline file for lensing spectrum");
  }

  std::vector<double> multipoles, spectra;

  // Skip first line
  std::string str;
  std::getline(inFile, str);

  while (true)
  {
    double l, p;
    inFile >> l >> p;
    if (inFile.eof())
      break;

    multipoles.push_back(l);
    spectra.push_back(p);
  }

  s = std::shared_ptr<Spline1D>(new Spline1D(multipoles.data(), spectra.data(), spectra.size()));
  lmin = multipoles.front();
  lmax = multipoles.back();
  std::cout << "Done setting up lensing spectrum\n";
}

double LensingSpectrum::operator()(double l) const
{
  if ((l < lmin) || (l > lmax))
  {
    std::cerr << "Lensing spectrum spline called with l out of range"
              << " l = " << l << " lmin = " << lmin << " lmax = " << lmax;
    throw std::runtime_error("Lensing spectrum spline called with l out of range");
  }
  return s->operator()(l);
}

CAMBSpectrum::CAMBSpectrum(const std::string &filename)
{
  std::ifstream inFile;
  inFile.open(filename);
  if (!inFile)
  {
    std::cerr << "Unable to open spline file in CAMB spectrum with filename "<<filename<<"\n";
    throw std::runtime_error("Unable to open spline file in CAMB spectrum");
  }

  std::vector<double> momenta, spectra;

  while (true)
  {
    double k, tree;
    inFile >> k >> tree;
    if (inFile.eof())
      break;

    // Multiply by h_hubble because the CAMB codes actually return P = P(k/h_hubble)
    momenta.push_back(k);
    spectra.push_back(tree);
  }

  s = std::shared_ptr<Spline1D>(new Spline1D(momenta.data(), spectra.data(), spectra.size()));
  kmin = momenta.front();
  kmax = momenta.back();
  pmin = spectra.front();
  pmax = spectra.back();
  double dk = 1e-8;

  // Compute power laws for extrapolation below kmin and above kmax
  alphal = (log((*s)(kmin + dk)) - log(pmin)) / (log(kmin + dk) - log(kmin));
  coeffl = pmin / pow(kmin, alphal);
  alphah = -8;
  coeffh = pmax / pow(kmax, alphah);

  std::cout << "Done setting up CAMB spectrum\n";
}

double CAMBSpectrum::P0(double k) const
{
  if ((k > kmin) && (k < kmax))
    return (*s)(k);
  else if (k <= kmin)
    return coeffl * pow(k, alphal);
  else
    return coeffh * pow(k, alphah);
}

double CAMBSpectrum::operator()(double k, const CosmoUtil &cu) const
{
  return P0(k) * pow(cu.D(k, cu.eta), 2);
}
namespace CDM
{
  // CDM growth factor for matter dominated universe with Omega_m = 1
  // corresponds to D = a/a_in
  double D(double eta, double eta_in)
  {
    return pow(eta / eta_in, 2);
  }

  /***
   * Define helper functions for numerically integrating CDM time evolution equation
   ***/

  double aux_c0(double a)
  {
    return -1.5 * const_omega_m / pow(a, 5) / pow(hubble(a), 2);
  }

  double aux_c1(double a)
  {
    return 3.0 / a + dloghubble(a);
  }

  double daux_c0(double a)
  {
    double da = 1e-8;
    return (aux_c0(a + da) - aux_c0(a - da)) / (2 * da);
  }

  double daux_c1(double a)
  {
    return -3.0 / (a * a) + d2loghubble(a);
  }

  /* --- d_plus_function [defines f0 = dy1/da and f1 = dy2/da] --- */
  /* Growth equation is second-order ODE given as y'' + aux_c1(a) y' + aux_c0(a) y = 0*/
  int d_function(double a, const double y[], double f[], void *params)
  {
    /* derivatives f_i = dy_i/dt */
    f[0] = y[1];
    f[1] = -aux_c0(a) * y[0] - aux_c1(a) * y[1];

    return (GSL_SUCCESS);
  }
  /* --- end of function dplus_function --- */

  /* --- function dplus_jacobian --- */
  int d_jacobian(double a, const double y[], double *dfdy, double dfdt[], void *params)
  {
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array(dfdy, 2, 2);
    gsl_matrix *m = &dfdy_mat.matrix;

    /* jacobian df_i(t,y(a)) / dy_j */
    gsl_matrix_set(m, 0, 0, 0.0);
    gsl_matrix_set(m, 0, 1, 1.0);
    gsl_matrix_set(m, 1, 0, -aux_c0(a));
    gsl_matrix_set(m, 1, 1, -aux_c1(a));

    /* gradient df_i/da, explicit dependence */
    dfdt[0] = 0.0;
    dfdt[1] = -daux_c0(a) * y[0] - daux_c1(a) * y[1];

    return (GSL_SUCCESS);
  }
  /* --- end of function dplus_jacobian --- */

  /************************************************
   *  Integrand for d_plus
   *  Obtained via integrating governing ODE
   *  Initial scale factor is 10^(const_a_min)
   ***********************************************/
  double aux_d(double a, double *result_d_plus, double *result_d_plus_prime, bool growingMode = true)
  {
    double result;
    int status;
    const gsl_odeiv_step_type *T = gsl_odeiv_step_bsimp;
    gsl_odeiv_step *s = gsl_odeiv_step_alloc(T, 2);
    gsl_odeiv_control *c = gsl_odeiv_control_y_new(0.0, epsrel);
    gsl_odeiv_evolve *e = gsl_odeiv_evolve_alloc(2);
    double h, a0, y[2]; /* initial conditions */

    if (growingMode)
    {
      // Get growing mode by assuming IC for matter-dominated universe D+(a) = a*/
      h = 1e-5;
      y[0] = 1e-4;
      y[1] = 1;
      a0 = y[0];
    }
    else
    {
      // Get decaying mode by assuming IC for matter-dominated universe D-(a) = a^(-3/2)*/
      h = -1e-5;
      y[0] = 11;
      y[1] = -1.5;
      a0 = y[0];
    }
    /* result from solution of a 2nd order differential equation, transformed to a system of 2 1st order deqs */
    gsl_odeiv_system sys = {d_function, d_jacobian, 2, NULL};

    if (growingMode)
    {
      while (a0 < a)
      {
        status = gsl_odeiv_evolve_apply(e, c, s, &sys, &a0, a, &h, y);
        if (status != GSL_SUCCESS)
          break;
      }
    }
    else
    {
      while (a0 > a)
      {
        status = gsl_odeiv_evolve_apply(e, c, s, &sys, &a0, a, &h, y);
        if (status != GSL_SUCCESS)
          break;
      }
    }

    gsl_odeiv_evolve_free(e);
    gsl_odeiv_control_free(c);
    gsl_odeiv_step_free(s);

    result = *result_d_plus = y[0]; /* d_plus */
    *result_d_plus_prime = y[1];    /* d(d_plus)/da */

    return (result);
  }

  double get_ic(double a_in, bool growingMode)
  {
    double result, d, dd;

    aux_d(a_in, &d, &dd, growingMode);

    result = dd / d;

    return (result);
  }

  double d_plus(double a, double a_in)
  {
    double result, dummy, norm;

    aux_d(a, &result, &dummy, true);
    aux_d(a_in, &norm, &dummy, true);

    result /= norm;

    return (result);
  }

  double d_minus(double a, double a_in)
  {
    double result, dummy, norm;

    aux_d(a, &result, &dummy, false);
    aux_d(a_in, &norm, &dummy, false);

    result /= norm;

    return (result);
  }

  /***
   * Structure constructing and initialising spline with CDM growth function in constructor
   * Call as d = D(); d.s(a); to get CDM growth factor at scale_factor a
   ***/

  D_spline::D_spline(bool growingMode)
  {
    std::cout << "Setting up CDM spline" << std::endl;
    as = pyLogspace(const_a_min, const_a_max, const_a_res);
    if (as.front() > const_a_in)
    {
      as.insert(as.begin(), const_a_in - 1e-4);
    }
    if (as.back() < 1)
    {
      as.push_back(1);
    }

    amin_ = as.front();
    amax_ = as.back();

    Ds.reserve(as.size());

    for (double scale_factor : as)
    {
      if (growingMode)
      {
        Ds.push_back(d_plus(scale_factor, const_a_in));
      }
      else
      {
        Ds.push_back(d_minus(scale_factor, const_a_in));
      }
    }

    double *x0 = &as[0];
    double *y0 = &Ds[0];
    s = std::make_shared<Spline1D>(x0, y0, as.size());
  }

  double D_spline::operator()(double a)
  {
    if (a < amin_ + verysmalleps || a > amax_ - verysmalleps)
    {
      throw std::runtime_error("Value of a in D_spline out of range with a_min = " + std::to_string(amin_) + " amax = " + std::to_string(amax_) + " a = " + std::to_string(a));
    }
    return (*s)(a);
  }

  double alpha(const vec &k1, const vec &k2)
  {
    double k1s, result;

    k1s = dot(k1, k1);
    // Need this check for case x/x where x == 0 and expression should vanish
    if (k1s < verysmalleps)
    {
      return 0;
    }

    result = dot(k1 + k2, k1) / k1s;
    return (result);
  }

  double beta(const vec &k1, const vec &k2)
  {
    double k1s, k2s, result;
    k1s = dot(k1, k1);
    k2s = dot(k2, k2);
    // Need this check for case x/x where x == 0 and expression should vanish
    if ((k1s < verysmalleps) || (k2s < verysmalleps))
    {
      return 0;
    }
    result = ssum(k1, k2) * dot(k1, k2) / (2 * k1s * k2s);

    return (result);
  }

  // F2 symmetrised mode coupling, computed using recursion relations in Mathematica
  double F2s(const vec &k1, const vec &k2)
  {
    double result, a1, a2, b;
    a1 = alpha(k1, k2);
    a2 = alpha(k2, k1);
    b = beta(k1, k2);
    result = 5. / 14 * (a1 + a2) + 2. / 7 * b;
    return (result);
  }

  // F2 symmetrised mode coupling with time dependence, computed using perturbation code in Mathematica
  double F2s_td(const vec &k1, const vec &k2, const CosmoUtil &cu)
  {
    double eta, eta0;
    eta = cu.eta;
    eta0 = cu.eta_in;
    return ((25 * pow(eta, 7) - 21 * pow(eta, 5) * pow(eta0, 2) - 4 * pow(eta0, 7)) * dot(k2, k2) * dot(k1 + k2, k1) + (25 * pow(eta, 7) - 21 * pow(eta, 5) * pow(eta0, 2) - 4 * pow(eta0, 7)) * dot(k1, k1) * dot(k1 + k2, k2) + 2 * (5 * pow(eta, 7) - 7 * pow(eta, 5) * pow(eta0, 2) + 2 * pow(eta0, 7)) * dot(k1, k2) * dot(k1 + k2, k1 + k2)) / (70. * pow(eta, 7) * dot(k1, k1) * dot(k2, k2));
  }

  // F3 symmetrised mode coupling, computed using recursion relations in Mathematica
  double F3s(const vec &k1, const vec &k2, const vec &k3)
  {
    return (21 * alpha(k1, k2) * alpha(k1 + k2, k3) + 21 * alpha(k2, k1) * alpha(k1 + k2, k3) + 35 * alpha(k2, k1 + k3) * alpha(k3, k1) + 35 * alpha(k1, k2) * alpha(k3, k1 + k2) + 35 * alpha(k2, k1) * alpha(k3, k1 + k2) + 21 * alpha(k3, k1) * alpha(k1 + k3, k2) + 21 * alpha(k2, k3) * alpha(k2 + k3, k1) + 21 * alpha(k3, k2) * alpha(k2 + k3, k1) + 56 * alpha(k1 + k2, k3) * beta(k1, k2) + 28 * alpha(k3, k1 + k2) * beta(k1, k2) + 28 * alpha(k2, k1 + k3) * beta(k1, k3) + 56 * alpha(k1 + k3, k2) * beta(k1, k3) + 12 * alpha(k2, k3) * beta(k1, k2 + k3) + 12 * alpha(k3, k2) * beta(k1, k2 + k3) + 56 * alpha(k2 + k3, k1) * beta(k2, k3) + 32 * beta(k1, k2 + k3) * beta(k2, k3) + 7 * alpha(k1, k2 + k3) * (5 * alpha(k2, k3) + 5 * alpha(k3, k2) + 4 * beta(k2, k3)) + 12 * alpha(k3, k1) * beta(k2, k1 + k3) + 32 * beta(k1, k3) * beta(k2, k1 + k3) + alpha(k1, k3) * (35 * alpha(k2, k1 + k3) + 21 * alpha(k1 + k3, k2) + 12 * beta(k2, k1 + k3)) + 12 * alpha(k1, k2) * beta(k1 + k2, k3) + 12 * alpha(k2, k1) * beta(k1 + k2, k3) + 32 * beta(k1, k2) * beta(k1 + k2, k3)) / 756.;
  }

  // F3 symmetrised mode coupling with time dependence, computed using perturbation code in Mathematica
  double F3s_td(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    double eta, eta0, b1, b2, b3, s12, s23, s13;
    eta = cu.eta;
    eta0 = cu.eta_in;

    b1 = 0;
    b2 = 0;
    b3 = 0;

    s12 = dot(k1 + k2, k1 + k2);
    s23 = dot(k2 + k3, k2 + k3);
    s13 = dot(k1 + k3, k1 + k3);

    //double s11 = dot(k1, k1);
    //double s22 = dot(k2, k2);
    //double s33 = dot(k3, k3);

    if (s23 > verysmalleps)
    {
      b1 = (2 * dot(k2, k3) * dot(k2 + k3, k2 + k3) * ((35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k3, k3) * dot(k2 + k3, k2) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k2, k2) * dot(k2 + k3, k3) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k2 + k3, k2 + k3);
    }

    if (s13 > verysmalleps)
    {
      b2 = (2 * dot(k1, k3) * dot(k1 + k3, k1 + k3) * ((35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k3, k3) * dot(k1 + k3, k1) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k1, k1) * dot(k1 + k3, k3) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k1 + k3, k1 + k3);
    }

    if (s12 > verysmalleps)
    {
      b3 = (2 * dot(k2, k1) * dot(k1 + k2, k1 + k2) * (10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k2, k2) * dot(k1 + k2, k1) * (3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k1, k1) * dot(k1 + k2, k2) * (3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k1 + k2, k1 + k2);
    }
    return (b1 + b2 + b3) / (3780. * pow(eta, 9) * dot(k1, k1) * dot(k2, k2) * dot(k3, k3));
  }

  // F4 symmetrised mode coupling, computed using recursion relations in Mathematica
  double F4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4)
  {
    return (810 * alpha(k1, k2) * alpha(k1 + k2, k3 + k4) * alpha(k3, k4) + 810 * alpha(k2, k1) * alpha(k1 + k2, k3 + k4) * alpha(k3, k4) + 735 * alpha(k1, k4) * alpha(k2, k1 + k3 + k4) * alpha(k3, k1 + k4) + 735 * alpha(k1, k2 + k4) * alpha(k2, k4) * alpha(k3, k1 + k2 + k4) + 735 * alpha(k1, k4) * alpha(k2, k1 + k4) * alpha(k3, k1 + k2 + k4) + 441 * alpha(k1, k2) * alpha(k1 + k2, k4) * alpha(k3, k1 + k2 + k4) + 441 * alpha(k2, k1) * alpha(k1 + k2, k4) * alpha(k3, k1 + k2 + k4) + 441 * alpha(k1, k3) * alpha(k2, k1 + k3 + k4) * alpha(k1 + k3, k4) + 441 * alpha(k2, k1 + k3 + k4) * alpha(k3, k1) * alpha(k1 + k3, k4) + 810 * alpha(k1, k3) * alpha(k2, k4) * alpha(k1 + k3, k2 + k4) + 810 * alpha(k2, k4) * alpha(k3, k1) * alpha(k1 + k3, k2 + k4) + 810 * alpha(k1, k4) * alpha(k2, k3) * alpha(k2 + k3, k1 + k4) + 810 * alpha(k1, k4) * alpha(k3, k2) * alpha(k2 + k3, k1 + k4) + 315 * alpha(k1, k2 + k3) * alpha(k2, k3) * alpha(k1 + k2 + k3, k4) + 315 * alpha(k1, k3) * alpha(k2, k1 + k3) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k1, k2) * alpha(k1 + k2, k3) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k2, k1) * alpha(k1 + k2, k3) * alpha(k1 + k2 + k3, k4) + 315 * alpha(k2, k1 + k3) * alpha(k3, k1) * alpha(k1 + k2 + k3, k4) + 315 * alpha(k1, k2 + k3) * alpha(k3, k2) * alpha(k1 + k2 + k3, k4) + 315 * alpha(k1, k2) * alpha(k3, k1 + k2) * alpha(k1 + k2 + k3, k4) + 315 * alpha(k2, k1) * alpha(k3, k1 + k2) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k1, k3) * alpha(k1 + k3, k2) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k3, k1) * alpha(k1 + k3, k2) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k2, k3) * alpha(k2 + k3, k1) * alpha(k1 + k2 + k3, k4) + 189 * alpha(k3, k2) * alpha(k2 + k3, k1) * alpha(k1 + k2 + k3, k4) + 735 * alpha(k2, k1 + k3 + k4) * alpha(k3, k1 + k4) * alpha(k4, k1) + 735 * alpha(k2, k1 + k4) * alpha(k3, k1 + k2 + k4) * alpha(k4, k1) + 810 * alpha(k2, k3) * alpha(k2 + k3, k1 + k4) * alpha(k4, k1) + 810 * alpha(k3, k2) * alpha(k2 + k3, k1 + k4) * alpha(k4, k1) + 735 * alpha(k1, k2 + k4) * alpha(k3, k1 + k2 + k4) * alpha(k4, k2) + 810 * alpha(k1, k3) * alpha(k1 + k3, k2 + k4) * alpha(k4, k2) + 810 * alpha(k3, k1) * alpha(k1 + k3, k2 + k4) * alpha(k4, k2) + 735 * alpha(k1, k2) * alpha(k3, k1 + k2 + k4) * alpha(k4, k1 + k2) + 735 * alpha(k2, k1) * alpha(k3, k1 + k2 + k4) * alpha(k4, k1 + k2) + 810 * alpha(k1, k2) * alpha(k1 + k2, k3 + k4) * alpha(k4, k3) + 810 * alpha(k2, k1) * alpha(k1 + k2, k3 + k4) * alpha(k4, k3) + 735 * alpha(k1, k3) * alpha(k2, k1 + k3 + k4) * alpha(k4, k1 + k3) + 735 * alpha(k2, k1 + k3 + k4) * alpha(k3, k1) * alpha(k4, k1 + k3) + 735 * alpha(k1, k2 + k3) * alpha(k2, k3) * alpha(k4, k1 + k2 + k3) + 735 * alpha(k1, k3) * alpha(k2, k1 + k3) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k1, k2) * alpha(k1 + k2, k3) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k2, k1) * alpha(k1 + k2, k3) * alpha(k4, k1 + k2 + k3) + 735 * alpha(k2, k1 + k3) * alpha(k3, k1) * alpha(k4, k1 + k2 + k3) + 735 * alpha(k1, k2 + k3) * alpha(k3, k2) * alpha(k4, k1 + k2 + k3) + 735 * alpha(k1, k2) * alpha(k3, k1 + k2) * alpha(k4, k1 + k2 + k3) + 735 * alpha(k2, k1) * alpha(k3, k1 + k2) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k1, k3) * alpha(k1 + k3, k2) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k3, k1) * alpha(k1 + k3, k2) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k2, k3) * alpha(k2 + k3, k1) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k3, k2) * alpha(k2 + k3, k1) * alpha(k4, k1 + k2 + k3) + 441 * alpha(k1, k4) * alpha(k3, k1 + k2 + k4) * alpha(k1 + k4, k2) + 441 * alpha(k3, k1 + k2 + k4) * alpha(k4, k1) * alpha(k1 + k4, k2) + 441 * alpha(k1, k4) * alpha(k2, k1 + k3 + k4) * alpha(k1 + k4, k3) + 441 * alpha(k2, k1 + k3 + k4) * alpha(k4, k1) * alpha(k1 + k4, k3) + 810 * alpha(k1, k4) * alpha(k2, k3) * alpha(k1 + k4, k2 + k3) + 810 * alpha(k1, k4) * alpha(k3, k2) * alpha(k1 + k4, k2 + k3) + 810 * alpha(k2, k3) * alpha(k4, k1) * alpha(k1 + k4, k2 + k3) + 810 * alpha(k3, k2) * alpha(k4, k1) * alpha(k1 + k4, k2 + k3) + 441 * alpha(k2, k4) * alpha(k3, k1 + k2 + k4) * alpha(k2 + k4, k1) + 441 * alpha(k3, k1 + k2 + k4) * alpha(k4, k2) * alpha(k2 + k4, k1) + 810 * alpha(k1, k3) * alpha(k2, k4) * alpha(k2 + k4, k1 + k3) + 810 * alpha(k2, k4) * alpha(k3, k1) * alpha(k2 + k4, k1 + k3) + 810 * alpha(k1, k3) * alpha(k4, k2) * alpha(k2 + k4, k1 + k3) + 810 * alpha(k3, k1) * alpha(k4, k2) * alpha(k2 + k4, k1 + k3) + 315 * alpha(k1, k2 + k4) * alpha(k2, k4) * alpha(k1 + k2 + k4, k3) + 315 * alpha(k1, k4) * alpha(k2, k1 + k4) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k1, k2) * alpha(k1 + k2, k4) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k2, k1) * alpha(k1 + k2, k4) * alpha(k1 + k2 + k4, k3) + 315 * alpha(k2, k1 + k4) * alpha(k4, k1) * alpha(k1 + k2 + k4, k3) + 315 * alpha(k1, k2 + k4) * alpha(k4, k2) * alpha(k1 + k2 + k4, k3) + 315 * alpha(k1, k2) * alpha(k4, k1 + k2) * alpha(k1 + k2 + k4, k3) + 315 * alpha(k2, k1) * alpha(k4, k1 + k2) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k1, k4) * alpha(k1 + k4, k2) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k4, k1) * alpha(k1 + k4, k2) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k2, k4) * alpha(k2 + k4, k1) * alpha(k1 + k2 + k4, k3) + 189 * alpha(k4, k2) * alpha(k2 + k4, k1) * alpha(k1 + k2 + k4, k3) + 441 * alpha(k2, k1 + k3 + k4) * alpha(k3, k4) * alpha(k3 + k4, k1) + 441 * alpha(k2, k1 + k3 + k4) * alpha(k4, k3) * alpha(k3 + k4, k1) + 810 * alpha(k1, k2) * alpha(k3, k4) * alpha(k3 + k4, k1 + k2) + 810 * alpha(k2, k1) * alpha(k3, k4) * alpha(k3 + k4, k1 + k2) + 810 * alpha(k1, k2) * alpha(k4, k3) * alpha(k3 + k4, k1 + k2) + 810 * alpha(k2, k1) * alpha(k4, k3) * alpha(k3 + k4, k1 + k2) + 315 * alpha(k1, k4) * alpha(k3, k1 + k4) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k1, k3) * alpha(k1 + k3, k4) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k3, k1) * alpha(k1 + k3, k4) * alpha(k1 + k3 + k4, k2) + 315 * alpha(k3, k1 + k4) * alpha(k4, k1) * alpha(k1 + k3 + k4, k2) + 315 * alpha(k1, k3) * alpha(k4, k1 + k3) * alpha(k1 + k3 + k4, k2) + 315 * alpha(k3, k1) * alpha(k4, k1 + k3) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k1, k4) * alpha(k1 + k4, k3) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k4, k1) * alpha(k1 + k4, k3) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k3, k4) * alpha(k3 + k4, k1) * alpha(k1 + k3 + k4, k2) + 189 * alpha(k4, k3) * alpha(k3 + k4, k1) * alpha(k1 + k3 + k4, k2) + 315 * alpha(k2, k3 + k4) * alpha(k3, k4) * alpha(k2 + k3 + k4, k1) + 315 * alpha(k2, k4) * alpha(k3, k2 + k4) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k2, k3) * alpha(k2 + k3, k4) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k3, k2) * alpha(k2 + k3, k4) * alpha(k2 + k3 + k4, k1) + 315 * alpha(k3, k2 + k4) * alpha(k4, k2) * alpha(k2 + k3 + k4, k1) + 315 * alpha(k2, k3 + k4) * alpha(k4, k3) * alpha(k2 + k3 + k4, k1) + 315 * alpha(k2, k3) * alpha(k4, k2 + k3) * alpha(k2 + k3 + k4, k1) + 315 * alpha(k3, k2) * alpha(k4, k2 + k3) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k2, k4) * alpha(k2 + k4, k3) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k4, k2) * alpha(k2 + k4, k3) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k3, k4) * alpha(k3 + k4, k2) * alpha(k2 + k3 + k4, k1) + 189 * alpha(k4, k3) * alpha(k3 + k4, k2) * alpha(k2 + k3 + k4, k1) + 2160 * alpha(k1 + k2, k3 + k4) * alpha(k3, k4) * beta(k1, k2) + 1176 * alpha(k1 + k2, k4) * alpha(k3, k1 + k2 + k4) * beta(k1, k2) + 504 * alpha(k1 + k2, k3) * alpha(k1 + k2 + k3, k4) * beta(k1, k2) + 252 * alpha(k3, k1 + k2) * alpha(k1 + k2 + k3, k4) * beta(k1, k2) + 588 * alpha(k3, k1 + k2 + k4) * alpha(k4, k1 + k2) * beta(k1, k2) + 2160 * alpha(k1 + k2, k3 + k4) * alpha(k4, k3) * beta(k1, k2) + 1176 * alpha(k1 + k2, k3) * alpha(k4, k1 + k2 + k3) * beta(k1, k2) + 588 * alpha(k3, k1 + k2) * alpha(k4, k1 + k2 + k3) * beta(k1, k2) + 504 * alpha(k1 + k2, k4) * alpha(k1 + k2 + k4, k3) * beta(k1, k2) + 252 * alpha(k4, k1 + k2) * alpha(k1 + k2 + k4, k3) * beta(k1, k2) + 648 * alpha(k3, k4) * alpha(k3 + k4, k1 + k2) * beta(k1, k2) + 648 * alpha(k4, k3) * alpha(k3 + k4, k1 + k2) * beta(k1, k2) + 1176 * alpha(k2, k1 + k3 + k4) * alpha(k1 + k3, k4) * beta(k1, k3) + 2160 * alpha(k2, k4) * alpha(k1 + k3, k2 + k4) * beta(k1, k3) + 252 * alpha(k2, k1 + k3) * alpha(k1 + k2 + k3, k4) * beta(k1, k3) + 504 * alpha(k1 + k3, k2) * alpha(k1 + k2 + k3, k4) * beta(k1, k3) + 2160 * alpha(k1 + k3, k2 + k4) * alpha(k4, k2) * beta(k1, k3) + 588 * alpha(k2, k1 + k3 + k4) * alpha(k4, k1 + k3) * beta(k1, k3) + 588 * alpha(k2, k1 + k3) * alpha(k4, k1 + k2 + k3) * beta(k1, k3) + 1176 * alpha(k1 + k3, k2) * alpha(k4, k1 + k2 + k3) * beta(k1, k3) + 648 * alpha(k2, k4) * alpha(k2 + k4, k1 + k3) * beta(k1, k3) + 648 * alpha(k4, k2) * alpha(k2 + k4, k1 + k3) * beta(k1, k3) + 504 * alpha(k1 + k3, k4) * alpha(k1 + k3 + k4, k2) * beta(k1, k3) + 252 * alpha(k4, k1 + k3) * alpha(k1 + k3 + k4, k2) * beta(k1, k3) + 756 * alpha(k2, k3) * alpha(k1 + k2 + k3, k4) * beta(k1, k2 + k3) + 756 * alpha(k3, k2) * alpha(k1 + k2 + k3, k4) * beta(k1, k2 + k3) + 252 * alpha(k2, k3) * alpha(k4, k1 + k2 + k3) * beta(k1, k2 + k3) + 252 * alpha(k3, k2) * alpha(k4, k1 + k2 + k3) * beta(k1, k2 + k3) + 588 * alpha(k2, k1 + k3 + k4) * alpha(k3, k1 + k4) * beta(k1, k4) + 588 * alpha(k2, k1 + k4) * alpha(k3, k1 + k2 + k4) * beta(k1, k4) + 648 * alpha(k2, k3) * alpha(k2 + k3, k1 + k4) * beta(k1, k4) + 648 * alpha(k3, k2) * alpha(k2 + k3, k1 + k4) * beta(k1, k4) + 1176 * alpha(k3, k1 + k2 + k4) * alpha(k1 + k4, k2) * beta(k1, k4) + 1176 * alpha(k2, k1 + k3 + k4) * alpha(k1 + k4, k3) * beta(k1, k4) + 2160 * alpha(k2, k3) * alpha(k1 + k4, k2 + k3) * beta(k1, k4) + 2160 * alpha(k3, k2) * alpha(k1 + k4, k2 + k3) * beta(k1, k4) + 252 * alpha(k2, k1 + k4) * alpha(k1 + k2 + k4, k3) * beta(k1, k4) + 504 * alpha(k1 + k4, k2) * alpha(k1 + k2 + k4, k3) * beta(k1, k4) + 252 * alpha(k3, k1 + k4) * alpha(k1 + k3 + k4, k2) * beta(k1, k4) + 504 * alpha(k1 + k4, k3) * alpha(k1 + k3 + k4, k2) * beta(k1, k4) + 252 * alpha(k2, k4) * alpha(k3, k1 + k2 + k4) * beta(k1, k2 + k4) + 252 * alpha(k3, k1 + k2 + k4) * alpha(k4, k2) * beta(k1, k2 + k4) + 756 * alpha(k2, k4) * alpha(k1 + k2 + k4, k3) * beta(k1, k2 + k4) + 756 * alpha(k4, k2) * alpha(k1 + k2 + k4, k3) * beta(k1, k2 + k4) + 252 * alpha(k2, k1 + k3 + k4) * alpha(k3, k4) * beta(k1, k3 + k4) + 252 * alpha(k2, k1 + k3 + k4) * alpha(k4, k3) * beta(k1, k3 + k4) + 756 * alpha(k3, k4) * alpha(k1 + k3 + k4, k2) * beta(k1, k3 + k4) + 756 * alpha(k4, k3) * alpha(k1 + k3 + k4, k2) * beta(k1, k3 + k4) + 140 * alpha(k2, k3 + k4) * alpha(k3, k4) * beta(k1, k2 + k3 + k4) + 140 * alpha(k2, k4) * alpha(k3, k2 + k4) * beta(k1, k2 + k3 + k4) + 84 * alpha(k2, k3) * alpha(k2 + k3, k4) * beta(k1, k2 + k3 + k4) + 84 * alpha(k3, k2) * alpha(k2 + k3, k4) * beta(k1, k2 + k3 + k4) + 140 * alpha(k3, k2 + k4) * alpha(k4, k2) * beta(k1, k2 + k3 + k4) + 140 * alpha(k2, k3 + k4) * alpha(k4, k3) * beta(k1, k2 + k3 + k4) + 140 * alpha(k2, k3) * alpha(k4, k2 + k3) * beta(k1, k2 + k3 + k4) + 140 * alpha(k3, k2) * alpha(k4, k2 + k3) * beta(k1, k2 + k3 + k4) + 84 * alpha(k2, k4) * alpha(k2 + k4, k3) * beta(k1, k2 + k3 + k4) + 84 * alpha(k4, k2) * alpha(k2 + k4, k3) * beta(k1, k2 + k3 + k4) + 84 * alpha(k3, k4) * alpha(k3 + k4, k2) * beta(k1, k2 + k3 + k4) + 84 * alpha(k4, k3) * alpha(k3 + k4, k2) * beta(k1, k2 + k3 + k4) + 2160 * alpha(k1, k4) * alpha(k2 + k3, k1 + k4) * beta(k2, k3) + 252 * alpha(k1, k2 + k3) * alpha(k1 + k2 + k3, k4) * beta(k2, k3) + 504 * alpha(k2 + k3, k1) * alpha(k1 + k2 + k3, k4) * beta(k2, k3) + 2160 * alpha(k2 + k3, k1 + k4) * alpha(k4, k1) * beta(k2, k3) + 588 * alpha(k1, k2 + k3) * alpha(k4, k1 + k2 + k3) * beta(k2, k3) + 1176 * alpha(k2 + k3, k1) * alpha(k4, k1 + k2 + k3) * beta(k2, k3) + 648 * alpha(k1, k4) * alpha(k1 + k4, k2 + k3) * beta(k2, k3) + 648 * alpha(k4, k1) * alpha(k1 + k4, k2 + k3) * beta(k2, k3) + 504 * alpha(k2 + k3, k4) * alpha(k2 + k3 + k4, k1) * beta(k2, k3) + 252 * alpha(k4, k2 + k3) * alpha(k2 + k3 + k4, k1) * beta(k2, k3) + 2016 * alpha(k1 + k2 + k3, k4) * beta(k1, k2 + k3) * beta(k2, k3) + 672 * alpha(k4, k1 + k2 + k3) * beta(k1, k2 + k3) * beta(k2, k3) + 1728 * alpha(k2 + k3, k1 + k4) * beta(k1, k4) * beta(k2, k3) + 1728 * alpha(k1 + k4, k2 + k3) * beta(k1, k4) * beta(k2, k3) + 224 * alpha(k2 + k3, k4) * beta(k1, k2 + k3 + k4) * beta(k2, k3) + 112 * alpha(k4, k2 + k3) * beta(k1, k2 + k3 + k4) * beta(k2, k3) + 756 * alpha(k1, k3) * alpha(k1 + k2 + k3, k4) * beta(k2, k1 + k3) + 756 * alpha(k3, k1) * alpha(k1 + k2 + k3, k4) * beta(k2, k1 + k3) + 252 * alpha(k1, k3) * alpha(k4, k1 + k2 + k3) * beta(k2, k1 + k3) + 252 * alpha(k3, k1) * alpha(k4, k1 + k2 + k3) * beta(k2, k1 + k3) + 2016 * alpha(k1 + k2 + k3, k4) * beta(k1, k3) * beta(k2, k1 + k3) + 672 * alpha(k4, k1 + k2 + k3) * beta(k1, k3) * beta(k2, k1 + k3) + 588 * alpha(k1, k2 + k4) * alpha(k3, k1 + k2 + k4) * beta(k2, k4) + 648 * alpha(k1, k3) * alpha(k1 + k3, k2 + k4) * beta(k2, k4) + 648 * alpha(k3, k1) * alpha(k1 + k3, k2 + k4) * beta(k2, k4) + 1176 * alpha(k3, k1 + k2 + k4) * alpha(k2 + k4, k1) * beta(k2, k4) + 2160 * alpha(k1, k3) * alpha(k2 + k4, k1 + k3) * beta(k2, k4) + 2160 * alpha(k3, k1) * alpha(k2 + k4, k1 + k3) * beta(k2, k4) + 252 * alpha(k1, k2 + k4) * alpha(k1 + k2 + k4, k3) * beta(k2, k4) + 504 * alpha(k2 + k4, k1) * alpha(k1 + k2 + k4, k3) * beta(k2, k4) + 252 * alpha(k3, k2 + k4) * alpha(k2 + k3 + k4, k1) * beta(k2, k4) + 504 * alpha(k2 + k4, k3) * alpha(k2 + k3 + k4, k1) * beta(k2, k4) + 1728 * alpha(k1 + k3, k2 + k4) * beta(k1, k3) * beta(k2, k4) + 1728 * alpha(k2 + k4, k1 + k3) * beta(k1, k3) * beta(k2, k4) + 672 * alpha(k3, k1 + k2 + k4) * beta(k1, k2 + k4) * beta(k2, k4) + 2016 * alpha(k1 + k2 + k4, k3) * beta(k1, k2 + k4) * beta(k2, k4) + 112 * alpha(k3, k2 + k4) * beta(k1, k2 + k3 + k4) * beta(k2, k4) + 224 * alpha(k2 + k4, k3) * beta(k1, k2 + k3 + k4) * beta(k2, k4) + 252 * alpha(k1, k4) * alpha(k3, k1 + k2 + k4) * beta(k2, k1 + k4) + 252 * alpha(k3, k1 + k2 + k4) * alpha(k4, k1) * beta(k2, k1 + k4) + 756 * alpha(k1, k4) * alpha(k1 + k2 + k4, k3) * beta(k2, k1 + k4) + 756 * alpha(k4, k1) * alpha(k1 + k2 + k4, k3) * beta(k2, k1 + k4) + 672 * alpha(k3, k1 + k2 + k4) * beta(k1, k4) * beta(k2, k1 + k4) + 2016 * alpha(k1 + k2 + k4, k3) * beta(k1, k4) * beta(k2, k1 + k4) + 756 * alpha(k3, k4) * alpha(k2 + k3 + k4, k1) * beta(k2, k3 + k4) + 756 * alpha(k4, k3) * alpha(k2 + k3 + k4, k1) * beta(k2, k3 + k4) + 336 * alpha(k3, k4) * beta(k1, k2 + k3 + k4) * beta(k2, k3 + k4) + 336 * alpha(k4, k3) * beta(k1, k2 + k3 + k4) * beta(k2, k3 + k4) + 140 * alpha(k1, k4) * alpha(k3, k1 + k4) * beta(k2, k1 + k3 + k4) + 84 * alpha(k1, k3) * alpha(k1 + k3, k4) * beta(k2, k1 + k3 + k4) + 84 * alpha(k3, k1) * alpha(k1 + k3, k4) * beta(k2, k1 + k3 + k4) + 140 * alpha(k3, k1 + k4) * alpha(k4, k1) * beta(k2, k1 + k3 + k4) + 140 * alpha(k1, k3) * alpha(k4, k1 + k3) * beta(k2, k1 + k3 + k4) + 140 * alpha(k3, k1) * alpha(k4, k1 + k3) * beta(k2, k1 + k3 + k4) + 84 * alpha(k1, k4) * alpha(k1 + k4, k3) * beta(k2, k1 + k3 + k4) + 84 * alpha(k4, k1) * alpha(k1 + k4, k3) * beta(k2, k1 + k3 + k4) + 84 * alpha(k3, k4) * alpha(k3 + k4, k1) * beta(k2, k1 + k3 + k4) + 84 * alpha(k4, k3) * alpha(k3 + k4, k1) * beta(k2, k1 + k3 + k4) + 224 * alpha(k1 + k3, k4) * beta(k1, k3) * beta(k2, k1 + k3 + k4) + 112 * alpha(k4, k1 + k3) * beta(k1, k3) * beta(k2, k1 + k3 + k4) + 112 * alpha(k3, k1 + k4) * beta(k1, k4) * beta(k2, k1 + k3 + k4) + 224 * alpha(k1 + k4, k3) * beta(k1, k4) * beta(k2, k1 + k3 + k4) + 336 * alpha(k3, k4) * beta(k1, k3 + k4) * beta(k2, k1 + k3 + k4) + 336 * alpha(k4, k3) * beta(k1, k3 + k4) * beta(k2, k1 + k3 + k4) + 756 * alpha(k1, k2) * alpha(k1 + k2 + k3, k4) * beta(k1 + k2, k3) + 756 * alpha(k2, k1) * alpha(k1 + k2 + k3, k4) * beta(k1 + k2, k3) + 252 * alpha(k1, k2) * alpha(k4, k1 + k2 + k3) * beta(k1 + k2, k3) + 252 * alpha(k2, k1) * alpha(k4, k1 + k2 + k3) * beta(k1 + k2, k3) + 2016 * alpha(k1 + k2 + k3, k4) * beta(k1, k2) * beta(k1 + k2, k3) + 672 * alpha(k4, k1 + k2 + k3) * beta(k1, k2) * beta(k1 + k2, k3) + 252 * alpha(k1, k2) * alpha(k3, k1 + k2 + k4) * beta(k1 + k2, k4) + 252 * alpha(k2, k1) * alpha(k3, k1 + k2 + k4) * beta(k1 + k2, k4) + 756 * alpha(k1, k2) * alpha(k1 + k2 + k4, k3) * beta(k1 + k2, k4) + 756 * alpha(k2, k1) * alpha(k1 + k2 + k4, k3) * beta(k1 + k2, k4) + 672 * alpha(k3, k1 + k2 + k4) * beta(k1, k2) * beta(k1 + k2, k4) + 2016 * alpha(k1 + k2 + k4, k3) * beta(k1, k2) * beta(k1 + k2, k4) + 216 * alpha(k1, k2) * alpha(k3, k4) * beta(k1 + k2, k3 + k4) + 216 * alpha(k2, k1) * alpha(k3, k4) * beta(k1 + k2, k3 + k4) + 216 * alpha(k1, k2) * alpha(k4, k3) * beta(k1 + k2, k3 + k4) + 216 * alpha(k2, k1) * alpha(k4, k3) * beta(k1 + k2, k3 + k4) + 576 * alpha(k3, k4) * beta(k1, k2) * beta(k1 + k2, k3 + k4) + 576 * alpha(k4, k3) * beta(k1, k2) * beta(k1 + k2, k3 + k4) + 648 * alpha(k1, k2) * alpha(k1 + k2, k3 + k4) * beta(k3, k4) + 648 * alpha(k2, k1) * alpha(k1 + k2, k3 + k4) * beta(k3, k4) + 1176 * alpha(k2, k1 + k3 + k4) * alpha(k3 + k4, k1) * beta(k3, k4) + 2160 * alpha(k1, k2) * alpha(k3 + k4, k1 + k2) * beta(k3, k4) + 2160 * alpha(k2, k1) * alpha(k3 + k4, k1 + k2) * beta(k3, k4) + 504 * alpha(k3 + k4, k1) * alpha(k1 + k3 + k4, k2) * beta(k3, k4) + 252 * alpha(k2, k3 + k4) * alpha(k2 + k3 + k4, k1) * beta(k3, k4) + 504 * alpha(k3 + k4, k2) * alpha(k2 + k3 + k4, k1) * beta(k3, k4) + 1728 * alpha(k1 + k2, k3 + k4) * beta(k1, k2) * beta(k3, k4) + 1728 * alpha(k3 + k4, k1 + k2) * beta(k1, k2) * beta(k3, k4) + 672 * alpha(k2, k1 + k3 + k4) * beta(k1, k3 + k4) * beta(k3, k4) + 2016 * alpha(k1 + k3 + k4, k2) * beta(k1, k3 + k4) * beta(k3, k4) + 112 * alpha(k2, k3 + k4) * beta(k1, k2 + k3 + k4) * beta(k3, k4) + 224 * alpha(k3 + k4, k2) * beta(k1, k2 + k3 + k4) * beta(k3, k4) + 2016 * alpha(k2 + k3 + k4, k1) * beta(k2, k3 + k4) * beta(k3, k4) + 896 * beta(k1, k2 + k3 + k4) * beta(k2, k3 + k4) * beta(k3, k4) + 224 * alpha(k3 + k4, k1) * beta(k2, k1 + k3 + k4) * beta(k3, k4) + 896 * beta(k1, k3 + k4) * beta(k2, k1 + k3 + k4) * beta(k3, k4) + 576 * alpha(k1, k2) * beta(k1 + k2, k3 + k4) * beta(k3, k4) + 576 * alpha(k2, k1) * beta(k1 + k2, k3 + k4) * beta(k3, k4) + 1536 * beta(k1, k2) * beta(k1 + k2, k3 + k4) * beta(k3, k4) + 7 * alpha(k1, k3 + k4) * (21 * alpha(k2, k1 + k3 + k4) + 9 * alpha(k1 + k3 + k4, k2) + 4 * beta(k2, k1 + k3 + k4)) * (5 * alpha(k3, k4) + 5 * alpha(k4, k3) + 4 * beta(k3, k4)) + 252 * alpha(k1, k4) * alpha(k2, k1 + k3 + k4) * beta(k3, k1 + k4) + 252 * alpha(k2, k1 + k3 + k4) * alpha(k4, k1) * beta(k3, k1 + k4) + 756 * alpha(k1, k4) * alpha(k1 + k3 + k4, k2) * beta(k3, k1 + k4) + 756 * alpha(k4, k1) * alpha(k1 + k3 + k4, k2) * beta(k3, k1 + k4) + 672 * alpha(k2, k1 + k3 + k4) * beta(k1, k4) * beta(k3, k1 + k4) + 2016 * alpha(k1 + k3 + k4, k2) * beta(k1, k4) * beta(k3, k1 + k4) + 336 * alpha(k1, k4) * beta(k2, k1 + k3 + k4) * beta(k3, k1 + k4) + 336 * alpha(k4, k1) * beta(k2, k1 + k3 + k4) * beta(k3, k1 + k4) + 896 * beta(k1, k4) * beta(k2, k1 + k3 + k4) * beta(k3, k1 + k4) + 756 * alpha(k2, k4) * alpha(k2 + k3 + k4, k1) * beta(k3, k2 + k4) + 756 * alpha(k4, k2) * alpha(k2 + k3 + k4, k1) * beta(k3, k2 + k4) + 336 * alpha(k2, k4) * beta(k1, k2 + k3 + k4) * beta(k3, k2 + k4) + 336 * alpha(k4, k2) * beta(k1, k2 + k3 + k4) * beta(k3, k2 + k4) + 2016 * alpha(k2 + k3 + k4, k1) * beta(k2, k4) * beta(k3, k2 + k4) + 896 * beta(k1, k2 + k3 + k4) * beta(k2, k4) * beta(k3, k2 + k4) + 140 * alpha(k1, k2 + k4) * alpha(k2, k4) * beta(k3, k1 + k2 + k4) + 140 * alpha(k1, k4) * alpha(k2, k1 + k4) * beta(k3, k1 + k2 + k4) + 84 * alpha(k1, k2) * alpha(k1 + k2, k4) * beta(k3, k1 + k2 + k4) + 84 * alpha(k2, k1) * alpha(k1 + k2, k4) * beta(k3, k1 + k2 + k4) + 140 * alpha(k2, k1 + k4) * alpha(k4, k1) * beta(k3, k1 + k2 + k4) + 140 * alpha(k1, k2 + k4) * alpha(k4, k2) * beta(k3, k1 + k2 + k4) + 140 * alpha(k1, k2) * alpha(k4, k1 + k2) * beta(k3, k1 + k2 + k4) + 140 * alpha(k2, k1) * alpha(k4, k1 + k2) * beta(k3, k1 + k2 + k4) + 84 * alpha(k1, k4) * alpha(k1 + k4, k2) * beta(k3, k1 + k2 + k4) + 84 * alpha(k4, k1) * alpha(k1 + k4, k2) * beta(k3, k1 + k2 + k4) + 84 * alpha(k2, k4) * alpha(k2 + k4, k1) * beta(k3, k1 + k2 + k4) + 84 * alpha(k4, k2) * alpha(k2 + k4, k1) * beta(k3, k1 + k2 + k4) + 224 * alpha(k1 + k2, k4) * beta(k1, k2) * beta(k3, k1 + k2 + k4) + 112 * alpha(k4, k1 + k2) * beta(k1, k2) * beta(k3, k1 + k2 + k4) + 112 * alpha(k2, k1 + k4) * beta(k1, k4) * beta(k3, k1 + k2 + k4) + 224 * alpha(k1 + k4, k2) * beta(k1, k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k2, k4) * beta(k1, k2 + k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k4, k2) * beta(k1, k2 + k4) * beta(k3, k1 + k2 + k4) + 112 * alpha(k1, k2 + k4) * beta(k2, k4) * beta(k3, k1 + k2 + k4) + 224 * alpha(k2 + k4, k1) * beta(k2, k4) * beta(k3, k1 + k2 + k4) + 896 * beta(k1, k2 + k4) * beta(k2, k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k1, k4) * beta(k2, k1 + k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k4, k1) * beta(k2, k1 + k4) * beta(k3, k1 + k2 + k4) + 896 * beta(k1, k4) * beta(k2, k1 + k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k1, k2) * beta(k1 + k2, k4) * beta(k3, k1 + k2 + k4) + 336 * alpha(k2, k1) * beta(k1 + k2, k4) * beta(k3, k1 + k2 + k4) + 896 * beta(k1, k2) * beta(k1 + k2, k4) * beta(k3, k1 + k2 + k4) + 252 * alpha(k1, k3) * alpha(k2, k1 + k3 + k4) * beta(k1 + k3, k4) + 252 * alpha(k2, k1 + k3 + k4) * alpha(k3, k1) * beta(k1 + k3, k4) + 756 * alpha(k1, k3) * alpha(k1 + k3 + k4, k2) * beta(k1 + k3, k4) + 756 * alpha(k3, k1) * alpha(k1 + k3 + k4, k2) * beta(k1 + k3, k4) + 672 * alpha(k2, k1 + k3 + k4) * beta(k1, k3) * beta(k1 + k3, k4) + 2016 * alpha(k1 + k3 + k4, k2) * beta(k1, k3) * beta(k1 + k3, k4) + 336 * alpha(k1, k3) * beta(k2, k1 + k3 + k4) * beta(k1 + k3, k4) + 336 * alpha(k3, k1) * beta(k2, k1 + k3 + k4) * beta(k1 + k3, k4) + 896 * beta(k1, k3) * beta(k2, k1 + k3 + k4) * beta(k1 + k3, k4) + 216 * alpha(k1, k3) * alpha(k2, k4) * beta(k1 + k3, k2 + k4) + 216 * alpha(k2, k4) * alpha(k3, k1) * beta(k1 + k3, k2 + k4) + 216 * alpha(k1, k3) * alpha(k4, k2) * beta(k1 + k3, k2 + k4) + 216 * alpha(k3, k1) * alpha(k4, k2) * beta(k1 + k3, k2 + k4) + 576 * alpha(k2, k4) * beta(k1, k3) * beta(k1 + k3, k2 + k4) + 576 * alpha(k4, k2) * beta(k1, k3) * beta(k1 + k3, k2 + k4) + 576 * alpha(k1, k3) * beta(k2, k4) * beta(k1 + k3, k2 + k4) + 576 * alpha(k3, k1) * beta(k2, k4) * beta(k1 + k3, k2 + k4) + 1536 * beta(k1, k3) * beta(k2, k4) * beta(k1 + k3, k2 + k4) + 756 * alpha(k2, k3) * alpha(k2 + k3 + k4, k1) * beta(k2 + k3, k4) + 756 * alpha(k3, k2) * alpha(k2 + k3 + k4, k1) * beta(k2 + k3, k4) + 336 * alpha(k2, k3) * beta(k1, k2 + k3 + k4) * beta(k2 + k3, k4) + 336 * alpha(k3, k2) * beta(k1, k2 + k3 + k4) * beta(k2 + k3, k4) + 2016 * alpha(k2 + k3 + k4, k1) * beta(k2, k3) * beta(k2 + k3, k4) + 896 * beta(k1, k2 + k3 + k4) * beta(k2, k3) * beta(k2 + k3, k4) + 21 * alpha(k1, k2 + k3 + k4) * (21 * alpha(k2, k3) * alpha(k2 + k3, k4) + 21 * alpha(k3, k2) * alpha(k2 + k3, k4) + 35 * alpha(k3, k2 + k4) * alpha(k4, k2) + 35 * alpha(k2, k3) * alpha(k4, k2 + k3) + 35 * alpha(k3, k2) * alpha(k4, k2 + k3) + 21 * alpha(k4, k2) * alpha(k2 + k4, k3) + 21 * alpha(k3, k4) * alpha(k3 + k4, k2) + 21 * alpha(k4, k3) * alpha(k3 + k4, k2) + 56 * alpha(k2 + k3, k4) * beta(k2, k3) + 28 * alpha(k4, k2 + k3) * beta(k2, k3) + 28 * alpha(k3, k2 + k4) * beta(k2, k4) + 56 * alpha(k2 + k4, k3) * beta(k2, k4) + 12 * alpha(k3, k4) * beta(k2, k3 + k4) + 12 * alpha(k4, k3) * beta(k2, k3 + k4) + 56 * alpha(k3 + k4, k2) * beta(k3, k4) + 32 * beta(k2, k3 + k4) * beta(k3, k4) + 7 * alpha(k2, k3 + k4) * (5 * alpha(k3, k4) + 5 * alpha(k4, k3) + 4 * beta(k3, k4)) + 12 * alpha(k4, k2) * beta(k3, k2 + k4) + 32 * beta(k2, k4) * beta(k3, k2 + k4) + alpha(k2, k4) * (35 * alpha(k3, k2 + k4) + 21 * alpha(k2 + k4, k3) + 12 * beta(k3, k2 + k4)) + 12 * alpha(k2, k3) * beta(k2 + k3, k4) + 12 * alpha(k3, k2) * beta(k2 + k3, k4) + 32 * beta(k2, k3) * beta(k2 + k3, k4)) + 216 * alpha(k1, k4) * alpha(k2, k3) * beta(k2 + k3, k1 + k4) + 216 * alpha(k1, k4) * alpha(k3, k2) * beta(k2 + k3, k1 + k4) + 216 * alpha(k2, k3) * alpha(k4, k1) * beta(k2 + k3, k1 + k4) + 216 * alpha(k3, k2) * alpha(k4, k1) * beta(k2 + k3, k1 + k4) + 576 * alpha(k2, k3) * beta(k1, k4) * beta(k2 + k3, k1 + k4) + 576 * alpha(k3, k2) * beta(k1, k4) * beta(k2 + k3, k1 + k4) + 576 * alpha(k1, k4) * beta(k2, k3) * beta(k2 + k3, k1 + k4) + 576 * alpha(k4, k1) * beta(k2, k3) * beta(k2 + k3, k1 + k4) + 1536 * beta(k1, k4) * beta(k2, k3) * beta(k2 + k3, k1 + k4) + 140 * alpha(k1, k2 + k3) * alpha(k2, k3) * beta(k1 + k2 + k3, k4) + 140 * alpha(k1, k3) * alpha(k2, k1 + k3) * beta(k1 + k2 + k3, k4) + 84 * alpha(k1, k2) * alpha(k1 + k2, k3) * beta(k1 + k2 + k3, k4) + 84 * alpha(k2, k1) * alpha(k1 + k2, k3) * beta(k1 + k2 + k3, k4) + 140 * alpha(k2, k1 + k3) * alpha(k3, k1) * beta(k1 + k2 + k3, k4) + 140 * alpha(k1, k2 + k3) * alpha(k3, k2) * beta(k1 + k2 + k3, k4) + 140 * alpha(k1, k2) * alpha(k3, k1 + k2) * beta(k1 + k2 + k3, k4) + 140 * alpha(k2, k1) * alpha(k3, k1 + k2) * beta(k1 + k2 + k3, k4) + 84 * alpha(k1, k3) * alpha(k1 + k3, k2) * beta(k1 + k2 + k3, k4) + 84 * alpha(k3, k1) * alpha(k1 + k3, k2) * beta(k1 + k2 + k3, k4) + 84 * alpha(k2, k3) * alpha(k2 + k3, k1) * beta(k1 + k2 + k3, k4) + 84 * alpha(k3, k2) * alpha(k2 + k3, k1) * beta(k1 + k2 + k3, k4) + 224 * alpha(k1 + k2, k3) * beta(k1, k2) * beta(k1 + k2 + k3, k4) + 112 * alpha(k3, k1 + k2) * beta(k1, k2) * beta(k1 + k2 + k3, k4) + 112 * alpha(k2, k1 + k3) * beta(k1, k3) * beta(k1 + k2 + k3, k4) + 224 * alpha(k1 + k3, k2) * beta(k1, k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k2, k3) * beta(k1, k2 + k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k3, k2) * beta(k1, k2 + k3) * beta(k1 + k2 + k3, k4) + 112 * alpha(k1, k2 + k3) * beta(k2, k3) * beta(k1 + k2 + k3, k4) + 224 * alpha(k2 + k3, k1) * beta(k2, k3) * beta(k1 + k2 + k3, k4) + 896 * beta(k1, k2 + k3) * beta(k2, k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k1, k3) * beta(k2, k1 + k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k3, k1) * beta(k2, k1 + k3) * beta(k1 + k2 + k3, k4) + 896 * beta(k1, k3) * beta(k2, k1 + k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k1, k2) * beta(k1 + k2, k3) * beta(k1 + k2 + k3, k4) + 336 * alpha(k2, k1) * beta(k1 + k2, k3) * beta(k1 + k2 + k3, k4) + 896 * beta(k1, k2) * beta(k1 + k2, k3) * beta(k1 + k2 + k3, k4)) / 232848.;
  }

  double D_plus(double k, double a, double a_in, double m) {
    return (a/a_in);
  }

  double dD_plus(double k, double a, double a_in, double m) {
    return 1./a_in;
  }
  double ddD_plus(double k, double a, double a_in, double m) {
    return 0;
  }
  double D_minus(double k, double a, double a_in, double m) {
    return pow(a/a_in, -1.5);
  }

  double dD_minus(double k, double a, double a_in, double m) {
    return pow(a, -5./2) * (-3./2) / pow(a_in, -3./2);
  }

  double ddD_minus(double k, double a, double a_in, double m) {
    return pow(a,-7./2) * (-3./2) * (-5./2) / pow(a_in,-3./2);
  }

  double propagator(double k, double s, double eta, double m) {
    double a = a_from_eta(s);
    double b = a_from_eta(eta);
    
    double Dp_a   = D_plus (k, a, const_a_in, m);
    double Dm_a   = D_minus(k, a, const_a_in, m);
    double Dp_b   = D_plus (k, b, const_a_in, m);
    double Dm_b   = D_minus(k, b, const_a_in, m);

    double dDp_a = dD_plus (k, a, const_a_in, m) * sqrt(a);
    double dDm_a = dD_minus(k, a, const_a_in, m) * sqrt(a);
    //double dDp_b = dD_plus (k, b, const_a_in, m) * sqrt(b);
    //double dDm_b = dD_minus(k, b, const_a_in, m) * sqrt(b);

    double wronskian =  Dm_a * dDp_a - dDm_a * Dp_a;
    double result    = (Dm_a *  Dp_b -  Dm_b * Dp_a) / wronskian;
    return result;
  }

  double d_s_propagator(double k, double s, double eta, double m) {
    double a = a_from_eta(s);
    double b = a_from_eta(eta);
    
    double Dp_a      = D_plus (k, a, const_a_in, m);
    double Dm_a      = D_minus(k, a, const_a_in, m);
    double Dp_b      = D_plus (k, b, const_a_in, m);
    double Dm_b      = D_minus(k, b, const_a_in, m);
    double dDp_a     = dD_plus (k, a, const_a_in, m) * sqrt(a);
    double dDm_a     = dD_minus(k, a, const_a_in, m) * sqrt(a);
    //double dDp_b     = dD_plus (k, b, const_a_in, m) * sqrt(b);
    //double dDm_b     = dD_minus(k, b, const_a_in, m) * sqrt(b);
    double ddDp_a    = 0.5 * dD_plus (k, a, const_a_in, m)  + a * ddD_plus (k, a, const_a_in, m);
    double ddDm_a    = 0.5 * dD_minus(k, a, const_a_in, m)  + a * ddD_minus(k, a, const_a_in, m);
    double wronskian =   Dm_a * dDp_a - dDm_a *  Dp_a;
    double result    = (dDm_a *  Dp_b -  Dm_b * dDp_a) / wronskian - (Dm_a * Dp_b - Dm_b * Dp_a) / (wronskian * wronskian) * (Dm_a * ddDp_a - ddDm_a * Dp_a);
    return result;
  }

  double d_eta_propagator(double k, double s, double eta, double m) {
    double a = a_from_eta(s);
    double b = a_from_eta(eta);
    
    double Dp_a   = D_plus (k, a, const_a_in, m);
    double Dm_a   = D_minus(k, a, const_a_in, m);
    //double Dp_b   = D_plus (k, b, const_a_in, m);
    //double Dm_b   = D_minus(k, b, const_a_in, m);
    double dDp_a  = dD_plus (k, a,  const_a_in, m) * sqrt(a);
    double dDm_a  = dD_minus (k, a, const_a_in, m) * sqrt(a);
    double dDp_b  = dD_plus (k, b,  const_a_in, m) * sqrt(b);
    double dDm_b  = dD_minus (k, b, const_a_in, m) * sqrt(b);
    double wronskian =  Dm_a * dDp_a - dDm_a * Dp_a;
    double result    = (Dm_a * dDp_b - dDm_b * Dp_a) / wronskian;
    return result;
  }

  double d_s_eta_propagator(double k, double  s, double eta, double  m) {
    double a = a_from_eta(s);
    double b = a_from_eta(eta);

    double Dp_a  = D_plus  (k, a, const_a_in, m);
    double Dm_a  = D_minus (k, a, const_a_in, m);
    //double Dp_b  = D_plus  (k, b, const_a_in, m);
    //double Dm_b  = D_minus (k, b, const_a_in, m);
    double dDp_a = dD_plus (k, a, const_a_in, m) * sqrt(a);
    double dDm_a = dD_minus(k, a, const_a_in, m) * sqrt(a);
    double dDp_b = dD_plus (k, b, const_a_in, m) * sqrt(b);
    double dDm_b = dD_minus(k, b, const_a_in, m) * sqrt(b);
    double ddDp_a         = 0.5 * dD_plus (k, a, const_a_in, m) + a * ddD_plus (k, a, const_a_in, m);
    double ddDm_a         = 0.5 * dD_minus(k, a, const_a_in, m) + a * ddD_minus(k, a, const_a_in, m);

    double wronskian =   Dm_a * dDp_a - dDm_a *  Dp_a;
    double result    = (dDm_a * dDp_b - dDm_b * dDp_a) / wronskian - (Dm_a * dDp_b - dDm_b * Dp_a) / (wronskian * wronskian) * (Dm_a * ddDp_a - ddDm_a * Dp_a);
    return result;
  }


  double fi_coupling(double k, double s, double eta, double m, size_t i) {
    double a = a_from_eta(s);
    double b = a_from_eta(eta);
    switch(i) {
      case 0:
      return  D_plus(k, a, const_a_in, m) / D_plus(k, b, const_a_in, m);
      case 1:
      return -dD_plus(k, a, const_a_in, m) / D_plus(k, b, const_a_in, m) * sqrt(a);
    }
    throw std::runtime_error("fi_coupling called for i > 1");
  }

  CDM_Analytical_CosmoUtil::CDM_Analytical_CosmoUtil(double eta, double eta_in) : CosmoUtil(0, eta, eta_in) {}
  double CDM_Analytical_CosmoUtil::D              (double k, double eta)                     const {return CDM::D       (eta, eta_in);}          
  double CDM_Analytical_CosmoUtil::greens         (double k, double s, double eta)           const {return 0;}
  double CDM_Analytical_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return 0;}
  double CDM_Analytical_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return 0;}
  double CDM_Analytical_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return 0;}
  double CDM_Analytical_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return 0;}

  CDM_Numerical_CosmoUtil::CDM_Numerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in) : CosmoUtil(1, eta, eta_in), d(new FDM::D_spline(eta_in, fdm_mass_id, true)) {}
  double CDM_Numerical_CosmoUtil::D              (double k, double eta)                     const {return (*d)(1e-3, eta);}       
  double CDM_Numerical_CosmoUtil::greens         (double k, double s, double eta)           const {return propagator         (k, s, eta, m);}
  double CDM_Numerical_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return d_s_propagator     (k, s, eta, m);}
  double CDM_Numerical_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return d_eta_propagator   (k, s, eta, m);}
  double CDM_Numerical_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return d_s_eta_propagator (k, s, eta, m);}
  double CDM_Numerical_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return fi_coupling        (k, s, eta, m, i);}
  
}

// Define namespace for all Fuzzy Dark Matter related functions
namespace FDM
{

  double effective_sound_speed_aux(double a, double k, double mass)
  {
    return pow(const_hbar_in_ev / mass, 2) * pow(k * const_clight / (2 * a * const_mparsec), 2);
  }

  double effective_sound_speed(double a, double k, double mass)
  {
    double cs = effective_sound_speed_aux(a, k, mass);
    return cs;// / (1.0 + cs);
  }

  double aux_c0(double a, double k, double mass)
  {
    return -1.5 * const_omega_m / pow(a, 5) / pow(hubble(a), 2) + 1 / pow(const_hubble * hubble(a) * a, 2) * effective_sound_speed(a, k, mass) * pow(k * const_clight / (a * const_mparsec), 2);
  }

  double daux_c0(double a, double k, double mass)
  {
    double da = 1e-8;
    return (aux_c0(a + da, k, mass) - aux_c0(a - da, k, mass)) / (2 * da);
  }

  double aux_c1(double a)
  {
    return 3.0 / a + dloghubble(a);
  }

  double daux_c1(double a)
  {
    return -3.0 / (a * a) + d2loghubble(a);
  }

  /* --- d_plus_function [defines f0 = dy1/da and f1 = dy2/da] --- */
  /* Growth equation is second-order ODE given as y(a, k)'' + aux_c1(a) y(a, k)' + (aux_c0(a) + aux_c0_fdm(a,k)) y(a, k) = 0*/
  int d_function(double a, const double y[], double f[], void *params)
  {
    double *par = (double *)params;
    double k = par[0];
    double mass = par[1];
    /* derivatives f_i = dy_i/dt */
    f[0] = y[1];
    f[1] = -aux_c0(a, k, mass) * y[0] - aux_c1(a) * y[1];

    return (GSL_SUCCESS);
  }
  /* --- end of function dplus_function --- */

  /* --- function dplus_jacobian --- */
  int d_jacobian(double a, const double y[], double *dfdy, double dfdt[], void *params)
  {
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array(dfdy, 2, 2);
    gsl_matrix *m = &dfdy_mat.matrix;

    double *par = (double *)params;
    double k = par[0];
    double mass = par[1];

    /* jacobian df_i(t,y(a)) / dy_j */
    gsl_matrix_set(m, 0, 0, 0.0);
    gsl_matrix_set(m, 0, 1, 1.0);
    gsl_matrix_set(m, 1, 0, -aux_c0(a, k, mass));
    gsl_matrix_set(m, 1, 1, -aux_c1(a));

    /* gradient df_i/da, explicit dependence */
    dfdt[0] = 0.0;
    dfdt[1] = -daux_c0(a, k, mass) * y[0] - daux_c1(a) * y[1];

    return (GSL_SUCCESS);
  }
  /* --- end of function dplus_jacobian --- */

  /************************************************
   *  Integrand for d_plus
   *  Obtained via integrating governing ODE
   *  Initial scale factor is 1e-4
   ***********************************************/
  void aux_d(const std::vector<double> &a, double k, double mass, std::vector<double> &result_d_plus, std::vector<double> &result_d_plus_prime, double growingMode = true)
  {
    const gsl_odeiv_step_type *T = gsl_odeiv_step_bsimp;
    gsl_odeiv_step *s = gsl_odeiv_step_alloc(T, 2);
    gsl_odeiv_control *c = gsl_odeiv_control_y_new(0.0, epsrel);
    gsl_odeiv_evolve *e = gsl_odeiv_evolve_alloc(2);
    double params[2] = {k, mass};
    gsl_odeiv_system sys = {d_function, d_jacobian, 2, params};
    double h, a0, y[2]; /* initial conditions */

    if (growingMode)
    {
      // Get growing mode by assuming IC for matter-dominated universe D+(a) = a*/
      h = 1e-4;
      a0 = const_a_in;
      y[0] = 1;
      y[1] = CDM::get_ic(const_a_in, true);
    }
    else
    {
      // Get decaying mode by assuming IC for matter-dominated universe D-(a) = a^(-3/2)*/
      h = -1e-4;
      a0 = 1;
      y[0] = pow(a0, -1.5);
      y[1] = -1.5 * pow(a0, -2.5);
    }

    result_d_plus.reserve(a.size());
    result_d_plus_prime.reserve(a.size());

    // std::cout << "as in d_plus fdm call"<<std::endl;
    // for (double x : a) {
    //   std::cout << " " <<x<<" ";
    // }
    // std::cout<<std::endl;

    if (growingMode)
    {
      for (size_t i = 0; i < a.size(); ++i)
      {
        while (a0 < a[i])
        {
          gsl_odeiv_evolve_apply(e, c, s, &sys, &a0, a[i], &h, y);
        }
        result_d_plus.push_back(y[0]);
        result_d_plus_prime.push_back(y[1]);
      }
    }
    else
    {
      for (int i = a.size() - 1; i >= 0; --i)
      {
        while (a0 > a[i])
        {
          gsl_odeiv_evolve_apply(e, c, s, &sys, &a0, a[i], &h, y);
        }

        result_d_plus.push_back(y[0]);
        result_d_plus_prime.push_back(y[1]);
      }
    }
    gsl_odeiv_evolve_free(e);
    gsl_odeiv_control_free(c);
    gsl_odeiv_step_free(s);
  }

  std::vector<double> d_plus(const std::vector<double> &a, double a_in, double k, double mass)
  {
    std::vector<double> result, dummy, norm, anorm = {a_in};
    aux_d(a, k, mass, result, dummy, true);
    aux_d(anorm, k, mass, norm, dummy, true);

    for (size_t i = 0; i < a.size(); i++)
    {
      result[i] /= norm[0];
    }

    return (result);
  }

  std::vector<double> d_minus(const std::vector<double> &a, double a_in, double k, double mass)
  {
    std::vector<double> result, dummy, norm, anorm = {a_in};
    aux_d(a, k, mass, result, dummy, false);
    aux_d(anorm, k, mass, norm, dummy, false);
    // Reverse result vector because we start integration from high scale factors
    std::reverse(result.begin(), result.end());

    for (size_t i = 0; i < a.size(); i++)
    {
      result[i] /= norm[0];
    }

    return (result);
  }

  double d_plus(double a, double a_in, double k, double mass)
  {
    std::vector<double> result, av = {a};
    result = d_plus(av, a_in, k, mass);
    return (result.at(0));
  }

  double d_minus(double a, double a_in, double k, double mass)
  {
    std::vector<double> result, av = {a};
    result = d_minus(av, a_in, k, mass);
    return (result.at(0));
  }

  /***
   * Structure constructing and initialising spline with FDM growth function in constructor
   * Call as d = D(); d.s(a, k); to get FDM growth factor at scale_factor a
   ***/

  D_spline::D_spline(double eta_in, int fdm_mass_id, bool growingMode)
  {
    double mass = fdm_masses[fdm_mass_id];
    std::cout << "Setting up D spline at eta_in = " << eta_in << " mass = " << mass << " for growingMode = " << growingMode << std::endl;
    std::vector<double> as, ks, z, ds;

    double a_in = a_from_eta(eta_in);
    as = pyLogspace(const_a_min, const_a_max, const_a_res);
    if (as.front() > const_a_in)
    {
      as.insert(as.begin(), const_a_in - 1e-4);
    }
    if (as.back() < 1)
    {
      as.push_back(1);
    }
    ks = pyLogspace(const_k_min, const_k_max, const_k_res);

    this->kmin_ = ks.front() + 1e-6;
    this->kmax_ = ks.back() - 1e-6;

    z.reserve(as.size() * ks.size());
    std::ifstream indata;
    std::string filename = "splines/D" + std::to_string(growingMode) + "_" + cosmo_string + "_" + fdm_mass_strings[fdm_mass_id] + ".dat";
    indata.open(filename); // opens the file
    std::cout << filename << "\n";

    if (LOAD_D_SPLINES && indata)
    {
      z.insert(z.begin(), std::istream_iterator<double>(indata), std::istream_iterator<double>());

      if (z.size() != ks.size() * as.size())
      {
        std::cerr << "z size " << z.size() << " whereas it should be " << ks.size() * as.size() << std::endl;
        throw std::runtime_error("Check number of elements in 2d spline");
      }
      else
      {
        std::cout << "Successfully loaded spline from disk!" << std::endl;
      }
    }
    else
    {
      for (double k : ks)
      {
        ds.clear();
        ds.reserve(as.size());

        if (growingMode)
        {
          ds = d_plus(as, a_in, k, mass);
        }
        else
        {
          ds = d_minus(as, a_in, k, mass);
        }

        if (ds.size() != as.size())
        {
          throw std::runtime_error("Check integration of scale factor for 2D spline");
        }
        z.insert(z.end(), ds.begin(), ds.end());
      }
    }

    indata.close();

    if (SAVE_D_SPLINES)
    {
      std::ofstream outdata;                                                                                     // outdata is like cin
      outdata.open("splines/D" + std::to_string(growingMode) + "_" + cosmo_string + "_" + fdm_mass_strings[fdm_mass_id] + ".dat"); // opens the file
      if (!outdata)
      { // file couldn't be opened
        std::cerr << "Error: file for storing D spline could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
      }
      std::copy(z.cbegin(), z.cend(), std::ostream_iterator<double>(outdata, " "));
      outdata.close();
    }

    double *x0 = &as[0];
    double *y0 = &ks[0];
    double *z0 = &z[0];
    s = std::make_shared<Spline2D>(x0, y0, z0, as.size(), ks.size());

    amin_ = as.front();
    amax_ = as.back();
  }

  double D_spline::operator()(double k, double eta)
  {
    double a, result;
    a = a_from_eta(eta);


    if (a < amin_ || a > amax_)
    {
      throw std::runtime_error("Value of a in D_spline second derivative at k = " +std::to_string(k) + " out of range with a_min = " + std::to_string(amin_) + " amax = " + std::to_string(amax_) + " a = " + std::to_string(a));
    }

    if (k < kmin_)
    {
      k = kmin_;
    }
    result = (*s)(a, (k < kmax_) ? k : kmax_);
    if (k >= kmax_)
    {
      result = result * pow(k / kmax_, -4);
    }
    return result;
  }

  // Apply chain rule df/ deta = da / deta * df/ da = sqrt(a) * df/da
  double D_spline::deta(double k, double eta)
  {
    double a, result;
    a = a_from_eta(eta);

    if (a < amin_ || a > amax_)
    {
      throw std::runtime_error("Value of a in D_spline second derivative at k = " +std::to_string(k) + " out of range with a_min = " + std::to_string(amin_) + " amax = " + std::to_string(amax_) + " a = " + std::to_string(a));
    }

    result = (*s).dx(a, (k < kmax_) ? k : kmax_) * sqrt(a);
    if (k >= kmax_)
    {
      result = result * pow(k / kmax_, -4);
    }
    return result;
  }

  // Apply chain rule d^2f / d^2eta = d/deta (eta/2 * df/da) = 1/2 df/a + a * d^2f/da^2
  double D_spline::d2eta(double k, double eta)
  {
    double a, result;
    a = a_from_eta(eta);

    if (a < amin_ || a > amax_)
    {
      throw std::runtime_error("Value of a in D_spline second derivative at k = " +std::to_string(k) + " out of range with a_min = " + std::to_string(amin_) + " amax = " + std::to_string(amax_) + " a = " + std::to_string(a));
    }

    result = ((*s).d2x(a, (k < kmax_) ? k : kmax_) * a + 0.5 * (*s).dx(a, (k < kmax_) ? k : kmax_));
    if (k >= kmax_)
    {
      result = result * pow(k / kmax_, -4);
    }
    return result;
  }


  D_hybrid::D_hybrid(int fdm_mass_id) : s(), fdm_mass_id(fdm_mass_id)
  {
  }

  double D_hybrid::operator()(double k, double eta)
  {
    double a = a_from_eta(eta);
    double kj = jeans_scale(const_eta_in, fdm_masses[fdm_mass_id]);

    return 1 / (1 + fdm_growth_factor_fit_alpha[fdm_mass_id] * pow(k / kj, fdm_growth_factor_fit_beta[fdm_mass_id])) * s(a);
  }


  G_spline::G_spline(double eta_in, int fdm_mass_id) : dp(eta_in, fdm_mass_id, true), dm(eta_in, fdm_mass_id, false)
  {
  }

  double G_spline::propagator(double k, double s, double eta)
  {
    double wronskian, result, p, q, dp_p, dp_q, dm_p, dm_q, ddp_p, ddm_p;

    p = s;
    q = eta;
    dp_p = dp(k, p);
    dp_q = dp(k, q);
    dm_p = dm(k, p);
    dm_q = dm(k, q);
    ddp_p = dp.deta(k, p);
    ddm_p = dm.deta(k, p);

    wronskian = dm_p * ddp_p - ddm_p * dp_p;
    result = (dm_p * dp_q - dm_q * dp_p) / wronskian;
    return result;
  }

  double G_spline::d_s_propagator(double k, double s, double eta)
  {
    double wronskian, result, p, q, dp_p, dp_q, dm_p, dm_q, ddp_p, ddm_p, dddp_p, dddm_p;
    // double ds = 1e-8;
    // return (propagator(k, s+ds, eta) - propagator(k, s-ds, eta))/(2*ds);
    p = s;
    q = eta;
    dp_p = dp(k, p);
    dp_q = dp(k, q);
    dm_p = dm(k, p);
    dm_q = dm(k, q);
    ddp_p = dp.deta(k, p);
    ddm_p = dm.deta(k, p);
    dddp_p = dp.d2eta(k, p);
    dddm_p = dm.d2eta(k, p);

    wronskian = dm_p * ddp_p - ddm_p * dp_p;
    result = (ddm_p * dp_q - dm_q * ddp_p) / wronskian - (dm_p * dp_q - dm_q * dp_p) / (wronskian * wronskian) * (dm_p * dddp_p - dddm_p * dp_p);
    return result;
  }

  double G_spline::d_eta_propagator(double k, double s, double eta)
  {
    // double deta = 1e-8;
    // return (propagator(k, s, eta+deta) - propagator(k, s, eta- deta))/(2*deta);
    double wronskian, result, p, q, dp_p, dm_p, ddp_p, ddm_p, ddp_q, ddm_q;

    p = s;
    q = eta;
    dp_p = dp(k, p);
    dm_p = dm(k, p);
    ddp_p = dp.deta(k, p);
    ddm_p = dm.deta(k, p);
    ddp_q = dp.deta(k, q);
    ddm_q = dm.deta(k, q);

    wronskian = dm_p * ddp_p - ddm_p * dp_p;
    result = (dm_p * ddp_q - ddm_q * dp_p) / wronskian;

    return result;
  }

  double G_spline::d_s_eta_propagator(double k, double s, double eta)
  {
    // double ds   = 1e-8;
    // double deta = 1e-8;
    // double dsp1 = ds_propagator(k, s, eta + deta);
    // double dsp2 = ds_propagator(k, s, eta - deta);
    // return (dsp1 - dsp2)/(2*deta);
    double wronskian, result, p, q, dp_p, dm_p, ddp_p, ddm_p, ddp_q, ddm_q, dddp_p, dddm_p;

    p = s;
    q = eta;
    dp_p = dp(k, p);
    dm_p = dm(k, p);
    ddp_p = dp.deta(k, p);
    ddm_p = dm.deta(k, p);
    ddp_q = dp.deta(k, q);
    ddm_q = dm.deta(k, q);
    dddp_p = dp.d2eta(k, p);
    dddm_p = dm.d2eta(k, p);

    wronskian = dm_p * ddp_p - ddm_p * dp_p;
    result = (ddm_p * ddp_q - ddm_q * ddp_p) / wronskian - (dm_p * ddp_q - ddm_q * dp_p) / (wronskian * wronskian) * (dm_p * dddp_p - dddm_p * dp_p);
    return result;
  }

  double G_spline::f_i(double k, double s, double eta, size_t i)
  {
    if (i == 0)
    {
      return dp(k, s) / dp(k, eta);
    }
    else if (i == 1)
    {
      return -dp.deta(k, s) / dp(k, eta);
    }
    else
    {
      throw std::invalid_argument("Argument a in f_i is invalid.");
    }
  }
  // Clever numerical implemention of Bessel function derivatives taken from Python's numpy library
  //double my_bessel_diff_formula(double v, double z, size_t n, double phase)
  //{
  //  double p = 1.0;
  //  double s = boost::math::cyl_bessel_j(v - n, z);
  //  for (size_t i = 1; i < n + 1; ++i)
  //  {
  //    p = phase * (p * (n - i + 1)) / i;
  //    s += p * boost::math::cyl_bessel_j(v - n + i * 2, z);
  //  }
  //  return s / pow(2., n);
  //}
//
  //// nth derivative of cylindrical Bessel function of fractional order
  //double my_jvp(double v, double z, size_t n)
  //{
  //  // Return the nth derivative of Jv(z) with respect to z.
  //  if (n == 0)
  //  {
  //    return boost::math::cyl_bessel_j(v, z);
  //  }
  //  else
  //  {
  //    return my_bessel_diff_formula(v, z, n, -1.);
  //  }
  //}

  double m_bar(double m)
  {
    return m * const_electron_charge / const_hbar; // 1/s
  }

  double mbH0(double m)
  {
    // 2*k^2 / (H0*Omega_m^1/2 mb) * c^2 = 2*k^2 1/257.3 Mpc^2
    return (m_bar(m) * const_hubble_bar) / pow(const_clight, 2) * pow(const_mparsec, 2); // 1/m**2
  }

  // Quantum jeans scale as function of scale factor and mass
  // Equation (18) from Li 2020
  double jeans_scale(double eta, double m)
  {
    double a = a_from_eta(eta);
    return 44.7 * const_h_hubble * pow(6 * a * const_omega_m / 0.3, 0.25) * pow(m / (1e-22), 0.5); // #Mpc^-1
  }

  // Mass and momentum dependent FDM-scale
  double b_f(double k, double m)
  {
    return pow(k, 2) * 2 / mbH0(m);
  }

  // First three terms in Taylor expansion of fractional Bessel function
  double jv_taylor(double n, double x)
  {
    double p = gsl_sf_gamma(1. - n);
    double t1 = pow(2, n);
    double t2 = pow(2, n - 2) * pow(x, 2) / (n - 1);
    double t3 = pow(2, n - 5) * pow(x, 4) / (n - 2) / (n - 1.);
    return pow(x, -n) * (t1 + t2 + t3) / p;
  }

  // Growth function in FDM
  // Renormalised according to Lagüe paper
  double D_plus_renormalised(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    return sqrt(eta_in / eta) * boost::math::cyl_bessel_j(-2.5, bk / eta) / jv_taylor(2.5, bk / eta_in);
  }

  // Growth function in FDM
  double D_plus_analytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    return sqrt(eta_in / eta) * boost::math::cyl_bessel_j(-2.5, bk / eta) / boost::math::cyl_bessel_j(-2.5, bk / eta_in);
  }

  double D_minus_analytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    return sqrt(eta_in / eta) * boost::math::cyl_bessel_j(2.5, bk / eta) / boost::math::cyl_bessel_j(2.5, bk / eta_in);
  }


  double D_minus_renormalised(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    return sqrt(eta_in / eta) * boost::math::cyl_bessel_j(2.5, bk / eta) / jv_taylor(-2.5, bk / eta_in);
  }


  // Derivative of renormalised growth factor
  double d_D_plus_analytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    double p1 = sqrt(eta_in / eta);
    double p2 = (-bk) / pow(eta, 2);                             // Contribution from chain rule
    double p3 = boost::math::cyl_bessel_j_prime(-2.5, bk / eta); // Derivative of Bessel function
    double p4 = boost::math::cyl_bessel_j(-2.5, bk / eta_in);
    double p5 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    return p1 * p2 * p3 / p4 - 1. / 2. * p1 / eta * p5 / p4;
  }

  // Growth function in FDM
  double D_plus_semianalytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    return sqrt(1 / eta) * boost::math::cyl_bessel_j(-2.5, bk / eta) / (eta_in * eta_in / 4);
  }

  // Derivative of renormalised growth factor
  double d_D_plus_semianalytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    double p1 = sqrt(1 / eta);
    double p2 = (-bk) / pow(eta, 2);                             // Contribution from chain rule
    double p3 = boost::math::cyl_bessel_j_prime(-2.5, bk / eta); // Derivative of Bessel function
    double p4 = eta_in * eta_in / 4;
    double p5 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    return p1 * p2 * p3 / p4 - 1. / 2. * p1 / eta * p5 / p4;
  }

  double d_D_minus_analytical(double k, double eta, double eta_in, double m)
  {
    double bk = b_f(k, m);
    double p1 = sqrt(eta_in / eta);
    double p2 = (-bk) / pow(eta, 2);                            // Contribution from chain rule
    double p3 = boost::math::cyl_bessel_j_prime(2.5, bk / eta); // Derivative of Bessel function
    double p4 = boost::math::cyl_bessel_j_prime(2.5, bk / eta_in);
    double p5 = boost::math::cyl_bessel_j(2.5, bk / eta);
    return p1 * p2 * p3 / p4 - 1. / 2. * p1 / eta * p5 / p4;
  }

  // Use empirical transfer function
  // from Hu et al. 2000
  double empirical_transfer(double k, double eta, double eta_in, double m)
  {
    double kJQ = jeans_scale(eta_in, m); // where k in 1/Mpc and m in eV
    double x = 1.61 * pow(m / 1e-22, 1. / 18) * k / kJQ;
    return cos(pow(x, 3)) / (1 + pow(x, 8));
  }

  double greens_analytical(double k, double s, double eta, double m)
  {
    double bk = b_f(k, m);
    double p1 = M_PI * pow(s, 2) / 2. * pow(s, -0.5) * pow(eta, -0.5);

    double j1 = boost::math::cyl_bessel_j(2.5, bk / s);
    double j2 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    double j3 = boost::math::cyl_bessel_j(2.5, bk / eta);
    double j4 = boost::math::cyl_bessel_j(-2.5, bk / s);

    return p1 * (j1 * j2 - j3 * j4);
  }

  // s-derivative of greens
  double d_s_greens_analytical(double k, double s, double eta, double m)
  {
    double bk = b_f(k, m);
    double p1 = 3 * M_PI_4 * pow(s, 0.5) * pow(eta, -0.5);
    double p2 = M_PI_2 * bk * pow(s, -0.5) * pow(eta, -0.5);

    double j1 = boost::math::cyl_bessel_j(2.5, bk / s);
    double j2 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    double j3 = boost::math::cyl_bessel_j(2.5, bk / eta);
    double j4 = boost::math::cyl_bessel_j(-2.5, bk / s);
    double j5 = boost::math::cyl_bessel_j_prime(2.5, bk / s);
    double j8 = boost::math::cyl_bessel_j_prime(-2.5, bk / s);

    return p1 * (j1 * j2 - j3 * j4) + p2 * (j3 * j8 - j5 * j2);
  }

  // eta-derivative of greens
  double d_eta_greens_analytical(double k, double s, double eta, double m)
  {
    double bk = b_f(k, m);
    double p1 = -M_PI_4 * pow(s, 1.5) * pow(eta, -1.5);
    double p2 = M_PI_2 * bk * pow(s, 1.5) * pow(eta, -2.5);

    double j1 = boost::math::cyl_bessel_j(2.5, bk / s);
    double j2 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    double j3 = boost::math::cyl_bessel_j(2.5, bk / eta);
    double j4 = boost::math::cyl_bessel_j(-2.5, bk / s);
    double j6 = boost::math::cyl_bessel_j_prime(-2.5, bk / eta);
    double j7 = boost::math::cyl_bessel_j_prime(2.5, bk / eta);
    return p1 * (j1 * j2 - j3 * j4) + p2 * (j4 * j7 - j6 * j1);
  }

  // s- and eta- second derivative of greens
  double d_s_eta_greens_analytical(double k, double s, double eta, double m)
  {
    double bk = b_f(k, m);
    double p1 = -3 * M_PI / 8 * pow(s, 0.5) * pow(eta, -1.5);
    double p2 = M_PI_4 * pow(s, -0.5) * pow(eta, -1.5) * bk;

    double p3 = -3 * M_PI_4 * pow(s, 0.5) * pow(eta, -2.5) * bk;
    double p4 = M_PI_2 * pow(s, -0.5) * pow(eta, -2.5) * pow(bk, 2);

    double j1 = boost::math::cyl_bessel_j(2.5, bk / s);
    double j2 = boost::math::cyl_bessel_j(-2.5, bk / eta);
    double j3 = boost::math::cyl_bessel_j(2.5, bk / eta);
    double j4 = boost::math::cyl_bessel_j(-2.5, bk / s);
    double j5 = boost::math::cyl_bessel_j_prime(2.5, bk / s);
    double j6 = boost::math::cyl_bessel_j_prime(-2.5, bk / eta);
    double j7 = boost::math::cyl_bessel_j_prime(2.5, bk / eta);
    double j8 = boost::math::cyl_bessel_j_prime(-2.5, bk / s);

    return p1 * (j1 * j2 - j3 * j4) + p2 * (j5 * j2 - j3 * j8) + p3 * (j6 * j1 - j4 * j7) + p4 * (j5 * j6 - j7 * j8);
  }

  double f_i_analytical(double k, double s, double eta, double m, size_t i)
  {
    // Value of f_a is independent of eta_i in integration context
    double eta_in = eta_from_z(1);

    if (i == 0)
    {
      return D_plus_analytical(k, s, eta_in, m) / D_plus_analytical(k, eta, eta_in, m);
    }
    else if (i == 1)
    {
      return -d_D_plus_analytical(k, s, eta_in, m) / D_plus_analytical(k, eta, eta_in, m);
    }
    else
    {
      throw std::invalid_argument("Argument a in f_i is invalid.");
    }
  }


  FDM_Analytical_CosmoUtil::FDM_Analytical_CosmoUtil(int fdm_mass_id, double eta, double eta_in) : CosmoUtil(fdm_masses[fdm_mass_id], eta, eta_in) {}
  double FDM_Analytical_CosmoUtil::D              (double k, double eta)                     const {return D_plus_renormalised       (k, eta, eta_in, m);}          
  double FDM_Analytical_CosmoUtil::greens         (double k, double s, double eta)           const {return greens_analytical         (k, s, eta, m)   ;}
  double FDM_Analytical_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return d_s_greens_analytical     (k, s, eta, m)   ;}
  double FDM_Analytical_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return d_eta_greens_analytical   (k, s, eta, m)   ;}
  double FDM_Analytical_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return d_s_eta_greens_analytical (k, s, eta, m)   ;}
  double FDM_Analytical_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return f_i_analytical            (k, s, eta, m, i);}
  

  FDM_SemiNumerical_CosmoUtil::FDM_SemiNumerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in) : CosmoUtil(fdm_masses[fdm_mass_id], eta, eta_in), g(new G_spline(eta_in, fdm_mass_id)) {}
  double FDM_SemiNumerical_CosmoUtil::D              (double k, double eta)                     const {return g->dp                  (k, eta);}          
  double FDM_SemiNumerical_CosmoUtil::greens         (double k, double s, double eta)           const {return greens_analytical         (k, s, eta, m);}
  double FDM_SemiNumerical_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return d_s_greens_analytical     (k, s, eta, m);}
  double FDM_SemiNumerical_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return d_eta_greens_analytical   (k, s, eta, m);}
  double FDM_SemiNumerical_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return d_s_eta_greens_analytical (k, s, eta, m);}
  double FDM_SemiNumerical_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return f_i_analytical            (k, s, eta, m, i);}
  
  FDM_FullyNumerical_CosmoUtil::FDM_FullyNumerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in) : CosmoUtil(fdm_masses[fdm_mass_id], eta, eta_in), g(new G_spline(eta_in, fdm_mass_id)) {}
  double FDM_FullyNumerical_CosmoUtil::D              (double k, double eta)                     const {return g->dp                  (k, eta);}          
  double FDM_FullyNumerical_CosmoUtil::greens         (double k, double s, double eta)           const {return g->propagator         (k, s, eta);}
  double FDM_FullyNumerical_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return g->d_s_propagator     (k, s, eta);}
  double FDM_FullyNumerical_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return g->d_eta_propagator   (k, s, eta);}
  double FDM_FullyNumerical_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return g->d_s_eta_propagator (k, s, eta);}
  double FDM_FullyNumerical_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return g->f_i                (k, s, eta, i);}


  FDM_Fit_CosmoUtil::FDM_Fit_CosmoUtil(int fdm_mass_id, double eta, double eta_in) : CosmoUtil(fdm_masses[fdm_mass_id], eta, eta_in), d(new D_hybrid(fdm_mass_id)) {}
  double FDM_Fit_CosmoUtil::D              (double k, double eta)                     const {return (*d)(k, eta);} 
  double FDM_Fit_CosmoUtil::greens         (double k, double s, double eta)           const {return greens_analytical         (k, s, eta, m)   ;}
  double FDM_Fit_CosmoUtil::d_s_greens     (double k, double s, double eta)           const {return d_s_greens_analytical     (k, s, eta, m)   ;}
  double FDM_Fit_CosmoUtil::d_eta_greens   (double k, double s, double eta)           const {return d_eta_greens_analytical   (k, s, eta, m)   ;}
  double FDM_Fit_CosmoUtil::d_s_eta_greens (double k, double s, double eta)           const {return d_s_eta_greens_analytical (k, s, eta, m)   ;}
  double FDM_Fit_CosmoUtil::f_i            (double k, double s, double eta, size_t i) const {return f_i_analytical            (k, s, eta, m, i);}


  // FDM mode coupling function to linear order
  // Involves contributions from derivatives of density and velocity field
  // as well as linear contribution from quantum pressure
  double gamma_iab(const vec &k, const vec &k1, const vec &k2, double eta, double mass, size_t i, size_t a, size_t b)
  {
    // Gamma_0
    if (i == 0)
    {

      if ((a == 0) && (b == 1))
      {
        double k2s = dot(k2, k2);
        if (k2s < verysmalleps)
        {
          return 0;
        }
        return -dot(k, k2) / (2 * k2s);
      }

      if ((a == 1) && (b == 0))
      {
        double k1s = dot(k1, k1);
        if (k1s < verysmalleps)
        {
          return 0;
        }
        return -dot(k, k1) / (2 * k1s);
      }

      if ((a == 0) && (b == 0))
      {
        return 0;
      }

      if ((a == 1) && (b == 1))
      {
        return 0;
      }
    }

    else if (i == 1)
    {
      if ((a == 0) && (b == 0))
      {

        double ks = dot(k, k);
        if (ks < verysmalleps)
        {
          return 0;
        }

        double bk = b_f(k.norm2(), mass);
        double p1 = pow(bk, 2) / (4 * pow(eta, 4));
        double p2 = 1 + (dot(k1, k1) + dot(k2, k2)) / ks;
        return -p1 * p2;
      }

      if ((a == 1) && (b == 1))
      {
        double k1s = dot(k1, k1);
        double k2s = dot(k2, k2);
        if ((k1s < verysmalleps) || (k2s < verysmalleps))
        {
          return 0;
        }
        return -dot(k, k) * dot(k1, k2) / (2 * k1s * k2s);
      }
      if ((a == 0) && (b == 1))
      {
        return 0;
      }

      if ((a == 1) && (b == 0))
      {
        return 0;
      }
    }
    else
    {
      throw std::invalid_argument("Argument a, b in gamma_iab is invalid.");
    }

    throw std::invalid_argument("Argument i in gamma_iab is invalid.");
  }

  double theta2111(const vec &k, const vec &k1, const vec &k2, const vec &k3, double s, double eta, double m)
  {
    double ks = dot(k, k);
    if (ks < verysmalleps)
    {
      return 0;
    }
    double bk = b_f(k.norm2(), m);
    return pow(bk, 2) / pow(eta, 4) / 8. * (1. + (dot(k1, k1) + dot(k2, k2) + dot(k3, k3)) / ks + 1 / 3. * (ssum(k1, k2) + ssum(k2, k3) + ssum(k1, k3)) / ks);
  }

  double xi2111(const vec &k, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s, double eta, double m)
  {
    double ks = dot(k, k);
    if (ks < verysmalleps)
    {
      return 0;
    }
    double bk = b_f(k.norm2(), m);
    return -pow(bk, 2) / pow(eta, 4) * 3. / 32. * (1. + 2. / 3 * (dot(k1, k1) + dot(k2, k2) + dot(k3, k3) + dot(k4, k4)) / ks + 1. / 3 * (ssum(k1, k2) + ssum(k1, k3) + ssum(k1, k4) + ssum(k3, k2) + ssum(k2, k4) + ssum(k4, k3)) / ks);
  }

  double W_coupling(const vec &k, const vec &k1, const vec &k2, double s, double eta, const CosmoUtil &cu, size_t i, size_t a, size_t b)
  {
    double k_norm = k.norm2();

    // When deriving W_c, we assumed k1, k2 and k to be != 0
    // If they are zero, we need to explicitly return zero or will obtain divisions by zero
    if (k_norm < verysmalleps || k1.norm2() < verysmalleps || k2.norm2() < verysmalleps)
      return 0;

    if (i == 0)
    {
      double dg = cu.d_s_greens(k_norm, s, eta);
      double g = cu.greens(k_norm, s, eta);

      double ga1 = gamma_iab(k, k1, k2, s, cu.m, 0, a, b);
      double ga2 = gamma_iab(k, k1, k2, s, cu.m, 1, a, b);
      return (-ga1 * dg - ga2 * g + 2 / s * ga1 * g);
    }

    if (i == 1)
    {
      double dg = cu.d_eta_greens(k_norm, s, eta);
      double ddg = cu.d_s_eta_greens(k_norm, s, eta);

      double ga1 = gamma_iab(k, k1, k2, s, cu.m, 0, a, b);
      double ga2 = gamma_iab(k, k1, k2, s, cu.m, 1, a, b);
      return (ga1 * ddg + ga2 * dg - 2 / s * ga1 * dg);
    }

    throw std::invalid_argument("Argument i in W_coupling is invalid.");
  }

  double U_coupling(const vec &k, const vec &k1, const vec &k2, const vec &k3, double s, double eta, const CosmoUtil &cu, size_t i)
  {
    if (i == 0)
      return -theta2111(k, k1, k2, k3, s, eta, cu.m) * cu.greens(k.norm2(), s, eta);

    if (i == 1)
      return theta2111(k, k1, k2, k3, s, eta, cu.m) * cu.d_eta_greens(k.norm2(), s, eta);

    throw std::invalid_argument("Argument i in U_coupling is invalid.");
  }

  double V_coupling(const vec &k, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s, double eta, const CosmoUtil &cu, size_t i)
  {
    if (i == 0)
      return -xi2111(k, k1, k2, k3, k4, s, eta, cu.m) * cu.greens(k.norm2(), s, eta);

    if (i == 1)
      return xi2111(k, k1, k2, k3, k4, s, eta, cu.m) * cu.d_eta_greens(k.norm2(), s, eta);

    throw std::invalid_argument("Argument i in V_coupling is invalid.");
  }

  double F2(const vec &k1, const vec &k2, double s, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t a = 0; a < 2; ++a)
    {
      for (size_t b = 0; b < 2; ++b)
      {
        res += W_coupling(k1 + k2, k1, k2, s, cu.eta, cu, 0, a, b) * cu.f_i(k1.norm2(), s, cu.eta, a) * cu.f_i(k2.norm2(), s, cu.eta, b);
      }
    }

    return res;
  }

  double J3(const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double eta, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t b = 0; b < 2; ++b)
    {
      for (size_t c = 0; c < 2; ++c)
      {
        for (size_t d = 0; d < 2; ++d)
        {
          for (size_t e = 0; e < 2; ++e)
          {
            res += 2 * W_coupling(k1 + k2 + k3, k1, k2 + k3, s1, eta, cu, 0, b, c) * cu.f_i(k1.norm2(), s1, eta, b) * W_coupling(k2 + k3, k2, k3, s2, s1, cu, c, d, e) * cu.f_i(k2.norm2(), s2, eta, d) * cu.f_i(k3.norm2(), s2, eta, e);
          }
        }
      }
    }

    return res;
  }

  double I3(const vec &k1, const vec &k2, const vec &k3, double s, double eta, const CosmoUtil &cu)
  {
    return U_coupling(k1 + k2 + k3, k1, k2, k3, s, eta, cu, 0) * cu.f_i(k1.norm2(), s, eta, 0) * cu.f_i(k2.norm2(), s, eta, 0) * cu.f_i(k3.norm2(), s, eta, 0);
  }

  double F3(const vec &k1, const vec &k2, const vec &k3, double s1, double s2, const CosmoUtil &cu)
  {
    double res = 0;
    double interval = cu.eta - cu.eta_in;

    if (s2 < s1)
      res += 1. / 3 * (J3(k1, k2, k3, s1, s2, cu.eta, cu) + J3(k2, k1, k3, s1, s2, cu.eta, cu) + J3(k3, k2, k1, s1, s2, cu.eta, cu));

    res += I3(k1, k2, k3, s1, cu.eta, cu) / interval; // F3 function of s1, s2, but MC-integration is in 3D, divide by 1 integration interval in order to not have to integrate I3 separately

    return res;
  }

  double W4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double eta, const CosmoUtil &cu)
  {
    return V_coupling(k1 + k2 + k3 + k4, k1, k2, k3, k4, s1, eta, cu, 0) * cu.f_i(k1.norm2(), s1, eta, 0) * cu.f_i(k2.norm2(), s1, eta, 0) * cu.f_i(k3.norm2(), s1, eta, 0) * cu.f_i(k4.norm2(), s1, eta, 0);
  }

  double I4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t b = 0; b < 2; ++b)
    {
      for (size_t c = 0; c < 2; ++c)
      {
        for (size_t d = 0; d < 2; ++d)
        {
          for (size_t e = 0; e < 2; ++e)
          {
            for (size_t f = 0; f < 2; ++f)
            {
              for (size_t g = 0; g < 2; ++g)
              {
                res += W_coupling(k1 + k2 + k3 + k4, k1 + k2, k3 + k4, s1, eta, cu, 0, b, c) *
                       W_coupling(k1 + k2, k1, k2, s2, s1, cu, b, d, e) *
                       W_coupling(k3 + k4, k3, k4, s3, s1, cu, c, f, g) *
                       cu.f_i(k1.norm2(), s2, eta, d) *
                       cu.f_i(k2.norm2(), s2, eta, e) *
                       cu.f_i(k3.norm2(), s3, eta, f) *
                       cu.f_i(k4.norm2(), s3, eta, g);
              }
            }
          }
        }
      }
    }

    return res;
  }

  double I4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu)
  {
    return 1. / 3 * (I4(k1, k2, k3, k4, s1, s2, s3, eta, cu) + I4(k1, k3, k2, k4, s1, s2, s3, eta, cu) + I4(k1, k4, k3, k2, s1, s2, s3, eta, cu));
  }

  double J4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t b = 0; b < 2; ++b)
    {
      for (size_t c = 0; c < 2; ++c)
      {
        for (size_t d = 0; d < 2; ++d)
        {
          for (size_t e = 0; e < 2; ++e)
          {
            for (size_t f = 0; f < 2; ++f)
            {
              for (size_t g = 0; g < 2; ++g)
              {
                res += W_coupling(k1 + k2 + k3 + k4, k2 + k3 + k4, k1, s1, eta, cu, 0, b, c) *
                       W_coupling(k2 + k3 + k4, k3 + k4, k2, s2, s1, cu, b, d, e) *
                       W_coupling(k3 + k4, k3, k4, s3, s2, cu, d, f, g) *
                       cu.f_i(k1.norm2(), s1, eta, c) *
                       cu.f_i(k2.norm2(), s2, eta, e) *
                       cu.f_i(k3.norm2(), s3, eta, f) *
                       cu.f_i(k4.norm2(), s3, eta, g);
              }
            }
          }
        }
      }
    }

    return 4 * res;
  }

  double J4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu)
  {
    return 1. / 12 * (J4(k1, k2, k3, k4, s1, s2, s3, eta, cu) + J4(k1, k3, k2, k4, s1, s2, s3, eta, cu) + J4(k1, k4, k3, k2, s1, s2, s3, eta, cu) + J4(k2, k3, k1, k4, s1, s2, s3, eta, cu) + J4(k2, k4, k1, k3, s1, s2, s3, eta, cu) + J4(k3, k4, k1, k2, s1, s2, s3, eta, cu) + J4(k2, k1, k3, k4, s1, s2, s3, eta, cu) + J4(k3, k1, k2, k4, s1, s2, s3, eta, cu) + J4(k4, k1, k3, k2, s1, s2, s3, eta, cu) + J4(k3, k2, k1, k4, s1, s2, s3, eta, cu) + J4(k4, k2, k1, k3, s1, s2, s3, eta, cu) + J4(k4, k3, k1, k2, s1, s2, s3, eta, cu));
  }

  double K4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double eta, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t b = 0; b < 2; ++b)
    {
      for (size_t c = 0; c < 2; ++c)
      {
        res += W_coupling(k1 + k2 + k3 + k4, k2 + k3 + k4, k1, s1, eta, cu, 0, b, c) *
               U_coupling(k2 + k3 + k4, k2, k3, k4, s2, s1, cu, b) *
               cu.f_i(k1.norm2(), s1, eta, c) *
               cu.f_i(k2.norm2(), s2, eta, 0) *
               cu.f_i(k3.norm2(), s2, eta, 0) *
               cu.f_i(k4.norm2(), s2, eta, 0);
      }
    }

    return 2 * res;
  }

  double K4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double eta, const CosmoUtil &cu)
  {
    return 1. / 4 * (K4(k1, k2, k3, k4, s1, s2, eta, cu) + K4(k2, k1, k3, k4, s1, s2, eta, cu) + K4(k3, k2, k1, k4, s1, s2, eta, cu) + K4(k4, k2, k3, k1, s1, s2, eta, cu));
  }

  double H4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double eta, const CosmoUtil &cu)
  {
    double res = 0;

    for (size_t b = 0; b < 2; ++b)
    {
      for (size_t c = 0; c < 2; ++c)
      {
        res += U_coupling(k1 + k2 + k3 + k4, k3 + k4, k1, k2, s1, eta, cu, 0) *
               W_coupling(k3 + k4, k3, k4, s2, s1, cu, 0, b, c) *
               cu.f_i(k1.norm2(), s1, eta, 0) *
               cu.f_i(k2.norm2(), s1, eta, 0) *
               cu.f_i(k3.norm2(), s2, eta, b) *
               cu.f_i(k4.norm2(), s2, eta, c);
      }
    }

    return 3 * res;
  }

  double H4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double eta, const CosmoUtil &cu)
  {
    return 1. / 6 * (H4(k1, k2, k3, k4, s1, s2, eta, cu) + H4(k1, k3, k2, k4, s1, s2, eta, cu) + H4(k1, k4, k3, k2, s1, s2, eta, cu) + H4(k2, k3, k1, k4, s1, s2, eta, cu) + H4(k2, k4, k1, k3, s1, s2, eta, cu) + H4(k3, k4, k1, k2, s1, s2, eta, cu));
  }

  double F4(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    double res = 0;
    double interval = cu.eta - cu.eta_in;

    // Sum up contributions for coupling kernel F4
    // Again divide by integration interval:
    // W4 is function of s1 and cu.a, but MC integration is 4D
    // Divide by interval^2 in order to account for extra contribution from MC
    res += W4(k1, k2, k3, k4, s1, cu.eta, cu) / pow(interval, 2);

    if ((s3 < s2) && (s2 < s1))
      res += J4s(k1, k2, k3, k4, s1, s2, s3, cu.eta, cu);

    if (s2 < s1)
      res += H4s(k1, k2, k3, k4, s1, s2, cu.eta, cu) / interval;

    if ((s2 < s1) && (s3 < s1))
      res += I4s(k1, k2, k3, k4, s1, s2, s3, cu.eta, cu);

    if (s2 < s1)
      res += K4s(k1, k2, k3, k4, s1, s2, cu.eta, cu) / interval;

    return res;
  }

}

/**
 * MODE COUPLING END
 * **/

/***
 * SPECTRUM
 * **/

vec Spectrum::get_result(double k, const CosmoUtil &cu) const
{
  vec result(3);
  result[0] = this->operator()(k, cu);
  result[1] = 0;
  result[2] = 0;

  return result;
}

namespace CDM
{


  double ScaleFreeSpectrum::operator()(double k, const CosmoUtil &cu) const
  {
    return pow(k, n_) * pow(cu.D(k, cu.eta), 2.);
  }
}

namespace FDM
{
  double ScaleFreeSpectrum::operator()(double k, const CosmoUtil &cu) const
  {
    return pow(k, n_) * pow(cu.D(k, cu.eta), 2.);
  }
}
/***
 * SPECTRUM END
 * **/

/***
 * BISPECTRUM
 ****/

// From above eq. 177 in Bernardeau 2002
double ST(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
{
  double s1, s2, s3, p1, p2, p3;
  s1 = P(k1.norm2(), cu);
  s2 = P(k2.norm2(), cu);
  s3 = P(k3.norm2(), cu);
  p1 = s1 * s2;
  p2 = s2 * s3;
  p3 = s3 * s1;
  return p1 + p2 + p3;
}

// Sigma tree
double ST(const Spectrum &P, double k1, double k2, double k3, const CosmoUtil &cu)
{
  return P(k1, cu) * P(k2, cu) + P(k3, cu) * P(k2, cu) + P(k1, cu) * P(k3, cu);
}

// Sigma 1-loop
vec S1L(const Spectrum &P, const Spectrum &P1L, double k1, double k2, double k3, const CosmoUtil &cu)
{
  vec p1(P1L.get_result(k1, cu));
  vec p2(P1L.get_result(k2, cu));
  vec p3(P1L.get_result(k3, cu));

  double p01 = P(k1, cu);
  double p02 = P(k2, cu);
  double p03 = P(k3, cu);

  vec result(2);
  // Compute Sigma 1
  result[0] = p1[0] * p02 + p3[0] * p02 + p1[0] * p03 + p01 * p2[0] + p03 * p2[0] + p01 * p3[0];
  // Compute error of Sigma 1
  result[1] = sqrt(pow(p1[1] * p02, 2) + pow(p3[1] * p02, 2) + pow(p1[1] * p03, 2) + pow(p01 * p2[1], 2) + pow(p03 * p2[1], 2) + pow(p01 * p3[1], 2));
  return result;
}

// From above eq. 177 in Bernardeau 2002
// P1L is nonlinear power spectrum with one-loop corrections
double B1LR(double b, double b1, double s, double s1)
{
  return (b1 - s1 * b) / s;
}

double B1LR_err(double b, double b1_err, double s, double s1_err)
{
  return sqrt(b1_err * b1_err + s1_err * s1_err * b * b) / s;
}

double BFR(double b, double b1, double s, double s1)
{
  return (b + b1) / (s + s1);
}

double BFR_err(double b, double b1, double b1_err, double s, double s1, double s1_err)
{
  return sqrt(pow((b1_err) / (b + b1), 2) + pow((s1_err) / (s + s1), 2)) * BFR(b, b1, s, s1);
}

std::array<vec, 3> generateTriangle(double theta, double r1, double r2)
{

  vec k1(3), k2(3), k3(3);

  k1[0] = 0;
  k1[1] = 0;
  k1[2] = r1;

  k2[0] = r2 * sin(theta);
  k2[1] = 0;
  k2[2] = r2 * cos(theta);

  k3 = -k1 - k2;

  std::array<vec, 3> triangle = {k1, k2, k3};
  return triangle;
}

std::array<vec, 4> generateRectangle(bool isEquilateral, double l1, double l2, double l4, double phi, double psi)
{

  vec k1(3), k2(3), k3(3), k4(3);

  if (isEquilateral)
  {
    l2 = l1;
    l4 = l1;
    phi = -M_PI / 2;
    psi = -M_PI / 2;
  }

  k1[0] = l1;
  k1[1] = 0;
  k1[2] = 0;

  k2[0] = l2 * cos(M_PI - psi);
  k2[1] = l2 * sin(M_PI - psi);
  k2[2] = 0;

  k4[0] = -l4 * cos(phi);
  k4[1] = -l4 * sin(phi);
  k4[2] = 0;

  k3 = -(k1 + k2 + k4);

  std::array<vec, 4> rect = {k1, k2, k3, k4};
  return rect;
}

namespace CDM
{
  // Cartesian bispectrum at tree level
  double BT(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    double s1, s2, s3, p1, p2, p3;
    s1 = P(k1.norm2(), cu);
    s2 = P(k2.norm2(), cu);
    s3 = P(k3.norm2(), cu);
    p1 = 2 * F2s_td(k1, k2, cu) * s1 * s2;
    p2 = 2 * F2s_td(k2, k3, cu) * s2 * s3;
    p3 = 2 * F2s_td(k3, k1, cu) * s3 * s1;
    return p1 + p2 + p3;
  }

  double TreeBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    // std::cout<<a_from_eta(cu.eta)<<"    "<<k1.norm2()<<"    "<<k2.norm2()<<"    "<<k3.norm2()<<std::endl;
    return BT(P, k1, k2, k3, cu);
  }

  vec TreeBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    vec result(3);
    result[0] = BT(P, k1, k2, k3, cu);
    result[1] = 0;
    result[2] = 0;
    return result;
  }

  double BR(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return BT(P, k1, k2, k3, cu) / ST(P, k1, k2, k3, cu);
  }

  double TreeTrispectrum::t2(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    
    double k14, k13, k23, k24, c1, c2;
    k14 = (k1 + k4).norm2();
    k13 = (k1 + k3).norm2();
    k23 = (k2 + k3).norm2();
    k24 = (k2 + k4).norm2();
    c1 = 0;
    c2 = 0;

    if ((k13 > verysmalleps) && (k24 > verysmalleps)) {
      c1 = F2s_td(k1 + k3, -k3, cu) * F2s_td(k2 + k4, -k4, cu) * P(k13, cu);
    }

    if ((k14 > verysmalleps) && (k23 > verysmalleps)) {
      c2 = F2s_td(k1 + k4, -k4, cu) * F2s_td(k2 + k3, -k3, cu) * P(k14, cu);
    }

    return 4 * P(k3.norm2(), cu) * P(k4.norm2(), cu) * (c1 + c2);
  }

  double TreeTrispectrum::t3(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    double p_1, p_2, p_3, f3;
    p_1 = P(k1.norm2(), cu);
    p_2 = P(k2.norm2(), cu);
    p_3 = P(k3.norm2(), cu);
    f3 = F3s_td(k1, k2, k3, cu);

    return 6 * p_1 * p_2 * p_3 * f3;
  }

  double TreeTrispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    double r1, r2;
    // 12 pair-wise permutations
    r1 = (t2(k1, k2, k3, k4, cu) +
          t2(k1, k3, k2, k4, cu) + // 1 <-> 3
          t2(k1, k4, k2, k3, cu) + // 2 <-> 1 + 3 <-> 4
          t2(k2, k3, k1, k4, cu) + // 2 <-> 4
          t2(k2, k4, k1, k3, cu) + // 2 <-> 4
          t2(k3, k4, k1, k2, cu));
    // 4 cyclic permutations
    r2 = (t3(k1, k2, k3, k4, cu) +
          t3(k4, k1, k2, k3, cu) +
          t3(k3, k4, k1, k2, cu) +
          t3(k2, k3, k4, k1, cu));
    return r1 + r2;
  }

  vec TreeTrispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    vec result(3);
    result[0] = this->operator()(k1, k2, k3, k4, cu);
    result[1] = 0;
    result[2] = 0;
    return result;
  }

  double b222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const CosmoUtil &cu)
  {
    return 8 * F2s(-q, q + k1) * F2s(q + k1, -q + k2) * F2s(k2 - q, q) * P(q.norm2(), cu) * P((q + k1).norm2(), cu) * P((q - k2).norm2(), cu);
  }

  double bt222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const CosmoUtil &cu)
  {
    double res = 0;
    if (((k1 + q).norm2() > q.norm2()) && ((k2 - q).norm2() > q.norm2()))
      res += b222(P, q, k1, k2, cu);
    if (((k1 - q).norm2() > q.norm2()) && ((k2 + q).norm2() > q.norm2()))
      res += b222(P, -q, k1, k2, cu);
    return res * 0.5;
  }

  double B222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return (
        bt222(P, q, k1, k2, cu) +
        bt222(P, q, k3, k2, cu) +
        bt222(P, q, k1, k3, cu));
  }

  double b3211(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return 6 * P(k3.norm2(), cu) * F3s(-q, q - k2, -k3) * F2s(q, k2 - q) * P(q.norm2(), cu) * P((q - k2).norm2(), cu);
  }

  double bt3211(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    double res = 0;
    if ((k2 - q).norm2() > q.norm2())
      res += b3211(P, q, k2, k3, cu);
    if ((k2 + q).norm2() > q.norm2())
      res += b3211(P, -q, k2, k3, cu);
    return res;
  }

  double B3211(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return (
        bt3211(P, q, k2, k3, cu) +
        bt3211(P, q, k3, k2, cu) +
        bt3211(P, q, k1, k3, cu) +
        bt3211(P, q, k3, k1, cu) +
        bt3211(P, q, k1, k2, cu) +
        bt3211(P, q, k2, k1, cu));
  }
  double b3212(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return 6 * F2s(k2, k3) * P(k2.norm2(), cu) * P(k3.norm2(), cu) * F3s(k3, q, -q) * P(q.norm2(), cu);
  }
  double B3212(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return (
        b3212(P, q, k2, k3, cu) +
        b3212(P, q, k3, k2, cu) +
        b3212(P, q, k1, k3, cu) +
        b3212(P, q, k3, k1, cu) +
        b3212(P, q, k1, k2, cu) +
        b3212(P, q, k2, k1, cu));
  }

  double b411(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return 12 * P(k2.norm2(), cu) * P(k3.norm2(), cu) * F4s(q, -q, -k2, -k3) * P(q.norm2(), cu);
  }
  double B411(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return (
        b411(P, q, k2, k3, cu) +
        b411(P, q, k1, k2, cu) +
        b411(P, q, k3, k1, cu));
  }

  double B1L(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return (
        B222(P, q, k1, k2, k3, cu) +
        B3211(P, q, k1, k2, k3, cu) +
        B3212(P, q, k1, k2, k3, cu) +
        B411(P, q, k1, k2, k3, cu));
  }
}

namespace FDM
{

  double F2i(const vec &k1, const vec &k2, const CosmoUtil &cu)
  {
    double result, error;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(INTEGRATION_WORKSPACE_SIZE);

    // Define integrand
    auto integrand = [](double s, void *params)
    {
      F2_C *f2_c = (struct F2_C *)params;
      return F2(f2_c->k1, f2_c->k2, s, f2_c->cu);
    };

    gsl_function F;
    F.function = integrand;

    F2_C f2_c(k1, k2, cu);
    F.params = (void *)&f2_c;

    // Perform integration
    gsl_integration_qags(&F, cu.eta_in, cu.eta, F2_INTEGRATION_ABSERR, F2_INTEGRATION_RELERR, F2_SUBDIVISIONS,
                         w, &result, &error);

    gsl_integration_workspace_free(w);
    return result;
  }

  double F3i(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {

    double p, p_error, p_Q;
    F3_C ic(k1, k2, k3, cu);
    void *data = (void *)&ic;
    cuba_integrate(F3i_cuba, 2, data, p, p_error, p_Q, 4, 0, CUBA_ALGORITHMS::CUHRE);
    return p;
  }

  // Cartesian bispectrum at tree level
  double BT(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    double s1, s2, s3, p1, p2, p3;
    s1 = P(k1.norm2(), cu);
    s2 = P(k2.norm2(), cu);
    s3 = P(k3.norm2(), cu);
    p1 = 2 * F2i(k1, k2, cu) * s1 * s2;
    p2 = 2 * F2i(k2, k3, cu) * s2 * s3;
    p3 = 2 * F2i(k3, k1, cu) * s3 * s1;
    return p1 + p2 + p3;
  }

  double b0_integrand(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, double s, const CosmoUtil &cu)
  {
    double s1, s2, s3, p1, p2, p3;
    s1 = P(k1.norm2(), cu);
    s2 = P(k2.norm2(), cu);
    s3 = P(k3.norm2(), cu);
    p1 = 2 * F2(k1, k2, s, cu) * s1 * s2;
    p2 = 2 * F2(k2, k3, s, cu) * s2 * s3;
    p3 = 2 * F2(k3, k1, s, cu) * s3 * s1;
    return p1 + p2 + p3;
  }

  double TreeBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return BT(P, k1, k2, k3, cu);
  }

  vec TreeBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    vec result(3);

    result[0] = BT(P, k1, k2, k3, cu);
    result[1] = 0;
    result[2] = 0;
    return result;
  }

  double VegasTreeBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return this->get_result(k1, k2, k3, cu)[0];
  }

  vec VegasTreeBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    vec result(3);

    double p, p_error, p_Q;
    B0_IC ic(k1, k2, k3, P, cu);
    void *data = (void *)&ic;
    cuba_integrate(B0i_cuba, 1, data, p, p_error, p_Q, 4, 0, CUBA_ALGORITHMS::VEGAS, 1e-3, 1e-36);

    result[0] = p;
    result[1] = p_error;
    result[2] = p_Q;
    return result;
  }

  double BR(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu)
  {
    return BT(P, k1, k2, k3, cu) / ST(P, k1, k2, k3, cu);
  }

  double t2(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, const CosmoUtil &cu)
  {
    double k14, k13, k23, k24, c1, c2;
    k14 = (k1 + k4).norm2();
    k13 = (k1 + k3).norm2();
    k23 = (k2 + k3).norm2();
    k24 = (k2 + k4).norm2();
    c1 = 0;
    c2 = 0;

    if ((k13 > verysmalleps) && (k24 > verysmalleps)) {
      c1 = F2(k1 + k3, -k3, s1, cu) * F2(k2 + k4, -k4, s2, cu) * P(k13, cu);
    }

    if ((k14 > verysmalleps) && (k23 > verysmalleps)) {
      c2 = F2(k1 + k4, -k4, s1, cu) * F2(k2 + k3, -k3, s2, cu) * P(k14, cu);
    }

    return 4 * P(k3.norm2(), cu) * P(k4.norm2(), cu) * (c1 + c2);
  }

  double t3(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, const CosmoUtil &cu)
  {
    double p_1, p_2, p_3, f3;
    p_1 = P(k1.norm2(), cu);
    p_2 = P(k2.norm2(), cu);
    p_3 = P(k3.norm2(), cu);
    f3 = F3(k1, k2, k3, s1, s2, cu);
    return 6 * p_1 * p_2 * p_3 * f3;
  }

  double t0_integrand(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, const CosmoUtil &cu)
  {
    double r1, r2;

    // 6 pair-wise permutations
    r1 = (t2(P, k1, k2, k3, k4, s1, s2, cu) +
          t2(P, k1, k3, k2, k4, s1, s2, cu) + 
          t2(P, k1, k4, k2, k3, s1, s2, cu) + 
          t2(P, k2, k3, k1, k4, s1, s2, cu) + 
          t2(P, k2, k4, k1, k3, s1, s2, cu) + 
          t2(P, k3, k4, k1, k2, s1, s2, cu));
    // 4 cyclic permutations
    r2 = (t3(P, k1, k2, k3, k4, s1, s2, cu) +
          t3(P, k4, k1, k2, k3, s1, s2, cu) +
          t3(P, k3, k4, k1, k2, s1, s2, cu) +
          t3(P, k2, k3, k4, k1, s1, s2, cu));

    return r1 + r2;
  }

  double T0i(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu)
  {

    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(INTEGRATION_WORKSPACE_SIZE);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(INTEGRATION_WORKSPACE_SIZE);

    // Outer integrand
    auto outer_integrand = [](double eta, void *params) -> double
    {
      // Inner integrand
      auto inner_integrand = [](double s, void *params) -> double
      {
        T0_IC *ic = (struct T0_IC *)params;
        return t0_integrand(ic->P, ic->k1, ic->k2, ic->k3, ic->k4, s, ic->eta, ic->cu);
      };

      double r1, r2, e1, e2;
      gsl_function F_inner;
      F_inner.function = inner_integrand;
      F_inner.params = params;
      T0_IC *ic = (struct T0_IC *)params;
      ic->eta = eta;

      gsl_integration_qags(&F_inner, ic->cu.eta_in, eta, T0_INTEGRATION_ABSERR, T0_INTEGRATION_RELERR, T0_SUBDIVISIONS, ic->w, &r1, &e1);
      gsl_integration_qags(&F_inner, eta, ic->cu.eta, T0_INTEGRATION_ABSERR, T0_INTEGRATION_RELERR, T0_SUBDIVISIONS, ic->w, &r2, &e2);
      return r1 + r2;
    };

    double result, error;
    T0_IC ic(k1, k2, k3, k4, P, cu, w2);

    gsl_function F_outer;
    F_outer.function = outer_integrand;
    F_outer.params = (void *)&ic;

    // Perform integration
    gsl_integration_qags(&F_outer, cu.eta_in, cu.eta, T0_INTEGRATION_ABSERR, T0_INTEGRATION_RELERR, T0_SUBDIVISIONS,
                         w1, &result, &error);

    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);

    return result;
  }

  double TreeTrispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    return T0i(P, k1, k2, k3, k4, cu);
  }

  vec TreeTrispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    vec result(3);

    result[0] = this->operator()(k1, k2, k3, k4, cu);
    result[1] = 0;
    result[2] = 0;
    return result;
  }

  double TreeTrispectrumCuba::operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    return this->get_result(k1, k2, k3, k4, cu)[0];
  }

  vec TreeTrispectrumCuba::get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const
  {
    vec result(3);

    double p, p_error, p_Q;
    T0_IC ic(k1, k2, k3, k4, P, cu, NULL);
    void *data = (void *)&ic;
    cuba_integrate(T0i_cuba, 2, data, p, p_error, p_Q, 4, 2, CUBA_ALGORITHMS::CUHRE, 1e-2, 0);

    result[0] = p;
    result[1] = p_error;
    result[2] = p_Q;
    return result;
  }

  double b222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return 8 * F2(-q, q + k1, s1, cu) * F2(q + k1, -q + k2, s2, cu) * F2(k2 - q, q, s3, cu) * P(q.norm2(), cu) * P((q + k1).norm2(), cu) * P((q - k2).norm2(), cu);
  }

  double bt222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    double res = 0;
    if (((k1 + q).norm2() > q.norm2()) && ((k2 - q).norm2() > q.norm2()))
      res += b222(P, q, k1, k2, s1, s2, s3, cu);
    if (((k1 - q).norm2() > q.norm2()) && ((k2 + q).norm2() > q.norm2()))
      res += b222(P, -q, k1, k2, s1, s2, s3, cu);
    return res * 0.5;
  }

  double B222(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return (
        bt222(P, q, k1, k2, s1, s2, s3, cu) +
        bt222(P, q, k3, k2, s1, s2, s3, cu) +
        bt222(P, q, k1, k3, s1, s2, s3, cu));
  }

  double b3211(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return 6 * P(k3.norm2(), cu) * F3(-q, q - k2, -k3, s1, s2, cu) * F2(q, k2 - q, s3, cu) * P(q.norm2(), cu) * P((q - k2).norm2(), cu);
  }

  double bt3211(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    double res = 0;
    if ((k2 - q).norm2() > q.norm2())
      res += b3211(P, q, k2, k3, s1, s2, s3, cu);
    if ((k2 + q).norm2() > q.norm2())
      res += b3211(P, -q, k2, k3, s1, s2, s3, cu);
    return res;
  }

  double B3211(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return (
        bt3211(P, q, k2, k3, s1, s2, s3, cu) +
        bt3211(P, q, k3, k2, s1, s2, s3, cu) +
        bt3211(P, q, k1, k3, s1, s2, s3, cu) +
        bt3211(P, q, k3, k1, s1, s2, s3, cu) +
        bt3211(P, q, k1, k2, s1, s2, s3, cu) +
        bt3211(P, q, k2, k1, s1, s2, s3, cu));
  }
  double b3212(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return 6 * F2(k2, k3, s1, cu) * P(k2.norm2(), cu) * P(k3.norm2(), cu) * F3(k3, q, -q, s2, s3, cu) * P(q.norm2(), cu);
  }
  double B3212(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return (
        b3212(P, q, k2, k3, s1, s2, s3, cu) +
        b3212(P, q, k3, k2, s1, s2, s3, cu) +
        b3212(P, q, k1, k3, s1, s2, s3, cu) +
        b3212(P, q, k3, k1, s1, s2, s3, cu) +
        b3212(P, q, k1, k2, s1, s2, s3, cu) +
        b3212(P, q, k2, k1, s1, s2, s3, cu));
  }

  double b411(const Spectrum &P, const vec &q, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return 12 * P(k2.norm2(), cu) * P(k3.norm2(), cu) * F4(q, -q, -k2, -k3, s1, s2, s3, cu) * P(q.norm2(), cu);
  }
  double B411(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return (
        b411(P, q, k2, k3, s1, s2, s3, cu) +
        b411(P, q, k1, k2, s1, s2, s3, cu) +
        b411(P, q, k3, k1, s1, s2, s3, cu));
  }

  double B1L(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu)
  {
    return (
        B222(P, q, k1, k2, k3, s1, s2, s3, cu) +
        B3211(P, q, k1, k2, k3, s1, s2, s3, cu) +
        B3212(P, q, k1, k2, k3, s1, s2, s3, cu) +
        B411(P, q, k1, k2, k3, s1, s2, s3, cu));
  }

}

/***
 * BISPECTRUM END
 * ****/

P1L_IC::P1L_IC(double k_norm, const Spectrum &P, const CosmoUtil &cu, double r_low, double r_high) : k_norm(k_norm), P(P), cu(cu)
{
  ib = {0, 2 * M_PI, -1, 1, r_low, r_high, cu.eta_in, cu.eta};
}

B1L_IC::B1L_IC(vec k1, vec k2, vec k3, const Spectrum &P, const CosmoUtil &cu, double r_low, double r_high) : k1(k1), k2(k2), k3(k3), P(P), cu(cu)
{
  ib = {0, 2 * M_PI, -1, 1, r_low, r_high, cu.eta_in, cu.eta};
}

namespace CDM
{
  int P1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    P1L_IC *c = (struct P1L_IC *)userdata;

    // To -1, 1
    double cosine_theta = xx[0] * (c->ib[3] - c->ib[2]) + c->ib[2];
    // To 0 to infinity
    double temp         = xx[1] * (c->ib[5] - c->ib[4]) + c->ib[4];
    double r            = temp / (1 - temp);

    // Jacobi determinant
    double det = (c->ib[5] - c->ib[4]) * (c->ib[3] - c->ib[2]) / pow(1. - temp, 2);

    vec k(3), k1(3), k2(3), k3(3);

    double kn = c->k_norm;
    double k1n = r;

    k[0] = kn * sqrt(1 - cosine_theta * cosine_theta);
    k[1] = 0;
    k[2] = kn * cosine_theta;

    k1[0] = 0;
    k1[1] = 0;
    k1[2] = k1n;

    k2 = k - k1;
    k3 = k + k1;

    // Lengths of remaining vectors
    double k2n = k2.norm2();
    double k3n = k3.norm2();

    double res = 6 * c->P(kn, c->cu) * c->P(k1n, c->cu) * F3s_td(k, k1, -k1, c->cu);

    if (k1n < k2n)
      res += 2 * c->P(k1n, c->cu) * c->P(k2n, c->cu) * F2s_td(k1, k2, c->cu) * F2s_td(k1, k2, c->cu);
    if (k1n < k3n)
      res += 2 * c->P(k1n, c->cu) * c->P(k3n, c->cu) * F2s_td(-k1, k3, c->cu) * F2s_td(-k1, k3, c->cu);

    // Jacobi-determinant, Fourier convention and phi-integration
    res *= det * k1n * k1n * 1 / (4 * M_PI * M_PI);

    ff[0] = res;

    return 0;
  }

  int B1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    B1L_IC *c = (struct B1L_IC *)userdata;

    // To 0, 2pi
    double phi = xx[0] * (c->ib[1] - c->ib[0]) + c->ib[0];
    // To -1, 1
    double cosine_theta = xx[1] * (c->ib[3] - c->ib[2]) + c->ib[2];
    // To 1/5 to 16/17
    double temp = xx[2] * (c->ib[5] - c->ib[4]) + c->ib[4];
    // To 1/4 to 16
    double r = temp / (1 - temp);

    // Jacobi determinant
    double det = (c->ib[1] - c->ib[0]) * (c->ib[3] - c->ib[2]) * (c->ib[5] - c->ib[4]) / pow(1 - temp, 2);

    vec q(3);
    q[0] = r * sqrt(1 - pow(cosine_theta, 2)) * cos(phi);
    q[1] = r * sqrt(1 - pow(cosine_theta, 2)) * sin(phi);
    q[2] = r * cosine_theta;

    // Jacobi-determinant, Fourier convention and phi-integration
    ff[0] = det * r * r / pow(2 * M_PI, 3) * B1L(c->P, q, c->k1, c->k2, c->k3, c->cu);

    return 0;
  }
}

namespace FDM
{

  int F2i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    F2_C *c = (struct F2_C *)userdata;

    double eta_f = c->cu.eta;
    double eta_i = c->cu.eta_in;

    // To a_i, a
    double s1 = xx[0] * (eta_f - eta_i) + eta_i;

    // Jacobi determinant
    double det = (eta_f - eta_i);

    ff[0] = F2(c->k1, c->k2, s1, c->cu) * det;

    return 0;
  }

  int F3i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    F3_C *c = (struct F3_C *)userdata;

    double eta_f = c->cu.eta;
    double eta_i = c->cu.eta_in;

    // To a_i, a
    double s1 = xx[0] * (eta_f - eta_i) + eta_i;
    // To a_i, a
    double s2 = xx[1] * (eta_f - eta_i) + eta_i;

    // Jacobi determinant
    double det = (eta_f - eta_i) * (eta_f - eta_i);

    ff[0] = F3(c->k1, c->k2, c->k3, s1, s2, c->cu) * det;

    return 0;
  }

  int B0i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    B0_IC *c = (struct B0_IC *)userdata;

    // CAUTION: ACTUALLY THE ANALYTICALLY CORRECT EXPRESSION WOULD BE eta_i = cu.eta_in, BUT FOR AGREEMENT WITH THE CDM F_i, WE NEED 0, THIS CAUSES AROUND 1% error
    // c->cu.eta_in = smalleps;

    double eta_f = c->cu.eta;
    double eta_i = c->cu.eta_in;

    // To a_i, a
    double s1 = xx[0] * (eta_f - eta_i) + eta_i;

    // Jacobi determinant
    double det = (eta_f - eta_i);

    ff[0] = det * b0_integrand(c->P, c->k1, c->k2, c->k3, s1, c->cu);

    return 0;
  }

  int T0i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    T0_IC *c = (struct T0_IC *)userdata;

    double eta_f = c->cu.eta;
    double eta_i = c->cu.eta_in;

    // To a_i, a
    double s1 = xx[0] * (eta_f - eta_i) + eta_i;
    // To a_i, a
    double s2 = xx[1] * (eta_f - eta_i) + eta_i;

    // Jacobi determinant
    double det = (eta_f - eta_i) * (eta_f - eta_i);

    ff[0] = det * t0_integrand(c->P, c->k1, c->k2, c->k3, c->k4, s1, s2, c->cu);

    return 0;
  }

  int P1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    P1L_IC *c = (struct P1L_IC *)userdata;

    // To -1, 1
    double cosine_theta = xx[0] * (c->ib[3] - c->ib[2]) + c->ib[2];
    // To 1/5 to 16/17
    double temp = xx[1] * (c->ib[5] - c->ib[4]) + c->ib[4];

    double r = temp / (1 - temp);
    // To a_i, a
    double s1 = xx[2] * (c->ib[7] - c->ib[6]) + c->ib[6];
    // To a_i, a
    double s2 = xx[3] * (c->ib[7] - c->ib[6]) + c->ib[6];

    // Jacobi determinant
    double det = (c->ib[5] - c->ib[4]) * (c->ib[3] - c->ib[2]) * (c->ib[7] - c->ib[6]) / (1. - temp) * (c->ib[7] - c->ib[6]) / (1. - temp);

    vec k(3), k1(3), k2(3), k3(3);

    double kn = c->k_norm;
    double k1n = r;

    k[0] = kn * sqrt(1 - cosine_theta * cosine_theta);
    k[1] = 0;
    k[2] = kn * cosine_theta;

    k1[0] = 0;
    k1[1] = 0;
    k1[2] = k1n;

    k2 = k - k1;
    k3 = k + k1;

    // Lengths of remaining vectors
    double k2n = k2.norm2();
    double k3n = k3.norm2();


    double res = 6 * c->P(kn, c->cu) * c->P(k1n, c->cu) * F3(k, k1, -k1, s1, s2, c->cu);

    if (k1n < k2n) {
      if ((k1n > verysmalleps) && (k2n > verysmalleps)) {
      res += 2 * c->P(k1n, c->cu) * c->P(k2n, c->cu) * F2(k1, k2, s1, c->cu) * F2(k1, k2, s2, c->cu);
      }
    }
    if (k1n < k3n) {
      if ((k1n > verysmalleps) && (k3n > verysmalleps)) {
      res += 2 * c->P(k1n, c->cu) * c->P(k3n, c->cu) * F2(-k1, k3, s1, c->cu) * F2(-k1, k3, s2, c->cu);
      }
    }

    // Jacobi-determinant, Fourier convention and phi-integration and variable change from eta to a
    res *= det * k1n * k1n * 1 / (4 * M_PI * M_PI);

    ff[0] = res;

    return 0;
  }

  int B1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata)
  {

    B1L_IC *c = (struct B1L_IC *)userdata;

    // To 0, 2pi
    double phi = xx[0] * (c->ib[1] - c->ib[0]) + c->ib[0];
    // To -1, 1
    double cosine_theta = xx[1] * (c->ib[3] - c->ib[2]) + c->ib[2];
    // To 1/5 to 16/17
    double temp = xx[2] * (c->ib[5] - c->ib[4]) + c->ib[4];
    // To 1/4 to 16
    double r = temp / (1 - temp);
    // To a_i, a
    double s1 = xx[3] * (c->ib[7] - c->ib[6]) + c->ib[6];
    // To a_i, a
    double s2 = xx[4] * (c->ib[7] - c->ib[6]) + c->ib[6];
    // To a_i, a
    double s3 = xx[5] * (c->ib[7] - c->ib[6]) + c->ib[6];

    // Jacobi determinant
    double det = (c->ib[1] - c->ib[0]) * (c->ib[3] - c->ib[2]) * (c->ib[5] - c->ib[4]) / pow(1 - temp, 2) * pow(c->ib[7] - c->ib[6], 3);

    vec q(3);
    q[0] = r * sqrt(1 - pow(cosine_theta, 2)) * cos(phi);
    q[1] = r * sqrt(1 - pow(cosine_theta, 2)) * sin(phi);
    q[2] = r * cosine_theta;

    // Jacobi-determinant, Fourier convention and phi-integration
    ff[0] = det * r * r / pow(2 * M_PI, 3) * B1L(c->P, q, c->k1, c->k2, c->k3, s1, s2, s3, c->cu);

    return 0;
  }

  double B1Li_test(double phi, double cosine_theta, double r, double s1, double s2, double s3, B1L_IC *c)
  {
    vec q(3);
    q[0] = r * sqrt(1 - pow(cosine_theta, 2)) * cos(phi);
    q[1] = r * sqrt(1 - pow(cosine_theta, 2)) * sin(phi);
    q[2] = r * cosine_theta;

    // Jacobi-determinant, Fourier convention and phi-integration
    return r * r / pow(2 * M_PI, 3) * B1L(c->P, q, c->k1, c->k2, c->k3, s1, s2, s3, c->cu);
  }
}

int cuba_integrate(integrand_t Integrand, int ndim, void *userdata, double &result, double &uncertainty, double &probability, int gridno, int VERBOSE, CUBA_ALGORITHMS alg, double relerr, double abserr, void *spin, size_t maxeval)
{
  int neval, fail, nregions;
  cubareal integral[NCOMP], error[NCOMP], prob[NCOMP];

  switch (alg)
  {
  case CUBA_ALGORITHMS::VEGAS:
    Vegas(ndim, NCOMP, Integrand, userdata, NVEC,
          relerr, abserr, VERBOSE, SEED,
          MINEVAL, maxeval, NSTART, NINCREASE, NBATCH,
          gridno, STATEFILE, spin,
          &neval, &fail, integral, error, prob);
    break;
  case CUBA_ALGORITHMS::SUAVE:
    Suave(ndim, NCOMP, Integrand, userdata, NVEC,
          relerr, abserr, VERBOSE | LAST, SEED,
          MINEVAL, maxeval, NNEW, NMIN, FLATNESS,
          STATEFILE, spin,
          &nregions, &neval, &fail, integral, error, prob);
    break;

  case CUBA_ALGORITHMS::DIVONNE:
    Divonne(ndim, NCOMP, Integrand, userdata, NVEC,
            relerr, abserr, VERBOSE, SEED,
            MINEVAL, maxeval, KEY1, KEY2, KEY3, MAXPASS,
            BORDER, MAXCHISQ, MINDEVIATION,
            NGIVEN, ndim, NULL, NEXTRA, NULL,
            STATEFILE, spin,
            &nregions, &neval, &fail, integral, error, prob);
    break;

  case CUBA_ALGORITHMS::CUHRE:
    Cuhre(ndim, NCOMP, Integrand, userdata, NVEC,
          relerr, abserr, VERBOSE | LAST,
          MINEVAL, maxeval, KEY,
          STATEFILE, spin,
          &nregions, &neval, &fail, integral, error, prob);
    break;

  default:
    throw std::runtime_error("No valid algorithm available.");
  }

  result = integral[0];
  uncertainty = error[0];
  probability = prob[0];

  return fail;
}

int cuba_verbose_integrate(integrand_t Integrand, int NDIM, void *USERDATA)
{
  int comp, neval, fail;
  cubareal integral[NCOMP], error[NCOMP], prob[NCOMP];

  double epsrel = 0.1;
  double epsabs = 1e-10;

  int VERBOSE = 13;

#if 1
  {
    ////boost::timer::auto_cpu_timer t;
    printf("-------------------- Vegas test --------------------\n");

    Vegas(NDIM, NCOMP, Integrand, USERDATA, NVEC,
          epsrel, epsabs, VERBOSE, SEED,
          MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH,
          GRIDNO, STATEFILE, SPIN,
          &neval, &fail, integral, error, prob);

    printf("VEGAS RESULT:\tneval %d\tfail %d\n",
           neval, fail);
    for (comp = 0; comp < NCOMP; ++comp)
      printf("VEGAS RESULT:\t%.8f +- %.8f\tp = %.3f\n",
             (double)integral[comp], (double)error[comp], (double)prob[comp]);
  }
#endif

#if 1
  {
    int nregions;
    // boost::timer::auto_cpu_timer t;
    printf("\n-------------------- Suave test --------------------\n");

    Suave(NDIM, NCOMP, Integrand, USERDATA, NVEC,
          epsrel, epsabs, VERBOSE | LAST, SEED,
          MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
          STATEFILE, SPIN,
          &nregions, &neval, &fail, integral, error, prob);

    printf("SUAVE RESULT:\tnregions %d\tneval %d\tfail %d\n",
           nregions, neval, fail);
    for (comp = 0; comp < NCOMP; ++comp)
      printf("SUAVE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
             (double)integral[comp], (double)error[comp], (double)prob[comp]);
  }
#endif

#if 1
  {
    int nregions;
    // boost::timer::auto_cpu_timer t;
    printf("\n------------------- Divonne test -------------------\n");

    Divonne(NDIM, NCOMP, Integrand, USERDATA, NVEC,
            epsrel, epsabs, VERBOSE, SEED,
            MINEVAL, MAXEVAL, KEY1, KEY2, KEY3, MAXPASS,
            BORDER, MAXCHISQ, MINDEVIATION,
            NGIVEN, NDIM, NULL, NEXTRA, NULL,
            STATEFILE, SPIN,
            &nregions, &neval, &fail, integral, error, prob);

    printf("DIVONNE RESULT:\tnregions %d\tneval %d\tfail %d\n",
           nregions, neval, fail);
    for (comp = 0; comp < NCOMP; ++comp)
      printf("DIVONNE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
             (double)integral[comp], (double)error[comp], (double)prob[comp]);
  }
#endif

#if 1
  {
    int nregions;
    // boost::timer::auto_cpu_timer t;
    printf("\n-------------------- Cuhre test --------------------\n");

    Cuhre(NDIM, NCOMP, Integrand, USERDATA, NVEC,
          epsrel, epsabs, VERBOSE | LAST,
          MINEVAL, MAXEVAL, KEY,
          STATEFILE, SPIN,
          &nregions, &neval, &fail, integral, error, prob);

    printf("CUHRE RESULT:\tnregions %d\tneval %d\tfail %d\n",
           nregions, neval, fail);
    for (comp = 0; comp < NCOMP; ++comp)
      printf("CUHRE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
             (double)integral[comp], (double)error[comp], (double)prob[comp]);
  }
#endif

  return 0;
}

namespace CDM
{

  NLSpectrum::NLSpectrum(const Spectrum &P, double r_low, double r_high, int gridno) : P(P), r_low(r_low), r_high(r_high), gridno(gridno)
  {
  }

  // Special constructor that initialises vegas integrator with the given k vector and integration context
  NLSpectrum::NLSpectrum(const Spectrum &P, double r_low, double r_high, double k, const CosmoUtil &cu, int gridno) : P(P), r_low(r_low), r_high(r_high), gridno(gridno)
  {
    double p, p_error, p_Q;
    P1L_IC ic(k, P, cu, r_low, r_high);
    void *data = (void *)&ic;
    cuba_integrate(P1Li_cuba, 2, data, p, p_error, p_Q, -gridno);
  }

  double NLSpectrum::operator()(double k, const CosmoUtil &cu) const
  {
    return get_result(k, cu)[0];
  }

  vec NLSpectrum::get_result(double k, const CosmoUtil &cu) const
  {
    // Dummy variables to store integration result
    double p, p_error, p_Q;

    // Provide integration context
    P1L_IC ic(k, P, cu, r_low, r_high);
    void *data = (void *)&ic;

    // Perform integration
    cuba_integrate(P1Li_cuba, 2, data, p, p_error, p_Q, gridno, 1, CUBA_ALGORITHMS::SUAVE, 0.01, 1e-20);

    vec result(3);
    result[0] = p;
    result[1] = p_error;
    result[2] = p_Q;
    return result;
  }

  NLBispectrum::NLBispectrum(const Spectrum &P, const Bispectrum &tree, double r_low, double r_high, int gridno) : Bispectrum(P), tree(tree), r_low(r_low), r_high(r_high), gridno(gridno)
  {
  }

  // Special constructor that initialises vegas integrator with the given k vectors and integration context
  NLBispectrum::NLBispectrum(const Spectrum &P, const Bispectrum &tree, double r_low, double r_high, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu, int gridno) : Bispectrum(P), tree(tree), r_low(r_low), r_high(r_high), gridno(gridno)
  {
    B1L_IC ic(k1, k2, k3, P, cu, r_low, r_high);
    void *USERDATA = (void *)&ic;
    double b1, b1_error, b1_Q;

    cuba_integrate(B1Li_cuba, 3, USERDATA, b1, b1_error, b1_Q, -gridno);
  }

  double NLBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return get_result(k1, k2, k3, cu)[0];
  }

  vec NLBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    std::cout << "Perform cuba integration for nonlinear bispectrum \n";
    // Optional parameters specify lower and upper boundaries for radial integration
    B1L_IC ic(k1, k2, k3, P, cu, r_low, r_high);
    void *USERDATA = (void *)&ic;
    double b1, b1_error, b1_Q;

    cuba_integrate(B1Li_cuba, 3, USERDATA, b1, b1_error, b1_Q, gridno, 1, CUBA_ALGORITHMS::VEGAS, 0.01, 1e-10);

    // Compute tree-level bispectrum
    vec r1 = tree.get_result(k1, k2, k3, cu);

    vec result(3);
    result[0] = b1 + r1[0];
    result[1] = sqrt(pow(b1_error, 2) + pow(r1[1], 2));
    result[2] = b1_Q;
    return result;
  }

  NLRBispectrum::NLRBispectrum(const Spectrum &P, const Spectrum &P1L, const Bispectrum &B1L) : Bispectrum(P), P1L(P1L), B1L(B1L)
  {
  }

  double NLRBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return get_result(k1, k2, k3, cu)[0];
  }

  // Return vector of length 8 with br, br_error, bt, b1, b1_error, st, s1, s1_error
  vec NLRBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    double k1n, k2n, k3n, q0, s0, q1, q1_sdev;
    vec b1, s1;
    k1n = k1.norm2();
    k2n = k2.norm2();
    k3n = k3.norm2();

    q0 = BR(P, k1, k2, k3, cu);
    b1 = B1L.get_result(k1, k2, k3, cu);

    s0 = ST(P, k1n, k2n, k3n, cu);
    s1 = S1L(P, P1L, k1n, k2n, k3n, cu);
    q1 = B1LR(q0, b1[0], s0, s1[0]);
    q1_sdev = B1LR_err(q0, b1[1], s0, s1[1]);

    std::cout << "br tree " << q0 << " br loop " << q1 << " br loop error " << q1_sdev << " sigma " << s0 << " sigma 1 l " << s1 << "\n";

    vec result(4);
    result[0] = q0;
    result[1] = q1;
    result[2] = q1_sdev;
    result[3] = b1[2];
    return result;
  }
}

namespace FDM
{

  NLSpectrum::NLSpectrum(const Spectrum &P, double r_low, double r_high, int gridno) : P(P), r_low(r_low), r_high(r_high), gridno(gridno)
  {
  }

  // Special constructor that initialises vegas integrator with the given k vector and integration context
  NLSpectrum::NLSpectrum(const Spectrum &P, double r_low, double r_high, double k, const CosmoUtil &cu, int gridno) : P(P), r_low(r_low), r_high(r_high), gridno(gridno)
  {
    double p, p_error, p_Q;
    P1L_IC ic(k, P, cu, r_low, r_high);
    void *data = (void *)&ic;
    cuba_integrate(P1Li_cuba, 4, data, p, p_error, p_Q, -gridno);
  }

  double NLSpectrum::operator()(double k, const CosmoUtil &cu) const
  {
    return get_result(k, cu)[0];
  }

  vec NLSpectrum::get_result(double k, const CosmoUtil &cu) const
  {
    // Dummy variables to store integration result
    double p, p_error, p_Q;

    // Provide integration context
    P1L_IC ic(k, P, cu, r_low, r_high);
    void *data = (void *)&ic;
    // Perform integration
    // cuba_verbose_integrate(P1Li, 4, data);

    for (int i = 0; i < 2; i++)
    {
      if (i > 0)
      {
        std::cout << "Troublesome value k = " << k << " for i = " << i << ", resetting integrator \n";
        cuba_integrate(P1Li_cuba, 4, data, p, p_error, p_Q, -gridno);
        std::cout << "Reset complete \n";
      }
      cuba_integrate(P1Li_cuba, 4, data, p, p_error, p_Q, gridno, 1, CUBA_ALGORITHMS::VEGAS, 0.1, 1e-20);
      if (((P(k, cu) + p) > 0))
      {
        break;
      }
      else
      {
        if ((P(k, cu) + abs(p)) < 1e-10)
        {
          p = 0;
          break;
        }
      }
    }

    vec result(3);
    result[0] = p;
    result[1] = p_error;
    result[2] = p_Q;
    return result;
  }

  void NLSpectrum::verbose_integration(double k, const CosmoUtil &cu)
  {
    // Provide integration context
    P1L_IC ic(k, P, cu, r_low, r_high);
    void *data = (void *)&ic;

    cuba_verbose_integrate(P1Li_cuba, 4, data);
  }

  NLBispectrum::NLBispectrum(const Spectrum &P, const Bispectrum &tree, double r_low, double r_high, int gridno) : Bispectrum(P), tree(tree), r_low(r_low), r_high(r_high), gridno(gridno)
  {
  }

  // Special constructor that initialises vegas integrator with the given k vectors and integration context
  NLBispectrum::NLBispectrum(const Spectrum &P, const Bispectrum &tree, double r_low, double r_high, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu, int gridno) : Bispectrum(P), tree(tree), r_low(r_low), r_high(r_high), gridno(gridno)
  {
    //B1L_IC ic(k1, k2, k3, P, cu, r_low, r_high);
    //void *USERDATA = (void *)&ic;
    //double b1, b1_error, b1_Q;

    //cuba_integrate(B1Li_cuba, 6, USERDATA, b1, b1_error, b1_Q, -gridno);
  }

  double NLBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return get_result(k1, k2, k3, cu)[0];
  }

  vec NLBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    // Optional parameters specify lower and upper boundaries for radial integration
    B1L_IC ic(k1, k2, k3, P, cu, r_low, r_high);
    void *USERDATA = (void *)&ic;
    double b1, b1_error, b1_Q;

    cuba_integrate(B1Li_cuba, 6, USERDATA, b1, b1_error, b1_Q, gridno, 1, CUBA_ALGORITHMS::DIVONNE, 0.01, 1e-20);

    // Compute tree-level bispectrum
    vec r1 = tree.get_result(k1, k2, k3, cu);

    vec result(3);
    result[0] = b1 + r1[0];
    result[1] = sqrt(pow(b1_error, 2) + pow(r1[1], 2));
    result[2] = b1_Q;
    return result;
  }

  NLRBispectrum::NLRBispectrum(const Spectrum &P, const Spectrum &P1L, const Bispectrum &B1L) : Bispectrum(P), P1L(P1L), B1L(B1L)
  {
  }

  double NLRBispectrum::operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    return get_result(k1, k2, k3, cu)[0];
  }

  // Return vector of length 8 with br, br_error, bt, b1, b1_error, st, s1, s1_error
  vec NLRBispectrum::get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const
  {
    double k1n, k2n, k3n, q0, s0, q1, q1_sdev;
    vec b1, s1;
    k1n = k1.norm2();
    k2n = k2.norm2();
    k3n = k3.norm2();

    q0 = BR(P, k1, k2, k3, cu);
    b1 = B1L.get_result(k1, k2, k3, cu);

    s0 = ST(P, k1n, k2n, k3n, cu);
    s1 = S1L(P, P1L, k1n, k2n, k3n, cu);
    q1 = B1LR(q0, b1[0], s0, s1[0]);
    q1_sdev = B1LR_err(q0, b1[1], s0, s1[1]);

    std::cout << "br tree " << q0 << " br loop " << q1 << " br loop error " << q1_sdev << " sigma " << s0 << " sigma 1 l " << s1 << "\n";

    vec result(4);
    result[0] = q0;
    result[1] = q1;
    result[2] = q1_sdev;
    result[3] = b1[2];
    return result;
  }
}

std::vector<double> pyLogspace(double start, double stop, int num, double base)
{
  double realStart = pow(base, start);
  double realBase = pow(base, (stop - start) / num);

  std::vector<double> retval;
  retval.reserve(num);
  std::generate_n(std::back_inserter(retval), num, Logspace<double>(realStart, realBase));
  return retval;
}

std::vector<double> pyLinspace(double start, double stop, int num)
{
  std::vector<double> retval;
  retval.reserve(num);
  std::generate_n(std::back_inserter(retval), num, Linspace<double>(start, (stop - start) / num));
  return retval;
}