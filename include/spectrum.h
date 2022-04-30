#ifndef __SPECTRUM_H__
#define __SPECTRUM_H__

#include <array>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>

#include <boost/math/special_functions/bessel_prime.hpp>
#include <gslwrap/vector_float.h>

#if 0
#include <bspline.h>
#endif 

#include "cuba.h"

typedef gsl::vector_float vec;

//Default maximum number of integrand evaluations for CUBA integrations
const size_t max_evaluations = 1000000;
#define SPLINE_RANGE_CHECK

/***
 * This header file defines linear and nonlinear spectra and bispectra for fuzzy dark matter. 
 * It uses the CUBA as well as the GSL integration library.
 * Further boost special functions are required for the analytical FDM time evolution in a matter-dominated universe. 
 */

//Overload vector_float operators to support regular vector addition, subtraction and multiplication with scalars, dot products and a squared sum
vec operator+(const vec &v1, const vec &v2);
vec operator-(const vec &v1, const vec &v2);
vec operator-(const vec &v);
vec operator*(float c, const vec &v);
vec operator*(const vec &v, float c);
double dot (const vec &k1, const vec &k2); //Dot product d = sum over i : k1_i * k2_i
double ssum(const vec &k1, const vec &k2); //Squared sum of two vectors

//Helper functions to convert between scale factor a, redshift z and time variable eta
double a_from_z  (double z);
double eta_from_z(double z);
double eta_from_a(double a);
double a_from_eta(double eta);
double z_from_eta(double eta);


//GSL spline to approximate 1D function
//Spline1D manages interpolation memory
//Pass list of x and y values and length of list to constructor
class Spline1D {
  private:
  gsl_interp_accel *acc_spline_;
  gsl_spline* spline_;
  size_t steps_;
  #ifdef SPLINE_RANGE_CHECK
  double xmin_, xmax_;
  #endif

  public:
  Spline1D(const double* x, const double* y, size_t steps);
  ~Spline1D();
  double operator() (double x);
  double dx         (double x);
};

//GSL spline to approximate 2D functions
//Spline2D manages interpolation memory
//Pass list of x, y and z values and as well as x and y size of arrays to constructor (z = f(x, y))
class Spline2D {
  private:
  gsl_interp_accel *xacc_spline_, *yacc_spline_;
  gsl_spline2d* spline_;
  size_t xsize_, ysize_;

  #ifdef SPLINE_RANGE_CHECK
  double xmin_, xmax_, ymin_, ymax_;
  #endif

  public:
  Spline2D(const double* x, const double* y, const double* z, size_t xsize, size_t ysize);
  ~Spline2D();
  double operator()   (double x, double y);
  double dx           (double x, double y);
  double dy           (double x, double y);
  double d2x          (double x, double y);
  double d2y          (double x, double y);
};

//Numerically integrate CDM growth function for given cosmology defined in hubble(a) in spectrum.cpp
double hubble(double a);

//Base class data structure for mass, times, growth factors and greens functions in CDM and FDM
struct CosmoUtil {
  double m;
  double eta, eta_in;

  CosmoUtil(double m, double eta, double eta_in) : m(m), eta(eta), eta_in(eta_in) {
  }

  ~CosmoUtil() {
  }
  virtual double D              (double k, double eta) const = 0;
  virtual double greens         (double k, double s, double eta) const = 0;
  virtual double d_s_greens     (double k, double s, double eta) const = 0;
  virtual double d_eta_greens   (double k, double s, double eta) const = 0;
  virtual double d_s_eta_greens (double k, double s, double eta) const = 0;
  virtual double f_i            (double k, double s, double eta, size_t i) const = 0;
};


//Base class for spectra
class Spectrum
{
public:
  virtual ~Spectrum(){};
  virtual double operator()(double k, const CosmoUtil &cu) const = 0;
  //Return vector with three entries: value of bispectrum, uncertainty and Q-value
  //Returns zero for uncertainty and Q-value by default
  virtual vec get_result(double k, const CosmoUtil &cu) const;
};

//Base class for bispectra
class Bispectrum
{
public:
  Bispectrum(const Spectrum& P) : P(P) {}
  virtual ~Bispectrum(){};
  //Return value of bispectrum
  virtual double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const = 0;
  //Return vector with three entries: value of bispectrum, uncertainty and Q-value
  virtual vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const = 0;
  const Spectrum& getSpectrum() const {return P;}

protected:
  const Spectrum &P;
};


//Base class for bispectra
class Trispectrum
{
public:
  Trispectrum(const Spectrum& P) : P(P) {}
  virtual ~Trispectrum(){};
  //Return value of bispectrum
  virtual double operator()(const vec &k1, const vec &k2, const vec &k3, const vec& k4, const CosmoUtil &cu) const = 0;
  //Return vector with three entries: value of trispectrum, uncertainty and Q-value
  virtual vec get_result(const vec &k1, const vec &k2, const vec &k3, const vec& k4, const CosmoUtil &cu) const = 0;
  const Spectrum& getSpectrum() const {return P;}

protected:
  const Spectrum &P;
};


//Use Bardeen fit to provide initial spectrum at z = 99
class BardeenSpectrum : public Spectrum
{

public:
  BardeenSpectrum();

  //Return CDM spectrum at z = 99 in given cosmology
  double P0(double k) const;
  double operator()(double k, const CosmoUtil &cu) const;

  // Bardeen - CDM transfer function
  static double transfer_function(double k);
  static double sigma8();
  static double aux_dsigma8(double k, void *params);
  static double spectrum_aux(double k);

protected:
  double anorm;

private:
  double d0;
};


//Interpolate lensing spectrum given by filename
//Expected file format is text file with two columns consisting of multipole moment l and P_kappa(l)
class LensingSpectrum
{
public:
  LensingSpectrum(const std::string& filename);
  double operator()(double l) const;


private:
  std::shared_ptr<Spline1D> s;
  double lmin, lmax;
};


//Interpolate power spectrum given by filename
//Expected file format is text file with two columns consisting of momentum k/h and P(k/h)
//Low and high momenta are linearly extrapolated in log-log space
class CAMBSpectrum : public Spectrum
{
public:
  CAMBSpectrum(const std::string& filename);
  double operator()(double k, const CosmoUtil &cu) const;

  //Interpolate CAMB spectrum
  double P0(double k) const;

private:
  std::shared_ptr<Spline1D> s;
  double kmin, kmax, pmin, pmax;
  double alphal, alphah, coeffl, coeffh;
};

class SplineSpectrum : public Spectrum {
public:
  SplineSpectrum(const std::string& filename);
  double operator()(double k, const CosmoUtil &cu) const;

private:
  std::shared_ptr<Spline2D> s;
  double kmin, kmax, tmin, tmax; //Coefficients for extrapolation of the CAMB spectrum below and above kmin and kmax with power laws coeff*k^alpha 
};


#if 0
///All high dimensional spline libraries did not work as well as necessary for loop bispectrum 
///SPLINTER did give the best results of all of them
class SplineBispectrum : public Bispectrum {
public:
  SplineBispectrum(const Spectrum& P, const std::string& filename);
  double operator()(const vec& k1, const vec& k2, const vec& k3, const CosmoUtil &cu) const;
  vec get_result   (const vec& k1, const vec& k2, const vec& k3, const CosmoUtil &cu) const;
  SPLINTER::BSpline s;
};
#endif 

namespace CDM
{
  //Numerically integrate growth factor in given cosmology
  double d_plus (double a, double a_in);
  //Numerically integrate decay factor in given cosmology
  double d_minus(double a, double a_in);
  //Return D'(a) for growingMode (growingMode = true) and decayingMode (growingMode = false) 
  //Used for providing initial conditions for FDM integration
  double get_ic(double a, bool growingMode);

  //Interpolate integrated growth (growingMode = true) and decay (growingMode = false) factor for given cosmology
  //Constructor calls integration routine for interval [10^(const_a_min), 10^const_a_max)] with const_a_res points
  struct D_spline {
    std::shared_ptr<Spline1D> s;
    std::vector<double> as, Ds;
    double amin_, amax_;
    D_spline(bool growingMode = true);
    double operator() (double a);
  };

  struct InitialCDMSpectrum : public BardeenSpectrum
  {
    //Return value of linear CDM power spectrum given by fitting form
    double operator()(double k, const CosmoUtil &cu) const;
  };

  struct InitialFDMSpectrum : public BardeenSpectrum
  {
    //Return value of linear CDM power spectrum given by fitting form
    double operator()(double k, const CosmoUtil &cu) const;
  };

  struct ScaleFreeSpectrum : public Spectrum
  {
    //Initialise n_ with spectral_index
    ScaleFreeSpectrum(double spectral_index) : n_(spectral_index){};
    //Return scale-free spectrum (k^spectal_index) at time given by c.eta_f after linear evolution from c.eta_i
    double operator()(double k, const CosmoUtil &cu) const;

  protected:
    double n_;
  };

  

  class NLSpectrum : public Spectrum
  {
  private:
    const Spectrum &P;
    //Lower and upper integration limit in MC integration, in [0, 1]
    //They are mapped to [0, infinity] via the rescaling k = r/(1-r)
    double r_low, r_high;
    //Internal variable that MC vegas algorithm to store information about integrand between different integrations
    //Integer from 0 to 9 for 10 different integration slots
    //Passing -gridno to MC vegas algorithm resets integration algorithm
    //Used to speed up integration for closeby values of k
    int gridno;

  public:
    //Standard constructor with low IR cutoff and no UV cutoff
    NLSpectrum(const Spectrum &P, double r_low = 1e-4, double r_high = 1., int gridno = 1);
    //Constructor that initialises MC vegas algorithm upon first run
    NLSpectrum(const Spectrum &P, double r_low, double r_high, double k, const CosmoUtil &cu, int gridno = 1);
    double operator()(double k, const CosmoUtil &cu) const;
    vec get_result(double k, const CosmoUtil &cu) const;
  };


  //Compute CDM loop bispectrum corrections
  class NLBispectrum : public Bispectrum
  {
  private:
    const Bispectrum& tree;
    //Lower and upper integration limit in MC integration, in [0, 1]
    //They are mapped to [0, infinity] via the rescaling k = r/(1-r)
    double r_low, r_high;
    //Internal variable that MC vegas algorithm to store information about integrand between different integrations
    //Integer from 0 to 9 for 10 different integration slots
    //Passing -gridno to MC vegas algorithm resets integration algorithm
    //Used to speed up integration for closeby values of k
    int gridno;

  public:
    NLBispectrum(const Spectrum &P, const Bispectrum& B, double r_low = 1e-4, double r_high = 1, int gridno = 2);
    //Constructor that initialises MC vegas algorithm upon first run
    NLBispectrum(const Spectrum &P, const Bispectrum& B, double r_low, double r_high, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu, int gridno = 2);
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };

  //Compute reduced CDM loop bispectrum all in one go (Slow! Carries out many loop integrations with relatively high accuracy. )
  class NLRBispectrum : public Bispectrum
  {
  private:
    const Spectrum &P1L;
    const Bispectrum &B1L;

  public:
    NLRBispectrum(const Spectrum &P, const Spectrum &P1L, const Bispectrum &B1L);
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };
}

namespace FDM
{
  //Numerically integrate growth factor in given cosmology
  std::vector<double> d_plus(const std::vector<double>& a, double a_in, double k, double mass);
  //Numerically integrate decay factor in given cosmology
  std::vector<double> d_minus(const std::vector<double>& a, double a_in, double k, double mass);

  //Interpolate integrated growth (growingMode = true) and decay (growingMode = false) factor for given cosmology
  //Constructor calls integration routine for interval [10^(const_a_min), 10^const_a_max)] with const_a_res points (constant logarithmic steps)
  //times [10^(const_k_min), 10^(const_k_max)] with const_k_res points (constant logarithmic steps)
  struct D_spline {
    std::shared_ptr<Spline2D> s;
    double kmax_, kmin_;
    double amin_, amax_;

    D_spline(double eta_in, int fdm_mass_id, bool growingMode);
    double operator() (double k, double eta);
    //Accounts for chain rule when taking derivative w.r.t. eta = 2 sqrt(a). 
    double deta       (double k, double eta);
    //Accounts for chain rule when taking second derivative w.r.t. eta = 2 sqrt(a). 
    double d2eta      (double k, double eta);
  };

  //Interpolated greens function and derivatives as well as f_i (f_i=0 = D(s)/D(eta), f_i=1 = -D'(s)/D(eta))
  struct G_spline {
    D_spline dp, dm;
    G_spline(double eta_in, int fdm_mass_id);
    double propagator         (double k, double s, double eta);
    double d_s_propagator     (double k, double s, double eta);
    double d_eta_propagator   (double k, double s, double eta);
    double d_s_eta_propagator (double k, double s, double eta);
    double f_i                (double k, double s, double eta, size_t i);
  };

  //Interpolated CDM growth factor with suppression below Jeans scale
  struct D_hybrid {
    CDM::D_spline s;
    int fdm_mass_id;
    D_hybrid(int fdm_mass_id);
    double operator() (double k, double eta);
  };

  //CDM spectrum with FDM transfer function at time const_eta_in
  struct InitialFDMSpectrum : public BardeenSpectrum
  {
    //Return spectrum at time given by c.eta_f after linear evolution from FDM-like initial conditions at c.eta_i
    double operator()(double k, const CosmoUtil &cu) const;
  };

  //CDM spectrum at time const_eta_in
  struct InitialCDMSpectrum : public BardeenSpectrum
  {
    //Return spectrum at time given by c.eta_f after linear evolution from CDM-like initial conditions at c.eta_i
    double operator()(double k, const CosmoUtil &cu) const;
  };

  struct ScaleFreeSpectrum : public Spectrum
  {
    //Initialise n_ with spectral_index
    ScaleFreeSpectrum(double spectral_index) : n_(spectral_index){};
    //Return scale-free spectrum (k^spectal_index) at time given by c.eta_f after linear evolution from c.eta_i
    double operator()(double k, const CosmoUtil &cu) const;

  protected:
    double n_;
  };

  //Data structure for bispectrum at tree level
  struct BT_C
  {
    double k_norm;
    const Spectrum &P;
    const CosmoUtil& cu;
    BT_C(double k_norm, const Spectrum &P, const CosmoUtil &cu) : k_norm(k_norm), P(P), cu(cu) {}
  };

  //Data structure for F2 mode coupling coefficient
  struct F2_C
  {
    const vec &k1;
    const vec &k2;
    const CosmoUtil& cu;
    F2_C(const vec &k1, const vec &k2, const CosmoUtil &cu) : k1(k1), k2(k2), cu(cu) {}
  };


  //Data structure for F3 mode coupling coefficient
  struct F3_C
  {
    //Integration limits
    const vec &k1;
    const vec &k2;
    const vec &k3;
    const CosmoUtil &cu;
    F3_C(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) : k1(k1), k2(k2), k3(k3), cu(cu) {}
  };

  //Matter power spectrum  with loop corrections
  class NLSpectrum : public Spectrum
  {
  private:
    const Spectrum &P;
    //Lower and upper integration limit in MC integration, in [0, 1]
    //They are mapped to [0, infinity] via the rescaling k = r/(1-r)
    double r_low, r_high;
    //Internal variable that MC vegas algorithm to store information about integrand between different integrations
    //Integer from 0 to 9 for 10 different integration slots
    //Passing -gridno to MC vegas algorithm resets integration algorithm
    //Used to speed up integration for closeby values of k
    int gridno;

  public:
    //Standard constructor with low IR cutoff and no UV cutoff
    NLSpectrum(const Spectrum &P, double r_low = 1e-4, double r_high = 1., int gridno = 1);
    //Constructor that initialises MC vegas algorithm upon first run
    NLSpectrum(const Spectrum &P, double r_low, double r_high, double k, const CosmoUtil &cu, int gridno = 1);
    double operator()(double k, const CosmoUtil &cu) const;
    vec get_result(double k, const CosmoUtil &cu) const;
    void verbose_integration(double k, const CosmoUtil &cu);
  };

  //Matter bispectrum with loop corrections
  class NLBispectrum : public Bispectrum
  {
  private:
    const Bispectrum& tree;
    //Lower and upper integration limit in MC integration, in [0, 1]
    //They are mapped to [0, infinity] via the rescaling k = r/(1-r)
    double r_low, r_high;
    //Internal variable that MC vegas algorithm to store information about integrand between different integrations
    //Integer from 0 to 9 for 10 different integration slots
    //Passing -gridno to MC vegas algorithm resets integration algorithm
    //Used to speed up integration for closeby values of k
    int gridno;

  public:
    NLBispectrum(const Spectrum &P, const Bispectrum& tree, double r_low, double r_high, int gridno = 2);
    //Constructor that initialises MC vegas algorithm upon first run
    NLBispectrum(const Spectrum &P, const Bispectrum& tree, double r_low, double r_high, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu, int gridno = 2);
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };

  //Class for nonlinear bispectrum, store linear and nonlinear power spectrum as well as nonlinear bispectrum
  class NLRBispectrum : public Bispectrum
  {
  private:
    const Spectrum &P1L;
    const Bispectrum &B1L;

  public:
    NLRBispectrum(const Spectrum &P, const Spectrum &P1L, const Bispectrum &B1L);
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };
}

//From above eq. 177 in Bernardeau 2002
double ST(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);
//Sigma tree
double ST(const Spectrum &P, double k1, double k2, double k3, const CosmoUtil &cu);

//Sigma 1-loop
vec S1L(const Spectrum &P, const Spectrum &P1L, double k1, double k2, double k3, const CosmoUtil &cu);
//From above eq. 177 in Bernardeau 2002
//B1L is nonlinear power spectrum with one-loop corrections
//Taylor expansion of reduced bispectrum at 1-loop level
double B1LR(double b, double b1, double s, double s1);
double B1LR_err(double b, double b1_err, double s, double s1_err);
//Full reduced bispectrum at 1-loop level
double BFR(double b, double b1, double s, double s1);
double BFR_err(double b, double b1, double b1_err, double s, double s1, double s1_err);
double BFR_err(double b, double b1, double b1_err, double s, double s1, double s1_err);

//Generate triangle configuration for bispectrum
std::array<vec, 3> generateTriangle(double theta = 2 * M_PI / 3, double r1 = 1., double r2 = 1.);
//Generate rectangle configuration for trispectrum
std::array<vec, 4> generateRectangle(bool isEquilateral, double l1, double l2, double l4, double phi, double psi);


//Integration context for tree-level trispectrum integration
struct B0_IC
{
  //Integration limits
  const vec& k1;
  const vec& k2;
  const vec& k3;
  const Spectrum &P; 
  const CosmoUtil &cu;
  B0_IC(const vec& k1, const vec& k2, const vec& k3, const Spectrum &P, const CosmoUtil &cu) : k1(k1), k2(k2), k3(k3), P(P), cu(cu) {}
};

//Integration context for tree-level trispectrum integration
struct T0_IC
{
  //Integration limits
  const vec& k1;
  const vec& k2;
  const vec& k3;
  const vec& k4;
  const Spectrum &P; 
  const CosmoUtil &cu;
  gsl_integration_workspace *w;
  double eta;
  T0_IC(const vec& k1, const vec& k2, const vec& k3, const vec& k4, const Spectrum &P, const CosmoUtil &cu, gsl_integration_workspace *w, double eta = 0) : k1(k1), k2(k2), k3(k3), k4(k4), P(P), cu(cu), w(w), eta(eta) {}
};

//Integration context for 1-loop power spectrum integration
struct P1L_IC
{
  //Integration limits
  //phi_low, phi_high, ctheta_low, ctheta_high, r_low, r_high, eta_in, eta_f
  std::array<double, 8> ib;
  double k_norm;
  const Spectrum &P; 
  const CosmoUtil &cu;
  P1L_IC(double k_norm, const Spectrum &P, const CosmoUtil &cu, double r_low = 1. / 5, double r_high = 16. / 17);
};

//Integration context for 1-loop bispectrum integration
struct B1L_IC
{
  //Integration limits
  //phi_low, phi_high, ctheta_low, ctheta_high, r_low, r_high, eta_in, eta_f
  std::array<double, 8> ib;
  vec k1, k2, k3;
  const Spectrum &P; 
  const CosmoUtil &cu;
  B1L_IC(vec k1, vec k2, vec k3, const Spectrum &P, const CosmoUtil &cu, double r_low = 1. / 5, double r_high = 16. / 17);
};

//Define namespace for all Cold Dark Matter related functions
namespace CDM
{
  //CDM growth factor for matter dominated universe with Omega_m = 1, corresponds to D = a/a_in
  double D(double eta, double eta_in);

  //Symmetrised F2 coupling from analytical expression for MDU
  double F2s(const vec &k1, const vec &k2);
  //Time-dependent symmetrised F2 coupling
  double F2s_td(const vec &k1, const vec &k2, const CosmoUtil& cu);
  //Symmetrised F3 coupling from analytical expression for MDU
  double F3s(const vec &k1, const vec &k2, const vec &k3);
  //Time-dependent symmetrised F3 coupling
  double F3s_td(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil& cu);
  //Symmetrised F4 coupling from analytical expression for MDU
  double F4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4);

  //Analytical growth function for MDU
  struct CDM_Analytical_CosmoUtil : public CosmoUtil {
    CDM_Analytical_CosmoUtil(double eta, double eta_in);
    double D              (double k, double eta)                     const;
    double greens         (double k, double s, double eta)           const;
    double d_s_greens     (double k, double s, double eta)           const;
    double d_eta_greens   (double k, double s, double eta)           const;
    double d_s_eta_greens (double k, double s, double eta)           const;
    double f_i            (double k, double s, double eta, size_t i) const;
  };

  //Numerical growth function for cosmology defined by hubble(a)
  //Use FDM spline evaluated at low k to make sure CDM and FDM agree exactly (important for chi-squared computations)
  struct CDM_Numerical_CosmoUtil : public CosmoUtil {
    //Use 2-dimensional FDM spline and evaluate at low k to make sure that CDM and FDM growth factors at low k are exactly equal
    std::shared_ptr<FDM::D_spline> d;
    CDM_Numerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in);
    double D              (double k, double eta)                     const;    
    double greens         (double k, double s, double eta)           const;
    double d_s_greens     (double k, double s, double eta)           const;
    double d_eta_greens   (double k, double s, double eta)           const;
    double d_s_eta_greens (double k, double s, double eta)           const;
    double f_i            (double k, double s, double eta, size_t i) const;
  };

  //Cartesian bispectrum at tree level
  double BT (const Spectrum &P,               const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);

  //Bispectrum at tree level
  class TreeBispectrum : public Bispectrum {
  public:
    TreeBispectrum(const Spectrum &P) : Bispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };

  //Reduced bispectrum at tree level
  double BR (const Spectrum &P,               const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);
  //Integrand for cartesian bispectrum at loop-level
  double B1L(const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);


  //Trispectrum at tree level
  class TreeTrispectrum : public Trispectrum {
  public:
    TreeTrispectrum(const Spectrum &P) : Trispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;

  private:
    double t2(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
    double t3(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
  };

  //CUBA integrand for power spectrum at 1-loop level
  int P1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for bispectrum at 1-loop level
  int B1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
}

//Define namespace for all Fuzzy Dark Matter related functions
namespace FDM
{
  //Clever numerical implemention of Bessel function derivatives taken from Python's numpy library
  double my_bessel_diff_formula(double v, double z, size_t n, double phase);
  //N-th derivative of cylindrical Bessel function of fractional order
  double my_jvp(double v, double z, size_t n = 1);
  //Return m/hbar * e
  double m_bar(double m);
  //Return m/hbar * e * hubble constant 
  double mbH0(double m);
  //Quantum jeans scale as function of scale factor and mass, equation (18) from Li 2020
  double jeans_scale(double a, double m);
  //Mass and momentum dependent FDM-scale
  double b_f(double k, double m);
  //First three terms in Taylor expansion of fractional Bessel function
  double jv_taylor(double n, double x);
  //Growth function in FDM for MDU
  //Bessel function in denominator approximated by third order Taylor expansion to improve numerical stability as suggested in Lague Paper
  double D_plus_renormalised  (double k, double eta, double eta_in, double m);
  //Decay function in FDM for MDU
  //Bessel function in denominator approximated by third order Taylor expansion to improve numerical stability as suggested in Lague Paper
  double D_minus_renormalised  (double k, double eta, double eta_in, double m);


  //Growth function in FDM for MDU
  double D_plus_analytical  (double k, double eta, double eta_in, double m);
  //Derivative of analytical growth factor
  double d_D_plus_analytical(double k, double eta, double eta_in, double m);

  //Growth function in FDM for MDU
  double D_plus_semianalytical  (double k, double eta, double eta_in, double m);
  //Derivative of analytical growth factor
  double d_D_plus_semianalytical(double k, double eta, double eta_in, double m);
  //Analytical decay factor
  double D_minus_analytical  (double k, double eta, double eta_in, double m);
  //Derivative of analytical decay factor
  double d_D_minus_analytical(double k, double eta, double eta_in, double m);
  //Use empirical transfer function from Hu et al. 2000
  double empirical_transfer (double k, double eta, double eta_in, double m);
  //Greens function for FDM time evolution in matter-dominated universe
  double greens_analytical  (double k, double s, double eta, double m);
  //S-derivative of greens
  double d_s_greens_analytical    (double k, double s, double eta, double m);
  //Eta-derivative of greens
  double d_eta_greens_analytical  (double k, double s, double eta, double m);
  //S- and eta- second derivative of greens
  double d_s_eta_greens_analytical(double k, double s, double eta, double m);
  //Helper function for time evolution in mode coupling functions
  double f_i_analytical           (double k, double s, double eta, double m, size_t i);


  //Analytical growth functions and propagators for MDU
  struct FDM_Analytical_CosmoUtil : public CosmoUtil {
    FDM_Analytical_CosmoUtil(int fdm_mass_id, double eta, double eta_in);
    double D              (double k, double eta)                     const;   
    double greens         (double k, double s, double eta)           const;
    double d_s_greens     (double k, double s, double eta)           const;
    double d_eta_greens   (double k, double s, double eta)           const;
    double d_s_eta_greens (double k, double s, double eta)           const;
    double f_i            (double k, double s, double eta, size_t i) const;
  };

  //Numerical growth functions for arbitrary cosmology and analytical propagators for MDU
  struct FDM_SemiNumerical_CosmoUtil : public CosmoUtil {
    std::shared_ptr<G_spline> g;
    FDM_SemiNumerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in);
    double D              (double k, double eta)                     const;
    double greens         (double k, double s, double eta)           const;
    double d_s_greens     (double k, double s, double eta)           const;
    double d_eta_greens   (double k, double s, double eta)           const;
    double d_s_eta_greens (double k, double s, double eta)           const;
    double f_i            (double k, double s, double eta, size_t i) const;
  };


  //Numerical growth functions and propagators for arbitrary cosmology
  //DOES NOT WORK SINCE WE WERE UNABLE TO NUMERICALLY OBTAIN TWO ORTHOGONAL FUNCTIONS D+(k) and D-(k) IN THE OSCILLATING REGIME
  //NUMERICALLY, THE LINEARLY INDEPENDENT OSCILLATING FUNCTIONS ARE NOT INDEPENDENT
  struct FDM_FullyNumerical_CosmoUtil : public CosmoUtil {
    std::shared_ptr<G_spline> g;
    FDM_FullyNumerical_CosmoUtil(int fdm_mass_id, double eta, double eta_in);
    double D              (double k, double eta)                     const;
    double greens         (double k, double s, double eta)           const;
    double d_s_greens     (double k, double s, double eta)           const;
    double d_eta_greens   (double k, double s, double eta)           const;
    double d_s_eta_greens (double k, double s, double eta)           const;
    double f_i            (double k, double s, double eta, size_t i) const;
  };

  //FDM mode coupling function to linear order
  //Involves contributions from derivatives of density and velocity field as well as linear contribution from quantum pressure
  double gamma_iab (const vec &k, const vec &k1, const vec &k2,                                         double eta, double m, size_t i, size_t a, size_t b);
  double theta2111 (const vec &k, const vec &k1, const vec &k2, const vec &k3,                double s, double eta, double m);
  double xi2111    (const vec &k, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s, double eta, double m);
  double W_coupling(const vec &k, const vec &k1, const vec &k2,                               double s, double eta, const CosmoUtil &cu, size_t i, size_t a, size_t b);
  double U_coupling(const vec &k, const vec &k1, const vec &k2, const vec &k3,                double s, double eta, const CosmoUtil &cu, size_t i);
  double V_coupling(const vec &k, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s, double eta, const CosmoUtil &cu, size_t i);

  //Mode coupling functions for two modes, symmetrised
  double F2(const vec &k1, const vec &k2, double s, const CosmoUtil &cu);

  //Mode coupling functions for three modes, symmetrised
  double J3(const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double eta, const CosmoUtil &cu);
  double I3(const vec &k1, const vec &k2, const vec &k3, double s, double eta, const CosmoUtil &cu);
  //F3 combines contributions I3 and J3,  symmetrised
  double F3(const vec &k1, const vec &k2, const vec &k3, double s1, double s2, const CosmoUtil &cu);

  //Mode coupling functions for four modes
  //The suffix "s" denotes explicitly symmetrised versions of the coupling functions
  //W4 is already symmetric
  double W4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1,                       double eta, const CosmoUtil &cu);
  double I4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu);
  double I4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu);
  double J4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu);
  double J4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3, double eta, const CosmoUtil &cu);
  double K4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2,            double eta, const CosmoUtil &cu);
  double K4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2,            double eta, const CosmoUtil &cu);
  double H4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2,            double eta, const CosmoUtil &cu);
  double H4s(const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2,            double eta, const CosmoUtil &cu);
  //F4 combines W4, I4s, J4s, K4s, H4s, symmetrised
  double F4 (const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, double s3,             const CosmoUtil &cu);

  //F2 mode coupling integrated using gsl 
  double F2i(const vec &k1, const vec &k2, const CosmoUtil &cu);
  //F2 mode coupling integrated using cuba
  double F3i(const vec &k1, const vec &k2, const vec& k3, const CosmoUtil &cu);
  //F2 mode coupling integrated using gsl 
  double T0i(const Spectrum&P, const vec &k1, const vec &k2, const vec& k3, const vec& k4, const CosmoUtil &cu);

  //Integrand for FDM bispectrum at tree level
  double b0_integrand(const Spectrum& P, const vec &k1, const vec &k2, const vec &k3, double s1, const CosmoUtil &cu);

  //Cartesian bispectrum at tree level
  double BT(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);

  //Bispectrum at tree level integrated using GSL
  class TreeBispectrum : public Bispectrum {
  public:
    TreeBispectrum(const Spectrum &P) : Bispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };


  //Bispectrum at tree level integrated using CUBA
  class VegasTreeBispectrum : public Bispectrum {
  public:
    VegasTreeBispectrum(const Spectrum &P) : Bispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu) const;
  };



  //Reduced bispectrum at tree level
  double BR(const Spectrum &P, const vec &k1, const vec &k2, const vec &k3, const CosmoUtil &cu);

  //Integrand for FDM trispectrum at tree level
  double t0_integrand(const Spectrum& P, const vec &k1, const vec &k2, const vec &k3, const vec &k4, double s1, double s2, const CosmoUtil &cu);


  //Tispectrum at tree level integrated using GSL
  class TreeTrispectrum : public Trispectrum {
  public:
    TreeTrispectrum(const Spectrum &P) : Trispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
  };



  //Trispectrum at tree level integrated using CUBA
  class TreeTrispectrumCuba : public Trispectrum {
  public:
    TreeTrispectrumCuba(const Spectrum &P) : Trispectrum(P) {}
    //Return value of bispectrum
    double operator()(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
    //Return vector with three entries: value of bispectrum, uncertainty and Q-value
    vec get_result(const vec &k1, const vec &k2, const vec &k3, const vec &k4, const CosmoUtil &cu) const;
  };


  //Contributions to bispectrum at 1-loop level
  double b222  (const Spectrum &P, const vec &q, const vec &k1, const vec &k2,                double s1, double s2, double s3, const CosmoUtil &cu);
  double bt222 (const Spectrum &P, const vec &q, const vec &k1, const vec &k2,                double s1, double s2, double s3, const CosmoUtil &cu);
  double B222  (const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double b3211 (const Spectrum &P, const vec &q,                const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double bt3211(const Spectrum &P, const vec &q,                const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double B3211 (const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double b3212 (const Spectrum &P, const vec &q,                const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double B3212 (const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double b411  (const Spectrum &P, const vec &q,                const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double B411  (const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);
  double B1L   (const Spectrum &P, const vec &q, const vec &k1, const vec &k2, const vec &k3, double s1, double s2, double s3, const CosmoUtil &cu);


  //CUBA integrand for coupling kernel F2
  int F2i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for coupling kernel F3
  int F3i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for trispectrum at Tree-level
  int B0i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for trispectrum at Tree-level
  int T0i_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for power spectrum at 1-loop level
  int P1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);
  //CUBA integrand for bispectrum at 1-loop level
  int B1Li_cuba(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *userdata);

  //Helper function used in unit_test.cpp
  double B1Li_test(double phi, double cosine_theta, double r, double s1, double s2, double s3, B1L_IC *c);
}

//CUBA integration routines
enum CUBA_ALGORITHMS {
  VEGAS,
  SUAVE,
  CUHRE,
  DIVONNE
};

int cuba_integrate(integrand_t Integrand, int NDIM, void *USERDATA, double &result, double &uncertainty, double &probability, int gridno = 1, int VERBOSE = 0, CUBA_ALGORITHMS alg = CUBA_ALGORITHMS::VEGAS, double RELERR = 1e-1, double ABSERR = 1e-20, void* spin = NULL, size_t maxeval = max_evaluations);
int cuba_verbose_integrate(integrand_t Integrand, int NDIM, void *USERDATA);

//Python logspace function template
//Credit to user Erbureth on https://stackoverflow.com/a/21429452
template <typename T>
class Logspace
{
private:
  T curValue, base;

public:
  Logspace(T first, T base) : curValue(first), base(base) {}

  T operator()()
  {
    T retval = curValue;
    curValue *= base;
    return retval;
  }
};

//Python linspace function template
template <typename T>
class Linspace
{
private:
  T curValue, step;

public:
  Linspace(T first, T step) : curValue(first), step(step) {}

  T operator()()
  {
    T retval = curValue;
    curValue += step;
    return retval;
  }
};

//Python logspace function
//Credit to user Erbureth on https://stackoverflow.com/a/21429452
std::vector<double> pyLogspace(double start, double stop, int num = 50, double base = 10);

//Python linspace function
std::vector<double> pyLinspace(double start, double stop, int num = 50);

#endif