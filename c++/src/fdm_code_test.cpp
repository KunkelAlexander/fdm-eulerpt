#define BOOST_TEST_MODULE FDMCodeTest

#include  <vector>
#include <boost/test/unit_test.hpp>

#include "spectrum.h"
//#include "tomo.h"

/******
 * DISCLAIMER:
 * The test cases rely on the unit test fiducial cosmology as set in constants.h
 * Define UNIT_TEST_COSMOLOGY in constants.h for tests to work!
**/

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

const double m = 1e-23, z_i = 99., z_f = 5., n = -2.0, k = 0.3, s = 0.2, a = a_from_z(z_f), a_in = a_from_z(z_i), eta = eta_from_a(a), eta_in = eta_from_a(a_in);
FDM::FDM_Analytical_CosmoUtil cu1(m, eta, eta_in);
FDM::FDM_Fit_CosmoUtil  cu2(m, eta, eta_in);

//Scale-free initial spectrum
const FDM::ScaleFreeSpectrum p_sc(-2.0);
//CDM-like initial spectrum
const FDM::InitialCDMSpectrum p;

//Generate bispectra configurations
const auto triangle  = generate_triangle(); 
const auto triangle2 = generate_triangle(0.3, 22., 31.); 

struct DATA *data;

BOOST_AUTO_TEST_CASE(ana_couplingtest, * utf::tolerance(0.0001))
{
  BOOST_TEST_MESSAGE("Analytical coupling test started.");
  vec k1(3), k2(3), k3(3), k4(3);

  //INITIALIZE VECTORS
  k1[0] = 2;
  k1[1] = 0;
  k1[2] = 0;

  k2[0] = 2;
  k2[1] = 1;
  k2[2] = 0;

  k3[0] = 1;
  k3[1] = 0;
  k3[2] = 0;

  k4[0] = 1;
  k4[1] = 3;
  k4[2] = 2;
  
  //Numerical results from python code
  std::vector <float> results = {
  13.932315199971079 , 134.71029139011478, 194.27461954077 , 
  1.519267447e-08 , -1674.780647400352 , 16.66666666666666 , 
  16.666610629114814 , 40.82468302152786 , 0.6660786151168298, 
  1.6351526688414155, -3.3450880013196027, -8.12175837820326 ,
  0.0 , -1.0 , -0.4 , 0.0 , 
  -0.0023845716435636852 , 0.0 , 0.0 , -0.8, 
0.44004642032418445 , -9.959713840257418 , -3.9838855361029677 , 0.5314722641923445 , 
-1.0798324418145517 , 24.34988392197223 , 9.739953568788891 , -1.3041828459296, 
-0.003379556508089739 , 0.008293113153135762 , 0.01633446474049803 , -0.03980574192924423, 
-0.025716114818557123, 0.06000020173586505 , -0.5999998712216029, 1.0390108220201149 , 
-4.869485703605372e-06 , 39065.65995352095, -7703163.395005878, 1.3411790054403123, 
0.011100196503938153 , -0.3429504082278544 , 2.423906139025725 , 10.039466933075015 , -0.16716997032367772 , 11.665128428093935};


  int i = 0;

  //TIME EVOLUTION
  BOOST_TEST(FDM::jeans_scale(eta, m) == results[i++]);
  BOOST_TEST(FDM::jv_taylor(2.5, .2) == results[i++]);
  BOOST_TEST(FDM::mbH0(m) == results[i++]);
  BOOST_TEST(FDM::m_bar(m) == results[i++]);
  BOOST_TEST(FDM::my_jvp(-2.5, 0.2) == results[i++]);
  BOOST_TEST(CDM::D(eta, eta_in) == results[i++]);
  BOOST_TEST(FDM::D_plus_analytical(k, eta, eta_in, m)== results[i++]);
  BOOST_TEST(FDM::d_D_plus_analytical(k, eta, eta_in, m) == results[i++]);
  BOOST_TEST(FDM::greens_analytical(k, s, eta, m ) == results[i++]);
  BOOST_TEST(FDM::d_eta_greens_analytical(k, s, eta, m ) == results[i++]);
  BOOST_TEST(FDM::d_s_greens_analytical(k, s, eta, m ) == results[i++]);
  BOOST_TEST(FDM::d_s_eta_greens_analytical(k, s, eta, m )== results[i++]);

  //COUPLING MATRICES

  //GAMMA
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 0, 0, 0 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 0, 0, 1 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 0, 1, 0 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 0, 1, 1 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 1, 0, 0 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 1, 0, 1 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1, 0 ) == results[i++]);
  BOOST_TEST       (FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1, 1 ) == results[i++]);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 2, 1,   1 ), std::invalid_argument);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1,  -1 ), std::invalid_argument);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 120, 1 ), std::invalid_argument);

  //W
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 0, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 0, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 0, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 0, 1, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 1, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 1, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 1, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu1, 1, 1, 1 ) == results[i++]);

  //U
  BOOST_TEST(FDM::U_coupling(k1, k2, k3, k4, s, eta, cu1, 0) == results[i++]);
  BOOST_TEST(FDM::U_coupling(k1, k2, k3, k4, s, eta, cu1, 1) == results[i++]);

  //V
  BOOST_TEST(FDM::V_coupling(k1 + k2, k1, k2, k3, k4, s, eta, cu1, 0) == results[i++]);
  BOOST_TEST(FDM::V_coupling(k1 + k2, k1, k2, k3, k4, s, eta, cu1, 1) == results[i++]);

  //XI
  BOOST_TEST(FDM::xi2111(k1 + k2, k1, k2, k3, k4, s, eta, m) == results[i++]);

  //f
  BOOST_TEST(FDM::f_i_analytical(k, s, eta, m, 0) == results[i++]);
  BOOST_TEST(FDM::f_i_analytical(k, s, eta, m, 1) == results[i++]);

  //COUPLING KERNELS
  BOOST_TEST(FDM::F2(k1, k2, s, cu1) == results[i++]);
  BOOST_TEST(FDM::F2(k1, -1.*k1, s, cu1) == 0.);
  BOOST_TEST(FDM::I3(k1, k2, k3, s, eta, cu1) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k2*108. + k3*63., k2*108., k3*63., s/2, s, cu1, 0, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::J3(24.*k1, 108.*k2, 63.*k3, s, s/2,   eta, cu1) == results[i++]);
  BOOST_TEST(FDM::J3(k1, k2, -1.*k2, s,     s/2,        eta, cu1) == 0.);
  BOOST_TEST(FDM::F3(k1, k2, k3, s        , s/2,             cu1) == results[i++]);
  BOOST_TEST(FDM::W4(k1, k2*2., k3, k4*3. , .5,         eta, cu1) == results[i++]);
  BOOST_TEST(FDM::H4s(k1, k2*2., k3, k4*3., .5, .4,     eta, cu1) == results[i++]);
  BOOST_TEST(FDM::H4s(k2*2., k3, k4*3., k1, .5, .5,     eta, cu1) == FDM::H4s(k1, k2*2., k3, k4*3., .5, .5, eta, cu1));
  BOOST_TEST(FDM::I4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu1) == results[i++]);
  BOOST_TEST(FDM::J4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu1) == results[i++]);
  BOOST_TEST(FDM::K4s(k1, k2*2., k3, k4*3., .5, .4,     eta, cu1) == results[i++]);
  BOOST_TEST(FDM::F4(k1, k2*2., k3, k4*3. , .5, .4, .3,      cu1) == results[i++]);
}

BOOST_AUTO_TEST_CASE(num_couplingtest, * utf::tolerance(0.01))
{
  vec k1(3), k2(3), k3(3), k4(3);

  //INITIALIZE VECTORS
  k1[0] = 2;
  k1[1] = 0;
  k1[2] = 0;

  k2[0] = 2;
  k2[1] = 1;
  k2[2] = 0;

  k3[0] = 1;
  k3[1] = 0;
  k3[2] = 0;

  k4[0] = 1;
  k4[1] = 3;
  k4[2] = 2;
  
  //Numerical results from python code
  std::vector <float> results = {
  13.932315199971079 , 134.71029139011478, 194.27461954077 , 
  1.519267447e-08 , -1674.780647400352 , 16.66666666666666 , 
  16.666610629114814 , 40.82468302152786 , 0.6660786151168298, 
  1.6351526688414155, -3.3450880013196027, -8.12175837820326 ,
  0.0 , -1.0 , -0.4 , 0.0 , 
  -0.0023845716435636852 , 0.0 , 0.0 , -0.8, 
0.44004642032418445 , -9.959713840257418 , -3.9838855361029677 , 0.5314722641923445 , 
-1.0798324418145517 , 24.34988392197223 , 9.739953568788891 , -1.3041828459296, 
-0.003379556508089739 , 0.008293113153135762 , 0.01633446474049803 , -0.03980574192924423, 
-0.025716114818557123, 0.06000020173586505 , -0.5999998712216029, 1.0390108220201149 , 
-4.869485703605372e-06 , 39065.65995352095, -7703163.395005878, 1.3411790054403123, 
0.011100196503938153 , -0.3429504082278544 , 2.423906139025725 , 10.039466933075015 , -0.16716997032367772 , 11.665128428093935};


  int i = 0;

  //TIME EVOLUTION
  BOOST_TEST(FDM::jeans_scale(eta, m)      == results[i++]);
  BOOST_TEST(FDM::jv_taylor(2.5, .2)       == results[i++]);
  BOOST_TEST(FDM::mbH0(m)                  == results[i++]);
  BOOST_TEST(FDM::m_bar(m)                 == results[i++]);
  BOOST_TEST(FDM::my_jvp(-2.5, 0.2)        == results[i++]);
  BOOST_TEST(CDM::D(eta, eta_in)           == results[i++]);
  BOOST_TEST(cu2.D  (k, eta)               == results[i++]);
  i++;
  //BOOST_TEST(cu2.d_D(k, eta)               == results[i++]);
  BOOST_TEST(cu2.D  (k4.norm2(), eta)      == cu1.D  (k4.norm2(), eta));
  i++;
  //BOOST_TEST(cu2.d_D(k4.norm2(), eta)      == cu1.d_D(k4.norm2(), eta));
  BOOST_TEST(cu2.D  (k3.norm2(), eta)      == cu1.D  (k3.norm2(), eta));
  i++;
  //BOOST_TEST(cu2.d_D(k3.norm2(), eta)      == cu1.d_D(k3.norm2(), eta));
  BOOST_TEST(cu2.D  ((2*k4).norm2(), eta)  == cu1.D  ((2*k4).norm2(), eta));
  i++;
  //BOOST_TEST(cu2.d_D((2*k4).norm2(), eta)  == cu1.d_D((2*k4).norm2(), eta));
  BOOST_TEST(cu2.greens(k, s, eta)         == results[i++]);
  BOOST_TEST(cu2.d_eta_greens(k, s, eta)   == results[i++]);
  BOOST_TEST(cu2.d_s_greens(k, s, eta)     == results[i++]);
  BOOST_TEST(cu2.d_s_eta_greens(k, s, eta) == results[i++]);

  //COUPLING MATRICES

  //GAMMA
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 0, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 0, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 0, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 0, 1, 1 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1, 1 ) == results[i++]);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 2, 1,   1 ), std::invalid_argument);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 1,  -1 ), std::invalid_argument);
  BOOST_CHECK_THROW(FDM::gamma_iab(k1, k2, k3, eta, m, 1, 120, 1 ), std::invalid_argument);

  //W
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 0, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 0, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 0, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 0, 1, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 1, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 1, 0, 1 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 1, 1, 0 ) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k1, k2, k3, s, eta, cu2, 1, 1, 1 ) == results[i++]);

  //W
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 0, 0, 0 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 0, 0, 0 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 0, 0, 1 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 0, 0, 1 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 0, 1, 0 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 0, 1, 0 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 0, 1, 1 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 0, 1, 1 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 1, 0, 0 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 1, 0, 0 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 1, 0, 1 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 1, 0, 1 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 1, 1, 0 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 1, 1, 0 ));
  BOOST_TEST(FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu1, 1, 1, 1 ) == FDM::W_coupling(2*k4, 3*k2, 5*k3, s, eta, cu2, 1, 1, 1 ));

  //U
  BOOST_TEST(FDM::U_coupling(k1, k2, k3, k4, s, eta, cu2, 0) == results[i++]);
  BOOST_TEST(FDM::U_coupling(k1, k2, k3, k4, s, eta, cu2, 1) == results[i++]);

  //V
  BOOST_TEST(FDM::V_coupling(k1 + k2, k1, k2, k3, k4, s, eta, cu2, 0) == results[i++]);
  BOOST_TEST(FDM::V_coupling(k1 + k2, k1, k2, k3, k4, s, eta, cu2, 1) == results[i++]);

  //XI
  BOOST_TEST(FDM::xi2111(k1 + k2, k1, k2, k3, k4, s, eta, m) == results[i++]);

  //f
  BOOST_TEST(cu2.f_i(k, s, eta, 0)              == results[i++]);
  BOOST_TEST(cu2.f_i(k, s, eta, 1)              == results[i++]);
  BOOST_TEST(cu1.f_i(k1.norm2(), s, eta, 0)     == cu2.f_i(k1.norm2(), s, eta, 0)) ;
  BOOST_TEST(cu1.f_i(k2.norm2(), s, eta, 1)     == cu2.f_i(k2.norm2(), s, eta, 1)) ;
  BOOST_TEST(cu1.f_i(k3.norm2(), s, eta, 0)     == cu2.f_i(k3.norm2(), s, eta, 0)) ;
  BOOST_TEST(cu1.f_i(k4.norm2(), s, eta, 1)     == cu2.f_i(k4.norm2(), s, eta, 1)) ;
  BOOST_TEST(cu1.f_i((2*k1).norm2(), s, eta, 0) == cu2.f_i((2*k1).norm2(), s, eta, 0));
  BOOST_TEST(cu1.f_i((2*k2).norm2(), s, eta, 1) == cu2.f_i((2*k2).norm2(), s, eta, 1));
  BOOST_TEST(cu1.f_i((2*k3).norm2(), s, eta, 0) == cu2.f_i((2*k3).norm2(), s, eta, 0));
  BOOST_TEST(cu1.f_i((2*k4).norm2(), s, eta, 1) == cu2.f_i((2*k4).norm2(), s, eta, 1));
  BOOST_TEST(cu1.f_i((3*k1).norm2(), s, eta, 0) == cu2.f_i((3*k1).norm2(), s, eta, 0));
  BOOST_TEST(cu1.f_i((3*k2).norm2(), s, eta, 1) == cu2.f_i((3*k2).norm2(), s, eta, 1));
  BOOST_TEST(cu1.f_i((3*k3).norm2(), s, eta, 0) == cu2.f_i((3*k3).norm2(), s, eta, 0));
  BOOST_TEST(cu1.f_i((3*k4).norm2(), s, eta, 1) == cu2.f_i((3*k4).norm2(), s, eta, 1));

  //COUPLING KERNELS
  BOOST_TEST(FDM::F2(k1, k2, s, cu2)          == results[i++]);
  BOOST_TEST(FDM::F2(k1, -1.*k1, s, cu2)      == 0.);
  BOOST_TEST(FDM::I3(k1, k2, k3, s, eta, cu2) == results[i++]);
  BOOST_TEST(FDM::W_coupling(k2*108. + k3*63., k2*108., k3*63., s/2, s, cu2, 0, 0, 0 ) == results[i++]);
  BOOST_TEST(FDM::J3(24.*k1, 108.*k2, 63.*k3, s, s/2,   eta, cu2) == results[i++]);
  BOOST_TEST(FDM::J3(k1, k2, -1.*k2, s,     s/2,        eta, cu2) == 0.);
  BOOST_TEST(FDM::F3(k1, k2, k3, s        , s/2,             cu2) == results[i++]);
  BOOST_TEST(FDM::W4(k1, k2*2., k3, k4*3. , .5,         eta, cu2) == results[i++]);
  BOOST_TEST(FDM::H4s(k1, k2*2., k3, k4*3., .5, .4,     eta, cu2) == results[i++]);
  BOOST_TEST(FDM::H4s(k2*2., k3, k4*3., k1, .5, .5,     eta, cu2) == FDM::H4s(k1, k2*2., k3, k4*3., .5, .5, eta, cu2));
  BOOST_TEST(FDM::I4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu2) == results[i++]);
  BOOST_TEST(FDM::J4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu2) == results[i++]);
  BOOST_TEST(FDM::K4s(k1, k2*2., k3, k4*3., .5, .4,     eta, cu2) == results[i++]);
  BOOST_TEST(FDM::F4(k1, k2*2., k3, k4*3. , .5, .4, .3,      cu2) == results[i++]);

  BOOST_TEST(FDM::I4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu1) == results[i++]);
  BOOST_TEST(FDM::J4s(k1, k2*2., k3, k4*3., .5, .4, .3, eta, cu1) == results[i++]);
  BOOST_TEST(FDM::K4s(k1, k2*2., k3, k4*3., .5, .4,     eta, cu1) == results[i++]);
  BOOST_TEST(FDM::F4(k1, k2*2., k3, k4*3. , .5, .4, .3,      cu1) == results[i++]);
}

//Test contributions to I4s
//BOOST_AUTO_TEST_CASE(num_I4s_test, * utf::tolerance(0.15)) {
//    vec k1(3), k2(3), k3(3), k4(3);
//
//    //INITIALIZE VECTORS
//    k1[0] = 2;
//    k1[1] = 0;
//    k1[2] = 0;
//
//    k2[0] = 2;
//    k2[1] = 1;
//    k2[2] = 0;
//
//    k3[0] = 1;
//    k3[1] = 0;
//    k3[2] = 0;
//
//    k4[0] = 1;
//    k4[1] = 3;
//    k4[2] = 2;
//
//    k1 *=1;
//    k2 *=2;
//    k3 *=1;
//    k4 *=3;
//
//    double s1, s2, s3;
//    s1 = .5;
//    s2 = .4;
//    s3 = .3;
//
//    for (size_t b = 0; b < 2; ++b)
//    {
//      for (size_t c = 0; c < 2; ++c)
//      {
//        for (size_t d = 0; d < 2; ++d)
//        {
//          for (size_t e = 0; e < 2; ++e)
//          {
//            for (size_t f = 0; f < 2; ++f)
//            {
//              for (size_t g = 0; g < 2; ++g)
//              {
//                BOOST_TEST(W_coupling(k1 + k2 + k3 + k4, k1 + k2, k3 + k4, s1, eta, cu1, 0, b, c) == W_coupling(k1 + k2 + k3 + k4, k1 + k2, k3 + k4, s1, eta, cu2, 0, b, c));
//                BOOST_TEST(W_coupling(k1 + k2, k1, k2, s2, s1, cu1, b, d, e) == W_coupling(k1 + k2, k1, k2, s2, s1, cu2, b, d, e));
//                BOOST_TEST(W_coupling(k3 + k4, k3, k4, s3, s1, cu1, c, f, g) == W_coupling(k3 + k4, k3, k4, s3, s1, cu2, c, f, g));
//                BOOST_TEST(W_coupling(k1 + k2 + k3 + k4, k1 + k3, k2 + k4, s1, eta, cu1, 0, b, c) == W_coupling(k1 + k2 + k3 + k4, k1 + k3, k2 + k4, s1, eta, cu2, 0, b, c));
//                BOOST_TEST(W_coupling(k1 + k3, k1, k3, s2, s1, cu1, b, d, e) == W_coupling(k1 + k3, k1, k3, s2, s1, cu2, b, d, e));
//                BOOST_TEST(W_coupling(k2 + k4, k2, k4, s3, s1, cu1, c, f, g) == W_coupling(k2 + k4, k2, k4, s3, s1, cu2, c, f, g));
//                BOOST_TEST(W_coupling(k1 + k2 + k3 + k4, k1 + k4, k3 + k2, s1, eta, cu1, 0, b, c) == W_coupling(k1 + k2 + k3 + k4, k1 + k4, k3 + k2, s1, eta, cu2, 0, b, c));
//                BOOST_TEST(W_coupling(k1 + k4, k1, k4, s2, s1, cu1, b, d, e) == W_coupling(k1 + k4, k1, k4, s2, s1, cu2, b, d, e));
//                BOOST_TEST(W_coupling(k3 + k2, k3, k2, s3, s1, cu1, c, f, g) == W_coupling(k3 + k2, k3, k2, s3, s1, cu2, c, f, g));
//                BOOST_TEST(cu1.f_i(k1.norm2(), s2, eta, d) == cu2.f_i(k1.norm2(), s2, eta, d));
//                BOOST_TEST(cu1.f_i(k2.norm2(), s2, eta, e) == cu2.f_i(k2.norm2(), s2, eta, e));
//                BOOST_TEST(cu1.f_i(k3.norm2(), s3, eta, f) == cu2.f_i(k3.norm2(), s3, eta, f));
//                BOOST_TEST(cu1.f_i(k4.norm2(), s3, eta, g) == cu2.f_i(k4.norm2(), s3, eta, g));
//              }
//            }
//          }
//        }
//      }
//    }
//}

BOOST_AUTO_TEST_CASE(ana_spectrumtest, * utf::tolerance(0.0001))
{
  std::vector <float> results = {42.23226579432116};
  int i = 0;

  BOOST_TEST(p_sc(k, cu1) == FDM::D_plus_analytical(k, cu1.eta, cu1.eta_in, cu1.m)*FDM::D_plus_analytical(k, cu1.eta, cu1.eta_in, cu1.m)*pow(k, n));
  BOOST_TEST(p(k, cu1) == results[i++]);
}

BOOST_AUTO_TEST_CASE(num_spectrumtest, * utf::tolerance(0.01))
{
  std::vector <float> results = {42.23226579432116};
  int i = 0;

  BOOST_TEST(p_sc(k, cu2) == cu2.D(k, cu2.eta)*cu2.D(k, cu2.eta)*pow(k, n));
  BOOST_TEST(p(k, cu2) == results[i++]);
}

//BOOST_AUTO_TEST_CASE(ana_bispectrumtest, * utf::tolerance(1e-7))
//{
//  vec k1(3), k2(3), k3(3);
//
//  k1[0] = 2;
//  k1[1] = 0;
//  k1[2] = 0;
//
//  k2[0] = 2;
//  k2[1] = 1;
//  k2[2] = 0;
//
//  k3[0] = 1;
//  k3[1] = 0;
//  k3[2] = 0;
//
//  std::vector <float> results = {1.725103983654164, 1.725103983654164, 
//  0.,0.,1., 0.8660254,  0. , -0.5, -0.8660254, -0. , -0.5, 0.5474816027542845, 
//  31.30356137271131 , 16.308895511263515 , -534.144993522024 , 791.2030301855106,
//  -253245.53825594645, 5.641e+06, 18.92, 0.3175757583786859};
//  int i = 0;
//
//  //F2 MODE COUPLING
//  BOOST_TEST(FDM::F2i(k1, k2, cu1) == results[i++]);
//  BOOST_TEST(FDM::F2i(k1, k2, cu1) == results[i++]);
//
//
//  //BISPECTRUM CONFIGURATIONS
//  BOOST_TEST(triangle[0][0]== results[i++]);
//  BOOST_TEST(triangle[0][1]== results[i++]);
//  BOOST_TEST(triangle[0][2]== results[i++]);
//  BOOST_TEST(triangle[1][0]== results[i++]);
//  BOOST_TEST(triangle[1][1]== results[i++]);
//  BOOST_TEST(triangle[1][2]== results[i++]);
//  BOOST_TEST(triangle[2][0]== results[i++]);
//  BOOST_TEST(triangle[2][1]== results[i++]);
//  BOOST_TEST(triangle[2][2]== results[i++]);
//
//
//  //REDUCED BISPECTRUM
//  BOOST_TEST((triangle2[0] + triangle2[1] + triangle2[2]).norm2() == 0);
//  BOOST_TEST(FDM::BR(p_sc, triangle[0], triangle[1], triangle[2], cu1) == results[i++]);
//  //1-LOOP BISPECTRUM
//  auto q = 2*triangle[0];
//  BOOST_TEST(FDM::B222 (p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu1) == results[i++]);
//  BOOST_TEST(FDM::B3211(p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu1) == results[i++]);
//  BOOST_TEST(FDM::B3212(p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu1) == results[i++]);
//  BOOST_TEST(FDM::B411 (p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu1) == results[i++]);
//
//  //B1L CUBA INTEGRATION CONTEXT
//  double phi = 0;
//  double cosine_theta = 0.5;
//  double r = 2.0;
//  B1L_IC ic(triangle[0], triangle[1], triangle[2], p_sc, cu1);
//  BOOST_TEST(FDM::B1Li_test(phi, cosine_theta, r, 0.5, 0.4, 0.3, &ic) == results[i++]);
//
//  //B1L CUBA INTEGRATION
//  FDM::NLBispectrum b1l_sc(p_sc, 1./5, 16./17);
//  BOOST_TEST(b1l_sc(triangle[0], triangle[1], triangle[2], cu1) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(5.0) ));
//  FDM::NLBispectrum b1l(p, 1e-4, 1.0);
//  BOOST_TEST(b1l(triangle[0], triangle[1], triangle[2], cu1) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(8.0) ));
//
//  //REDUCED BISPECTRUM
//  FDM::NLSpectrum p1l(p, 1e-4, 1.0);
//  FDM::NLRBispectrum b1lr(p, p1l, b1l);
//  BOOST_TEST(b1lr(triangle[0], triangle[1], triangle[2], cu1) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(5.0) ));
//}


BOOST_AUTO_TEST_CASE(num_bispectrumtest, * utf::tolerance(1e-4))
{
  vec k1(3), k2(3), k3(3);

  k1[0] = 2;
  k1[1] = 0;
  k1[2] = 0;

  k2[0] = 2;
  k2[1] = 1;
  k2[2] = 0;

  k3[0] = 1;
  k3[1] = 0;
  k3[2] = 0;

  std::vector <float> results = {1.725103983654164, 1.725103983654164, 
  0.,0.,1., 0.8660254,  0. , -0.5, -0.8660254, -0. , -0.5, 0.5474816027542845, 
  31.30356137271131 , 16.308895511263515 , -534.144993522024 , 791.2030301855106,
  -253245.53825594645, 5.641e+06, 18.92, 0.3175757583786859};
  int i = 0;

  //F2 MODE COUPLING
  BOOST_TEST(FDM::F2i(k1, k2, cu2) == results[i++]);
  BOOST_TEST(FDM::F2i(k1, k2, cu2) == results[i++]);


  //BISPECTRUM CONFIGURATIONS
  BOOST_TEST(triangle[0][0]== results[i++]);
  BOOST_TEST(triangle[0][1]== results[i++]);
  BOOST_TEST(triangle[0][2]== results[i++]);
  BOOST_TEST(triangle[1][0]== results[i++]);
  BOOST_TEST(triangle[1][1]== results[i++]);
  BOOST_TEST(triangle[1][2]== results[i++]);
  BOOST_TEST(triangle[2][0]== results[i++]);
  BOOST_TEST(triangle[2][1]== results[i++]);
  BOOST_TEST(triangle[2][2]== results[i++]);


  //REDUCED BISPECTRUM
  BOOST_TEST((triangle2[0] + triangle2[1] + triangle2[2]).norm2() == 0);
  BOOST_TEST(FDM::BR(p_sc, triangle[0], triangle[1], triangle[2], cu2) == results[i++]);
  //1-LOOP BISPECTRUM
  auto q = 2*triangle[0];
  BOOST_TEST(FDM::B222 (p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu2) == results[i++]);
  BOOST_TEST(FDM::B3211(p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu2) == results[i++]);
  BOOST_TEST(FDM::B3212(p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu2) == results[i++]);
  BOOST_TEST(FDM::B411 (p_sc, q, triangle[0], triangle[1], triangle[2], 0.5, 0.5, 0.5, cu2) == results[i++]);

  //B1L CUBA INTEGRATION CONTEXT
  double phi = 0;
  double cosine_theta = 0.5;
  double r = 2.0;
  B1L_IC ic(triangle[0], triangle[1], triangle[2], p_sc, cu2);
  BOOST_TEST(FDM::B1Li_test(phi, cosine_theta, r, 0.5, 0.4, 0.3, &ic) == results[i++]);

  //B1L CUBA INTEGRATION
  FDM::NLBispectrum b1l_sc(p_sc, 1./5, 16./17);
  BOOST_TEST(b1l_sc(triangle[0], triangle[1], triangle[2], cu2) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(5.0) ));
  FDM::NLBispectrum b1l(p, 1e-4, 1.0);
  BOOST_TEST(b1l(triangle[0], triangle[1], triangle[2], cu2) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(8.0) ));

  //REDUCED BISPECTRUM
  FDM::NLSpectrum p1l(p, 1e-4, 1.0);
  FDM::NLRBispectrum b1lr(p, p1l, b1l);
  BOOST_TEST(b1lr(triangle[0], triangle[1], triangle[2], cu2) == results[i++], tt::tolerance( tt::fpc::percent_tolerance(5.0) ));
}