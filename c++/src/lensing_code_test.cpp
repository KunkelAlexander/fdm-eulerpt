#define BOOST_TEST_MODULE LensingCodeTest

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iterator>

#include "spectrum.h"
#include "tomo.h"


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;


#define CHECK_CLOSE_COLLECTION(a, ae, b, be, tolerance) { \
    BOOST_REQUIRE_EQUAL(std::distance(a, ae), std::distance(b, be)); \
    for(; a != ae; ++a, ++b) { \
        BOOST_CHECK_CLOSE(*a, *b, tolerance); \
    } \
}

struct DATA *data;


struct F {
  F()  { 
        BOOST_TEST_MESSAGE( "setup fixture" ); 
        data = (struct DATA *)calloc(1, sizeof(struct DATA));
        gdata = data;
        data->cosmology = (struct COSMOLOGY *)calloc(1, sizeof(struct COSMOLOGY));
        gcosmo = data->cosmology;

        INIT_COSMOLOGY(data);
        PREPARE_NEW_COSMOLOGY(data);
        SPLIT_UP_GALAXY_SAMPLE(data);
      }
  ~F() { 
        BOOST_TEST_MESSAGE( "teardown fixture" );
        free(data);
  }
};



BOOST_AUTO_TEST_SUITE(s, * utf::fixture<F>())

BOOST_AUTO_TEST_CASE(testmodelensing, * utf::tolerance(1e-7))
{
    //Define streams for comparing files
    std::ifstream ifs1, ifs2;

    //mode lensing
    flag_mode = mode_lensing;
    flag_nonlinear = off;
    COMPUTE_S2N_SPECTRUM(data);
    SAVE_SIGNIFICANCE2DISK(data);

    ifs1.open("UnmodifiedTomo/data/s2n_lensing_1.data");
    ifs2.open("data/s2n_lensing_1.data");

    std::istream_iterator<double> b1(ifs1), e1;
    std::istream_iterator<double> b2(ifs2), e2;

    BOOST_CHECK_EQUAL_COLLECTIONS(b1, e1, b2, e2);

    ifs1.close();
    ifs2.close();
}

BOOST_AUTO_TEST_CASE(testmodeinflation, * utf::tolerance(1e-7) )
{
    //Define streams for comparing files
    std::ifstream ifs1, ifs2;

    //mode inflation
    flag_mode = mode_inflation;
    flag_nonlinear = off;

    //Set flag type to local at this point in order to compute CDM and not FDM lensing spectra
    flag_type = spectrum_type::type_local;
    COMPUTE_LENSING_SPECTRA(data);

    flag_type = spectrum_type::type_local;
    COMPUTE_S2N_BISPECTRUM(data);

    flag_type = spectrum_type::type_equil;
    COMPUTE_S2N_BISPECTRUM(data);

    flag_type = spectrum_type::type_ortho;
    COMPUTE_S2N_BISPECTRUM(data);

    SAVE_SIGNIFICANCE2DISK(data);

    ifs1.open("UnmodifiedTomo/data/s2n_inflation_1.data");
    ifs2.open("data/s2n_inflation_1.data");

    std::istream_iterator<double> b1(ifs1), e1;
    std::istream_iterator<double> b2(ifs2), e2;

    BOOST_CHECK_EQUAL_COLLECTIONS(b1, e1, b2, e2);

    ifs1.close();
    ifs2.close();
}

BOOST_AUTO_TEST_CASE(testmodesuyama, * utf::tolerance(1e-5) )
{
    //Define streams for comparing files
    std::ifstream ifs1, ifs2;

    //case mode_suyama
    flag_mode = mode_suyama;
    flag_nonlinear = off;
    COMPUTE_LENSING_SPECTRA(data);
    flag_type = spectrum_type::type_local;
    COMPUTE_S2N_BISPECTRUM(data);
    COMPUTE_S2N_TRISPECTRUM(data);
    SAVE_SIGNIFICANCE2DISK(data);


    ifs1.open("UnmodifiedTomo/data/s2n_suyama_1.data");
    ifs2.open("data/s2n_suyama_1.data");
    
    std::istream_iterator<double> b1(ifs1), e1;
    std::istream_iterator<double> b2(ifs2), e2;

    CHECK_CLOSE_COLLECTION(b1, e1, b2, e2, 1e-5);

    ifs1.close();
    ifs2.close();
}

BOOST_AUTO_TEST_CASE(testmodetomography, * utf::tolerance(1e-5) )
{
    //Define streams for comparing files
    std::ifstream ifs1, ifs2;

    //case tomography
    flag_mode = mode_tomography;
    flag_nonlinear = on;
    COMPUTE_LENSING_SPECTRA(data);
    SAVE_LENSING_SPECTRA2DISK(data);
    COMPUTE_LENSING_CORRELATION(data);
    SAVE_LENSING_CORRELATION2DISK(data);
    COMPUTE_LENSING_CCOEFFICIENTS(data);
    SAVE_LENSING_CCOEFFICIENTS2DISK(data);
    

    ifs1.open("UnmodifiedTomo/data/gamma_1.data");
    ifs2.open("data/gamma_1.data");
    std::istream_iterator<double> b1(ifs1), e1;
    std::istream_iterator<double> b2(ifs2), e2;
    CHECK_CLOSE_COLLECTION(b1, e1, b2, e2, 1e-5);
    ifs1.close();
    ifs2.close();

    ifs1.open("UnmodifiedTomo/data/kappa_1.data");
    ifs2.open("data/kappa_1.data");
    b1 = std::istream_iterator<double> (ifs1);
    b2 = std::istream_iterator<double> (ifs2);
    e1 = std::istream_iterator<double> ();
    e2 = std::istream_iterator<double> ();
    CHECK_CLOSE_COLLECTION(b1, e1, b2, e2, 1e-5);
    ifs1.close();
    ifs2.close();

    ifs1.open("UnmodifiedTomo/data/ccoefficient_nl_1.data");
    ifs2.open("data/ccoefficient_nl_1.data");
    b1 = std::istream_iterator<double> (ifs1);
    b2 = std::istream_iterator<double> (ifs2);
    e1 = std::istream_iterator<double> ();
    e2 = std::istream_iterator<double> ();
    CHECK_CLOSE_COLLECTION(b1, e1, b2, e2, 1e-5);
    ifs1.close();
    ifs2.close();
}

BOOST_AUTO_TEST_SUITE_END()