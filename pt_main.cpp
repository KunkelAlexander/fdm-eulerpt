#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/format.hpp>

#include "include/spectrum.h"
#include "include/constants.h"

using namespace std;


/****
 * GROWTH FACTORS
 ****/

void computeGrowthFactor(double k, int fdm_mass_id, const std::string filename, size_t n)
{
    std::vector<double> x1;
    double y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;
    x1 = pyLogspace(-2, 0, n);

    CDM::D_spline d_plus_cdm_num(true);
    FDM::D_spline d_plus_fdm_num(const_eta_in, fdm_mass_id, true);
    CDM::D_spline d_minus_cdm_num(false);
    FDM::D_spline d_minus_fdm_num(const_eta_in, fdm_mass_id, false);

    std::ofstream outdata;                                 
    outdata.open("data/growth/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing CDM P1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    // using Boost.Format
    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% %|75t|%6% %|90t|%7% %|105t|%8% %|120t|%9% %|135t|%10% %|150t|%11%\n") % "a" % "+_cdm_ana" % "+_fdm_ana" % "+_fdm_ren" % "+_cdm_num" % "+_fdm_num" % "-_cdm_ana" % "-_fdm_ana" % "-_fdm_ren" % "-_cdm_num" % "-_fdm_num" ;

    for (double a : x1)
    {
        double eta = eta_from_a(a);
        y1  = CDM::D(eta, const_eta_in);
        y2  = FDM::D_plus_analytical  (k, eta, const_eta_in, fdm_masses[fdm_mass_id]);
        y3  = FDM::D_plus_renormalised(k, eta, const_eta_in, fdm_masses[fdm_mass_id]);
        y4  = d_plus_cdm_num(a);
        y5  = d_plus_fdm_num(k, eta);
        y6  = pow((a/const_a_in), -1.5);
        y7  = FDM::D_minus_analytical  (k, eta, const_eta_in, fdm_masses[fdm_mass_id]);
        y8  = FDM::D_minus_renormalised(k, eta, const_eta_in, fdm_masses[fdm_mass_id]);
        y9  = d_minus_cdm_num(a);
        y10 = d_minus_fdm_num(k, eta);
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% %|75t|%6% %|90t|%7% %|105t|%8% %|120t|%9% %|135t|%10% %|150t|%11%\n") % a % y1 % y2 % y3 % y4 % y5 % y6 % y7 % y8 % y9 % y10;
    }
    outdata.close();
}


void computeGrowthFactors(int fdm_mass_id)
{
    computeGrowthFactor(1e-3, fdm_mass_id, cosmo_string+"_k = 1e-3_" + fdm_mass_strings[fdm_mass_id], 2000);
    computeGrowthFactor(5,    fdm_mass_id, cosmo_string+"_k = 5_"    + fdm_mass_strings[fdm_mass_id], 2000);
    computeGrowthFactor(25,   fdm_mass_id, cosmo_string+"_k = 25_"   + fdm_mass_strings[fdm_mass_id], 2000);
    computeGrowthFactor(100,  fdm_mass_id, cosmo_string+"_k = 100_"  + fdm_mass_strings[fdm_mass_id], 2000);
}

void computeFDMSuppression(double a, int fdm_mass_id, const std::string filename, size_t n)
{
    std::vector<double> x1, y1, y2, y3, y4;
    x1 = pyLogspace(0, 2, n);

    CDM::D_spline d1(true);
    FDM::D_spline d2(const_eta_in, fdm_mass_id, true);

    double eta = eta_from_a(a);

    for (double k : x1)
    {
        y1.push_back(FDM::D_plus_analytical(k, eta, eta_from_a(const_a_in),   fdm_masses[fdm_mass_id]) / CDM::D(eta, const_eta_in));
        y2.push_back(FDM::D_plus_renormalised(k, eta, eta_from_a(const_a_in), fdm_masses[fdm_mass_id]) / CDM::D(eta, const_eta_in));
        y3.push_back(d2(k, eta) / d1(a));
    }

    std::ofstream outdata;                                          
    outdata.open("data/fdm_suppression/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing CDM P1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5%\n") % "k" % "fdm_ana" % "fdm_ren" % "fdm_num" % "fdm_fit";

    for (size_t i = 0; i < x1.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5%\n") % x1[i] % y1[i] % y2[i] % y3[i] % y4[i];
    }
    outdata.close();
}

/****
 * SPECTRA
 ****/

/****
 * Tree-level spectrum
 ****/

struct NLScaleUtil_C {
    NLScaleUtil_C(const Spectrum &P, const CosmoUtil& cu) : P(P), cu(cu) {}
    const Spectrum &P;
    const CosmoUtil& cu;
};

void computeNonlinearScale(const CosmoUtil &cu, const Spectrum &P, const std::string filename, size_t n)
{
    std::cout << "Compute nonlinear scale by integrating tree power spectrum" << std::endl;

    std::vector<double> x(n), y1(n);
    x = pyLogspace(-3, 3, n, 10);


    gsl_integration_workspace *w = gsl_integration_workspace_alloc(INTEGRATION_WORKSPACE_SIZE);

    // Define integrand
    auto integrand = [](double k, void *params)
    {
      NLScaleUtil_C *c = (struct NLScaleUtil_C *)params;
      return c->P(k, c->cu) * k * k;
    };

    gsl_function F;
    F.function = integrand;

    NLScaleUtil_C c(P, cu);
    F.params = (void *)&c;

    double result, error;

    for (size_t i = 0; i < n; ++i)
    {
        double k = x[i];

        // Perform integration
        gsl_integration_qags(&F, 1e-4, k, 0, 1e-4, INTEGRATION_WORKSPACE_SIZE,
                            w, &result, &error);

        y1[i] = result / (2 * M_PI * M_PI);
    }


    gsl_integration_workspace_free(w);

    std::cout << "Done computing nonlinear scales" << std::endl;

    std::ofstream outdata;                                        
    outdata.open("data/spectrum/nonlinear_scales/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing nonlinear scales could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    // using Boost.Format
    outdata << boost::format("%1% %|15t|%2%\n") % "#k in h/Mpc" % "sigma^2";

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2%\n") % x[i] % y1[i];
    }
    outdata.close();
}

void computeSpectrum(const CosmoUtil &cu, const Spectrum &P, const std::string filename, size_t n)
{
    std::cout << "Create tree power spectrum" << std::endl;

    std::vector<double> x(n), y1(n);
    
    x = pyLogspace(-4, 3, n, 10);

    for (size_t i = 0; i < n; ++i)
    {
        double k = x[i];
        // Compute linear spectrum
        y1[i] = P(k, cu);
    }

    std::cout << "Done creating tree power spectrum" << std::endl;

    std::ofstream outdata;                                        
    outdata.open("data/spectrum/tree/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing FDM P1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    // using Boost.Format
    outdata << boost::format("%1% %|15t|%2%\n") % "#k in Mpc^-1" % "P0";

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2%\n") % x[i] % y1[i];
        ;
    }
    outdata.close();
}


/****
 * Compute loop-level spectra for n datapoints with radial integration limits specified by r_low and r_high
 ****/

void computeLoopPowerSpectrum(const CosmoUtil &cu_l, const CosmoUtil &cu_nl, const Spectrum &P, const Spectrum &P1L, const std::string &filename, size_t n)
{
    std::cout << "Create loop power spectrum" << std::endl;

    std::vector<double> x(n), y1(n), y2(n), y2_err(n), y2_Q(n);
    
    x = pyLogspace(-3, 3, n);

    for (size_t i = 0; i < n; ++i)
    {
        std::cout << i << "/" << n <<  ": " << std::setw(10) << "k = "<<x[i]<< " ";

        double k = x[i];

        // Compute linear spectrum
        y1[i] = P(k, cu_l);
        // Compute nonlinear spectrum
        vec result(P1L.get_result(k, cu_nl));
        y2[i]     = y1[i] + result[0]; // Linear spectrum + integration mean
        y2_err[i] = result[1];         // Standard deviation of MC integration
        y2_Q[i]   = result[2];         // Chi squared value of MC integration


        std::cout << std::setw(10) << "P0 = "  << y1[i] << std::setw(10) << " P1 = "  << result[0] << " +- " << result[1] << "\n";
    }

    std::ofstream outdata;                                        
    outdata.open("data/spectrum/loop/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing P1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    // using Boost.Format
    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% \n") % "#k in Mpc^-1" % "P0" % "P0 + P1L" % "P1L_err" % "Probability";

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% \n") % x[i] % y1[i] % y2[i] % y2_err[i] % y2_Q[i];
    }
    outdata.close();
}

void computeLoopPowerSpectrumSpline(CosmoUtil &cu_l, CosmoUtil &cu_nl, const Spectrum &P, const Spectrum &P1L, const std::string &filename)
{
    std::cout << "Create loop power spectrum spline" << std::endl;

    size_t nk = p1l_spline_nk;
    size_t na = p1l_spline_na;

    std::vector<double> ks(nk), as(na);
    ks = pyLogspace(-4, 3, nk - 1, 10);
    as = pyLogspace(-2, 0, na - 1);
    ks.push_back(pow(10, 3));
    as.push_back(1);

    double Pl[na][nk], Pnl[na][nk], sdev[na][nk];

    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nk; ++j)
        {
            double a = as[i];
            double k = ks[j];

            std::cout << j + i * nk << "/" << na * nk << "\n";

            cu_l.eta  = eta_from_a(a);
            cu_nl.eta = eta_from_a(a);

            // Compute linear spectrum
            Pl[i][j] = P(k, cu_l);
            // Compute nonlinear spectrum
            vec result(P1L.get_result(k, cu_nl));
            Pnl[i][j] = result[0];  // integration mean
            sdev[i][j] = result[1]; // Standard deviation of MC integration
        }
    }

    std::ofstream out;                             
    out.open("splines/" + filename + ".dat"); 

    if (!out)
    { // file couldn't be opened
        std::cerr << "Error: file for storing CDM P1L spline could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nk; ++j)
        {
            out << i * nk + j << " " << eta_from_a(as[i]) << " " << ks[j] << " " << Pl[i][j] << " " << Pnl[i][j] << " " << sdev[i][j];
            if (i * j != (na - 1) * (nk - 1))
            {
                out << "\n";
            }
        }
    }

    out.close();
}

/****
 * BISPECTRA
 ****/

/****
 * Show CDM and FDM tree-level reduced bispectra for different masses
 ****/
void computeAngularReducedBispectrum(const CosmoUtil &cu, const Bispectrum &B, const std::string filename, size_t n = 3, double r1 = .2, double r2 = .1)
{
    std::cout << "Create reduced tree bispectrum for m = " << cu.m << std::endl;

    std::vector<double> x(n), y1(n), y2(n);    
    x = pyLinspace(0, M_PI, n-1);
    x.push_back(M_PI - 1e-2);
    for (size_t i = 0; i < n; ++i)
    {
        auto triangle = generateTriangle(x[i], r1, r2);
        y1[i] = B(triangle[0], triangle[1], triangle[2], cu);
        y2[i] = y1[i] / ST(B.getSpectrum(), triangle[0], triangle[1], triangle[2], cu);
    }

    std::ofstream outdata;                                          
    outdata.open("data/bispectrum/angular/tree/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing tree bispectrum could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2% %|30t|%3%\n") % "Angle" % "BT" % "BTR";

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3%\n") % x[i] % y1[i] % y2[i];
    }
    outdata.close();
}

void computeEquilateralReducedBispectrum(const CosmoUtil &cu, const Bispectrum &B, const std::string filename, size_t n = 3)
{
    std::cout << "Create equilateral bispectra for m = " << cu.m << std::endl;

    std::vector<double> x(n), y1(n), y2(n), y3(n), y4(n), y5(n), y6(n);
    x = pyLogspace(-4, 3, n-1);
    x.push_back(1000);
    for (size_t i = 0; i < n; ++i)
    {

        auto triangle = generateTriangle();
        triangle[0] = triangle[0] * x[i];
        triangle[1] = triangle[1] * x[i];
        triangle[2] = triangle[2] * x[i];

        std::cout << i + 1 << "/"<<n<<": Triangle with side lengths: " << (triangle[0]).norm2() << " " << (triangle[1]).norm2() << " " << (triangle[2]).norm2() << ": ";

        vec result = B.get_result(triangle[0], triangle[1], triangle[2], cu);
        std::cout << result[0] << "+-" << result[1] << "\n";
        y1[i] = B.getSpectrum().operator()(x[i], cu);
        y2[i] = result[0];
        y3[i] = y2[i] / pow(y1[i], 1.5);
        y4[i] = y2[i] / ST(B.getSpectrum(), triangle[0], triangle[1], triangle[2], cu);
        y5[i] = result[1];
        y6[i] = result[2];
    }

    std::ofstream outdata;                                                 
    outdata.open("data/bispectrum/equilateral/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing tree bispectrum could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% %|75t|%6% %|90t|%7%\n") % "Momentum" % "Spectrum" % "Bispectrum" % "B(k,k,k)/P(k)^(3/2)" % "red. Bispectrum" % "B_error" % "B_Q";

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4% %|60t|%5% %|75t|%6% %|90t|%7%\n") % x[i] % y1[i] % y2[i] % y3[i] % y4[i] % y5[i] % y6[i];
    }
    outdata.close();
}

/****
 * Comute loop-level reduced bispectrum for n datapoints and given linear spectrum P
 ****/
void computeAngularReducedLoopBispectrum(const CosmoUtil &cu, const Bispectrum &B1Lred, const std::string filename, size_t n = 3, double r1 = 2, double r2 = 1)
{
    std::cout << "Create reduced loop bispectrum" << std::endl;

    std::vector<double> x(n), y1(n), y2(n), y3(n), y4(n), y5(n), y6(n), y7(n), y8(n);
    x = pyLinspace(0, M_PI, n-1);
    x.push_back(M_PI - 1e-2);
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << i +1<<"/"<<n<<"\n";

        auto triangle = generateTriangle(x[i], r1, r2);
        vec result(B1Lred.get_result(triangle[0], triangle[1], triangle[2], cu));
        y1[i] = result[0];             // Reduced bispectrum at tree level
        y2[i] = result[1];             // Reduced bispectrum 1-loop correction
        y3[i] = result[2];             // Error of 1-loop correction
    }

    std::ofstream outdata;                                         
    outdata.open("data/bispectrum/loop/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing CDM B1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4%\n") % "Angle" % "BTR" % "BNLR" % "BNLR_err" ;

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4%\n") % x[i] % y1[i] % y2[i] % y3[i];
    }
    outdata.close();
}


/****
 * Comute loop-level reduced bispectrum for n datapoints and given linear spectrum P
 ****/
void computeEquilateralReducedLoopBispectrum(const CosmoUtil &cu, const Bispectrum &B1Lred, const std::string filename, size_t n = 3)
{
    std::cout << "Create reduced loop bispectrum" << std::endl;

    std::vector<double> x(n), y1(n), y2(n), y3(n), y4(n), y5(n), y6(n), y7(n), y8(n);
    x = pyLogspace(-4, 3, n-1);
    x.push_back(1000);

    for (size_t i = 0; i < n; ++i)
    {
        std::cout << i +1<<"/"<<n<<"\n";

        auto triangle = generateTriangle();
        triangle[0] = triangle[0] * x[i];
        triangle[1] = triangle[1] * x[i];
        triangle[2] = triangle[2] * x[i];

        vec result(B1Lred.get_result(triangle[0], triangle[1], triangle[2], cu));
        y1[i] = result[0];             // Reduced bispectrum at tree level
        y2[i] = result[1];             // Reduced bispectrum 1-loop correction
        y3[i] = result[2];             // Error of 1-loop correction
    }

    std::ofstream outdata;                                         
    outdata.open("data/bispectrum/loop/reduced/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing CDM B1L could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4%\n") % "Angle" % "BTR" % "BNLR" % "BNLR_err" ;

    for (size_t i = 0; i < x.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2% %|30t|%3% %|45t|%4%\n") % x[i] % y1[i] % y2[i] % y3[i];
    }
    outdata.close();
}

/****
 * TRISPECTRA
 ****/

/****
 * Compute equilateral trispectra at tree-level
 ****/
void computeEquilateralTrispectra(const CosmoUtil &cu, const Trispectrum &T, const std::string filename, size_t n = 3)
{

    std::vector<double> r(n), y1(n);
    r = pyLogspace(-4, 3, n-1);
    r.push_back(1000);
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << "Rectangle with r = "<<r[i]<<" ";
        auto rect = generateRectangle(true, r[i], 0, 0, 0, 0);
        y1[i] = T(rect[0], rect[1], rect[2], rect[3], cu);
        std::cout << "T0 = "<<y1[i]<<"\n";
    }

    std::ofstream outdata;                                                  
    outdata.open("data/trispectrum/equilateral/" + filename + ".dat"); 
    if (!outdata)
    { // file couldn't be opened
        std::cerr << "Error: file for storing tree trispectrum could not be opened" << std::endl;
        throw std::invalid_argument("Invalid filename");
    }

    outdata << boost::format("%1% %|15t|%2%\n") % "k [Mpc^-1]" % "T";

    for (size_t i = 0; i < r.size(); ++i)
    {
        outdata << boost::format("%1% %|15t|%2%\n") % r[i] % y1[i];
    }
    outdata.close();
}



void computeTreeSpectrumDifference(int fdm_mass_id)
{
    std::cout << "Load CDM CAMB spectrum \n";
    CAMBSpectrum P_cdm(cdm_camb_path);
    std::cout << "Load FDM CAMB spectrum \n";
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::CDM_Numerical_CosmoUtil cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);

    size_t  n = 20;
    std::vector<double> x(n);
    x = pyLogspace(-4, 3, n, 10);

    for (size_t i = 0; i < n; ++i)
    {
        double k = x[i];

        double cdm = P_cdm(k, cu_cdm);
        double fdm = P_fdm(k, cu_fdm);
        double abserr = cdm - fdm;
        double relerr = abserr/cdm;
        std::cout << "k: " << k << " CDM: " << cdm << " FDM: " << fdm << " ABSERR: " << abserr <<  " RELERR: " << relerr  << "\n";
    }
    
}

void computeNonlinearScales(int fdm_mass_id) 
{
    int N = 1000;

    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    CDM::CDM_Numerical_CosmoUtil cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);

    computeNonlinearScale(cu_cdm, P_cdm, "cdm_z=0", N);
    computeNonlinearScale(cu_fdm, P_fdm, fdm_mass_strings[fdm_mass_id]+"_z=0", N);

    cu_cdm.eta = const_eta_in;
    cu_fdm.eta = const_eta_in;
    computeNonlinearScale(cu_cdm, P_cdm, "cdm_z=99", N);
    computeNonlinearScale(cu_fdm, P_fdm, fdm_mass_strings[fdm_mass_id]+"_z=99", N);
}

void computeTreeSpectra(int fdm_mass_id)
{
    cout << "Load CDM CAMB spectrum \n";
    CAMBSpectrum P_cdm(cdm_camb_path);
    cout << "Load FDM CAMB spectrum \n";
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);


    //Compute CDM tree spectra
    #if 0
    cout << "Integrate CDM growth factor \n";
    CDM::CDM_Numerical_CosmoUtil cu_cdm(const_eta_fin, const_eta_in);

    cout << "Compute CDM spectra\n";
    compute_spectrum(cu_cdm, P_cdm, "cdm", N);
    #endif 

    //Compute FDM tree spectra
    #if 0
    cout << "Integrate FDM growth factor \n";
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm_num(const_fdm_mass, const_eta_fin, const_eta_in);
    FDM::FDM_Fit_CosmoUtil           cu_fdm_fit(const_fdm_mass, const_eta_fin, const_eta_in);

    cout << "Compute FDM spectra\n";
    computeSpectrum(cu_fdm_num, P_fdm, "fdm_num_"+mass_string, N);
    computeSpectrum(cu_fdm_fit, P_fdm, "fdm_fit_"+mass_string, N);
    #endif

    //Compute FDM tree spectra
    #if 0
    cout << "Integrate FDM growth factor \n";
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm_num(const_fdm_mass, const_eta_fin, const_eta_in);
    FDM::FDM_Fit_CosmoUtil           cu_fdm_fit(const_fdm_mass, const_eta_fin, const_eta_in);

    cout << "Compute FDM spectra\n";
    computeSpectrum(cu_fdm_num, P_fdm, "fdm_num_"+mass_string, N);
    computeSpectrum(cu_fdm_fit, P_fdm, "fdm_fit_"+mass_string, N);
    #endif

    #if 0
    CDM::CDM_Numerical_CosmoUtil  cu_cdm(const_eta_fin, const_eta_in);
    FDM::FDM_Analytical_CosmoUtil cu_fdm(const_fdm_mass, const_eta_fin, const_eta_in);

    SplineSpectrum P_cdm_spl("splines/cdm.dat");
    computeSpectrum(cu_cdm,   P_cdm_spl, "cdm_cdm_for_eq",     N);
    SplineSpectrum P_fdm_spl1("splines/fdm_m23.dat");
    computeSpectrum(cu_fdm,   P_fdm_spl1, "fdm_m23_for_eq", N);
    SplineSpectrum P_fdm_spl2("splines/fdm_m22.dat");
    computeSpectrum(cu_fdm,   P_fdm_spl2, "fdm_m22_for_eq", N);
    SplineSpectrum P_fdm_spl3("splines/fdm_m21.dat");
    computeSpectrum(cu_fdm,   P_fdm_spl3, "fdm_m21_for_eq", N);
    #endif
}



void computeTreeDimensionlessEquilateral(int fdm_mass_id, size_t N = 100)
{

    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);

    CDM::TreeBispectrum B_cdm(P_cdm);
    FDM::VegasTreeBispectrum B_fdm(P_fdm);

    CDM::CDM_Numerical_CosmoUtil     cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);

    computeEquilateralReducedBispectrum(cu_cdm, B_cdm, "cdm", N);
    computeEquilateralReducedBispectrum(cu_fdm, B_fdm, "fdm_"+fdm_mass_strings[fdm_mass_id], N);
}

void computeLoopSpectra(int fdm_mass_id)
{
    // Use special constructor that initialises vegas integrator by running one integration with k0
    double k0 = 1e-4;
    int N = 50;

    #if 1
        cout << "Load CDM CAMB spectrum \n";
        CAMBSpectrum P_cdm(cdm_camb_path);
        CDM::CDM_Numerical_CosmoUtil cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
        CDM::NLSpectrum P1L_cdm(P_cdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, k0, cu_cdm, 1);
        cout << "Integrate CDM Loop corrections. \n";
        computeLoopPowerSpectrum(cu_cdm,   cu_cdm,     P_cdm, P1L_cdm, "cdm", N);
    #endif
    
    #if 1
        cout << "Load FDM CAMB spectrum \n";
        CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
        FDM::FDM_SemiNumerical_CosmoUtil cu_fdm_num(fdm_mass_id, const_eta_fin, const_eta_in);
        FDM::NLSpectrum P1L_fdm(P_fdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, k0, cu_fdm_num, 0);

        cout << "Integrate FDM Loop corrections. \n";
        computeLoopPowerSpectrum(cu_fdm_num, cu_fdm_num, P_fdm, P1L_fdm, "fdm_num_num_"+fdm_mass_strings[fdm_mass_id], N);
    #endif
}


void computeLoopSpectrumSplines()
{
    size_t N = 100;

    #if 0
    cout << "Integrate CDM growth factor \n";
    CDM::CDM_Numerical_CosmoUtil cu_cdm(const_eta_fin, const_eta_in);

    cout << "Load CDM CAMB spectrum \n";
    CAMBSpectrum P_cdm(cdm_camb_path);

    // Used for initialising VEGAS integrator
    double k0 = 1e-4;

    cout << "Compute CDM P1L spline \n";
    CDM::NLSpectrum P1L_cdm(P_cdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, k0, cu_cdm, 2);
    //compute_P1L_spline(cu_cdm, cu_cdm, P_cdm, P1L_cdm, "cdm");

    cu_cdm.eta    = const_eta_fin;

    SplineSpectrum P_cdm_spl("splines/cdm.dat");

    compute_spectrum(cu_cdm,   P_cdm_spl, "cdm_spline",     N);
    #endif 

    #if 0

    cout << "Load FDM CAMB spectrum \n";
    CAMBSpectrum P_fdm(fdm_camb_path);

    double k0 = 1e-4;

    FDM::FDM_SemiNumerical_CosmoUtil  cu_fdm(const_fdm_mass, const_eta_fin, const_eta_in);

    cout << "Compute FDM P1L spline \n";
    FDM::NLSpectrum P1L_fdm(P_fdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, k0, cu_fdm, 2);
    computeLoopPowerSpectrumSpline(cu_fdm, cu_fdm, P_fdm, P1L_fdm, "fdm_"+mass_string);

    cu_fdm.eta  = const_eta_fin;
    cu_fdm.eta  = const_eta_fin;

    SplineSpectrum P_fdm_spl("splines/fdm_"+mass_string+".dat");

    cout << "Integrate FDM growth factor \n";
    computeSpectrum(cu_fdm, P_fdm_spl, "fdm_spline_"+mass_string, N);
    
    #endif
}

#if 0

void computeLoopSpectrumAnimation() {
    std::vector<double> times = pyLinspace(0.2, 2, 150);
    //We can use the CDM cosmo util for FDM in this case because the spline spectra do not rely on D+ and the mass parameter
    CDM::CDM_Numerical_CosmoUtil     cu_cdm(const_eta_fin, const_eta_in);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(const_fdm_mass, const_eta_fin, const_eta_in);

    CAMBSpectrum P_cdm(cdm_camb_path);
    SplineSpectrum P1L_cdm(cdm_loop_spectrum_path);
    for (double eta : times) {
        cu_cdm.eta = eta;
        computeSpectrum(cu_cdm, P_cdm  , "animations/tree_cdm/" + std::to_string(z_from_eta(eta)), 300);
        computeSpectrum(cu_cdm, P1L_cdm, "animations/loop_cdm/" + std::to_string(z_from_eta(eta)), 300);
    }

    CAMBSpectrum P_fdm(fdm_camb_path);
    SplineSpectrum P1L_fdm(fdm_loop_spectrum_path);
    for (double eta : times) {
        cu_fdm.eta = eta;
        computeSpectrum(cu_fdm, P_fdm,   "animations/tree_" + fdm_string + "/" + std::to_string(z_from_eta(eta)), 300);
        computeSpectrum(cu_fdm, P1L_fdm, "animations/loop_" + fdm_string + "/" + std::to_string(z_from_eta(eta)), 300);
    }
}

#endif 

void computeTreeBispectra(int fdm_mass_id)
{
    int N = 100;

    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::CDM_Numerical_CosmoUtil     cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::TreeBispectrum B_cdm(P_cdm);
    FDM::TreeBispectrum B_fdm(P_fdm);

    computeAngularReducedBispectrum(cu_cdm, B_cdm, "cdm_r1=20_r2=10", N, 20, 10);
    computeAngularReducedBispectrum(cu_fdm, B_fdm, "fdm_r1=20_r2=10_"+fdm_mass_strings[fdm_mass_id], N, 20, 10);
    computeAngularReducedBispectrum(cu_cdm, B_cdm, "cdm_r1=02_r2=01", N, .2, .1);
    computeAngularReducedBispectrum(cu_fdm, B_fdm, "fdm_r1=02_r2=01_"+fdm_mass_strings[fdm_mass_id], N, .2, .1);
}

//Gives exactly 0 for low k only if we use the CDM mode coupling, FDM mode coupling consistently gives ~1 percent difference even for low k. Why?
void computeTreeBispectrumDifference(int fdm_mass_id)
{
    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::CDM_Numerical_CosmoUtil     cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::TreeBispectrum B_cdm(P_cdm);
    FDM::TreeBispectrum B_fdm(P_fdm);

    size_t  n = 10;
    std::vector<double> x(n), r1(3), r2(3);
    x = pyLinspace(0, M_PI, n-1);
    x.push_back(M_PI - 1e-2);
    r1[0] = 1e-3;
    r2[0] = 2e-3;
    r1[1] = 1e-1;
    r2[1] = 2e-1;
    r1[2] = 1e1;
    r2[2] = 2e1;

    for (size_t j = 0; j < 3; ++j) 
    {
    for (size_t i = 0; i < n; ++i)
    {
        auto triangle = generateTriangle(x[i], r1[j], r2[j]);

        double cdm = B_cdm(triangle[0], triangle[1], triangle[2], cu_cdm);
        double fdm = B_fdm(triangle[0], triangle[1], triangle[2], cu_fdm);
        double abserr = cdm - fdm;
        double relerr = abserr/cdm;
        std::cout << boost::format("r1 %1% %|15t| r2 %2% %|30t| theta %3% %|45t| cdm %4% %|60t| fdm %5% %|75t| abserr %6% %|90t| relerr %7%\n") % r1[j] % r2[j] % x[i] % cdm % fdm % abserr % relerr;

    }
    }
    
}



void computeLoopBispectra(int fdm_mass_id)
{
    #if 0
    CAMBSpectrum   P_cdm(cdm_camb_path);
    CDM::NLSpectrum P_cdm_1l(P_cdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, 5);
    //SplineSpectrum P_cdm_1l();
    CDM::CDM_Numerical_CosmoUtil cu_cdm(const_eta_fin, const_eta_in);
    CDM::TreeBispectrum B_cdm(P_cdm);
    CDM::NLBispectrum   B1L_cdm(P_cdm, B_cdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF);
    CDM::NLRBispectrum  B1L_red_cdm(P_cdm, P_cdm_1l, B1L_cdm);


    #if 0
    computeAngularReducedLoopBispectrum(cu_cdm, B1L_red_cdm, "cdm_r1=02_r2=01", 10, .2, .1);
    computeAngularReducedLoopBispectrum(cu_cdm, B1L_red_cdm, "cdm_r1=20_r2=10", 10, 20, 10);
    #endif

    #if 0
    computeEquilateralBispectrum(cu_cdm, P_cdm, B1L_cdm, "cdm_loop", 20);
    #endif

    #if 0
    computeEquilateralReducedLoopBispectrum(cu_cdm, B1L_red_cdm, "cdm_loop", 10);
    #endif

    #endif


    #if 1
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    FDM::NLSpectrum P_fdm_1l(P_fdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF, 5);
    //SplineSpectrum P_fdm_1l();
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);
    FDM::VegasTreeBispectrum B_fdm(P_fdm);
    FDM::NLBispectrum B1L_fdm(P_fdm, B_fdm, VEGAS_IR_CUTOFF, VEGAS_UV_CUTOFF);
    FDM::NLRBispectrum  B1L_red_fdm(P_fdm, P_fdm_1l, B1L_fdm);

    #if 0
    computeAngularReducedLoopBispectrum(cu_fdm, B1L_red_fdm, fdm_string + "_r1=02_r2=01", 10, .2, .1);
    computeAngularReducedLoopBispectrum(cu_fdm, B1L_red_fdm, fdm_string + "_r1=20_r2=10", 10, 20, 10);
    #endif
    #if 0
    computeEquilateralBispectrum(cu_fdm, P_fdm, B1L_fdm, "fdm_loop_"+mass_string, 10);
    #endif

    #if 1
    computeEquilateralReducedLoopBispectrum(cu_fdm, B1L_red_fdm, "loop_" + fdm_mass_strings[fdm_mass_id], 10);
    #endif
    #endif


}

void computeTreeTrispectra(int fdm_mass_id)
{
    int N = 100;

    
    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    CDM::CDM_Numerical_CosmoUtil     cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);

    #if 0
    CDM::TreeTrispectrum T_cdm(P_cdm);
    std::cout << "compute CDM Tree Trispectra \n";
    {
        //boost::timer::auto_cpu_timer t;
        computeEquilateralTrispectra(cu_cdm, T_cdm, "cdm", N);
    }
    #endif

    #if 1
    FDM::TreeTrispectrumCuba T_fdm_vegas(P_fdm);
    std::cout << "compute FDM Tree Trispectra by integrating using Vegas \n";
    {
        //boost::timer::auto_cpu_timer t;
        computeEquilateralTrispectra(cu_fdm, T_fdm_vegas, fdm_mass_strings[fdm_mass_id], N);
    }
    #endif

    #if 0
    FDM::TreeTrispectrum T_fdm(P_fdm);
    gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

    std::cout << "compute FDM Tree Trispectra by integrating using gsl \n";
    {
        //boost::timer::auto_cpu_timer t;
        computeEquilateralTrispectra(cu_fdm, T_fdm, "fdm_gsl", N);
    }
    gsl_set_error_handler(old_handler);
    #endif
}


void computeTreeTrispectrumDifference(int fdm_mass_id)
{

    CAMBSpectrum P_cdm(cdm_camb_path);
    CAMBSpectrum P_fdm(fdm_camb_paths[fdm_mass_id]);
    CDM::CDM_Numerical_CosmoUtil     cu_cdm(fdm_mass_id, const_eta_fin, const_eta_in);
    FDM::FDM_SemiNumerical_CosmoUtil cu_fdm(fdm_mass_id, const_eta_fin, const_eta_in);
    CDM::TreeTrispectrum      T_cdm(P_cdm);
    FDM::TreeTrispectrumCuba      T_fdm(P_fdm);

    size_t n = 3; 
    std::vector<double> x1(n), x2(n), r1(3), r2(3), r3(3);
    x1 = pyLinspace(3e-1, M_PI, n-1);
    x2 = pyLinspace(3e-1, M_PI, n-1);
    x1.push_back(M_PI - 3e-1);
    x2.push_back(M_PI - 3e-1);
    r1[0] = 1e-3;
    r2[0] = 2e-3;
    r3[0] = 1e-4;
    r1[1] = 1e-1;
    r2[1] = 2e-1;
    r3[1] = 2e-2;
    r1[2] = 1e1;
    r2[2] = 2e1;
    r3[2] = 2e-1;

    for (size_t j = 0; j < 3; ++j) 
    {
    for (size_t i = 0; i < n; ++i)
    {
    for (size_t k = 0; k < n; ++k)
    {
        auto rect = generateRectangle(false, r1[j], r2[j], r3[j], x1[i], x2[k]);
        double cdm = T_cdm(rect[0], rect[1], rect[2], rect[3], cu_fdm);
        double fdm = T_fdm(rect[0], rect[1], rect[2], rect[3], cu_fdm);
        double abserr = cdm - fdm;
        double relerr = abserr/cdm;
        std::cout << boost::format("r %1% %|25t| theta1 %6% %|50t| theta2 %7% %|75t| cdm %2% %|100t| fdm %3% %|125t| abserr %4% %|150t| relerr %5%\n") % r1[j] % cdm % fdm % abserr % relerr % x1[i] % x2[k];
    }
    }
    }

}

int main()
{
    for (int i = 0; i < 3; ++ i) {

    //Growth factors
    #if 0
        computeGrowthFactors(i);
    #endif 

    //Relative suppression between cdm and fdm growth factors
    #if 0 
        computeFDMSuppression(1, cosmo_string, 1000);
    #endif
    
    //Print difference between CDM and FDM CAMB spectra at different momenta
    #if 0
        computeTreeSpectrumDifference();
    #endif

    #if 1
        computeNonlinearScales(i);
    #endif

    //Tree power spectra
    #if 0
        computeTreeSpectra();
    #endif 

    //Loop power spectra
    #if 0
        computeLoopSpectra();
    #endif 

    //Loop spectrum splines
    #if 0
        computeLoopSpectrumSplines();
        computeLoopSpectrumAnimation();
    #endif 

    //Tree Bispectra
    #if 0
        computeTreeBispectra();
    #endif 

    //Print difference between CDM and FDM bispectra at different configurations
    #if 0
        computeTreeBispectrumDifference();
    #endif

    //Dimensionless equilateral tree bispectra
    #if 0
        computeTreeDimensionlessEquilateral();
    #endif 

    //Loop bispectra
    #if 0 
        computeLoopBispectra();
        //computeLoopBispectrumSpline();
        //readBSpline();
    #endif

    //Tree Trispectra
    #if 0
        computeTreeTrispectra();
    #endif
    
    //Print difference between CDM and FDM trispectra at different configurations
    #if 0
        computeTreeTrispectrumDifference();
    #endif
    }
    return 0;
}
