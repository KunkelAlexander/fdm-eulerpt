#ifndef _TIME_DEPENDENT_F2_
#define _TIME_DEPENDENT_F2_

double F2s(const vec &k1, const vec &k2, const CosmoUtil& cu) {
    double eta, eta0;
    eta = cu.eta;
    eta0 = cu.eta_in;
    return ((25*pow(eta, 7) - 21*pow(eta, 5)*pow(eta0, 2) - 4*pow(eta0, 7))*dot(k2,k2)*dot(k1 + k2,k1) + (25*pow(eta, 7) - 21*pow(eta, 5)*pow(eta0, 2) - 4*pow(eta0, 7))*dot(k1,k1)*dot(k1 + k2,k2) + 2*(5*pow(eta, 7) - 7*pow(eta, 5)*pow(eta0, 2) + 2*pow(eta0, 7))*dot(k1,k2)*dot(k1 + k2,k1 + k2))/(70.*pow(eta, 7)*dot(k1,k1)*dot(k2,k2));
}

  double alpha(const vec &k1, const vec &k2)
  {
    double k1s, result;
    
    k1s = dot(k1, k1);
    //Need this check for case x/x where x == 0 and expression should vanish
    if (k1s < verysmalleps)
    {
      return 0;
    }

    result = dot(k1 + k2, k1) / k1s;
    return(result);
  }

  double beta(const vec &k1, const vec &k2)
  {
    double k1s, k2s, result;
    k1s = dot(k1, k1);
    k2s = dot(k2, k2);
    //Need this check for case x/x where x == 0 and expression should vanish
    if ((k1s < verysmalleps) || (k2s < verysmalleps)) {
      return 0;
    }
    result = ssum(k1, k2) * dot(k1, k2) / (2 * k1s * k2s);

    return(result);
  }

  //F2 symmetrised mode coupling, computed using recursion relations in Mathematica
  double F2s(const vec &k1, const vec &k2)
  {
    double result, a1, a2, b;
    a1 = alpha(k1, k2);
    a2 = alpha(k2, k1);
    b  = beta(k1, k2);
    result = 5. / 14 * (a1 + a2) + 2. / 7 * b;
    return(result);
  }
  
#endif 