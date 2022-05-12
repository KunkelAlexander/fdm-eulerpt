def k_nl(P, *args, upper_lim = 1e2):
  #Integrate linear spectrum from k_min (epsilon) to given k
  #k_nl is defined as scale where spatial integral over P yields one
  #since P only depends on modulus of k, we multiply by spatial angle
  #We expect nonlinear scale to be found within the interval [epsilon, 100]*Mpc^(-1)
  def target_function(k_upper):
    return 4*np.pi*integrate.quad(lambda k: P(k, *args) * k**2, epsilon, k_upper)[0] - 1.
  return optimize.root_scalar(target_function, bracket=[0, upper_lim], method='brentq')

def k_smoothing(P, *args, upper_lim = 1e2):
  #Integrate linear spectrum from k_min (epsilon) to given k
  #k_nl is defined as scale where spatial integral over P yields one
  #since P only depends on modulus of k, we multiply by spatial angle
  #We expect nonlinear scale to be found within the interval [epsilon, 100]*Mpc^(-1)
  def target_function(R):
    return 4*np.pi*integrate.quad(lambda k: P(k, *args) * k**2 * np.exp(-k**2*R**2/2), 1e-5, 1e8)[0] - 1.
  return optimize.root_scalar(target_function, bracket=[0, 1], method='brentq')


1/(4*np.pi*D_CDM(eta)**2)

cdm_k_nl = k_nl(CDM_P, eta)
print("Nonlinear scale of CDM k_nl = {} at z = {} ".format(cdm_k_nl, z_from_eta(eta)))

cdm_k_smoothing = k_smoothing(CDM_P, eta)
print("Nonlinear scale of CDM with Gaussian smoothing: k  = {} at z = {} ".format(cdm_k_smoothing, z_from_eta(eta)))

cdm_k_smoothing.root*cdm_k_nl.root*D_CDM(eta)**4

my_gamma(3./2)

cdm_sc_k_nl = k_nl(CDM_P_sc, eta)
print("Nonlinear scale of CDM sc k_nl = {} at z = {} ".format(cdm_sc_k_nl, z_from_eta(eta)))

cdm_sc_k_smoothing = k_smoothing(CDM_P_sc, eta)

print("Nonlinear scale of sc CDM with Gaussian smoothing: k  = {} at z = {} ".format(cdm_sc_k_smoothing, z_from_eta(eta)))

cdm_sc_k_smoothing.root*cdm_sc_k_nl.root*D_CDM(eta)**4

fdm_k_nl = k_nl(FDM_P, eta, FDM_m)
print("Nonlinear scale of FDM k_nl = {} at z = {} ".format(fdm_k_nl, z_from_eta(eta)))

fdm_sc_k_nl = k_nl(FDM_P_sc, eta, FDM_m, upper_lim = 1e5)
print("Nonlinear scale of FDM sc k_nl = {} at z = {} ".format(fdm_sc_k_nl, z_from_eta(eta)))