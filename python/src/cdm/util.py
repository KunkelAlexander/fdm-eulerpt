from numba import njit 

#Settings mass and momentum dependent FDM-scale to zero in coupling code
#is equal to considering the IR limit where FDM dynamics equal CDM dynamics
#This enables us to compare the expressions for F2, F3, F4 obtained for FDM with
#the analytical expressions in CDM obtained using the recursion relations (see Bernardeau 2002)
@njit
def b_f(k, m):
  return 0