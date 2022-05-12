#from numba import njit 
#import numpy as np
#
#import constants as c
#from fdm.util import li_kj
#from special_functions import k_norm
#import cdm.couplings.analytical as cca

import greens 
from fdm.growth.suppressed_cdm import D_plus, dD_plus, ddD_plus, D_minus, dD_minus, ddD_minus

G, ds_G, deta_G, dseta_G, f1, f2 = greens.make_greens_utilities(D_plus, dD_plus, ddD_plus, D_minus, dD_minus, ddD_minus)