# importing libraries we will be using

import numpy as np
import pandas as pd
import math
from scipy.stats import norm

implied_vol_cube = {
    "1M" : [0.06, 0.11, 0.18, 0.27, 0.37, 0.67, 1.1, 1.7, 2.17, 2.94], 
    "3M" : 
}

# Defining the function that calculates the ATM SABR sigma parameter
def sabr_atm_vol(K, T, F_0, sigma_0, beta, upsilon, rho):
    sigma = sigma_0*F_0**(beta-1)*(1+(((1-beta)**2/24)*(sigma_0**2*(F_0)**(2*beta-2)) + (rho*beta*upsilon*sigma_0/4)*(F_0)**(beta-1) + (2-3*rho**2)/24*upsilon**2)*T)
    return sigma 


# Defining the original black-76 model for pricing swaptions (assuming constant vol.)
def black_swaption_premium(sigma, T, K, S, R): 
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    price = S*(R*norm.cdf(d1) - K*norm.cdf(d2))
    return price

print(black_swaption_premium(0.2, 10, 0.03, 0.01, 0.05))

