import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

######## functions definitions ###############################################

def SABR(alpha, beta, rho, nu, F, K, time, MKT):
    if K <= 0:   # Negative rates' problem, need to shift the smile
        VOL = 0
        diff = 0
    elif F == K: # ATM formula
        V = (F*K)**((1-beta)/2.)
        A = 1 + (((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.)) * time
        VOL = (alpha/V) * A
        diff = VOL - MKT
    elif F != K: # Not-ATM formula
        V = (F*K)**((1-beta)/2.)
        z = (nu/alpha)*V*math.log(F/K)
        x = math.log((math.sqrt(1-2*rho*z+z**2) + z - rho) / (1-rho))
        A = 1 + (((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.)) * time
        VOL = (nu * math.log(F/K) * A) / x
        diff = VOL - MKT

    return VOL, diff

# Defining the function that creates our new output dataframe

def SABR_vol_matrix(alpha, beta, rho, nu, F, K, time, MKT):
    outvol_data = [['SABR VOLATILITIES', 'strikes:'] + label_strikes]
    vol_diff_data = [['VOLATILITY DIFFERENCES', 'strikes:'] + label_strikes]

    for i in range(len(F)):
        smile_data = [label_ten[i], label_exp[i]] + [None] * (len(label_strikes))  # Initialize a row with None values
        smile_diff = [label_ten[i], label_exp[i]] + [None] * (len(label_strikes))  # Initialize a row for volatility differences
        for j in range(len(K[i])):
            vol, diff = SABR(alpha[i], beta[i], rho[i], nu[i], F[i], K[i][j], time[i], MKT[i][j])
            if j + 2 < len(smile_data):  # Check if the index is within the range of smile_data
                smile_data[j + 2] = vol  # Set the SABR volatility value at the appropriate index
                smile_diff[j + 2] = diff  # Set the volatility difference value at the appropriate index
            else:
                break  # Break the loop if the index is out of range
        outvol_data.append(smile_data)
        vol_diff_data.append(smile_diff)

    parameters_data = [['PARAMETERS']]
    parameters_data.append(['tenor', 'expiry', 'alpha', 'beta', 'rho', 'nu'])
    for i in range(len(F)):
        parameters_data.append([label_ten[i], label_exp[i], alpha[i], beta[i], rho[i], nu[i]])

    outvol_df = pd.DataFrame(outvol_data)
    vol_diff_df = pd.DataFrame(vol_diff_data)
    parameters_df = pd.DataFrame(parameters_data)

    with pd.ExcelWriter("C:\\Users\\juliu\\OneDrive\\Skrivebord\\sabr script\\output.xlsx") as writer:
        outvol_df.to_excel(writer, sheet_name='outvol', index=False, header=False)
        vol_diff_df.to_excel(writer, sheet_name='vol_diff', index=False, header=False)
        parameters_df.to_excel(writer, sheet_name='parameters', index=False, header=False)

    print("Data Successfully Calculated! Output in directory!")

    return outvol_df, vol_diff_df, parameters_df


def objfunc(par, F, K, time, MKT):
    sum_sq_diff = 0
    for j in range(len(K)):
        if MKT[j] == 0:   
            diff = 0       
        else:
            vol, _ = SABR(par[0], par[1], par[2], par[3], F, K[j], time, MKT[j])
            diff = vol - MKT[j]
        sum_sq_diff += diff**2  
    obj = math.sqrt(sum_sq_diff)
    return obj


def calibration(starting_par, F, K, time, MKT):
    for i in range(len(F)):
        x0 = starting_par
        bnds = ((0.001,None), (0,1), (-0.999,0.999), (0.001,None))
        res = minimize(objfunc, x0, (F[i], K[i], time[i], MKT[i]), bounds=bnds, method='SLSQP')
        alpha[i] = res.x[0]
        beta[i] = res.x[1]
        rho[i] = res.x[2]
        nu[i] = res.x[3]
    
    print("Calibration Done!")

######## inputs and outputs #########################################

try:
    market_data = pd.read_excel("C:\\Users\\juliu\\OneDrive\\Skrivebord\\sabr script\\swapsdata.xlsx", sheet_name='Swaptions data')  # load market data
except FileNotFoundError:
    print("File 'Market_data.xlsx' not found. Please check the file path.")
    # Handle the error or exit the program gracefully

######## set swaptions characteristics ###############################
     
strike_spreads = [-100, -75, -50, -25, 0, 25, 50, 75, 100] 

num_strikes = len(strike_spreads)

expiries = market_data["Expiry"]

tenors = market_data["Tenor"]

F = market_data["Fwd"]

K = np.zeros((len(F), num_strikes))
for i in range(len(F)):
    for j in range(num_strikes):
        K[i][j] = F[i] + (strike_spreads[j])/10000

MKT = market_data.iloc[0:, 3:].values

starting_guess = np.array([0.001, 0.5, 0, 0.001])
alpha = len(F) * [starting_guess[0]]
beta = len(F) * [starting_guess[1]]
rho = len(F) * [starting_guess[2]]
nu = len(F) * [starting_guess[3]]

######## set labels ######## 

def get_label_exp(expiry):
    exp_dates = []
    for exp in expiry:
        if exp < 1:
            exp_dates.append(str(int(round(12 * exp))) + 'm')
        else:
            exp_dates.append(str(int(round(exp))) + 'y')
            if exp - round(exp) > 0:
                exp_dates[-1] += str(int(round((12 * (round(exp, 2) - int(exp)))))) + 'm'
            elif exp - round(exp) < 0:
                exp_dates[-1] = str(int(round(exp)) - 1) + 'y'
                exp_dates[-1] += str(int(round((12 * (round(exp, 2) - int(exp)))))) + 'm'
    return exp_dates

def get_label_ten(tenor):
    ten_dates = []
    for ten in tenor:
        if ten < 1:
            ten_dates.append(str(int(round(12 * ten))) + 'm')
        else:
            ten_dates.append(str(int(round(ten))) + 'y')
            if ten - round(ten) > 0:
                ten_dates[-1] += str(int(round((12 * (round(ten, 2) - int(ten)))))) + 'm'
            elif ten - round(ten) < 0:
                ten_dates[-1] = str(int(round(ten)) - 1) + 'y'
                ten_dates[-1] += str(int(round((12 * (round(ten, 2) - int(ten)))))) + 'm'
    return ten_dates

label_exp = get_label_exp(expiries)
label_ten = get_label_ten(tenors)


# Extracting unique strikes from the market data
# Define the basis point spreads
basis_point_spreads = [-100, -75, -50, -25, 0, 25, 50, 75, 100] 

# Define the ATM strike
# atm_strike = 'ATM'

# Combine the ATM strike and basis point spreads
label_strikes = [str(spread) for spread in basis_point_spreads]


######## Call the functions #########

calibration(starting_guess, F, K, expiries, MKT)

SABR_vol_matrix(alpha, beta, rho, nu, F, K, expiries, MKT)


