#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:43:12 2023
@author: ted

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import datetime as datetime
from scipy.interpolate import CubicSpline,interp1d
import yfinance as yf
from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv
from tools.Levenberg_Marquardt import levenberg_Marquardt
from tools.Heston_COS_METHOD import heston_cosine_method
from tools.Implied_Dividend_Yield import Implied_Dividend_Yield
from tools.stock_vol_correlation import calculate_yearly_correlation
from tools.clean_up_helpers import df_to_numpy,filter_option_chain,removing_nans
#%%
def options_chain(symbol):
    """

    Parameters
    ----------
    symbol : Str
        Stock Ticker.

    Returns
    -------
    options : DataFrame
       Options data i.e. bid-ask spread, strikes, expiries etc.
    S : Float
        Spot price of the Stock.

    """

    tk = yf.Ticker(symbol)
    # Expiration dates
    
    exps = tk.options
    if symbol=='^VVIX' or symbol=='^VIX':
        return tk

    S = (tk.info['bid'] + tk.info['ask'])/2 

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate'])
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['midPrice'] = (options['bid'] + options['ask']) / 2 
    
    options = options.drop(columns = ['contractSize', 'currency', 'impliedVolatility', 'inTheMoney', 'change', 'percentChange', 'lastTradeDate'])
    
    return options,S,tk


"""
Option chains and historical daily returns

"""
"""

SPX,S,SPX_info = options_chain("^SPX")
VIX_info = options_chain("^VIX")
VVIX_info = options_chain("^VVIX")

date_today = str(datetime.datetime.today().date())

VVIX_daily = yf.download('^VVIX', start="2004-01-03", end=date_today, interval='1d')
VIX_daily  = yf.download('^VIX',  start="2004-01-03", end=date_today, interval='1d')
SPX_daily  = yf.download('^SPX',  start="2004-01-03", end=date_today, interval='1d')
"""

"""
Using preloaded data
"""
#%%


SPX = pd.read_csv('Results_SPX_2023-11-28/SPX_options_chain.csv',index_col=0)
S = pd.read_csv('Results_SPX_2023-11-28/spot_prices.csv', index_col=0).iloc[0,0]

SPX_daily = pd.read_csv('Results_SPX_2023-11-28/SPX_daily.csv', parse_dates=['Date'], index_col='Date')
VIX_daily = pd.read_csv('Results_SPX_2023-11-28/VIX_daily.csv', parse_dates=['Date'], index_col='Date')
VVIX_daily= pd.read_csv('Results_SPX_2023-11-28/VVIX_daily.csv',parse_dates=['Date'], index_col='Date')

#%%

 
"""

Creating term-structure for the risk-free rate & implied dividend yields


"""

Treasury_Dates = [0, 1/12, 3/12, 6/12, 1, 2, 5, 10, 30]
Treasury_Rates = [0.05355,0.05355, 0.05445, 0.0545, 0.0525, 0.0490, 0.0443, 0.0440, 0.0454]
Treasury_Curve = CubicSpline(Treasury_Dates, Treasury_Rates, bc_type='natural')


t = 50/365

# Local-module Yield-curve generator. 
Implied_Dividend_Dates, Implied_Dividend_Rates = Implied_Dividend_Yield(SPX, t, S, Treasury_Curve)

Implied_Dividend_Curve = CubicSpline(Implied_Dividend_Dates, Implied_Dividend_Rates, bc_type='natural')

#%%

"""
Retrieving ATM (that is also OTM) options data for calculating ATM vol.
"""
price_type = 'lastPrice'


SPX_call_ATM = filter_option_chain(SPX, 0.000001, 0.03, S, S+50, 5, 'c', price_type, 0.10)
    
SPX_put_ATM = filter_option_chain(SPX, 0.000001, 0.03, S-50, S, 5, 'p', price_type, 0.10)

SPX_ATM = pd.concat([SPX_call_ATM,SPX_put_ATM])

"""
Retrieving liquid OTM options data for calibration.

"""
SPX_call_calib = filter_option_chain(SPX, 20/365, 10000, S+5, 9999999999, 600, 'c', price_type, 0.10)
SPX_put_calib  = filter_option_chain(SPX, 20/365, 10000, 0, S-5, 600, 'p', price_type, 0.10)

SPX_calib = pd.concat([SPX_call_calib,SPX_put_calib])

           
market_price = np.empty([len(SPX_ATM),1])

#%%

"""
Retrieving Strikes, expiries and option prices are NumPy arrays
"""

market_price = SPX_ATM.pivot_table(index='dte',columns='strike',values='midPrice')

# Creating DataFrame for the vol so it is like the options prie dataframe
vol_ATM = pd.DataFrame().reindex_like(market_price)

"""
Calculating V0 from ATM options data.

"""
T_ATM, K_ATM, r_ATM, q_ATM, option_prices_ATM, flag_ATM = df_to_numpy(SPX_ATM, Treasury_Curve, Implied_Dividend_Curve, price_type)

# Calculating implied volatility for ATM options.
imp_vol_ATM = calculate_iv(option_prices_ATM, S, K_ATM, T_ATM, r_ATM, flag_ATM, q_ATM, model='black_scholes_merton',return_as='numpy')

idx=0
for i in range(len(T_ATM)):
    vol_ATM.loc[T_ATM[i]][K_ATM[i]] = imp_vol_ATM[i]


"""
Interpolating the data with missing strikes. Each row is for each dte.
Then for each dte, interpolate and find ATM implied vol, then append for each dte.
"""

K_interpol=np.array(vol_ATM.columns)
vol_ATM_interpol=[]
for i in range(len(vol_ATM.index)):
    vol_row = vol_ATM.iloc[i]
    not_nan_indices = vol_row.notnull()  
    interpolated_nans = interp1d(np.where(not_nan_indices)[0], vol_row[not_nan_indices], kind='cubic', fill_value='extrapolate')
    nan_indices = vol_row.isnull()
    vol_row[nan_indices] = interpolated_nans(np.where(nan_indices)[0])
    
    interpolated_func = interp1d(vol_row.index,vol_row,kind='cubic')
    vol_ATM_interpol.append(interpolated_func(S))
    

v0_guess = np.mean(vol_ATM_interpol) # Interpolated volatility

# Turninng into volatility into variance for heston model
v0_guess = v0_guess**2 


"""
Calculating implied vol from the data we wish to calibrated to.
"""

T_calib, K_calib, r_calib, q_calib, option_prices_calib, flag_calib = df_to_numpy(SPX_calib, Treasury_Curve, Implied_Dividend_Curve, price_type)


# Calculating implied volatility for the options we are calibrating to.
imp_vol_calib = calculate_iv(option_prices_calib, S, K_calib, T_calib, r_calib, flag_calib, q_calib, model='black_scholes_merton',return_as='numpy').reshape(np.size(K_calib),1)

imp_vol_calib, option_prices_calib, K_calib, T_calib, r_calib, flag_calib, q_calib = removing_nans(imp_vol_calib, option_prices_calib, K_calib, T_calib, r_calib, flag_calib, q_calib)

imp_vol_calib = 100 * imp_vol_calib.reshape(np.size(imp_vol_calib),1)



"""
Calculating historical averages for initial guesses. Convert std to var for heston
"""
VVIX_mean = (np.mean(VVIX_daily['Close'])/100)**2
VIX_mean  = (np.mean(VIX_daily['Close'])/100)**2

# Local module to calculate the average correlation each year.
rho_mean  = calculate_yearly_correlation(SPX_daily,VIX_daily)


N = 240                  # Number of terms of summation during COS-expansion
L = 20                   # Length of truncation
I = 400                  # Max numbr of accepted iterations of calibration
w = 1.0                  # Weight of initial damping factor   
F = 10                   # Factor to reduce pre-calibration by
precision = 0.01         # Precision of numerical differentiation

"""
Initial Guesses
"""
v_bar_guess = VIX_mean   # v_bar : long-term vol
sigma_guess = VVIX_mean  # sigma : vol of vol
rho_guess = rho_mean     # rho   : correlation between S and V
kappa_guess = 2.5        # Kappa : rate of mean-reversion
v0_guess = v0_guess      # v0    : initial vol


initial_guesses = np.array([ v_bar_guess, 
                             sigma_guess,      
                             rho_guess,        
                             kappa_guess,         
                             v0_guess      ]).reshape(5,1)

initial_guesses = np.array([ 0.04385+0.004, 
                             1,      
                             -0.409,        
                             3.8 ,        
                             0.015675+0.004    ]).reshape(5,1)

"""
Choose params you want to calibrated. Params not in params_2b_calibrated will be fixed. 
put params to be calibrated: 'v0','vbar','sigma','rho','kappa'
"""

params_2b_calibrated = ['v0','vbar','sigma','rho','kappa']
w = w
#%%
error_array=np.empty([6,2])
for i in range(2):

    """
    Using an initial calibration with 1/10 of the data. 
    """
    if i==1:
        M = np.size(K_calib)
        
        K_ini = np.empty(M//F)
        T_ini = np.empty(M//F)
        q_ini = np.empty(M//F)
        r_ini = np.empty(M//F)
        flag_ini = np.empty(M//F, dtype = str)
        imp_vol_ini = np.empty(M//F).reshape(M//F,1)
        for k in range(M//F):
            K_ini[k] = K_calib[k*F]
            T_ini[k] = T_calib[k*F]
            q_ini[k] = q_calib[k*F]
            r_ini[k] = r_calib[k*F]
            flag_ini[k] = flag_calib[k*F]
            imp_vol_ini[k,0] = imp_vol_calib[k*F]
            
        
        I_ini = I/F
        
    
        ini_calibrated_params, counts_accepted, counts_rejected = levenberg_Marquardt(initial_guesses,imp_vol_ini,I_ini,w,S,K_ini,T_ini,N,L,r_ini,q_ini,0,0,0,0,0,flag_ini,precision,params_2b_calibrated)
        
        """
        Then use the resuls as the initial guess for full calibration with a smaller damping factor weight. 
        """ 
        initial_guesses = ini_calibrated_params 
        
    if i==1:
        w = w*1e-3
    
    # Start Calibration
    calibrated_params,counts_accepted,counts_rejected = levenberg_Marquardt(initial_guesses,imp_vol_calib,I,w,S,K_calib,T_calib,N,L,
                                                                            r_calib,q_calib,0,0,0,0,0,flag_calib,precision,params_2b_calibrated)
    
    # Removing nan values if there were any from small priced option 
  
    # Prices from the calibrated paramters.
    calibrated_prices = heston_cosine_method(S,K_calib,T_calib,N,L,r_calib,q_calib,calibrated_params[0],calibrated_params[4],calibrated_params[1]-1,calibrated_params[2],calibrated_params[3],flag_calib)
    calibrated_iv = (calculate_iv(calibrated_prices[0,:], S, K_calib, T_calib, r_calib, flag_calib, q_calib, model='black_scholes_merton',return_as='numpy')*100).reshape(np.size(K_calib),1)
    
    
    nan_count = 0
    M = np.size(K_calib)
    for j in range(M):
        if np.isnan(calibrated_iv[j]):
            calibrated_iv[j]=0.
            imp_vol_calib[j]=0.
            nan_count+=1
    
    
    print('\nCalibrated_Params:\n', calibrated_params)
    print('\ncost_function_error (implied vol): ', (1/(M - nan_count)) * np.sum(calibrated_iv - imp_vol_calib))
    
    error_array[0,i] = (1/(M - nan_count)) * np.sum(calibrated_iv - imp_vol_calib)            
    error_array[1:,i] = np.squeeze(calibrated_params)
    
print("error options without a pre-calibration: ", error_array[0,0])
print("error options using a pre-calibration:  ", error_array[0,1])

error_df = pd.DataFrame(error_array,index=['error','v_bar','sigma','rho','kappa','v0'],columns=['no pre-calibration','using pre-calibration'])

#%%

os.mkdir(f'Results_SPX_{date_today}')

error_df.to_csv(f'Results_SPX_{date_today}/Results.csv')
SPX.to_csv(f'Results_SPX_{date_today}/SPX_options_chain.csv',)
SPX_daily.to_csv(f'Results_SPX_{date_today}/SPX_daily.csv')
VIX_daily.to_csv(f'Results_SPX_{date_today}/VIX_daily.csv')
VVIX_daily.to_csv(f'Results_SPX_{date_today}/VVIX_daily.csv')

spot_prices_dict = {"SPX_price" : [S]}
spot_prices_df = pd.DataFrame(spot_prices_dict)
spot_prices_df.to_csv(f'Results_SPX_{date_today}/spot_prices.csv')



#%%

import matplotlib.pyplot as plt

sigma = np.linspace(0.01,2,100)

for i in range(13):
    plt.plot(sigma, heston_cosine_method(S,K_calib[i*10],T_calib[i*10],N,L,r_calib[i*10],q_calib[i*10],calibrated_params[0],calibrated_params[4],sigma,calibrated_params[2],calibrated_params[3],flag_calib[i*10])[0,:])
    plt.title("idx=" + str(i*10) +", " +str(K_calib[i*10])+", "+str(round(T_calib[i*10],4))+", "+str(round(r_calib[i*10],4)) +", "+str(round(q_calib[i*10],4)) +", "+str(flag_calib[i*10]))
    plt.show()

#%%
m=12
for i in np.arange(-200,200,20):
    plt.subplot(1, 2, 1)

    plt.plot(sigma, heston_cosine_method(S,S+i,T_calib[m*10],N,L,r_calib[m*10],q_calib[m*10],calibrated_params[0],calibrated_params[4],sigma,calibrated_params[2],calibrated_params[3],'c')[0,:])
    plt.title('c, K = ' + str(S+i))
    plt.subplot(1, 2, 2)

    plt.plot(sigma, heston_cosine_method(S,S+i,T_calib[m*10],N,L,r_calib[m*10],q_calib[m*10],calibrated_params[0],calibrated_params[4],sigma,calibrated_params[2],calibrated_params[3],'p')[0,:])
    plt.title('p, K = ' + str(S+i))
    
    plt.show()
