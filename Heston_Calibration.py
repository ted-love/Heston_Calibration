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
from tools.Heston_Calibration_Class import Data_Class


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
    options['dte'] = ((options['expirationDate'] - datetime.datetime.today()).dt.days + 1) / 365
    
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
if S==0:
   S= 4550.58

"""

#%%
"""
Using preloaded data
"""

SPX = pd.read_csv('Results_SPX_2023-11-26/SPX_options_chain.csv',index_col=0)
S = pd.read_csv('Results_SPX_2023-11-26/spot_prices.csv', index_col=0).iloc[0,0]

SPX_daily = pd.read_csv('Results_SPX_2023-11-26/SPX_daily.csv', parse_dates=['Date'], index_col='Date')
VIX_daily = pd.read_csv('Results_SPX_2023-11-26/VIX_daily.csv', parse_dates=['Date'], index_col='Date')
VVIX_daily= pd.read_csv('Results_SPX_2023-11-26/VVIX_daily.csv',parse_dates=['Date'], index_col='Date')
date_today = '2023-11-26'

#%%
 
"""
Creating term-structure for the risk-free rate & implied dividend yields
"""

Treasury_Dates = [0, 1/12, 3/12, 6/12, 1, 2, 5, 10, 30]
Treasury_Rates = [0.05355,0.05355, 0.05445, 0.0545, 0.0525, 0.0490, 0.0443, 0.0440, 0.0454]
Treasury_Curve = CubicSpline(Treasury_Dates, Treasury_Rates, bc_type='natural')

tmin = 1/365

# Local-module Yield-curve generator. 
Implied_Dividend_Dates, Implied_Dividend_Rates = Implied_Dividend_Yield(SPX, tmin, S, Treasury_Curve)

Implied_Dividend_Curve = CubicSpline(Implied_Dividend_Dates, Implied_Dividend_Rates, bc_type='natural')

#%%

"""
Retrieving liquid OTM options data for calibration.

"""

volume = 600
tmin = 50/365
tmax = 999999/365
price_type = 'lastPrice'


# Create a class to store all the data as we will be dynamically removing problematic options. 
Data_calib = Data_Class()
Data_calib.S = S

SPX_call_calib = Data_calib.filter_option_chain(SPX, tmin, tmax, S, 9999999999, volume, 'c', price_type, 0.10)
SPX_put_calib  = Data_calib.filter_option_chain(SPX, tmin, tmax, 0, S-5, volume, 'p', price_type, 0.10)

# Removing every m option up until time tt if there is a lot of data.
if volume < 200:
    tt=1
    m = 2
    SPX_put_calib = Data_calib.remove_every_2nd_option(SPX_put_calib,tt,m)
    m=3
    SPX_call_calib = Data_calib.remove_every_2nd_option(SPX_call_calib, tt, m)
    
SPX_calib = pd.concat([SPX_call_calib,SPX_put_calib])
Data_calib.option_chain = SPX_calib
           


"""
Calculating implied vol from the data we wish to calibrated to.
"""

Data_calib.df_to_numpy(Treasury_Curve, Implied_Dividend_Curve, price_type)

Data_calib.calculate_implied_vol()
Data_calib.removing_iv_nan()

#%%

"""
Retrieving ATM (that is also OTM) options data for calculating ATM vol for initial calibration guesses.
"""

tmin_atm = 0.000001
tmax_atm = 0.03

Data_ATM = Data_Class()
Data_ATM.S = S

SPX_call_ATM = Data_ATM.filter_option_chain(SPX, tmin_atm, tmax_atm, S, S+50, 5, 'c', price_type, 0.10)
SPX_put_ATM = Data_ATM.filter_option_chain(SPX, tmin_atm, tmax_atm, S-50, S, 5, 'p', price_type, 0.10)

Data_ATM.option_chain=pd.concat([SPX_call_ATM,SPX_put_ATM])


"""
Retrieving Strikes, expiries and option prices are NumPy arrays
"""
market_price = np.empty([len(Data_ATM.option_chain),1])

market_price = (Data_ATM.option_chain).pivot_table(index='dte',columns='strike',values='midPrice')

# Creating DataFrame for the vol so it is like the options prie dataframe
vol_ATM = pd.DataFrame().reindex_like(market_price)

"""
Calculating implied vol from ATM options data to use as V_0 initial guess 
"""
Data_ATM.df_to_numpy(Treasury_Curve, Implied_Dividend_Curve, price_type)

# Calculating implied volatility for ATM options.
imp_vol_ATM = calculate_iv(Data_ATM.market_prices, Data_ATM.S, Data_ATM.K, Data_ATM.T, Data_ATM.r, Data_ATM.flag, Data_ATM.q, model='black_scholes_merton',return_as='numpy')

idx=0
for i in range(len(Data_ATM.T)):
    vol_ATM.loc[Data_ATM.T[i]][Data_ATM.K[i]] = imp_vol_ATM[i]


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
Calculating historical averages for sigma and v_bar initial guesses.
"""
VVIX_mean = (np.mean(VVIX_daily['Close'])/100)**2
VIX_mean  = (np.mean(VIX_daily['Close'])/100)**2

# Local module to calculate the average correlation each year.
rho_mean  = calculate_yearly_correlation(SPX_daily,VIX_daily)


N = 240                  # Number of terms of summation during COS-expansion
L = 20                   # Length of truncation
I = 600                  # Max numbr of accepted iterations of calibration
w = 1e-3                 # Weight of initial damping factor   
F = 10                   # Factor to reduce pre-calibration by
precision = 0.01         # Precision of numerical differentiation

"""
Initial Guesses
"""
v_bar_guess = VIX_mean   # v_bar : long-term vol
sigma_guess = VVIX_mean  # sigma : vol of vol
rho_guess = rho_mean     # rho   : correlation between S and V
kappa_guess = 0.655      # Kappa : rate of mean-reversion
v0_guess = v0_guess      # v0    : initial vol


initial_guesses = np.array([ v_bar_guess, 
                             sigma_guess,      
                             rho_guess,        
                             kappa_guess,         
                             v0_guess      ]).reshape(5,1)

"""
Choose params you want to calibrated. Params not in params_2b_calibrated will be fixed. 
put params to be calibrated: params_2b_calibrated = ['v0','vbar','sigma','rho','kappa']
"""

params_2b_calibrated = ['v0','vbar','sigma','rho','sigma']
#%%

error_array=np.empty([7])

min_acc = 1e-3
accelerator = 1


calibrated_params,counts_accepted,counts_rejected,RMSE = levenberg_Marquardt(Data_calib,initial_guesses,I,w,N,L,precision,params_2b_calibrated,accelerator,min_acc)

calibrated_prices = heston_cosine_method(Data_calib.S,Data_calib.K,Data_calib.T,N,L,Data_calib.r,Data_calib.q,calibrated_params[0],calibrated_params[4],calibrated_params[1],calibrated_params[2],calibrated_params[3],Data_calib.flag)
calibrated_iv = 100*calculate_iv(calibrated_prices[0,:], Data_calib.S, Data_calib.K, Data_calib.T, Data_calib.r, Data_calib.flag, Data_calib.q, model='black_scholes_merton',return_as='numpy')

# Removing nan values if there were any from small priced options 
calibrated_iv = Data_calib.check_4_calibrated_nans(calibrated_iv)

M=np.size(Data_calib.K)

rmse = np.sqrt((1/M) * np.sum((calibrated_iv - Data_calib.market_vol)**2))

print('\nCalibrated_Params:\n', calibrated_params)
print('\ncost_function_error (implied vol as a %): ', rmse)

error_array[0] = rmse
error_array[1] = int(M)
error_array[2:] = np.squeeze(calibrated_params)

error_df = pd.DataFrame(error_array,index=['RMSE','No. Options','v_bar','sigma','rho','kappa','v0'],columns=['Results'])
Data_calib.plot_save_surface(calibrated_iv, date_today)


#%%

try:
    os.mkdir(f'Results_SPX_{date_today}')
except:
    pass

error_df.to_csv(f'Results_SPX_{date_today}/Results.csv')
SPX.to_csv(f'Results_SPX_{date_today}/SPX_options_chain.csv',)
SPX_daily.to_csv(f'Results_SPX_{date_today}/SPX_daily.csv')
VIX_daily.to_csv(f'Results_SPX_{date_today}/VIX_daily.csv')
VVIX_daily.to_csv(f'Results_SPX_{date_today}/VVIX_daily.csv')

spot_prices_dict = {"SPX_price" : [S]}
spot_prices_df = pd.DataFrame(spot_prices_dict)
spot_prices_df.to_csv(f'Results_SPX_{date_today}/spot_prices.csv')



