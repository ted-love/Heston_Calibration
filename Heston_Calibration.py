#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:28:48 2023

@author: ted
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from tools.Heston_Real_Solution import heston_price_rec,heston_price_quad,heston_price_trapezoid
import pandas as pd
import datetime as datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
from tools.Jaeckel_method import Jaeckel_method


import yfinance as yf

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
  #  dividend = tk.info["dividendYield"]
    S=(tk.info['bid']+tk.info['ask'])/2

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])

    return options,S

SPX,S = options_chain("^SPX")



SPX_c = SPX.loc[(SPX['volume']>100) & (SPX['dte']>0) & (SPX['lastPrice']>0.1)]

ATM_strikes=np.empty((1,2))

for i in np.array(SPX["strike"].index):
    if SPX.loc[i]["strike"]>S:
        print(i)
        ATM_strikes[0,1]=SPX.loc[i]["strike"]
        ATM_strikes[0,0]=SPX.loc[i-1]["strike"]
        break


SPX_v_c=SPX.loc[(SPX['strike']==ATM_strikes[0,1]) & (SPX['CALL']==True) & (SPX['dte']<(21/365)) & (SPX['dte']>0)]
SPX_v_p=SPX.loc[(SPX['strike']==ATM_strikes[0,0]) & (SPX['CALL']==False) & (SPX['dte']<(21/365)) & (SPX['dte']>0)]
SPX_v=SPX_v_c.append(SPX_v_p)

market_price = SPX_v.pivot_table(index='dte',columns='strike',values='lastPrice')


r=0.054


start_1 = time.time()

params = {"V_0": {"x0": 0.1, "bound": [1e-4,0.5]}, 
          "kappa": {"x0": 2.5, "bound": [0.00001,5]},
          "theta": {"x0": 0.07, "bound": [0.00001,0.1]},
          "sigma": {"x0": 0.4, "bound": [0.00001,1]},
          "rho": {"x0": -0.75, "bound": [-1,-0.0001]},
          "lambd": {"x0": 0.4, "bound": [-1,1]},
          }
x0 = [param["x0"] for key, param in params.items()]
bnds = [param["bound"] for key, param in params.items()]


def SqErr(x):
    V_0, kappa, theta, sigma, rho, lambd = [param for param in x]

    w = 1/np.shape(SPX_c)[0]
    error=0
    
    for i in SPX_c.index.values:

        
        if SPX_c.loc[i]['CALL'] ==  True:
            
            error = error + (1/w)*(SPX_c.loc[i]['lastPrice']-
                               
                         heston_price_trapezoid(S, SPX_c.loc[i]['strike'], V_0, kappa, 
                                           theta, sigma, rho, lambd, SPX_c.loc[i]['dte'], r))**2
        else:
            D=np.exp(-r*SPX_c.loc[i]['strike'])
            
            # Put-Call parity : D*K - P = S - C
            # Thus, P = -S + D*K + C
            
            error = error + (1/w)*(SPX_c.loc[i]['lastPrice'] - (-S + D*SPX_c.loc[i]['strike'] \
                               +heston_price_trapezoid(S, SPX_c.loc[i]['strike'], V_0, kappa, 
                                                 theta, sigma, rho, lambd, SPX_c.loc[i]['dte'], r)))**2
        
    penalty = np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)])
    
    error = error + penalty
    return error

result = minimize(SqErr, x0, tol = 1e-9, method='Nelder-Mead', options={'maxiter': 30}, bounds=bnds)
V_0_1, kappa_1, theta_1, sigma_1, rho_1, lambd_1 = [param for param in result.x]

end_1 = time.time()
total_1 = end_1 - start_1


"""

Calibrating but using V_0 as the implied vol 

"""

start_2 = time.time()

vol = pd.DataFrame().reindex_like(market_price)
idx=0

r=0.054
d=0
tol=0.0000001
I=100

idx=0

for i,j in market_price.iterrows():
    expiry=i
    price=j[market_price.columns[1]]
    strike = market_price.columns[1]
    theta=1
    

    vol.loc[expiry][strike] = Jaeckel_method(S,strike,d,expiry,r,price,theta,tol,I)

for i,j in market_price.iterrows():
    expiry=i
    price=j[market_price.columns[0]]
    strike = market_price.columns[0]
    theta=-1

    vol.loc[expiry][strike] = Jaeckel_method(S,strike,d,expiry,r,price,theta,tol,I)

K_interpol=np.array(vol.columns)
vol_interpol=[]
for i in range(len(vol.index)):
    f = interp1d(K_interpol, vol.iloc[i])
    vol_interpol.append(f(S))

V_0_2 = np.mean(vol_interpol)

params = { 
          "kappa": {"x0": 2.5, "bound": [0.00001,5]},
          "theta": {"x0": 0.07, "bound": [0.00001,0.1]},
          "sigma": {"x0": 0.4, "bound": [0.00001,1]},
          "rho": {"x0": -0.75, "bound": [-1,-0.0001]},
          "lambd": {"x0": 0.4, "bound": [-1,1]},
          }
x0 = [param["x0"] for key, param in params.items()]
bnds = [param["bound"] for key, param in params.items()]


def SqErr(x):
    kappa, theta, sigma, rho, lambd = [param for param in x]

    w = 1/np.shape(SPX_c)[0]
    error=0
    
    for i in SPX_c.index.values:

        
        if SPX_c.loc[i]['CALL'] ==  True:
            
            error = error + (1/w)*(SPX_c.loc[i]['lastPrice']-
                               
                         heston_price_trapezoid(S, SPX_c.loc[i]['strike'], V_0_2, kappa, 
                                           theta, sigma, rho, lambd, SPX_c.loc[i]['dte'], r))**2
        else:
            D=np.exp(-r*SPX_c.loc[i]['strike'])
            
            # Put-Call parity : D*K - P = S - C
            # Thus, P = -S + D*K + C
            
            error = error + (1/w)*(SPX_c.loc[i]['lastPrice'] - (-S + D*SPX_c.loc[i]['strike'] \
                               +heston_price_trapezoid(S, SPX_c.loc[i]['strike'], V_0_2, kappa, 
                                                 theta, sigma, rho, lambd, SPX_c.loc[i]['dte'], r)))**2
        
    penalty = np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)])
    
    error = error + penalty
    return error

result = minimize(SqErr, x0, tol = 1e-9, method='Nelder-Mead', options={'maxiter': 30}, bounds=bnds)
kappa_2, theta_2, sigma_2, rho_2, lambd_2 = [param for param in result.x]

end_2 = time.time()
total_2 = end_2 - start_2
V_0_1, kappa_1, theta_1, sigma_1, rho_1, lambd_1

results = {'v_0': [V_0_1, V_0_2], 'kappa': [kappa_1, kappa_2], 'theta':[theta_1,theta_2],'sigma':[sigma_1,sigma_2],
           "rho":[rho_1,rho_2],"lamda":[lambd_1,lambd_2],"time":[total_1,total_2]}

Results = pd.DataFrame(results)



