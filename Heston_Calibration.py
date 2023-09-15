#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:28:48 2023

@author: ted
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from Heston_Real_Solution import heston_price_rec,heston_price_quad,heston_price_trapezoid
import pandas as pd
import datetime as datetime
from scipy.optimize import minimize 
import time
from Jaeckel_method import Jaeckel_method


import yfinance as yf

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
  #  dividend = tk.info["dividendYield"]
    S=tk.info['bid']

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

def intersection(l1, l2, l3, l4, l5):
    return list(set(l1) & set(l2) & set(l3) & set(l4) & set(l5))

idx_vol_c = set(SPX.index[SPX['volume'] >=100].tolist())
idx_type_c = set(SPX.index[SPX['CALL'] ==  True].tolist())
idx_date_c = set(SPX.index[SPX['dte']>0].tolist())
idx_price_c = set(SPX.index[SPX['lastPrice']>0.1].tolist())
idx_OTM_c = set(SPX.index[SPX['inTheMoney']==0].tolist())

indices_call = intersection(((idx_vol_c, idx_type_c,idx_date_c,idx_price_c,idx_OTM_c)))


idx_vol_p = set(SPX.index[SPX['volume'] >=100].tolist())
idx_type_p = set(SPX.index[SPX['CALL'] ==  False].tolist())
idx_date_p = set(SPX.index[SPX['dte']>0].tolist())
idx_price_p = set(SPX.index[SPX['lastPrice']>0.1].tolist())
idx_OTM_p = set(SPX.index[SPX['inTheMoney']==0].tolist())

indices_put = intersection(idx_vol_p, idx_type_p,idx_date_p,idx_price_p,idx_OTM_p)

indices = indices_call + indices_put


r=0.054
tol=0.000001


T=np.empty([len(indices),1])
K=np.empty([len(indices),1])
market_price=np.empty([len(indices),1])

count=0

for i in indices:
    T[count] = SPX.iloc[i]['dte']
    K[count] = SPX.iloc[i]['strike']
    market_price[count] = SPX.iloc[i]['lastPrice']
    
    count+=1

start_1 = time.time()

params = {"V_0": {"x0": 0.1, "bound": [1e-4,0.2]}, 
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

    w = 1/len(T)
    err=0
    idx=0
    for i in indices:
        
        if SPX.iloc[i]['CALL'] ==  True:
            err = err + (1/w)*(market_price[idx].item()-
                         heston_price_trapezoid(S, K[idx].item(), V_0, kappa, 
                                           theta, sigma, rho, lambd, T[idx].item(), r))**2
        else:
            D=np.exp(-r*T[idx])
            err = err + (1/w)*(market_price[idx].item() - (-S + D*K[idx].item() \
                               +heston_price_trapezoid(S, K[idx].item(), V_0, kappa, 
                                           theta, sigma, rho, lambd, T[idx].item(), r)))**2
        idx+=1
        
    return err

result = minimize(SqErr, x0, tol = 1e-4, method='Nelder-Mead', options={'maxiter': 30 }, bounds=bnds)
V_0_1, kappa_1, theta_1, sigma_1, rho_1, lambd_1 = [param for param in result.x]
V_0_1, kappa_1, theta_1, sigma_1, rho_1, lambd_1
end_1 = time.time()
total_1 = end_1 - start_1

start_2 = time.time()



idx_upper_time= set(SPX.index[SPX['dte']<0.384].tolist())
idx_lower_time = set(SPX.index[SPX['dte']>0].tolist())
idx_lower_price = set(SPX.index[SPX['strike']>0.9*S].tolist())
idx_upper_price = set(SPX.index[SPX['strike']<0.9*S].tolist())

indices_new = intersection(indices,idx_upper_time,idx_lower_time,idx_lower_price,idx_upper_price)


d=0
tol=0.0000001
I=100

iv_array = np.empty([len(indices),1])
idx=0

for i in indices_new:
    
     expiry = T[idx]
     strike = K[idx]
     
     
    # market_price = (SPX.iloc[i]['ask'] + SPX.iloc[i]['bid'])/2
    
    # using last price as there is a bug with Yahoo finance where it sometimes won't return bid and call prices
     market_price = SPX.iloc[i]["lastPrice"]
     
     # Use weighted price instead : market_price = P_a * V_b / (V_a+V_b) + P_b * V_a (V_a+V_b)
     # But Yahoo finance does not provide buy/ask volume
     
     if SPX.iloc[i]['CALL'] ==  True:
         theta=1
         imp_vol =  Jaeckel_method(S,strike,d,expiry,r,market_price,theta,tol,I)
     else:
         theta=-1
         imp_vol =  Jaeckel_method(S,strike,d,expiry,r,market_price,theta,tol,I)
     iv_array[idx] = imp_vol
     
     idx+=1



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

    w = 1/len(T)
    err=0   
    for i in range(len(T)):
        err = err + (1/w)*(market_price[i].item()-
                     heston_price_trapezoid(S, K[i].item(), 0.155, kappa, 
                                       theta, sigma, rho, lambd, T[i].item(), r))**2
    return err 

result = minimize(SqErr, x0, tol = 1e-4, method='Nelder-Mead', options={'maxiter': 1 }, bounds=bnds)
kappa_2, theta_2, sigma_2, rho_2, lambd_2 = [param for param in result.x]


end_2 = time.time()

total_2 = end_2 - start_2

