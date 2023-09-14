#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:43:12 2023

@author: ted
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from heston_model import heston_model
from bs_model import bs_call,bs_put
from Heston_Real_Solution import heston_price_rec,heston_price_quad,heston_price_trapezoid
import pandas as pd
import datetime as datetime
from scipy.optimize import minimize 
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from Jackel_method import Jackel_method
import yfinance as yf
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol


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

    # Bizarre error in yfinance that gives the wrong expiration date
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
idx_price_c = set(SPX.index[SPX['bid']>0.1].tolist())
idx_OTM_c = set(SPX.index[SPX['inTheMoney']==0].tolist())

indices_call = intersection(idx_vol_c, idx_type_c,idx_date_c,idx_price_c,idx_OTM_c)


idx_vol_p = set(SPX.index[SPX['volume'] >=100].tolist())
idx_type_p = set(SPX.index[SPX['CALL'] ==  False].tolist())
idx_date_p = set(SPX.index[SPX['dte']>0].tolist())
idx_price_p = set(SPX.index[SPX['bid']>0.1].tolist())
idx_OTM_p = set(SPX.index[SPX['inTheMoney']==0].tolist())

indices_put = intersection(idx_vol_p, idx_type_p,idx_date_p,idx_price_p,idx_OTM_p)

indices = indices_call + indices_put

r=0.054
V_0=0.1
t=0
tol=0.000001
q=0


q=0
T=np.empty([len(indices),1])
K=np.empty([len(indices),1])
market_price=np.empty([len(indices),1])

count=0

for i in indices:
    T[count] = SPX.iloc[i]['dte']   
    K[count] = SPX.iloc[i]['strike']
    market_price[count] = (SPX.iloc[i]['bid'] +SPX.iloc[i]['ask'])/2 
    
    count+=1

size=np.size(indices)
iv_array=np.empty((size,1))

idx=0

r=0.054
d=0
tol=0.0000001
I=100
idx=0

for i in indices:
    
     expiry = T[idx]
     strike = K[idx]
     
     
    # market_price = (SPX.iloc[i]['ask'] + SPX.iloc[i]['bid'])/2
    
    # using last price as there is a bug with Yahoo finance where it sometimes won't return bid and call prices
     market_price = SPX.iloc[i]["lastPrice"]
     
     # Use weighted price instead : market_price = P_a * V_b / (V_a+V_b) + P_b * V_a (V_a+V_b)
     # But Yahoo finance does not provide buy/ask volume
     
     if SPX.iloc[i]['CALL'] ==  True:
         theta=1
         imp_vol =  Jackel_method(S,strike,d,expiry,r,market_price,theta,tol,I)
     else:
         theta=-1
         imp_vol =  Jackel_method(S,strike,d,expiry,r,market_price,theta,tol,I)
     iv_array[idx] = imp_vol
     
     idx+=1


T = T[~np.isnan(iv_array)]
K = K[~np.isnan(iv_array)]
iv_array = iv_array[~np.isnan(iv_array)]
T=T[:,]
K=K[:,]

T2=np.linspace(T.min(),T.max(),100)
K2=np.linspace(K.min(),K.max(),4000)


iv_interpol=griddata((T,K), iv_array[~np.isnan(iv_array)], (T2[None,:],K2[:,None]),method='linear')

T2,K2=np.meshgrid(T2,K2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X=T2, Y=K2, Z=iv_interpol)
plt.show()
