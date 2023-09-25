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
import numpy.ma as ma
from scipy import interpolate
from Heston_Real_Solution import heston_price_rec,heston_price_quad,heston_price_trapezoid
import pandas as pd
import datetime as datetime
from scipy.interpolate import griddata,bisplrep,bisplev
from Jaeckel_method import Jaeckel_method
import yfinance as yf

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
  #  dividend = tk.info["dividendYield"]
    S=(tk.info['bid'] + tk.info['ask'])/2 

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])

    return options,S,tk

SPX,S = options_chain("^SPX")

SPX=SPX.loc[(SPX['dte']<(21/365)) & (SPX['dte']>0)]
                    
            
SPX=SPX.loc[(SPX['strike']>S-50) & (SPX['strike']<S+50)]


r=0.054
V_0=0.1
t=0
tol=0.000001
q=0


q=0

T=np.empty([len(SPX),1])
K=np.empty([len(SPX),1])
market_price=np.empty([len(SPX),1])

count=0

T=SPX['dte'].to_numpy()
K=SPX['strike'].to_numpy()



market_price = SPX.pivot_table(index='dte',columns='strike',values='lastPrice')

vol =pd.DataFrame().reindex_like(market_price)
idx=0

r=0.054
d=0
tol=0.0000001
I=100

idx=0
for i,j in market_price.items():
    strike = i
    expiries = j.index    
    price = j.values
     
     
    # market_price = (SPX.iloc[i]['ask'] + SPX.iloc[i]['bid'])/2
    
    # using last price as there is a bug with Yahoo finance where it sometimes won't return bid and call prices
     
     
     # Use weighted price instead : market_price = P_a * V_b / (V_a+V_b) + P_b * V_a (V_a+V_b)
     # But Yahoo finance does not provide buy/ask volume
     
    if strike>S:
        idx=0
        for expiry in expiries:
            theta=1

            vol.loc[expiry][strike]=  Jaeckel_method(S,strike,d,expiry,r,price[idx],theta,tol,I)
            idx+=1
    else:
        idx=0
        for expiry in expiries:
             theta=-1
             vol.loc[expiry][strike] =  Jaeckel_method(S,strike,d,expiry,r,price[idx],theta,tol,I)
             idx+=1



T=np.array(vol.index)
K=np.array(vol.columns)
valid_vol=ma.masked_invalid(vol).T

Ti=np.linspace(float(T.min()),float(T.max()),len(T))
Ki=np.linspace(float(K.min()),float(K.max()),len(K))

Ti,Ki = np.meshgrid(Ti,Ki)
T,K = np.meshgrid(T,K)

valid_Ti = Ti[~valid_vol.mask]
valid_Ki = Ki[~valid_vol.mask]
valid_vol = valid_vol[~valid_vol.mask]

iv_interpol=griddata((valid_Ti,valid_Ki), valid_vol, (Ti, Ki),method='linear')
tck = bisplrep(Ti, Ki, iv_interpol, s=2)

T_new, K_new = np.mgrid[Ti.min():T.max():80j, Ki.min():Ki.max():80j]


znew = bisplev(T_new[:,0], K_new[0,:], tck)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_new[:,0], K_new[0,:]/S, znew)
ax.set_ylabel('Monyness K/S')
ax.set_xlabel('Expiry (Years)')
ax.set_zlabel('Implied Vol')
ax.set_title("Implied Vol of SPX")
