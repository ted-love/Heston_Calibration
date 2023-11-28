#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:40:59 2023

@author: ted
"""
import numpy as np

def df_to_numpy(DF, Treasury_Curve, Implied_Dividend_Curve, price_type):
    
    T = DF['dte'].to_numpy()
    K = DF['strike'].to_numpy()
    r = Treasury_Curve(T)
    q = Implied_Dividend_Curve(T)
    option_prices = DF[price_type].to_numpy()
    flag = DF['CALL']
    flag = flag.replace({True:'c'})
    flag = flag.replace({False:'p'}).to_numpy()
    return T,K,r,q,option_prices,flag


def removing_options_with_nans(nans,F,market_vol,K,T,r,q,flag):
    for i in nans:
        T = np.delete(T,i*F)
        K = np.delete(K,i*F)
        r = np.delete(r,i*F)
        q = np.delete(q,i*F)
        flag = np.delete(flag,i*F)
        market_vol = np.delete(market_vol,i*F,axis=0)
    
    return T,K,r,q,flag,market_vol

def filter_option_chain(df, tmin, tmax, Kmin, Kmax, volume, option_type, price_type, min_price):
    
    if option_type=='c':
        option_flag=True
    if option_type=='p':
        option_flag=False
        
    filtered_chain = df.loc[(df['dte'] >= tmin) & (df['dte'] <= tmax) & (df[price_type] >= 0.10)
                            & (df['volume']>=volume) & (df['CALL']==option_flag) & (df['strike'] >= Kmin)
                            & (df['strike'] <= Kmax)]
    
    return filtered_chain


def removing_nans(vol, option_prices, K, T, r, flag, q):
    idx=0
    for implied_vol in vol:
        if np.isnan(implied_vol):
            vol = np.delete(vol,idx)
            option_prices = np.delete(option_prices,idx)
            K = np.delete(K,idx)
            T = np.delete(T,idx)
            r = np.delete(r,idx)
            flag = np.delete(flag,idx)
            q = np.delete(q,idx)
        idx+=1
    
    return vol, option_prices, K, T, r, flag, q
            
    












