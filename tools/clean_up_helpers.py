#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:40:59 2023

@author: ted
"""
import numpy as np

def df_to_numpy(DF, Treasury_Curve, Implied_Dividend_Curve):
    
    T = DF['dte'].to_numpy()
    K = DF['strike'].to_numpy()
    r = Treasury_Curve(T)
    q = Implied_Dividend_Curve(T)
    option_prices = DF['midPrice'].to_numpy()
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