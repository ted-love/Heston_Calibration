#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:00:51 2023

@author: ted
"""
import numpy as np
from scipy.stats import norm

def bs_call(sigma,q,t,T,r,S,K):
    
    d1 = 1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r - q + 0.5*(sigma**2))*(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)

    F = S*np.exp((r-q)*(T-t))

    C = np.exp(-r*(T-t)) * (F*norm.cdf(d1,0,1) - K*norm.cdf(d2,0,1))

    return C

def bs_put(sigma,q,t,T,r,S,K):
    
    d1 = 1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r - q + 0.5*(sigma**2))*(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    
    F= S*np.exp((r-q)*(T-t))
    
    P = np.exp(-r*(T-t)) * (K*norm.cdf(-d2,0,1)-F*norm.cdf(-d1,0,1))
    
    return P

def vega(sigma,q,t,T,r,S,K):
    
    d1 =  1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r -q+ 0.5*sigma**2)*(T-t))
    v = S*np.exp(-(r-q)*(T-t))*norm.pdf(d1)*np.sqrt(T-t)
    
    return v


from Jackel_method import Jackel_method
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
import time

C=42.9262345428
S=100
K=50
d=0
r=0.05
T=1


start = time.time()
imp_pvv = implied_vol(C,S,K,T,r,'c',d,model='black_scholes',return_as='numpy')
end = time.time()
time_pvv = end-start


start = time.time()
imp_jm = Jackel_method(S,K,d,T,r,C,1,0.0000001,200)
end = time.time()
time_jm = end-start
