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


def black(sigma,q,t,T,r,S,K):
    F = S*np.exp(r*T)
    x = np.log(F/K)
    
    t = sigma/2
    h = x / sigma
    
    b = 0.5 * np.xp(-0.5*(h**2 + t**2))* 9
 
    return b
