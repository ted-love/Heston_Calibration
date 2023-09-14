#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:07:34 2023

@author: ted
"""
from bs_model import bs_call,vega,bs_put
from AMO_BIN import AMO_BIN,vega_bin
import numpy as np


def implied_vol(q,t,T,r,S0,K,V0,market_price,tol):
    
    max_iter = 100 #max number of iterations
    vol_old = V0 #initial guess
    
    for k in range(max_iter):
        
        C = bs_call(vol_old,q,t,T,r,S0,K)

        C0 = C
        Cprime= vega(vol_old,q,t,T,r,S0,K)
        C = C0 - market_price


        
        vol_new = np.max(0.0001,vol_old - C[0]/Cprime[0])

        C_new = bs_call(vol_new,q,t,T,r,S0,K)
        
        
        
        if (abs(vol_old - vol_new) < tol or abs(C_new - market_price) < tol):
            break
        
        vol_old = vol_new
        
    
    return vol_old

def AMO_implied_vol(sigma,q,t,T,r,S0,K,V0,market_price,tol):
    
    max_iter = 100 #max number of iterations
    vol_old = V0 #initial guess
    Nsteps=10
    h=0.001
    for k in range(max_iter):

        C = AMO_BIN(vol_old,K,T,S0,r,q,Nsteps,'C')
        C0 = C
        Cprime= vega_bin(vol_old,h,K,T,S0,r,q,Nsteps,'C')

        C = C0 - market_price
        
        vol_new = vol_old - C/Cprime

        C_new = bs_call(vol_new,q,t,T,r,S0,K)

        if (abs(vol_old - vol_new) < tol or abs(C_new - market_price) < tol):
            break
        
        vol_old = vol_new
        
    
    return vol_old


