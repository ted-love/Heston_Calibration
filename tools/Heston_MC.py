#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:04:29 2023

@author: ted
"""
import numpy as np

def heston_model(kappa, omega, N, Nsims, V0, S0, p, r, K, T,theta):
    # initialise other parameters
    dt = T/N

    
    ####### FULL TRUNCATION ########
    f1 = lambda x : x
    f2 = lambda x : np.maximum(x,0)
    f3 = lambda x : np.maximum(x,0)
    
    Z_v = np.random.normal(0,1,[Nsims,N])
    Z_2 = np.random.normal(0,1,[Nsims,N])
    Z_s = p*Z_v + np.sqrt(1-(p**2))*Z_2
    
    V = np.full(shape=(Nsims,N+1),fill_value=V0)
    V_tilde = V

    S = np.full(shape=(Nsims,N+1),fill_value=S0)
 
    for i in range(1,N+1):

        S[:,i] = S[:,i-1]*np.exp((r-(0.5*V[:,i-1]))*dt + np.sqrt(V[:,i-1])*np.sqrt(dt)*Z_s[:,i-1])
        
        V_tilde[:,i] = f1(V_tilde[:,i-1]) + kappa*dt*(theta - f2(V_tilde[:,i-1])) \
            + omega*np.sqrt(f3(V_tilde[:,i-1]))*Z_v[:,i-1]*np.sqrt(dt)
            
        V[:,i] = f3(V_tilde[:,i])
        
    
    C = np.exp(-r*T)*np.maximum(S[:,-1]-K,0)
    C = np.mean(C)

    return C

