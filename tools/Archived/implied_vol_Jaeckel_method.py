#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:33:22 2023

@author: ted
"""

"""
This module calculates implied volatility. Based on "By Implication" by Peter Jackel
http://www.jaeckel.org/ByImplication.pdf

"""



import numpy as np
from scipy.stats import norm


def bs_price_normal(x,v,t):
    '''
    PARAMETERS
    ----------
    
    x : Normalized log-forward stock price, x=log(F/K) for F=Se^(rt)
    v : Volatility
    t : option type, t=1 for call, t=-1 for put
    
    
    Returns
    ----------
    log-normalized option price 
    
    '''
    b = t*np.exp(0.5*x) * norm.cdf(t*(x/v + 0.5*v) ,0 , 1) \
        - t*np.exp(-0.5*x)*norm.cdf(t*(x/v - 0.5*v) ,0 , 1)
    return b

def implied_vol(S,K,d,T,r,C,theta,tol,I):
    '''
    Based on the paper "By Implication" (Jaeckel, 2006). Will be able to calculate OTM options within 
    4 iterations. See: http://www.jaeckel.org/ByImplication.pdf 
    
    
    PARAMETERS
    ----------
    
    S : Price of stock
    K : Strike
    d : dividend
    T : Time till expiry
    r : risk-free rate
    C : market price of option
    theta : CALL or PUT. theta = 1 for CALL, theta = -1 for PUT
    tol : tolerance of volatility estimation
    I : iterations to estimate volatility
    
    Returns
    -------
    Implied volatility
    
    '''
    
    F= S * np.exp((r-d)*T)
    x=np.log(F/K)
    
    beta=C/(np.exp(-r*T)*np.sqrt(F*K))
    
    sigma_c = np.sqrt(2*abs(x))
    
    # Normalised intrinsic value
    iota = np.heaviside(theta*x,0)*theta * (np.exp(0.5 * x) - np.exp(-0.5*x))   
    
    
    # Normalised black-scholes   
      
    b_c =  bs_price_normal(x,sigma_c,theta)

    if beta<b_c:
        
        # sigma_low
        sigma_0 =  np.sqrt( (2*x**2)/ (abs(x) - 4*np.log( (beta - iota) / (b_c - iota))))

    else:
        
        # sigma_high
        inside = norm.cdf(-np.sqrt(abs(x)/2))
        sigma_0 = -2*(norm.ppf( (np.exp(0.5*theta*x) - beta) / (np.exp(0.5*theta*x) - b_c) * inside))
        
    
    b = bs_price_normal(x,sigma_0,theta)
    
    sigma_old = sigma_0
    
    k=0
    while k<I:

        b_prime = np.exp(-0.5*((x/sigma_old)**2) -0.5*((0.5*sigma_old)**2)) / np.sqrt(2*np.pi)

        b=bs_price_normal(x,sigma_old,theta)
        
        if beta < b_c:
            v_tilde = np.log( (beta - iota) / (b - iota)) * ((b-iota)/b_prime)
            
        else:
            v_tilde = (b - beta)/b_prime 

        sigma_new = sigma_old + v_tilde
        
        if abs(sigma_new / sigma_old - 1) < tol:
            break
        
        sigma_old = sigma_new
        print(sigma_old)
        k+=1
        
    return sigma_old/np.sqrt(T)  
