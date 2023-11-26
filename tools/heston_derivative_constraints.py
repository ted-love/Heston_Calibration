#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:13:22 2023

@author: ted
"""

import numpy as np
from  py_vollib_vectorized import vectorized_implied_volatility as iv
from tools.Heston_COS_METHOD import heston_cosine_method


def heston_implied_vol_derivative(r,K,T,N,L,q,S,flag,sigma,rho,v0,vbar,kappa):
    """
    

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    flag : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    v0 : TYPE
        DESCRIPTION.
    vbar : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.

    Returns
    -------
    deriv_array : TYPE
        DESCRIPTION.

    """
    
    up=1.01 
    down=0.99

    price_up = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar*up,v0,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    price_down = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar*down,v0,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    
    deriv_vbar = (price_up - price_down)/((up-down)*vbar)


    price_up = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma*up,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    price_down = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma*down,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    
    deriv_sigma = (price_up - price_down)/((up-down)*sigma)


    price_up = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho*up,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    price_down = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho*down,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
   
    deriv_rho = (price_up - price_down)/((up-down)*rho)
           
    
    price_up = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho,kappa*up,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    price_down = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho,kappa*down,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
  
    deriv_kappa = (price_up - price_down)/((up-down)*kappa)
    
    
    price_up = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0*up,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    price_down = iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0*down,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100
    
    deriv_v0 = (price_up - price_down) / ((up-down)*v0)


    deriv_array = np.array([deriv_vbar,deriv_sigma,deriv_rho,deriv_kappa,deriv_v0])
    return deriv_array


def heston_constraints(new_params, old_params):
    """
    Applying constraints to the new parameters. If the new parameter value is outside the constraint bounds,
    the new parameter is changed to be the midpoint between the old parameter value and the boundary it exceeds.
    

    Parameters
    ----------
    new_params : NumPy Array
        New parameters before constraints are appliedd.
    old_params : NumPy Array
        Old parameters before adding delta_params. 

    Returns
    -------
    new_params : NumPy array
        new parameters that satisfy the constraints.

    """
    
    vbar_c = [0.001,1]
    sigma_c = [0.001,2]
    rho_c =[-1,0.001]
    kappa_c = [0.001,10]
    v0_c = [0.001,1]
    
    
    if new_params[0,0] < vbar_c[0]:
        new_params[0,0] = (old_params[0,0] + vbar_c[0]) / 2
        
    if new_params[0,0] > vbar_c[1]:
        new_params[0,0] = (old_params[0,0] + vbar_c[1]) / 2
    
    if new_params[1,0] < sigma_c[0]:
        new_params[1,0] = (old_params[1,0] + sigma_c[0]) / 2  
        
    if new_params[1,0] > sigma_c[1]:
        new_params[1,0] = (old_params[1,0] + sigma_c[1]) / 2  
        
    if new_params[2,0] < rho_c[0]:
        new_params[2,0] = (old_params[2,0] + rho_c[0]) / 2
        
    if new_params[2,0] > rho_c[1]:
        new_params[2,0] = (old_params[2,0] + rho_c[1]) / 2
        
    if new_params[3,0] < kappa_c[0]:
        new_params[3,0] = (old_params[3,0] + kappa_c[0]) / 2
        
    if new_params[3,0] > kappa_c[1]:
        new_params[3,0] = (old_params[3,0] + kappa_c[1]) / 2
    
    if new_params[4,0] < v0_c[0]:
        new_params[4,0] = (old_params[4,0] + v0_c[0]) / 2
        
    if new_params[4,0] > v0_c[1]:
        new_params[4,0] = (old_params[4,0] + v0_c[1]) / 2
    
    return new_params