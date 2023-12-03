#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:13:22 2023

@author: ted
"""

import numpy as np
from  py_vollib_vectorized import vectorized_implied_volatility as calculate_iv
from tools.Heston_COS_METHOD import heston_cosine_method


def heston_implied_vol_derivative(r,K,T,N,L,q,S,flag,sigma,rho,v0,vbar,kappa,precision,params_2b_calibrated):
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
    precision : Float
        precision of numerical differentiation

    Returns
    -------
    deriv_array : TYPE
        DESCRIPTION.

    """
    
    up = 1 + precision
    down= 1 - precision
    if 'vbar' in params_2b_calibrated:
        price_up = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar*up,v0,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        price_down = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar*down,v0,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        
        deriv_vbar = (price_up - price_down)/((up-down)*vbar)
    else:
        deriv_vbar = np.zeros(np.size(K))

    if 'sigma' in params_2b_calibrated:
        price_up = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma*up,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        price_down = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma*down,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        
        deriv_sigma = (price_up - price_down)/((up-down)*sigma)
    else:
        deriv_sigma = np.zeros(np.size(K))
    
    if 'rho' in params_2b_calibrated:
        price_up = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho*up,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        price_down = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho*down,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
       
        deriv_rho = (price_up - price_down)/((up-down)*rho)
    else:
        deriv_rho = np.zeros(np.size(K))
    
    if 'kappa' in params_2b_calibrated:
        price_up = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho,kappa*up,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        price_down = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0,sigma,rho,kappa*down,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
      
        deriv_kappa = (price_up - price_down)/((up-down)*kappa)
    else:
        deriv_kappa =np.zeros(np.size(K))
    
    if 'v0' in params_2b_calibrated:
        price_up = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0*up,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        price_down = 100*calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,vbar,v0*down,sigma,rho,kappa,flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy') 
        
        deriv_v0 = (price_up - price_down) / ((up-down)*v0)
    else:
        deriv_v0 = np.zeros(np.size(K))
        
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
    
    vbar_c = np.array([0.001,1])
    sigma_c = np.array([0.001,2])
    rho_c = np.array([-1,0.001])
    kappa_c = np.array([0.001,10])
    v0_c = np.array([0.001,1])
    
    constraints = np.stack((vbar_c,sigma_c,rho_c,kappa_c,v0_c))
    
    
    for i in range(5):
        
        if new_params[i,0] < constraints[i,0]:
            new_params[i,0] = (old_params[i,0] + constraints[i,0]) / 2
        
        if new_params[i,0] > constraints[i,1]:
            new_params[i,0] = (old_params[i,0] + constraints[i,1]) / 2
   
    return new_params