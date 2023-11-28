#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:29 2023

@author: ted
"""


"""
This module is a box-constrained Levenberg-Marquardt Algorithm for the Heston Model. 

"""


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tools.Heston_COS_METHOD import heston_cosine_method
from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv
from scipy.linalg import inv
from tools.heston_derivative_constraints import heston_constraints,heston_implied_vol_derivative


def levenberg_Marquardt(old_params,C_market,I,w,S,K,T,N,L,r,q,v_bar,v0,sigma,rho,kappa,flag,precision,params_2b_calibrated):
    """
    

    Parameters
    ----------
    old_params : NumPy Array
        Parameters to be calibrated.
    C_market : NumPy Array
        Implied vol of market data to be calibrated on.
    I : Int
        Number of iterations.
    w : Float
        Initial damping factor weight.
    S : Float
        Spot price.
    K : NumPy Array
        Strike.
    T : NumPy Array
        Expiry.
    N : Int
        Number of steps of summation in the COS-expansion.
    L : Float
        Range of truncation in the COS-expansion.
    r : NumPy Array
        Interest Rate
    q : Float
        Dividend yield.
    v_bar : Float
        Long-Term vol.
    v0 : Float
        Initial vol.
    sigma : Float
        vol of vol.
    rho : Float
        Correlation between Stock and Volatility.
    kappa : Float
        Rate of mean-reversion.
    flag : Str
        Option typ, 'c' for call and 'p' for put.
    precision : Float
        precision of numerical differentiation

    Returns
    -------
    old_params : NumPy Array
        Calibrated parameters.
    counts_accepted : Int
        Number of iterations accepted.
    counts_rejected : Int
        number of iterations rejected.

    """
    
    skip=1
    
    nu = 2
  
    M = np.size(K)
    eps_1 = 1e-5
    eps_2 = eps_1
    eps_3 = 1e-10
    f_x = C_market - (calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,old_params[0,0],old_params[4,0],old_params[1,0],old_params[2,0],old_params[3,0],flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100).reshape(np.size(K),1)
    F_x = 0.5 * (1/M) * f_x.T @ f_x
    J = -1*heston_implied_vol_derivative(r,K,T,N,L,q,S,flag,old_params[1,0],old_params[2,0],old_params[4,0],old_params[0,0],old_params[3,0], precision, params_2b_calibrated)
    
    g = (1/M) * J @ f_x

    A = J@J.T 
    mu = w * np.amax(np.diag(J@J.T))
    print('mu: ', mu)
    
    counts_accepted=0
    counts_rejected=0
    k=0
    while k<I:
        
        # Calculating step of the parameters. inv is linalg.inv
        delta_params = inv((A + mu*np.eye(np.size(old_params)))) @ -g
        
        new_params = heston_constraints(old_params + delta_params, old_params)
        
        # Cost-Function of new step
        f_xh = C_market - (calculate_iv(heston_cosine_method(S,K,T,N,L,r,q,new_params[0],new_params[4],new_params[1],new_params[2],new_params[3],flag),S, K, T, r, flag, q, model='black_scholes_merton',return_as='numpy')*100).reshape(np.size(K),1)
        F_xh = 0.5 * (1/M) * f_xh.T@f_xh
  
        gain_ratio = (F_x[0] - F_xh[0]) / (0.5*delta_params.T @ (mu*delta_params - g))
   
        if gain_ratio > 0:
            counts_accepted+=1
            
            
            old_params = new_params[:]


            J = -1*heston_implied_vol_derivative(r,K,T,N,L,q,S,flag,old_params[1,0],old_params[2,0],old_params[4,0],old_params[0,0],old_params[3,0], precision, params_2b_calibrated)
            
            f_x = f_xh[:]
            F_x = 0.5 * (1/M) * f_x.T @ f_x
                        
            g = (1/M) * J @ f_x
            A = J@J.T 
            
            # Adjust damping factor
            mu = mu*np.maximum(1/3 , 1-(2*gain_ratio - 1)**3)[0,0]
            nu = 2

             
            if k % 10 == 0:
                print('\nIteration: ', k,'\n', old_params)

            if mu==np.inf:
                print('overflow')
                break
        else:

            counts_rejected +=1
            try:
                mu=mu*nu
                nu*=2
               
            
            # If we the damping factor goes off to infinity
            except:
                print("overflow")
                skip=0
                break
            if mu==np.inf:
                print("overflow")
                skip=0
                break
            continue

            
        if F_xh <= eps_1:
            print("Loss function close to zero")
            skip=0
            break
        
        if np.amax(g)<=eps_2:
            print("Small J")
            skip=0
            break
        """
        if np.linalg.norm(delta_params)/np.linalg.norm(old_params) < eps_3:
            print("Steps converging to 0!")
            skip = 0
            break
        """

        k+=1
        
    if skip==1:
        print('Exceeded maximum iterations')

    return old_params, counts_accepted, counts_rejected
