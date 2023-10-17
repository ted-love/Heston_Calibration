#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:31:40 2023

@author: ted
"""

import numpy as np

def chi_k(k,a,b,c,d):
    """

    Parameters
    ----------
    k : int
        Summation index.
    a : float
        lower bound of truncation.
    b : float
        upper bound of truncation.
    c : float
        lower bound of interal.
    d : float
        upper bound of truncation.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    M = (k*np.pi)/(b-a)

    return (1/(1+M**2)) * (np.cos(M*(d-a))*np.exp(d) - np.cos(M*(c-a))*np.exp(c)
                          + M*np.sin(M*(d-a))*np.exp(d) - M*np.sin(M*(c-a))*np.exp(c))
 

def psi_k(k,a,b,c,d):
    
    M = k*np.pi/(b-a)
    M[0] = 2
    psi = (1/M)*(np.sin(M*(d-a)) - np.sin(M*(c-a)))
    psi[0] = d-c
    return psi


def Characteric_Function(omega,r,rho,eta,lambd,u_0,u_bar,T):
    """
    The characteristic function of the Heston Model
    

    Parameters
    ----------
    omega : Float
        Input of the Characteristic function.
    r : Float
        Interest Rate.
    rho : Float
        Correlation between Stock and Volatility.
    eta : Float
        Vol of Vol.
    lambd : Float
        Rate of mean-reversion.
    u_0 : Float
        Initial Volatility.
    u_bar : Float
        Long-term volatility.
    T : Float
        Stirke.

    Returns
    -------
    psi_hes : Float
        Value of the Characteristic function.

    """
    
    W = lambd - 1j*rho*eta*omega
    
    D = np.sqrt( W**2 + (omega**2 + 1j*omega) * (eta**2))

    G = (W - D) / (W + D)
    
    exp_1 = np.exp(1j*omega*r*T + (u_0/(eta**2)) * ((1-np.exp(-D*T))/(1 - G*np.exp(-D*T))) * (W-D))
    
    exp_2 = np.exp( (lambd*u_bar)/(eta**2) * ((T * (W - D)) - 2*np.log( (1-G*np.exp(-D*T)) / (1-G) )))
    
    psi_hes = exp_1 * exp_2
    
    return psi_hes


def U_k(k,a,b):
         return 


def heston_cosine_method(S_0,K,T,N,r,q,u_bar,u_0,eta,rho,lambd,flag):
    """
    Vectorised Heston Cosine Expansion

    Parameters
    ----------
    x : Float
        Spot price of Stock.
    K : NumPy Array
        Numpy array of strikes.
    T : Float
        Expiry.
    N : Float
        Number of steps for the summation.
    r : Float
        Interest Rate.
    u_bar : Float
        Long-term volatility.
    u_0 : Float
        Initial Volatility.
    eta : Float
        Vol of Vol.
    rho : Float
        Correlation between Stock and Volatility.
    lambd : Float
        Rate of mean-reversion.
    flag : Int
        Type of European option. flag=1 for call option and flag=-1 for put option.

    Returns
    -------
    v : NumPy array
        Value of the European Options.

    """
    
    
    """
    Calculating the lower and upper bounds of the truncation
    """
    c_1 = r*T + (1-np.exp(-lambd*T)) * ((u_bar-u_0)/(2*lambd)) - 0.5*u_bar*T

    c2_scalar = 1/(8*lambd**3)
    
    c2_term1 = eta*T*lambd*np.exp(-lambd*T) * (u_0 - u_bar) * (8*lambd*rho - 4*eta)
    c2_term2 = lambd*rho*eta*(1-np.exp(-lambd*T)) * (16*u_bar - 8*u_0)
    c2_term3 = 2 * u_bar * lambd * T * (-4*lambd*rho*eta + eta**2 + 4*lambd**2)
    c2_term4 = (eta**2) * ( (u_bar - 2*u_0) * np.exp(-2*lambd*T) + u_bar * (6*np.exp(-lambd*T) - 7) + 2*u_0)
    c2_term5 = (8*lambd**2) * (u_0 - u_bar) * (1-np.exp(-lambd*T))
    
    c_2 = c2_scalar * (c2_term1 +  c2_term2 + c2_term3 + c2_term4 + c2_term5)

    L = 12

    a = c_1 - L*np.sqrt(abs(c_2))
    b = c_1 + L*np.sqrt(abs(c_2))
    
    
    k = np.linspace(0,N-1,N)

    omega = k*np.pi / (b-a)
    Uk = (2/(b-a)) * (-chi_k(k,a,b,a,0) + psi_k(k,a,b,a,0))
    
    x = np.log(S_0 / K)
    
    character_func = Characteric_Function(omega, r, rho, eta, lambd, u_0, u_bar, T)
    integrand = character_func * Uk * np.exp(1j*omega*(x -a))
    
    v= K * np.exp(-r*T) * np.real(0.5 * integrand[0] + np.sum(integrand[1:]))
    
    """
    Put-Call parity as the put is more accurate than the call for this method
    """
    if flag == 1:
        return v + S_0*np.exp(-q*T) - K*np.exp(-r*T)
    return v
