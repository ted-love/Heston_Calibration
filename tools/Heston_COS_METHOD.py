#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:31:40 2023

@author: ted
"""
import warnings
warnings.filterwarnings('ignore')
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
        lower bound of integral.
    d : float
        upper bound of integral.

    Returns
    -------
    float
        Cosine series coefficients.

    """
        
    M = (k*np.pi)/(b-a)
    
    return (1/(1+M**2)) * (np.cos(M*(d-a))*np.exp(d) - np.cos(M*(c-a))*np.exp(c)
                          + M*np.sin(M*(d-a))*np.exp(d) - M*np.sin(M*(c-a))*np.exp(c))
 

def psi_k(k,a,b,c,d):
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
        lower bound of integral.
    d : float
        upper bound of integral.

    Returns
    -------
    psi : float
        Cosine series coefficients.

    """
    M = k*np.pi/(b-a)
    M[0] = 2
    psi = (1/M)*(np.sin(M*(d-a)) - np.sin(M*(c-a)))
    psi[0] = d-c

    return psi


def charact_func(omega,r,q,rho,sigma,kappa,v0,v_bar,T):
    """
    
    The characteristic function of the Heston Model

    Parameters
    ----------
    omega : NumPy Array
        Input of the Characteristic function.
    r : NumPy array/Float
        Interest Rate.
    q : NumPy array/Float
        Dividend Rate.
    rho : Float
        Correlation between Stock and Volatility.
    sigma : Float
        Vol of Vol.
    kappa : Float
        Rate of mean-reversion.
    v0 : Float
        Initial Volatility.
    v_bar : Float
        Long-term volatility.
    T : Float
        Stirke.

    Returns
    -------
    charact_func : float
        Value of the Characteristic function.

    """
   
    W = kappa - 1j*rho*sigma*omega
    
    D = np.sqrt( W**2 + (omega**2 + 1j*omega) * (sigma**2))
    
    G = (W - D) / (W + D)
    
    exp_1 = np.exp(1j*omega*(r-q)*T + (v0/(sigma**2)) * ((1-np.exp(-D*T))/(1 - G*np.exp(-D*T))) * (W-D))
    
    exp_2 = np.exp( (kappa*v_bar)/(sigma**2) * ((T * (W - D)) - 2*np.log( (1-G*np.exp(-D*T)) / (1-G) )))
    
    charact_func = exp_1 * exp_2
    
    return charact_func


def U_k(k,a,b):
    """
    

    Parameters
    ----------
    k : int
        Summation index.
    a : float
        lower bound of truncation.
    b : float
        upper bound of truncation.

    Returns
    -------
    float
        U_k.

    """
    return (2/(b-a)) * (-chi_k(k,a,b,a,0) + psi_k(k,a,b,a,0))

def cumulants_truncation(L,T,r,q,v_bar,v0,sigma,rho,kappa):
    """
    Cumulants determine the truncation length of the characteristic function.

    Parameters
    ----------
    L : float
        Truncation range magnitude.
    T : float
        Expiry.
    r : float
        Interest rate.
    v_bar : float
        long-term vol.
    v0 : float
        Initial vol.
    sigma : float
        vol of vol.
    rho : float
        correlation betwen stock and vol.
    kappa : float
        Rate of mean-reversion.

    Returns
    -------
    a : float
        Lower bound of truncation.
    b : float
        upper bound of truncation.

    """

    c_1 = (r-q)*T + (1-np.exp(-kappa*T)) * ((v_bar-v0)/(2*kappa)) - 0.5*v_bar*T


    c2_scalar = 1/(8*kappa**3)
    c2_term1 = sigma*T*kappa*np.exp(-kappa*T) * (v0 - v_bar) * (8*kappa*rho - 4*sigma)
    c2_term2 = kappa*rho*sigma*(1-np.exp(-kappa*T)) * (16*v_bar - 8*v0)
    c2_term3 = 2 * v_bar * kappa * T * (-4*kappa*rho*sigma + sigma**2 + 4*kappa**2)
    c2_term4 = (sigma**2) * ( (v_bar - 2*v0) * np.exp(-2*kappa*T) + v_bar * (6*np.exp(-kappa*T) - 7) + 2*v0)
    c2_term5 = (8*kappa**2) * (v0 - v_bar) * (1-np.exp(-kappa*T))
    c_2 = c2_scalar * (c2_term1 +  c2_term2 + c2_term3 + c2_term4 + c2_term5)

    a = c_1 - L*np.sqrt(abs(c_2))
    b = c_1 + L*np.sqrt(abs(c_2))
    
    return a,b


def heston_cosine_method(S,K,T,N,L,r,q,v_bar,v0,sigma,rho,kappa,flag):
    """
    
    Vectorised Heston Cosine Expansion.
    

    Parameters
    ----------
    S : float
        Spot price of Stock.
    K : NumPy Array
        Numpy array of strikes.
    T : NumPy array/Float
        Expiry.
    N : float
        Number of steps for the summation.
    L : float
        Truncation range magnitude.
    r : NumPy Array/Float
        Interest Rate.
    q : NumPy Array/Float
        Diidend Rate.
    v_bar : Float
        Long-term volatility.
    v0 : Float
        Initial Volatility.
    sigma : Float
        Vol of Vol.
    rho : Float
        Correlation between Stock and Volatility.
    kappa : Float
        Rate of mean-reversion of the volatility.
    flag : int
        Type of European option. flag=1 for call option and flag=-1 for put option.

    Returns
    -------
    v : NumPy array
        Value of the European Options.

    """
    
    
    a, b = cumulants_truncation(L,T,r,q,v_bar,v0,sigma,rho,kappa)

 
    k = np.linspace(0,N-1,N).reshape(N,1)
    omega = k*np.pi / (b-a)
    character_func = charact_func(omega, r, q, rho, sigma, kappa, v0, v_bar, T)
    Uk = U_k(k,a,b)
    
    x = np.log(S/K)
    
    integrand = character_func * Uk * np.exp(1j*omega*(x-a))
    
    v =  K * np.exp(-r*T) * np.real( 0.5*integrand[0,:] \
                                + np.sum(integrand[1:,:],axis=0,keepdims=True))

    if np.size(K) > 1:
      
        for i in range(np.size(K)):
            
            if flag[i] == 'c':
                
                v[0,i] = v[0,i] + S*np.exp(-q[i]*T[i]) - K[i]*np.exp(-r[i]*T[i])
                
        return v
    
    else:
        
        if flag == 'c':
            
            return v + S*np.exp(-q*T) - K*np.exp(-r*T)
        
        return v



def charact_deriv(omega,sigma,T,rho,v0,v_bar,kappa):
    """
    The derivative of the characteristic function wrt its parameters

    Parameters
    ----------
    omega : NumPy array
        Independent variable of the characteristinc function as an nxm array.
    sigma : Float
        Vol of Vol.
    T : Float
        Expiration.
    rho : Float
        Correlation between stock and volatility.
    v0 : Float
        Initial volatility.
    v_bar : Float
        Long-term volatility.
    kappa : Float
        Rate of mean-reversion of the volatility.

    Returns
    -------
    NumPy Array
        The derivatives of the characteristic function in a 3-dim array.

    """
    xi = kappa - sigma*rho*1j*omega
    d = np.sqrt(xi**2 + (sigma**2)*(omega**2+1j*omega))
    
    A1 = (omega**2 + 1j*omega)*np.sinh(d*T/2)
    A2 = (d/v0) * np.cosh(d*T/2) + (xi/v0) * np.sinh(d*T/2)
    A=A1/A2
    
    B = d*np.exp(kappa*T/2)/(v0*A2)
    
    D = np.log(d/v0) + ((kappa-d)*T)/2 - np.log((d+xi)/(2*v0) + ((d-xi)/(2*v0)) * np.exp(-d*T))
    
    # Derivatives where the subscript is what the derivative depends on.
    d_rho = -xi*sigma*1j*omega/d
    d_sigma = (rho/sigma - (1/xi)) * d_rho + (sigma*omega**2)/d
    
    A1_rho = -((1j*omega*(omega**2 + 1*omega)*T*xi*sigma)/(2*d)) * np.cosh(d*T/2)
    A2_rho = -(sigma*omega*1j*(2 + xi*T))/(2*d*v0) * (xi * np.cosh(d*T/2) + d*np.sinh(d*T/2))
    A_rho = A1_rho / A2 - (A/A2)*A2_rho
    
    A1_sigma = (((omega**2 + 1j*omega)*T)/2) * (d_sigma*np.cosh(d*T/2))
    A2_sigma = rho*A2_rho/sigma - (((2+T*xi)/(v0*T*xi*omega*1j)) * A1_rho) + sigma*T*A1/(2*v0)
    A_sigma = A1_sigma/A2 - (A/A2)*A2_sigma
    
    B_rho = (np.exp(kappa*T/2)/v0) * (d_rho/A2 - d*A2_rho/(A2**2))
    B_kappa = 1j*B_rho/(sigma*omega) + B*T/2    
    
    # charact_v0
    h1 = -A/v0
    
    # charact_v_bar
    h2 = 2*kappa*D/(sigma**2) - kappa*rho*T*1j*omega/sigma
    
    # charact_rho
    h3 = -A_rho + ((2*kappa*v_bar)/(d*sigma**2)) * (d_rho - (d/A2) * A2_rho)\
        - kappa*v_bar*T*1j*omega/sigma
    
    # charact_kappa
    h4 = A_rho/(sigma*1j*omega) + 2*v_bar*D/(sigma**2) + \
        (2*kappa*v_bar*B_kappa)/(B*sigma**2) - v_bar*rho*T*1j*omega/sigma
    
    # charact_sigma
    h5 = -A_sigma - 4*kappa*v_bar*D/(sigma**3) + ((2*kappa*v_bar)/(d*sigma**2))*(d_sigma - d*A2_sigma/A2) \
        + kappa*v_bar*rho*T*1j*omega/(sigma**2)
    return  np.array([h1,h2,h3,h4,h5])
  #  return np.array([h1,h2,h3,h4,h5])
       


def heston_cosine_derivatives(S,K,T,N,L,r,q,v_bar,v0,sigma,rho,kappa):
    """
    
    Derivative of the vectorised Heston Cosine Expansion.
    

    Parameters
    ----------
    S : float
        Spot price of Stock.
    K : NumPy Array
        Strike prices.
    T : Float
        Expiry.
    N : float
        Number of steps for the summation.
    L : float
        Truncation range magnitude.
    r : Float
        Interest Rate.
    v_bar : Float
        Long-term volatility.
    v0 : Float
        Initial Volatility.
    sigma : Float
        Vol of Vol.
    rho : Float
        Correlation between Stock and Volatility.
    kappa : Float
        Rate of mean-reversion.
    
    Returns
    -------
    v : NumPy array
        Call Option Derivatives.

    """
    
    
    a, b = cumulants_truncation(L,T,r,v_bar,v0,sigma,rho,kappa)
    
    
    x = np.log(S/K)
    k = np.linspace(0.000001,N-1,N).reshape(N,1)
    
    
    
    omega = k*np.pi / (b-a)
    character_func = charact_func(omega, r, rho, sigma, kappa, v0, v_bar, T)
    Uk = U_k(k,a,b)
    
    integrand = character_func * Uk * np.exp(1j*omega*(x-a))
    character_derivatives = charact_deriv(omega,sigma,T,rho,v0,v_bar,kappa)
    
    v=np.empty([5,np.size(K)])
    for i in range(5):
    
        v[i,:] =  K * np.exp(-r*T) * np.real( 0.5*character_derivatives[i,0,:]*integrand[0,:] \
                                + np.sum(character_derivatives[i,1:,:]*integrand[1:,:],axis=0,keepdims=True))
    return v
   