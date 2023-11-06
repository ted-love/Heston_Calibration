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
 

def psi_k(k,a,b,c,d,mode):
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
    mode : int
        Direction of vectorisation.
        1 for vectorising the strikes K.
        0 for vectorising the N summation

    Returns
    -------
    psi : float
        Cosine series coefficients.

    """
    if mode==1:
        if k==0:
            return d - c
        
        M = k*np.pi/(b-a)
        psi = (1/M)*(np.sin(M*(d-a)) - np.sin(M*(c-a)))
        return psi
    if mode == 0:
        M = k*np.pi/(b-a)
        M[0] = 2
        psi = (1/M)*(np.sin(M*(d-a)) - np.sin(M*(c-a)))
        psi[0] = d-c
    
        return psi


def charact_func(omega,r,rho,eta,lambd,u_0,u_bar,T):
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
    charact_func : float
        Value of the Characteristic function.

    """
    
    W = lambd - 1j*rho*eta*omega
    
    D = np.sqrt( W**2 + (omega**2 + 1j*omega) * (eta**2))

    G = (W - D) / (W + D)
    
    exp_1 = np.exp(1j*omega*r*T + (u_0/(eta**2)) * ((1-np.exp(-D*T))/(1 - G*np.exp(-D*T))) * (W-D))
    
    exp_2 = np.exp( (lambd*u_bar)/(eta**2) * ((T * (W - D)) - 2*np.log( (1-G*np.exp(-D*T)) / (1-G) )))
    
    charact_func = exp_1 * exp_2
    
    return charact_func


def U_k(k,a,b,mode):
    """
    

    Parameters
    ----------
    k : int
        Summation index.
    a : float
        lower bound of truncation.
    b : float
        upper bound of truncation.
    mode : int
        Direction of vectorisation.
        1 for vectorising the strikes K.
        0 for vectorising the N summation

    Returns
    -------
    float
        U_k.

    """
    return (2/(b-a)) * (-chi_k(k,a,b,a,0) + psi_k(k,a,b,a,0,mode))

def cumulants_truncation(L,T,r,u_bar,u_0,eta,rho,lambd):
    """
    Cumulants determine the truncation length of the characteristic function

    Parameters
    ----------
    L : float
        Truncation range magnitude.
    T : float
        Expiry.
    r : float
        Interest rate.
    u_bar : float
        long-term vol.
    u_0 : float
        Initial vol.
    eta : float
        vol of vol.
    rho : float
        correlation betwen stock and vol.
    lambd : float
        Rate of mean-reversion.

    Returns
    -------
    a : float
        Lower bound of truncation.
    b : float
        upper bound of truncation.

    """
    c_1 = r*T + (1-np.exp(-lambd*T)) * ((u_bar-u_0)/(2*lambd)) - 0.5*u_bar*T
    
    c2_scalar = 1/(8*lambd**3)
    c2_term1 = eta*T*lambd*np.exp(-lambd*T) * (u_0 - u_bar) * (8*lambd*rho - 4*eta)
    c2_term2 = lambd*rho*eta*(1-np.exp(-lambd*T)) * (16*u_bar - 8*u_0)
    c2_term3 = 2 * u_bar * lambd * T * (-4*lambd*rho*eta + eta**2 + 4*lambd**2)
    c2_term4 = (eta**2) * ( (u_bar - 2*u_0) * np.exp(-2*lambd*T) + u_bar * (6*np.exp(-lambd*T) - 7) + 2*u_0)
    c2_term5 = (8*lambd**2) * (u_0 - u_bar) * (1-np.exp(-lambd*T))
    c_2 = c2_scalar * (c2_term1 +  c2_term2 + c2_term3 + c2_term4 + c2_term5)

    a = c_1 - L*np.sqrt(abs(c_2))
    b = c_1 + L*np.sqrt(abs(c_2))
    
    return a,b


def heston_cosine_method(S_0,K,T,N,L,r,q,u_bar,u_0,eta,rho,lambd,flag):
    """
    
    Vectorised Heston Cosine Expansion.
    

    Parameters
    ----------
    S_0 : float
        Spot price of Stock.
    K : NumPy Array
        Numpy array of strikes.
    T : Float
        Expiry.
    N : float
        Number of steps for the summation.
    L : float
        Truncation range magnitude.
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
        Rate of mean-reversion of the volatility.
    flag : int
        Type of European option. flag=1 for call option and flag=-1 for put option.

    Returns
    -------
    v : NumPy array
        Value of the European Options.

    """
    
    
    a, b = cumulants_truncation(L,T,r,u_bar,u_0,eta,rho,lambd)

 
    k = np.linspace(0,N-1,N).reshape(N,1)
    omega = k*np.pi / (b-a)
    character_func = charact_func(omega, r, rho, eta, lambd, u_0, u_bar, T)
    Uk = U_k(k,a,b,0)
    
    x = np.log(S_0/K)
    
    integrand = character_func * Uk * np.exp(1j*omega*(x-a))
    
    v =  K * np.exp(-r*T) * np.real( 0.5*integrand[0,:] \
                                + np.sum(integrand[1:,:],axis=0,keepdims=True))

    if flag == 1:
        
        return v + S_0*np.exp(-q*T) - K*np.exp(-r*T)
    
    return v



def charact_deriv(omega,eta,T,rho,u_0,u_bar,lambd):
    """
    The derivative of the characteristic function wrt its parameters

    Parameters
    ----------
    omega : NumPy array
        Independent variable of the characteristinc function as an nxm array.
    eta : Float
        Vol of Vol.
    T : Float
        Expiration.
    rho : Float
        Correlation between stock and volatility.
    u_0 : Float
        Initial volatility.
    u_bar : Float
        Long-term volatility.
    lambd : Float
        Rate of mean-reversion of the volatility.

    Returns
    -------
    NumPy Array
        The derivatives of the characteristic function in a 3-dim array.

    """
    xi = lambd - eta*rho*1j*omega
    d = np.sqrt(xi**2 + (eta**2)*(omega**2+1j*omega))
    
    A1 = (omega**2 + 1j*omega)*np.sinh(d*T/2)
    A2 = (d/u_0) * np.cosh(d*T/2) + (xi/u_0) * np.sinh(d*T/2)
    A=A1/A2
    
    B = d*np.exp(lambd*T/2)/(u_0*A2)
    
    D = np.log(d/u_0) + ((lambd-d)*T)/2 - np.log((d+xi)/(2*u_0) + ((d-xi)/(2*u_0)) * np.exp(-d*T))
    
    # Derivatives where the subscript is what the derivative depends on
    d_rho = -xi*eta*1j*omega/d
    d_eta = (rho/eta - (1/xi)) * d_rho + (eta*omega**2)/d
    
    A1_rho = -(1j*omega*(omega**2 + 1*omega)*T*xi*eta)/(2*d) * np.cosh(d*T/2)
    A2_rho = -(eta*omega*1j*(2 + xi*T))/(2*d*u_0) * (xi * np.cosh(d*T/2) + d*np.sinh(d*T/2))
    A_rho = A1_rho / A2 - (A/A2)*A2_rho
    
    A1_eta = (((omega**2 + 1j*omega)*T)/2) * (d_eta*np.cosh(d*T/2))
    A2_eta = rho*A2_rho/eta - ((2+T*xi)/(u_0*T*xi*omega*1j) * A1_rho) + eta*T*A1/(2*u_0)
    A_eta = A1_eta/A2 - (A/A2)*A2_eta
    
    B_rho = (np.exp(lambd*T/2)/u_0) * (d_rho/A2 - d*A2_rho/(A2**2))
    B_lambd = 1j*B_rho/(eta*omega) + B*T/2    
    
    h1 = -A/u_0
    h2 = 2*lambd*D/(eta**2) - lambd*rho*T*1j*omega/eta
    h3 = -A_rho + ((2*lambd*u_bar)/(d*eta**2)) * (d_rho - (d/A2) * A2_rho) - lambd*u_bar*T*1j*omega/eta
    h4 = A_rho/(eta*1j*omega) + 2*u_bar*D/(eta**2) + \
        (2*lambd*u_bar*B_lambd)/(B*eta**2) - u_bar*rho*T*1j*omega/eta
    h5 = -A_eta - 4*lambd*u_bar*D/(eta**3) + ((2*lambd*u_bar)/(d*eta**2))*(d_eta - d*A2_eta/A2) \
        + lambd*u_bar*rho*T*1j*omega/(eta**2)

    return np.array([h1,h2,h3,h4,h5])
       

def heston_cosine_derivatives(S_0,K,T,N,L,r,q,u_bar,u_0,eta,rho,lambd):
    """
    
    Derivative of the vectorised Heston Cosine Expansion.
    

    Parameters
    ----------
    S_0 : float
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
    
    Returns
    -------
    v : NumPy array
        Call Option Derivatives.

    """
    
    
    a, b = cumulants_truncation(L,T,r,u_bar,u_0,eta,rho,lambd)
    
    
    x = np.log(S_0/K)
    k = np.linspace(0.000001,N-1,N).reshape(N,1)
    
    omega = k*np.pi / (b-a)
    character_func = charact_func(omega, r, rho, eta, lambd, u_0, u_bar, T)
    Uk = U_k(k,a,b,0)
    
    integrand = character_func * Uk * np.exp(1j*omega*(x-a))
    character_derivatives=charact_deriv(omega,eta,T,rho,u_0,u_bar,lambd)

    v=np.empty([5,np.size(K)])
    for i in range(5):
    
        v[i,:] =  K * np.exp(-r*T) * np.real( 0.5*character_derivatives[i,0,:]*integrand[0,:] \
                                + np.sum(character_derivatives[i,1:,:]*integrand[1:,:],axis=0,keepdims=True))
    return v
    