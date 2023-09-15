#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:00:36 2023

@author: ted
"""
import numpy as np
from scipy.integrate import quad,trapezoid


def heston_charfunc(phi, S_0, V_0, kappa, theta, sigma, rho, lambd, T, r):
    
    a = kappa*theta
    b = kappa+lambd
    
    rspi = rho*sigma*phi*1j
    
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )
    
    g = (b-rspi+d)/(b-rspi-d)
    
    exp1 = np.exp(r*phi*1j*T)
    term2 = S_0**(phi*1j) * ( (1-g*np.exp(d*T))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*T*(b-rspi+d)/sigma**2 + V_0*(b-rspi+d)*( (1-np.exp(d*T))/(1-g*np.exp(d*T)) )/sigma**2)
    
    return exp1*term2*exp2


def integrand(phi, S_0, V_0, kappa, theta, sigma, rho, lambd, T, r, K):
    
    args = (S_0, V_0, kappa, theta, sigma, rho, lambd, T, r)
    
    numerator = np.exp(r*T)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    
    denominator = 1j*phi*K**(1j*phi)
    
    return numerator/denominator


def heston_price_quad(S_0, K, V_0, kappa, theta, sigma, rho, lambd, T, r):
    args = (S_0, V_0, kappa, theta, sigma, rho, lambd, T, r, K)
    
    real_integral, err = np.real( quad(integrand, 0, 100, args=args) )
    
    return (S_0 - K*np.exp(-r*T))/2 + real_integral/np.pi



def heston_price_trapezoid(S_0, K, V_0, kappa, theta, sigma, rho, lambd, T, r):
    
    eps=0.0005
    phi=np.linspace(eps,100,10000)
    args = (phi,S_0, V_0, kappa, theta, sigma, rho, lambd, T, r, K)
    
    
    
    real_integral = np.real( trapezoid(integrand(*args), phi) )
    
    return (S_0 - K*np.exp(-r*T))/2 + real_integral/np.pi



def heston_price_rec(S_0, K, V_0, kappa, theta, sigma, rho, lambd, T, r):
    args = (S_0, V_0, kappa, theta, sigma, rho, lambd, T, r)
    
    P, umax, N = 0, 100, 10000
    
    dphi=umax/N 
    
    for i in range(1,N):

        phi = dphi * (2*i + 1)/2 
        
        numerator = np.exp(r*T)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)
        
        P += dphi * numerator/denominator
        
    return np.real((S_0 - K*np.exp(-r*T))/2 + P/np.pi)





