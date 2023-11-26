#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:27:00 2023

@author: ted
"""
import numpy as np
import pandas as pd



def calculate_yield(C,P,r,T,S,K):
    """
    

    Parameters
    ----------
    C : Float
        Price of call.
    P : Float
        Price of put.
    r : Float
        Interest rate.
    T : Float
        Time till expiratio.
    S : Float
        Price of Stock.
    K : Float
        Strike price.

    Returns
    -------
    q : Float
        Dividend Yield Rate

    """
    
    q = np.log( (C - P + K*np.exp(-r*T))/S)/-T
    return q

def Implied_Dividend_Yield(SPX, t, S, Treasury_Curve):
    """
    
    
    Parameters
    ----------
    SPX : DataFrame
        Options Chain.
    t : Float
        Minimum time of interest. (>t).
    S : Float
        Price of stock.
    Treasury_Curve : scipy.interpolate class object
        Interpolated interest-rate yield curve.

    Returns
    -------
    Implied_Dividend_Dates : NumPy array
        Dates of yield.
    Implied_Dividend_Rates : NumPy array
        Rates for each date.

    """
    
    SPX_dividends_short_term_calls = SPX.loc[(SPX['strike'] <S+4) & (SPX['strike'] > S-1)
                                             & (SPX['dte']>t) & (SPX['dte']<0.5) & (SPX['CALL']==True)]
    SPX_dividends_short_term_puts = SPX.loc[(SPX['strike'] == SPX_dividends_short_term_calls['strike'].values[0])
                                   & (SPX['dte']>t) & (SPX['dte']<0.5) & (SPX['CALL']==False)]
    
    SPX_dividends_short_term_calls = SPX_dividends_short_term_calls[SPX_dividends_short_term_calls['dte'].isin(SPX_dividends_short_term_puts['dte'])]
    
    SPX_dividends_short_term_puts = SPX_dividends_short_term_puts[SPX_dividends_short_term_puts['dte'].isin(SPX_dividends_short_term_calls['dte'])]
    
    
    q_short = np.empty(np.size(SPX_dividends_short_term_calls.index))
    
    for i in range(np.size(q_short)):
        C = SPX_dividends_short_term_calls['midPrice'].values[i]
        P = SPX_dividends_short_term_puts['midPrice'].values[i]
        K = SPX_dividends_short_term_calls['strike'].values[i]
        r = Treasury_Curve(SPX_dividends_short_term_calls['dte'].values[i])
        T = SPX_dividends_short_term_calls['dte'].values[i]
        
        q_short[i] = calculate_yield(C, P, r, T, S, K)
        

    SPX_dividends_long_term_calls = SPX.loc[(SPX['strike'] == 4500 ) & (SPX['dte']>0.5) & (SPX['CALL']==True)]
    SPX_dividends_long_term_puts = SPX.loc[(SPX['strike'] == 4500 ) & (SPX['dte']>0.5) & (SPX['CALL']==False)]
    
    q_long = np.empty(np.size(SPX_dividends_long_term_calls.index))
    for i in range(np.size(q_long)):  
        
        C = SPX_dividends_long_term_calls['midPrice'].values[i] 
        P = SPX_dividends_long_term_puts['midPrice'].values[i]
        K = SPX_dividends_long_term_calls['strike'].values[i]
        r = Treasury_Curve(SPX_dividends_long_term_calls['dte'].values[i])
        T = SPX_dividends_long_term_calls['dte'].values[i]
        q_long[i] = calculate_yield(C, P, r, T, S, K)
    
                
    Implied_Dividend_Dates = np.concatenate([np.array(SPX_dividends_short_term_puts['dte'].values),np.array(SPX_dividends_long_term_puts['dte'].values)])
    Implied_Dividend_Rates = np.concatenate([q_short,q_long])
    
    return Implied_Dividend_Dates, Implied_Dividend_Rates
    