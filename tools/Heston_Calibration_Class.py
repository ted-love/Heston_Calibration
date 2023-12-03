#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:05:38 2023

@author: ted
"""

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv


class Data_Class:
    """
    
    This class holds and operates all the market data information on the options. 
    We use a class because throughout the calibration, we will be removing problematics 
    options that return negative/nan volatilities. The class allows us to keep the amount
    of strikes, expiries, prices etc consistent throughout the calibtation procedure.
    """
    
    def __init__(self,**kwargs):

        self.option_chain = kwargs.get('option_chain')     # Dataframe of option chain
        self.market_vol = kwargs.get('market_vol')         # NumPy array of implied vols of market data              
        self.market_prices = kwargs.get('market_prices')   # NumPy array of market prices
        self.S = kwargs.get('S')                           # Price of stock 
        self.K = kwargs.get('K')                           # Strike 
        self.T = kwargs.get('T')                           # Expiry 
        self.r = kwargs.get('r')                           # Interest rate 
        self.q = kwargs.get('q')                           # Dividend yield
        self.flag = kwargs.get('f')                        # Option type as str 
        
 
    def filter_option_chain(self,df,tmin,tmax,Kmin,Kmax,volume,option_type,price_type,min_price):
        """
            
        
        Parameters
        ----------
        df : Pandas DataFrame
            Option chain.
        tmin : float
            minimum expiry.
        tmax : float
            maximum expiry.
        Kmin : float
            minimum strike.
        Kmax : float
            maximum strike.
        volume : float
            maximum volume.
        option_type : str
            flag of option type.
        price_type : str
            price type, i.e. lastPrice, bid, midPrice or ask.
        min_price : float
            minimum price of option.

        Returns
        -------
        filtered_chain : Pandas DataFrame
            A filtered option chain.

        """
        if option_type=='c':
            option_flag=True
        if option_type=='p':
            option_flag=False
            
        filtered_chain = df.loc[(df['dte'] >= tmin) & (df['dte'] <= tmax) & (df[price_type] >= min_price)
                                & (df['volume']>=volume) & (df['CALL']==option_flag) & (df['strike'] >= Kmin)
                                & (df['strike'] <= Kmax)]
        
        return filtered_chain
        

    def df_to_numpy(self,Treasury_Curve,Implied_Dividend_Curve,price_type):
        """
        

        Parameters
        ----------
        Treasury_Curve : scipy.interpolate.CubicSpline
            A CubicSpline object that is the interest rate curve. 
        Implied_Dividend_Curve : scipy.interpolate.CubicSpline
            A CubicSpline object that is the dividend yield curve. 
        price_type : str
            price type, i.e. lastPrice, bid, midPrice or ask.

        Returns
        -------
        None.

        """
        
        self.T = self.option_chain['dte'].to_numpy()
        self.K = self.option_chain['strike'].to_numpy()
        self.r = Treasury_Curve(self.T)
        self.q = Implied_Dividend_Curve(self.T)
        self.market_prices = self.option_chain[price_type].to_numpy()
        ini_flag = self.option_chain['CALL']
        ini_flag = ini_flag.replace({True:'c'})
        self.flag = ini_flag.replace({False:'p'}).to_numpy()
    
    
    def calculate_implied_vol(self):
        """
        
        Calculate implied volatility from the market prices as a %.

        """
        imp_vol = calculate_iv(self.market_prices, self.S, self.K,self.T,self.r,self.flag,self.q, model='black_scholes_merton',return_as='numpy')
        self.market_vol = 100 * imp_vol
        
    
    def delete_nan_options(self,idx):
        """
        
        Is a helper to delete problematic options.        
        
        Parameters
        ----------
        idx : int
            index of the numpy arrays to be deleted.

        Returns
        -------
        None.

        """
        self.market_vol = np.delete(self.market_vol,idx)
        self.market_prices = np.delete(self.market_prices,idx)
        self.K = np.delete(self.K,idx)
        self.T = np.delete(self.T,idx)
        self.r = np.delete(self.r,idx)
        self.flag = np.delete(self.flag,idx)
        self.q = np.delete(self.q,idx)
    
    
    def removing_iv_nan(self):
        
        """
        Removing implied volatilities that = nan and corresponding options data.
        """
        idx=0
        M = np.size(self.K)
        while idx < M:
            
            if np.isnan(self.market_vol[idx]):
                self.delete_nan_options(idx)
                M-=1
                continue
            idx+=1
            
        
    def removing_nans_fx(self,f_x):
        """
        
        Parameters
        ----------
        f_x : NumPy array
            Difference of the market vol and current guess of vol during calibration.
            (F_x = 1/2 f_x @ f_x.T, where F_x is the cost-function)

        Returns
        -------
        f_x : NumPy array
            removed nan values of array and corresponding options data.

        """
        i=0
        M=np.size(self.K)
        while i < M:

            if np.isnan(f_x[i]):
                f_x = np.delete(f_x,i,axis=0)
                self.delete_nan_options(i)

                M-=1
                continue
            i+=1
        return f_x
        
    
    def removing_nans_J(self, J, f_x):
        """
        

        Parameters
        ----------
        J : NumPy array
            gradient of the option.
        f_x : NumPy array
            Difference of the market vol and current guess of vol during calibration.
            (F_x = 1/2 f_x @ f_x.T, where F_x is the cost-function)
            
        Returns
        -------
        J : NumPy array
            nan values removed and corresponding options data
        f_x : NumPy array
            nan values removed and corresponding options data
        """
        i=0
        M=np.size(self.K)
    
        while i < M:
            
            if np.isnan(J[:,i]).any():
                J = np.delete(J,i,axis=1) 
                f_x = np.delete(f_x,i,axis=0)
                self.delete_nan_options(i)

                M-=1
                continue
            i+=1
    
        return J, f_x
        
        
    def remove_every_2nd_option(self,DF,t,m):     
        """
        

        Parameters
        ----------
        DF : Pandas DataFrame
            Options chain.
        t : float
            filtering chain until time t.
        m : int
            removing every m options data.

        Returns
        -------
        DF : TYPE
            DESCRIPTION.

        """
        idx=0
        M=np.size(DF,axis=0)
        while idx < M:
    
            if idx % m == 0:
                idx+=1
                continue
            if DF.iloc[idx]['dte']<t:
                DF = DF.drop(DF.index[idx])
                M-=1
            idx+=1    
        
        return DF
    
        
    def check_4_calibrated_nans(self,iv):
        """
        

        Parameters
        ----------
        iv : NumPy array
            calibrated implied volatility.

        Returns
        -------
        iv : NumPy array
            removed nans of calibrated iv and their corresponding data.

        """
        i=0
        M=np.size(self.K)
    
        while i < M:
            
            if np.isnan(iv[i]):
                iv = np.delete(iv,i)
                self.delete_nan_options(i)
                M-=1
                continue
            i+=1
    
        return iv
    
    def plot_save_surface(self,calibrated_iv,date_today):
        """
        
        Plotting and saving the calibrated implied vol surface vs the market implied vol surface

        Parameters
        ----------
        calibrated_iv : NumPy array
            calibrated implied volatilities.
        date_today : str
            date of calibration.

        Returns
        -------
        None.

        """
        moneyness = self.K/self.S
        K_vals, T_vals = np.meshgrid(np.linspace(min(moneyness), max(moneyness), 100),
                                     np.linspace(min(self.T), max(self.T), 100))
        
        # Interpolating the calibrated surface
        calibrated_iv_interp = griddata((moneyness, self.T), calibrated_iv, (K_vals, T_vals), method='cubic')

        # Interpolating the original implied volatilities
        imp_vol_interp = griddata((moneyness, self.T), self.market_vol, (K_vals, T_vals), method='cubic')

            
        fig = plt.figure(figsize=(13, 5),dpi=300)
        
        plt.subplots_adjust(wspace=-0.5)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(K_vals, T_vals, np.squeeze(calibrated_iv_interp), cmap='viridis')
        
        
        ax1.set_title('calibrated surface',y=0.95,fontsize=16)
        ax1.set_xlabel('Moneyness (K/S)',fontsize=12)
        ax1.set_ylabel('Expiration (T)',fontsize=12)
        ax1.zaxis.set_rotate_label(False)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_zlabel('implied vol',rotation=90)
        ax1.azim+=-10
        ax1.elev+=-12
        ax1.w_yaxis.set_major_locator(MultipleLocator(1))
        ax1.w_zaxis.set_major_locator(MultipleLocator(10))
        ax1.w_xaxis.set_major_locator(MultipleLocator(0.5))
        
        ax1.zaxis.label.set_verticalalignment('center')  
        ax1.set_box_aspect(aspect=None, zoom=0.9) 
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(K_vals, T_vals, np.squeeze(imp_vol_interp), cmap='plasma')
        ax2.set_title('iv surface',y=0.95,fontsize=16)
        ax2.set_xlabel('Moneyness (K/S)',fontsize=12)
        ax2.set_ylabel('Expiration (T)',fontsize=12)
        ax2.zaxis.set_rotate_label(False)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_zlabel('implied vol',rotation=90)
        ax2.azim+=-10
        ax2.elev+=-12
        ax2.w_yaxis.set_major_locator(MultipleLocator(1))
        ax2.w_zaxis.set_major_locator(MultipleLocator(10))
        ax2.w_xaxis.set_major_locator(MultipleLocator(0.5))
        
        
        ax2.zaxis.label.set_verticalalignment('center')  
        ax2.set_box_aspect(aspect=None, zoom=0.9)
        fig.tight_layout()
        fig1 = plt.gcf()
        plt.show()
        
        try:
            os.mkdir(f'Results_SPX_{date_today}')
        except:
            pass
     
        fig1.savefig(f'Results_SPX_{date_today}/surface.png',dpi=300)
        


                
                