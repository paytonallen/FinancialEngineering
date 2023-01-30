import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import math
import scipy.optimize as optimize


def zero_coup_bond(par,y,t):
    """
    Pricing a zero coupon bond
    """

    return par/(1+y)**t

#YIELD TO MATURITY
def bond_ytm(price,par,T,coup,freq = 2, guess = 0.05):
    freq = float(freq)
    periods = T *2
    coupon = coup/100.*par
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y : sum([coupon/freq/(1+y/freq)**(freq*t)for t in dt])+par/(1+y/freq)**(freq*T) - price

    return optimize.newton(ytm_func,guess)
    




class BootstrapYieldCurve(object):

    def __init__(self):
        self.zero_rates = dict()
        self.intruments = dict()
        
    def add_instrument(self,par,T,coup,price,compounding_freq = 2):
        self.intruments[T] = (par,coup,price,compounding_freq)

    def get_maturities(self):
        """
        returns list of maturities of added securities
        """
        return sorted(self.instruments.key())

    def get_zero_rates(self):
        """
        list of spot rates on the yield curve
        """
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()

        return [self.zero_rates[T] for T in self.get_maturities()]

    def bootstrap_zero_coupons(self):
        """
        Bootstrap the yield curve with zero coupoon instruments first
        """
        for (T,instrument) in self.instruments.items():
            (par,coup,price,freq) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par,price,T)
                self.zero_rates[T] = spot_rate

    def zero_coupon_spot_rate(self,par,price,T):
        """
        return: zero coupon spot rate with continuos compounding
        """
        spot_rate = math.log(par/price)/T
        return spot_rate
    
    def get_bond_spot_rates(self):
        """
        Spot rates implied by bonds, using short-term instruments
        """
        for T in self.get_maturities():
            instrument = self.intruments[T]
            (par,coup,price,freq)= instrument
            if coup != 0:
                spot_rate= self.calculate_bond_spot_rate(T,instrument)
                self.zero_rates[T]= spot_rate
    def calculate_bond_spot_rate(self,T,instrument):
        """
        calculate spot rate at a particular maturity period
        """
        try:
            (par,coup,price,freq)= instrument
            periods = T * freq
            value = price
            per_coupon = coup/freq
            for i in range(int(periods)-1):
                t = (i+1)/float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * math.exp(-spot_rate*t)
                value -= discounted_coupon
            last_period = int(periods)/float(freq)
            spot_rate = -math.log(value/(par+per_coupon))/last_period
            return spot_rate
        except:
            print("Error: spot rate not found for T=",t)
## Forward Rates
class ForwardRates(object):
    def __init__(self):

        self.forward_rates = []
        self.spot_rates = dict()
    def add_spot_rate(self,T,spot_rate):
        self.spot_rates[T] = spot_rate
    def get_forward_rates(self):
        periods = sorted(self.spot_rates.keys())
        for T2,T1 in zip(periods,periods[1:]):
            forward_rate = self.calculate_forward_rate(T1,T2)
            self.forward_rates.append(forward_rate)
        return self.forward_rates
    def calculate_forward_rate(self,T1,T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2*T2-R1*T1)/(T2-T1)
        return forward_rate









