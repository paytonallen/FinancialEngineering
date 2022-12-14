import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute 
import yfinance as yf
plt.style.use('seaborn')

class BBStrategy():
    """
    Class for mean reversion of an asset or asset pair trading
    """
    def __init__(self, symbol,SMA,dev,start,end,tc =0.01):
        self.symbol = symbol
        self.SMA = SMA
        self.dev = dev
        self.start = start
        self.end = end
        self.tc = tc

    def __repr__(self):
        rep = "BBStrategy(symbol =  {}, SMA ={}, dev = {}, start = {}, end= {}"
        return rep.format(self.symbol, self.SMA, self.dev,self.start, self.end)

    def get_data(self):
        df = yf.download(self.symbol,self.start,self.end)['Adj Close']
        df = df[self.symbol].to_frame().dropna()
        df = df.loc[self.start:self.end]
        df.rename(columns={self.symbol: 'price'}, inplace = True)
        df['returns'] = np.log(df/df.shift(1))
        df['SMA'] = df['price'].rolling(self.SMA).mean()
        df['UpperBand'] = df['SMA'] + df['price'].std()*self.dev
        df['Lowerband'] = df['SMA'] - df['price'].std()*self.dev
        self.data = df
        return df

    def set_parameters(self, SMA = None, dev = None):
        """
        update parameters and resp. time series
        """

        if SMA is not None :
            self.SMA = SMA 
            self.data['SMA'] = self.data['price'].rolling(self.SMA).mean()
            self.data['UpperBand'] = self.data['price']+self.data['price'].rollling(self.SMA).std()*self.dev
            self.data['LowerBand'] = self.data['price']- self.data['price'].rolling(self.SMA).std()*self.dev
        if dev is not None:
            self.dev = dev
            self.data['LowerBand'] = self.data['SMA']-self.data.rolling(self.SMA).std()*self.dev
            self.data['UpperBand'] = self.data['SMA']+self.data.rolling(self.SMA).std()*self.dev

    def test_strategy(self):
        """
        Backtesting trading results
        """

        data = self.data.copy().dropna()
        data['distance'] = data.price - data.SMA
        data['position'] = np.where(data.price < data.lower, 1, np.nan )
        data['position'] = np.where(data.price>data.upper,-1,data['position'])
        data['position'] = np.where(data.distance * data.distance.shift(1)<0,0,data['position'])
        data['position'] = data.position.ffill().fillna(0)
        data['strategy'] = data.position.shift(1)*data['returns']
        data.dropna(inplace = True)
        # when do trades take place
        data['trades'] = data.position.diff().fillna(0).abs()

        #transaction costs if trading FOREX or DERIVATIVES

        data.strategy = data.strategy-data.trades*self.tc

        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data

        #Performace vs Buy and Hold

        perf = data['cstrategy'].iloc[-1]
        outperf = perf - data['creturns'].iloc[-1]

        return round(perf,6), round(outperf,6)

    def plot_results(self):

        if self.results is None:
            print('Nothing to plot')
        else:
            title = '{} | SMA = {} | dev = {} | TC = {}'.format(self.symbol,self.SMA,self.dev,self.tc)
            self.results[['creturns','cstrategy']].plot(title = title, figsize =(12,8))


    def update_and_run(self,boll):

        self.set_parameters(int(boll[0]), int(boll[1]))
        return -self.test_strategy()[0]

    def optimize_parameters(self,SMA_range,dev_range):
        """
        Finds global maximum given the parameters range

        Parameters
        ==========
        SMA_range, dist_range: tuple
            tuples of the form(start,end,step size)
        """
        opt = brute(self.update_and_run,(SMA_range,dev_range),finish = None)
        return opt, -self.update_and_run(opt)



