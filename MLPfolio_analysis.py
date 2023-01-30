import numpy as np
import pandas as pd
from pandas_datareader import data as wb

import nasdaqdatalink
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

import yfinance as yf

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression 
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, mean_squared_error, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor







def indicators(asset):
    
    asset_II = ((asset.AdjClose - asset.Low)-(asset.High - asset.Close)/(asset.High - asset.Low))*asset.Volume
    asset_vpt_su = (asset.Close/asset.Close.shift(1)-1)*asset.Volume
    asset_vpt = asset_vpt_su.cumsum()
    asset_bp = asset.High -asset.Open
    asset_sp = asset.Close - asset.Low
    asset_drf = (asset_bp-asset_sp)/2*(asset.High - asset.Low) 
    asset_pr = (asset.High.rolling(10).max()-asset.Close/(asset.High.rolling(10).max()-asset.Low.rolling(10).min()))
    asset_rvi = (asset.Close - asset.Open)/(asset.High - asset.Low)
    asset_mom = asset.Close - asset.Close.shift(20)
    asset_av = asset.Volume.rolling(20).mean()
    asset_tr = asset.High - asset.Low
    asset_atr = asset_tr.rolling(20).mean()
    asset_trmv = asset_mom/asset_atr * asset_av
    
    
    return pd.DataFrame({'Momentum': asset_mom,
                         'PercentR':asset_pr,
                         'TrendMomentum':asset_trmv,
                         'IntradayIntensity': asset_II,
                         'AvgTrueRange': asset_atr})


assets = ['TLT','EMB','IWM','DIA','SPY','QQQ',
          'EFA','EEM','VGK','REM','VNQ','SMH',
          'JNK','IJH','LQD','HYG','IGSB','XLF',
          'EUFN','AAPL','AMZN','GOOGL','MSFT',
          'NVDA','BIL','SHV','SHY','IEI','STIP','UUP',
          'FXF','FXY','GLD','SLV','CASH']

#import assets
portfolio = yf.download(assets,start = '2010-01-04')['Adj Close']
portfolio
portfolio = portfolio.resample('W-FRI').last()
portfolio_returns = portfolio.pct_change()
#import 10yr risk free rate divided by weekly basis
rfr = wb.DataReader('DGS10','fred',start = '2010-01-04')
rfr = rfr.resample('W-FRI').last()
rfr = rfr/100/52
rfr
#join for cleaning
variables = portfolio_returns.join(rfr).dropna()
#seperate after cleaning
riskfree = pd.DataFrame(variables['DGS10'])
portfolio_assets = variables.drop(['DGS10','SPY'],axis = 1).dropna()
mkt_return = pd.DataFrame(variables['SPY'])
portfolio
#frequency of data after cleaning
freq = 52
#setting the testing data indexes
start_date = '2010-01-01'
end_date = '2022-01-01'

ind = (portfolio_assets.index >= start_date)*(portfolio_assets.index<=end_date)

portfolio_assets = portfolio_assets[ind]
mkt_return = mkt_return[ind]
riskfree = riskfree[ind]
#setting training data indexes
start_date_train = '2010-01-01'
end_date_train = '2022-01-01'

ind = (portfolio_assets.index >= start_date_train)*(portfolio_assets.index<=end_date_train)
#save dataset for analysis
portfolio_assets_train = portfolio_assets[ind].copy()
mkt_return_train = mkt_return[ind].copy()
riskfree_train = riskfree[ind].copy()



def ann_ret(x):
    return (x+1)**freq-1


def ann_std(x):
    return x*np.sqrt(freq)

def ann_geo_mean(x):
    n = len(x)
    return np.exp(np.sum(np.log(1+x)) * freq / n ) -1

def ann_sr(x,rf):
    n= len(x)
    ret_expected = np.sum(x-rf)/n
    ret_avg = np.sum(x)/n
    std_dev = np.sqrt(np.sum((x-ret_avg)**2)/n)
    ann_ret_expected = (ret_expected+1)**freq-1
    ann_std_dev = std_dev *np.sqrt(freq)
    return ann_ret_expected/ann_std_dev

def mdd(x):
    wealth = 1*(1+x).cumprod()
    peak = wealth.cummax()
    drawdown = wealth/peak
    return drawdown.min()

def LR(X,y):
    reg = LinearRegression().fit(X.reshape(-1,1),y.reshape(-1,1))
    return reg.coef_,reg.intercept_

#get the mean and covariance matrix of the training data
portfolio_train_mean = portfolio_assets_train.mean()
portfolio_train_cov_mat = portfolio_assets_train.cov()

def EReturn(w):
    EReturn = w @ portfolio_train_mean
    return EReturn
#function to compute the portfolio standard deviation
def PVol(w):
    pvar = w @ portfolio_train_cov_mat @ w
    return np.sqrt(pvar)

#function to solve for the optimal solution to Markowitz Portfolio Optimization Problem
#with specificed target return r
def MarkPortOpt(r,silent = False):
    #constraints
    def constraint1(w):
        return np.sum(w) - 1.0 #budget constraint
    def constraint2(w):
        return 1.0 - np.sum(w) #budget constraint
    def constraint3(w):
        return w #nonnegative constraint
    def constraint4(w):
        diff = EReturn(w)-r
        return diff
    
    con1 = {'type': 'ineq','fun': constraint1}
    con2 = {'type': 'ineq','fun': constraint2}
    con3 = {'type': 'ineq','fun': constraint3}
    con4 = {'type': 'ineq','fun': constraint4}
    cons = ([con1,con2,con3,con4])
    
    #initial x0
    w0 = np.ones(len(portfolio_assets_train.columns))
    
    #solve the problem
    sol = minimize(PVol,w0,method='SLSQP',constraints= cons)
    
    #whether the solution will be printed
    if(not silent):
        print('Solution to the Markowitz Problem with r=', round(r*100,3), '%')
        print(sol)
        print("")
    elif (not sol['success']):
        print('WARNING: the optimizer did NOT exit successfully!!')
        
    return sol


#Annual Sharpe Ratio
def AnnSR(w, data = portfolio_assets_train, rf = riskfree_train['DGS10'].mean()):
    excess_ret = data @ w - rf
    AnnSR = ann_ret(excess_ret.mean())/ann_std(PVol(w))
    return AnnSR

# function to find the optimal portfolio the maximize the sharpe ratio

def MaxSR(data = portfolio_assets_train, rf = riskfree_train['DGS10'].mean(), silent=False):
    #Objective Function
    def SR(w):
        excess_ret = data @ w - rf
        SR = (excess_ret.mean())/(PVol(w))
        return -SR
    n = len(data.columns)
    
    #Bounds
    bnds = tuple((0,1) for i in range(n)) # nonnegativity constraint
    
    #constraints
    def constraint1(w):
        return np.sum(w) - 1.0 #budget constraint
    cons = {'type': 'eq','fun': constraint1}
    #initial x0
    w0 = np.array(np.ones(n))
    #solve the problem
    
    sol = minimize(SR, w0, method = 'SLSQP',bounds = bnds, constraints = cons)
    
    #whether the solution will be printed
    if(not silent):
        print('Solution to the Sharpe Ratio Problem is:')
        print(sol)
        print('')
    elif (not sol['success']): #check if optimizer exit successfully
        print('Warning: the optimizer did not exit successfully')
        
    return sol

#function to compute the expected return for the portfolio
def EReturn_with_rf(w):
    EReturn = w @ portfolio_assets_train_withrf_mean
    return EReturn

def PVol_with_rf(w):
    pvar = w @ portfolio_assets_train_withrf_covmat @ w
    return np.sqrt(pvar)

#function to solve for the optimal solution to the markowitz portfolio optimization problem
#with specified target return r

def MarkPortOpt_with_rf(r,silent=False):
    
    #constraints
    def constraint1(w):
        return np.sum(w) - 1.0 #budget constraint
    def constraint2(w):
        return 1.0 - np.sum(w) #budget constraint
    def constraint3(w):
        return w[:-1] #nonnegative constraint
    def constraint4(w):
        diff = EReturn_with_rf(w)-r
        return diff #return constraint
    
    con1 = {'type': 'ineq','fun': constraint1}
    con2 = {'type': 'ineq','fun': constraint2}
    con3 = {'type': 'ineq','fun': constraint3}
    con4 = {'type': 'ineq','fun': constraint4}
    cons = ([con1,con2,con3,con4])
    
    #initial x0
    w0 = np.ones(len(portfolio_assets_train_withrf.columns))
    
    #solve the problem
    sol = minimize(PVol_with_rf,w0,method='SLSQP',constraints= cons)
    
    #whether the solution will be printed
    if(not silent):
        print('Solution to the Markowitz Problem with r=', round(r*100,3), '%')
        print(sol)
        print("")
    elif (not sol['success']):
        print('WARNING: the optimizer did NOT exit successfully!!')
        
    return sol
    
def portRet_BH(w):
    n = pa_bt.shape[0]
    PR = np.zeros(n)
    X = w * 1
    for i in range(n):
        W = (1+pa_bt.iloc[i]) @ X
        PR[i] = (W - np.sum(X))/np.sum(X)
        X = (1+pa_bt.iloc[i]) * X
    return PR

#Function to display the summary statistics of the backtest results
def DisplaySummary_BT(PR, n_dec=2):
    col_names = PR.columns
    
    #compute and display summary stats for each portfolio
    print('Summary Statistic of various allocations for the back test (from '+str(BT_startdate)+' to '+str(BT_enddate)+'):')
    PR_mean = PR.mean()
    PR_std = PR.std()
    BT = pd.DataFrame(index =  col_names)
    BT['Geo Mean(Annu,%)'] = np.round(PR.apply(ann_geo_mean) * 100, n_dec)
    BT['Std(Annu,%)'] = np.round(ann_std(PR.std())*100,n_dec)
    BT['Sharpe Ratio(Annu)'] = np.round(PR.apply(ann_sr, rf= riskfree['DGS10']),n_dec)
    BT['Max Drawdown(%)'] = np.round(PR.apply(mdd)*100, n_dec)
    display(BT)
    return PR_mean,PR_std

