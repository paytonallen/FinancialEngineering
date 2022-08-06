    #import modules needed
from statistics import LinearRegression
import numpy as np
import matplotlib as plt
from pandas_datareader import data as wb
import yfinance as yf
import datetime as dt
import pandas as pd
from scipy.optimize import minimize
import scipy.stats


#Function for getting asset returns you want for your portfolio
def AssetReturns(PortfolioAssets):
    Assets = [PortfolioAssets]
#create empyty dataset
    hist_data = {}
    for asset in Assets:
        df = yf.download(asset, start = 2000-1-1)
        hist_data[asset] = df['Adj Close']
    df_returns = np.log(df/df.shift())
    df_returns.index = pd.to_datetime(df_returns.index).to_period('W-Fri').last()
    df.returns = df_returns.dropna()
    return df_returns

#view the potential drawdowns of adding an asset to your portfolio
def drawdown(r: pd.Series):
    'Creates wealth index, Previous peaks, and Percentage Drawdown'
    wealth = 1*(1+r).cumprod()
    previous_peaks= wealth.cummax()
    drawdown = (wealth - previous_peaks)/previous_peaks
    pd.DataFrame({'Weatlh':wealth,
                  'Peaks': previous_peaks,
                  'Drawdown': drawdown})

def Return_Statistics(r,periods):
    vol = r.std()*np.sqrt(periods)
    annual_return = r.mean()*periods
    sharpe = annual_return/vol
    demeanedr = r - r.mean()
    sigmar = r.std(ddof=0)
    exp = (demeanedr**3).mean()
    skew = exp/sigmar**3
    exp1 = (demeanedr**4).mean()
    kurtosis = exp1/sigmar**4
    return pd.DataFrame({'Skew':skew,'Kurtosis':kurtosis,'Vol': vol,'Return':annual_return,'Sharpe':sharpe})

def is_normal(r,level = 0.01):
    """
    Applies the Jarque-Bera test to determine if series is normal or not
    Test is applied at 1% level by default
    Return True if hypothis of normality is accepted, False otherwise
    """
    statistic,p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    is_negative= r<0
    return r[is_negative].std(ddof=0)

def var_historic(r,level,modified=False):
    """
    Returns Value at Risk at specified level
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level = level)
    elif  isinstance(r,pd.Series):
        return -np.percentile(r,level)

    else:
        raise TypeError('Expected r to be a series or DataFrame')
def skewness(r):
    demeanedr=r-r.mean()
    sigmar=r.std(ddof=0)
    exp = (demeanedr**3).mean()
    skew = exp/sigmar**3
    
    return exp/sigmar**3

def kurtosis(r):
    demeanedr=r-r.mean()
    sigmar=r.std(ddof=0)
    exp = (demeanedr**4).mean()
    kurtosis = exp/sigmar**4
    
    return exp/sigmar**4

from scipy.stats import norm

def var_gaussian(r,level = 5, modified = False):
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z+(z**2-1)*s/6+(z**3-3*z)*(k-3)/24-(2*z**3-5*z)*(s**2)/36)
    return -(r.mean()+z*r.std(ddof=0))

def cvar(r, level = 5):
    """
    Conditional Value at Risk
    """

    if isinstance(r,pd.Series):
        lessthan = r<= var_historic(r,level = level)
        return -r[lessthan].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar,level = level)
    else:
        raise TypeError('Expected r to be a series or DataFrame')
def annual_return(r,periods):

    cg = (1+r).prod()
    n_periods = r.shape[0]
    return cg**(periods/n_periods)-1

def annual_vol(r,periods):
    return r.std()*np.sqrt(periods)


def Sharpe_Ratio(r,rfr=0.05,periods): 
    
    rfr_pp = (1+rfr)*(1/periods)-1
    excess_ret = r - rfr_pp
    rets = annual_return(excess_ret,periods)
    ann_vol = annual_vol(r,periods)
    return rets/ann_vol

def portfolio_return(weights,returns):
    """
    Weights --> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    """
    Portfolio Volatility
    """
    return(weights.T @ covmat @ weights)**0.5
def minimize_vol(target_return,er,cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds =((0.0,1.0),) *n
    return_is_target = {'type':'eq','fun':lambda weights,er:target_return-portfolio_return(weights,er)}
    weights_sum_to_1 = {'type':'eq','fun':lambda weights: np.sum(weights)-1}
    results = minimize(portfolio_vol,init_guess,args = (cov,),method = 'SLSQP',options{'disp':False},constraints = (return_is_target,weights_sum_to_1),bounds= bounds)
    return results.x

def optimal_weights(n_points,er,cov):
    """
    Optimal asset weights based on minimizing the variance of a portfolio
    """

    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return,er,cov)for target_return in target_rs]
    return weights

def MaxSharpeRatio(riskfree_rate=0.05,er,cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        """
        returns neg sharpe ratio
        """
        r=portfolio_return(weights,er)
        vol= portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio,init_guess,
                       args = (riskfree_rate,er,cov,),method='SLSQP',
                       options={'disp':False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x
def gmv(cov):
    n = cov.shape[0]
    return MaxSharpeRatio(0,np.repeat(1,n),cov)
                
def plot_ef(n_points,er,cov,show_cml=False,style='.-',riskfree_rate = 0.05,show_ew=False,show_gmv = False):
    
    """
    Plots N-Asset efficient Frontier
    """
    weights = optimal_weights(n_points,er,cov)
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({'Returns':rets,
                    'Volatilty':vols})
    ax = ef.plot.line(x = 'Volatilty',y = 'Returns',style =style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        vol_ew = portfolio_vol(w_ew,cov)
        ax.plot([vol_ew],[r_ew],color = 'midnightblue',marker = 'o',markersize =10)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv= portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv],color='goldenrod',marker = 'o',markersize = 10)

    if show_cml:
        ax.set_xlim(left=0)
        rf = 0.03 
        w_msr = MaxSharpeRatio(rf,er,cov)
        r_msr = portfolio_return(w_msr,er)
        vol_msr = portfolio_vol(w_msr,cov)

        cml_x = [0,vol_msr]
        cml_y = [riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color='red',marker='o',linestyle='dashed',marker = 12,linewidth=2)
        return ax
from sklearn.linear_model import LinearRegression

def LR(X,y):
    reg = LinearRegression().fit(X.reshape(-1,1),y.reshape(-1,1))
    return reg.coef_,reg.intercept_





    






    



