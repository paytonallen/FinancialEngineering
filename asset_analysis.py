#import modules needed

from IPython.display import display
import ipywidgets as widgets
import numpy as np
import matplotlib as plt
from pandas_datareader import data as wb
import yfinance as yf
import datetime as dt
import pandas as pd
from scipy.optimize import minimize
import scipy.stats


#Function for getting asset returns you want for your portfolio
def AssetReturns(*args,date):
    assets=[*args]
    portfolio ={}
    for asset in assets:
        df = yf.download(asset,start=date)
        portfolio[asset] = df['Adj Close']
    portfolio = pd.concat(portfolio,axis =1)
        
    df_returns = portfolio/portfolio.shift()-1
    df_returns = df_returns.resample('W-FRI').last()
    df_returns = df_returns.dropna()
    return df_returns



#view the potential drawdowns of adding an asset to your portfolio
def drawdown(r:pd.Series):
    'Creates wealth index, Previous peaks, and Percentage Drawdown'
    wealth = 1*(1+r).cumprod()
    previous_peaks= wealth.cummax()
    drawdown = (wealth - previous_peaks)/previous_peaks
    return pd.DataFrame({'Weatlh':wealth,
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


def Sharpe_Ratio(r,rfr=0.05,periods=52): 
    
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
def minimize_vol(target_return,r,cov):
    n = r.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds =((0.0,1.0),) *n
    return_is_target = {'type':'eq','args':(r,),'fun':lambda weights,r:target_return-portfolio_return(weights,r)}
    weights_sum_to_1 = {'type':'eq','fun':lambda weights: np.sum(weights)-1}
    results = minimize(portfolio_vol,init_guess,args = (cov,),method = 'SLSQP',options={'disp':False},constraints = (return_is_target,weights_sum_to_1),bounds= bounds)
    return results.x

def optimal_weights(n_points,r,cov):
    """
    Optimal asset weights based on minimizing the variance of a portfolio
    """

    target_rs = np.linspace(r.min(),r.max(),n_points)
    weights = [minimize_vol(target_return,r,cov)for target_return in target_rs]
    return weights

def msr(riskfree_rate,r,cov):
    """
    Returns weights of the portfolio that five max sharpe ratio given rfr and er and a covmat
    """
    n = r.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights,riskfree_rate,r,cov):
        """
        returns neg sharpe ratio
        """
        r=portfolio_return(weights,r)
        vol= portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    
    results = minimize(neg_sharpe_ratio,init_guess,
                       args = (riskfree_rate,r,cov,),method='SLSQP',
                       options={'disp':False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x
def gmv(cov):
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
                
def plot_ef(n_points,r,cov,show_cml=False,style='.-',riskfree_rate=0,show_ew=False, show_gmv=False):
    """
    Plots N-Asset efficient frontier
    """
    weights = optimal_weights(n_points,r,cov)
    rets = [portfolio_return(w,r) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility':vols
    })
    ax=ef.plot.line(x = 'Volatility',y='Returns',style=style)
    if show_ew:
        n=r.shape[0]
        w_ew= np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,r)
        vol_ew = portfolio_vol(w_ew,cov)
        ax.plot([vol_ew],[r_ew],color='midnightblue',marker='o',markersize=10)
    
    if show_gmv:
        w_gmv =gmv(cov)
        r_gmv = portfolio_return(w_gmv,r)
        vol_gmv = portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv],color='goldenrod',marker='o',markersize=10)
    
    if show_cml:
        ax.set_xlim(left=0)
        rf = 0.03
        w_msr=msr(rf,r,cov)
        r_msr=portfolio_return(w_msr,r)
        vol_msr= portfolio_vol(w_msr,cov)

        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color="red",marker='o',linestyle='dashed',markersize=12,linewidth=2)
        return ax
from sklearn.linear_model import LinearRegression

def LR(X,y):
    reg = LinearRegression().fit(X.reshape(-1,1),y.reshape(-1,1))
    return reg.coef_,reg.intercept_

#DownSideProtection Constant 
def run_cppi(r, safe_r = None, m = 3, start = 1000,floor = 0.8,riskfree_rate = 0.03,drawdown=None):
    dates = r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    if isinstance(r,pd.Series):
        r = pd.Dataframe(r,columns['R'])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(r)
        safe_r.values[:] = riskfree_rate/12
    account_history = pd.DataFrame().reindex_like(r)
    cushion_history = pd.DataFrame().reindex_like(r)
    risky_w_history = pd.DataFrame().reindex_like(r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak,account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w,1)
        risky_w = np.maximum(risky_w,0)
        safe_w = 1-risky_w
        risky_allocation = account_value * risky_w
        safe_allocation = account_value * safe_w
        #update account value for this timestamp
        account_value = risky_allocation*(1+r.iloc[step])+safe_allocation*(1+safe_r.iloc[step])
        #save values to look at history and plot
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start*(1+r).cumprod()
    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'RiskBudget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm':m,
        'start':start,
        'floor':floor,
        'r':r,
        'safe_r': safe_r
    }
    return backtest_result


def gbm(n_years = 10, n_scenarios = 1000, mu = 0.07, sigma = 0.15,steps_per_year = 252, s_0 = 100.0,prices = True):
    """
    Geometric Brownian Motion Model
    """
    delta_time = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) +1
    rets_plus_1 = np.random.normal(loc = (1+mu)**delta_time, scale = (sigma*np.sqrt(delta_time)),size=(n_steps,n_scenarios))
    rets_plus_1[0]=1
    
    prices = s_0*pd.DataFrame(rets_plus_1).cumprod() 
    return prices




##Double Scalar Issue with model, please advise, will continue to work on at later date. 
def show_cppi(n_scenarios=50,mu=0.07,sigma=0.15,m=3,floor = 0.,riskfree_rate=0.03,y_max=100):
    
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios,mu=mu,sigma=sigma,prices = False,steps_per_year = 252)
    risky_r = pd.DataFrame(sim_rets)
    
    btr = run_cppi(r = pd.DataFrame(risky_r),riskfree_rate = riskfree_rate,m=m,start=start,floor=floor)
    wealth = btr['Wealth']
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    ax = wealth.plot(legend=False,alpha = 0.2,color='indianred',figsize=(12,6))
    ax.axhline(y=start,ls=':',color='black')
    ax.axhline(y=start*floor,ls=':',color = 'red')
cppi_controls = widgets.interactive(show_cppi,
                                   n_scenarios = widgets.IntSlider(min=1,max=1000,step=5,value=50),
                                   mu = (0.,+.2,0.01),
                                   sigma=(0,0.3,.05),
                                   floor = (0,2,0.1),
                                   m = (1,5,.5),
                                   riskfree_rate=(0,0.1,0.01),
                                   y_max = widgets.IntSlider(min=0,max=100,step=1,value=100,
                                                            description="zoom y axis")
)
    


## Present Value of Liabilities and the Funding ration


def discount(t,r):
    return (1+r)**(-t)

def pv(l,r):
    """
    computes pv of a sequence of liabilities
    l in indexed by the time and the values are the amounts of each liability
    returns the pv of the sequence
    """
    dates = l.index
    discounts = discount(dates,r)
    return (discounts*l).sum()

def funding_ratio(assets,liabilities,r=0.03):
    """
    computes the funding ratio of some assets given liabilities and interest rate
    """
    return assets/pv(liabilities,r)

def show_funding_ratio(assets,r):

    fr = funding_ratio(assets,liabilities,r)
    print(f'{fr*100:.2f}')

controls = widgets.interactive(show_funding_ratio,assets = widgets.IntSlider(min=1,max=10,step=1,value = 5),r=(0,.20,.01))
display(controls)


