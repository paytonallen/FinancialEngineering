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
import statsmodels.api as sm
import math



#Function for getting asset returns you want for your portfolio
def AssetReturns(*args,date):
    assets=[*args]
    df = yf.download(assets,start=date)['Adj Close']    
    df_returns = df/df.shift()-1
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

def compound(r):
    return np.expm1(np.log1p(r).sum())

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
        r = pd.Dataframe(r,columns=['R'])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(r)
        safe_r.values[:] = riskfree_rate/12
    account_history = pd.DataFrame().reindex_like(r)
    cushion_history = pd.DataFrame().reindex_like(r)
    risky_w_history = pd.DataFrame().reindex_like(r)
    floorval_history = pd.DataFrame().reindex_like(r)
    peak_history = pd.DataFrame().reindex_like(r)

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
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+r).cumprod()
    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm':m,
        'start':start,
        'floor':floor,
        'r':r,
        'safe_r': safe_r,
        'drawdown':drawdown,
        'peak':peak_history,
        'floor':floorval_history
    }
    return backtest_result
def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annual_return, periods_per_year=12)
    ann_vol = r.aggregate(annual_vol, periods_per_year=12)
    ann_sr = r.aggregate(Sharpe_Ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
    

def gbm(n_years = 10, n_scenarios = 1000, mu = 0.07, sigma = 0.15,steps_per_year = 252, s_0 = 100.0,prices = True):
    """
    Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) +1
    rets_plus_1 = np.random.normal(loc = (1+mu)**dt, scale = (sigma*np.sqrt(dt)),size=(n_steps,n_scenarios))
    rets_plus_1[0]=1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1
    return ret_val


def regress(dependent, explanatory_variables, alpha = True):
    if alpha: 
        explanatory_variables= explanatory_variables.copy()
        explanatory_variables['Alpha'] = 1
    lm = sm.OLS(dependent, explanatory_variables).fit()
    return lm

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
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows,r):
    """
    computes pv of a sequence of liabilities
    l in indexed by the time and the values are the amounts of each liability
    returns the pv of the sequence
    """
    dates = flows.index
    discounts = discount(dates,r)
    return discounts.multiply(flows, axis = 'rows').sum()

def funding_ratio(assets,liabilities,r=0.03):
    """
    computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets,r)/pv(liabilities,r)


def inst_to_ann(r):
    """
    convert short rate to an annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    convert annualized rate to short rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, r_0=None):
    """
    CIR model for interest rates
    """
    if r_0 is None: r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    num_steps = int(n_years*steps_per_year)+1
    shock = np.random.normal(0,scale = np.sqrt(dt),size = (num_steps,n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    #generate prices
    h = math.sqrt(a**2 +2*sigma**2)
    prices = np.empty_like(shock)
    ###
    
    def price(ttm,r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        
        _B = (2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years,r_0)
    
    for step in range(1,num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt+sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t+d_r_t)
        #generate prices at time t as well
        prices[step] = price(n_years-step*dt,rates[step])
    rates = pd.DataFrame(data = inst_to_ann(rates),index = range(num_steps))
    prices = pd.DataFrame(data = prices,index = range(num_steps))
    
    return rates, prices

def bond_cash_flows(maturity,principal= 100,coupon_rate = 0.03,coupons_per_year = 12):
    """
    Series of cash flows generated by a bond,
    indexed by a coupon number
    """
    
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1,n_coupons+1)
    cash_flows = pd.Series(data = coupon_amt,index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100,coupon_rate = 0.03, coupons_per_year = 12, discount_rate = 0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and final coupon payment is returned. 
    This is not designed to be efficient, rather, to illustrate the underlying
    principal in bond pricing. 
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date and the bond value is computed over time. 
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates,columns = discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,discount_rate.loc[t])
        return prices
    else:#base case... single time periods
        if maturity <=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity,principal,coupon_rate,coupons_per_year)
        return pv(cash_flows,discount_rate/coupons_per_year)
    
    

def macaulay_duration(flows,discount_rate):
    """
    completes the macauley duration of cash flows
    """
    discounted_flows = discount(flows.index,discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index,weights=weights)

def match_durations(cf_t,cf_s,cf_l,discount_rate):
    
    d_t = macaulay_duration(cf_t,discount_rate)
    d_s = macaulay_duration(cf_s,discount_rate)
    d_l = macaulay_duration(cf_l,discount_rate)
    return (d_l - d_t)/(d_l-d_s)

def bond_total_return(monthly_prices,principal,coupon_rate,coupons_per_year):
    """
    Computes the total return of a bond based on month bond prices and coupon payments
    Assumes that dividends(coupons) are paid out at the end of the period
    and that dividends are reinvested
    """
    coupons = pd.DataFrame(data = 0, index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year,t_max,int(coupons_per_year*t_max/12),dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices +coupons)/monthly_prices.shift()-1
    return total_returns.dropna()



def bt_mix(r1,r2,allocator,**kwargs):
    """
    Runs a back test of allocating between two sets of returns r1 and r2
    arem T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio the rest in r2(GHP) AS Tx1 DataFrame
    Returns a TxN Data Frame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 need to be same shape')
        
    weights = allocator(r1,r2,**kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    r_mix = weights*r1+(1-weights)*r2
    return r_mix

def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    Produces a time series over T steps of allocations between PSP and GHP accros N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP & GHP such that:
     each column is a scenario
     each row is the price for a stimestep
    Returns an T x N DataFrame of PSP Weights
    """
    
    return pd.DataFrame(data=w1,index=r1.index,columns = r1.columns)

def bt_mix(r1,r2,allocator,**kwargs):
    """
    Runs a back test of allocating between two sets of returns r1 and r2
    arem T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio the rest in r2(GHP) AS Tx1 DataFrame
    Returns a TxN Data Frame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 need to be same shape')
        
    weights = allocator(r1,r2,**kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    r_mix = weights*r1+(1-weights)*r2
    return r_mix

def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    Produces a time series over T steps of allocations between PSP and GHP accros N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP & GHP such that:
     each column is a scenario
     each row is the price for a stimestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1,index=r1.index,columns = r1.columns)

def terminal_values(rets):
    """
    Returns the final values of a dollar at the end of each  return period
    """
    return (rets+1).prod()


def terminal_stats(rets,floor=0.8,cap=np.inf,name="Stats"):
    """
    Produce Summary Statistics on the terminal values per investd dollar
    across a range of N scenarios rets is a T x N DataFrmae of returns, where T
    is the time step
    Returns a 1 column dataframe of summary stats indexed by the stat name
    """
    
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std' : terminal_wealth.std(),
        'p_breach':p_breach,
        'e_short' :e_short,
        'p_reach':p_reach,
        'e_surplus':e_surplus
    },orient = 'index',columns = [name])
    return sum_stats

def glidepath_allocator(r1,r2,start_glide =1,end_glide=0):
    """
    Simulates a target-date-fund style graual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide,end_glide,num=n_points))
    paths = pd.concat([path]*n_col,axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths



def floor_allocator(psp_r,ghp_r,floor,zc_prices,m=3):
    """
    Allocatore between PSP and GHP with the goal to provide exposure to the upside of the PSP withought violating the floor. 
    Uses CPI style dynamic risk budgting algo by investing a multiple
    of the cushion in the PSP
    Returns a dataframe with the same shape as the php/ghp representing the weights in PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP and ZC prices must have same shape')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index,columns = psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step]
        cushion = (account_value-floor_value)/account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc= account_value*psp_w
        ghp_alloc = account_value*ghp_w
        
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r,ghp_r,maxdd,m=3):
    
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1,n_scenarios)
    peak_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index,columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value
        cushion = (account_value-floor_value)/account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value,account_value)
        w_history.iloc[step] = psp_w
    return w_history