import math
from turtle import pd
import pandas as pd
import numpy as np
from decimal import Decimal
from abc import ABC, abstractmethod
import scipy.linalg as linalg
import sys

"""
class will store factors included stock option
"""
#Base Stock Option Class
class StockOption(object):
    def __init__(self,S0,K,r=0.05,T=1,N=2,pu=0,pd=0,div=0,sigma=0,is_put=False,is_am=False):
        """
        Deafults to European unless indicated
        S0 : initial stock price
        K : Strike Price
        r : risk free rate
        T : time to maturity
        N : number of steps
        pu : probability of up state
        pd : probability of down state
        div : dividend yield, if any
        sigma : volatility 
        is_put : True for put option, False for call
        is_am : True for american option, False for European
        """
        
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1,N)
        self.STs = [] # Stock prices tree
        
        """
        Optional params used by derives classes
        """
        
        self.pu,self.pd = pu,pd
        self.div = div
        self.sigma=sigma
        self.is_call = not is_put
        self.is_european = not is_am
        
    @property
    def dt(self):
        """Single time step, in years"""
        return self.T/float(self.N)
    
    @property
    def df(self):
        """The discount factor"""
        return math.exp(-(self.r-self.div)*self.dt)

class BinomialEuropeanOption(StockOption):
    
    def setup_parameters(self):
        #Calculations for model
        self.M = self.N+1
        self.u = 1+self.pu
        self.d = 1 - self.pd
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu
        
    def init_stock_price_tree(self):
        #initialize nodes to zeros
        
        self.STs = np.zeros(self.M)
        #expected stock prices for each node
        for i in range(self.M):
            self.STs[i]=self.S0 *(self.u**(self.N-i))*(self.d**i)
            
    def init_payoffs_tree(self):
        """Payoffs when the option expires at terminal nodes"""
        if self.is_call:
            return np.maximum(0,self.STs-self.K)
        else:
            return np.maximum(0,self.K-self.STs)
        
    def traverse_tree(self,payoffs):
        """traverses backwards from the time the option expires and caluclate the discounted payoffs at each node"""
        for i in range(self.N):
            payoffs = (payoffs[:-1]*self.qu+payoffs[1:]*self.qd)*self.df
            
        return payoffs
        
    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)
    
    def price(self):
        """Entry point of the pricing implementation"""
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        return payoffs[0]

#Change class to American pricing Model by removing M
class BinomialTreeOption(StockOption):
    
    def setup_parameters(self):
        #Calculations for model
        self.M = self.N+1
        self.u = 1+self.pu
        self.d = 1 - self.pd
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu
        
    def init_stock_price_tree(self):
        #initialize 2d tree at T=0
        self.STs = [np.array([self.S0])]
        
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u,[prev_branches[-1]*self.d]))
            self.STs.append(st)
        
        
            
    def init_payoffs_tree(self):
        """Payoffs when the option expires at terminal nodes"""
        if self.is_call:
            return np.maximum(0,self.STs[self.N]-self.K)
        else:
            return np.maximum(0,self.K-self.STs[self.N])
        
    def check_early_exercise(self,payoffs,node):
        if self.is_call:
            return np.maximum(payoffs, self.STs[node]-self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])
        
    def traverse_tree(self,payoffs):
        """traverses backwards from the time the option expires and caluclate the discounted payoffs at each node"""
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-1]*self.qu+payoffs[1:]*self.qd)*self.df
            
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs,i)
            
        return payoffs
        
    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)
    
    def price(self):
        """Entry point of the pricing implementation"""
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        return payoffs[0]
    

#COX ROSS RUBINSTEIN MODEL(CRR)
class BinomialCRROption(BinomialTreeOption):
    
    def setup_parameters(self):
        
        self.u = math.exp(self.sigma*math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd= 1-self.qu
#LEISEN-REIMER TREE MODEL(LR)
class BinomialLROption(BinomialTreeOption):
    
    def setup_parameters(self):
        odd_N = self.N if (self.N%2==0) else (self.N+1)
        d1 = (math.log(self.S0/self.K)+((self.r-self.div)+(self.sigma**2)/2.)*self.T)/(self.sigma*math.sqrt(self.T))
        d2 = (math.log(self.S0/self.K)+((self.r-self.div)-(self.sigma**2)/2.)*self.T)/(self.sigma * math.sqrt(self.T))
        pbar = self.pp_2_inversion(d1,odd_N)
        self.p = self.pp_2_inversion(d2,odd_N)
        self.u = 1/self.df *pbar/self.p
        self.d = (1/self.df-self.p*self.u)/(1-self.p)
        self.qu = self.p
        self.qd = 1 - self.p
        
    def pp_2_inversion(self,z,n):
        return 0.5 + math.copysign(1,z)*math.sqrt(0.25-0.25*math.exp(-((z/(n+1.0/3.+0.1/(n+1)))**2.0)*(n+1.0/6.0)))
#BINOMIAL LR WITH GREEKS
class BinomialLRwithGreeks(BinomialLROption):
    
    def greek_stock_price_tree(self):
        self.STs = [np.array([self.S0*self.u/self.d,self.S0,self.S0*self.d/self.u])]
        
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches * self.u,[prev_branches[-1]*self.d]))
            self.STs.append(st)
            
    def price(self):
        self.setup_parameters()
        self.greek_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        
        option_value = payoffs[len(payoffs)//2]
        
        payoff_up = payoffs[0]
        payoff_down = payoffs[-1]
        S_up = self.STs[0][0]
        S_down = self.STs[0][-1]
        dS_up = S_up - S_down
        dS_down = self.S0 - S_down
        
        #calculate delta values
        dS = S_up - S_down
        dV = payoff_up - payoff_down
        delta = dV/dS
        
        #gamma
        gamma = ((payoff_up - option_value)/dS_up - (option_value - payoff_down)/dS_down)/((self.S0+S_up)/2. -(self.S0+S_down)/2.)
        return option_value,delta,gamma
#TRINOMIAL TREE OPTION
class TriTreeOption(BinomialTreeOption):
    
    def setup_parameters(self):
        self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
        self.d = 1/self.u
        self.m = 1 
        self.qu = ((math.exp((self.r-self.div)*self.dt/2.)-math.exp(-self.sigma*math.sqrt(self.dt/2.)))/(math.exp(self.sigma *math.sqrt(self.dt/2.))-math.exp(-self.sigma*math.sqrt(self.dt/2.))))**2
        self.qd = ((math.exp(self.sigma * math.sqrt(self.dt/2.))-math.exp((self.r-self.div)*self.dt/2.))/(math.exp(self.sigma*math.sqrt(self.dt/2.))-math.exp(-self.sigma*math.sqrt(self.dt/2.))))**2
        self.qm = 1-self.qu - self.qd
        
    def init_stock_price_tree(self):
        self.STs = [np.array([self.S0])]
        
        for i in range(self.N):
            prev_nodes = self.STs[-1]
            self.ST = np.concatenate((prev_nodes*self.u, [prev_nodes[-1]*self.m,prev_nodes[-1]*self.d]))
            self.STs.append(self.ST)
            
    def traverse_tree(self, payoffs):
        
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-2]*self.qu +payoffs[1:-1] * self.qm +payoffs[2:] *self.qd)*self.df
            
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs,i)
                
        return payoffs
#BINOMIAL LATTICE
class BinomialCRRLattice(BinomialCRROption):
    
    def setup_parameters(self):
        super(BinomialCRRLattice,self).setup_parameters()
        self.M = 2*self.N+1
        
    def init_stock_price_tree(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0 * self.u**self.N
            
        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d
            
    def init_payoffs_tree(self):
        odd_nodes = self.STs[::2]
        if self.is_call:
            return np.maximum(0,odd_nodes - self.K)
        else:
            return np.maximum(0,self.K-odd_nodes)
        
    def check_early_exercise(self,payoffs,node):
        self.STs = self.STs[1:-1]
        odd_STs = self.STs[::2]
        if self.is_call:
            return np.maximum(payoffs,odd_STs - self.K)
        else:
            return np.maximum(payoffs, self.K - odd_STs)
#TRINOMIAL LATTICE MODEL
class TrinomialLattice(TriTreeOption):
    
    def setup_parameters(self):
        super(TrinomialLattice,self).setup_parameters()
        self.M = 2*self.N+1
        
    def init_stock_price_tree(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0 * self.u** self.N
        
        for i in range(self.M)[1:]:
            
            self.STs[i] = self.STs[i-1]*self.d
    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.STs - self.K)
        else:
            return np.maximum(0,self.K - self.STs)
    
    def check_early_exercise(self,payoffs,node):
        self.STs = self.STs[1:-1]
        if self.is_call:
            return np.maximum(payoffs,self.STs - self.K)
        else:
            return np.maximum(payoffs,self.K-self.STs)
#FINITE DIFFERENCES BASE CLASS
class FiniteDifferences(object):
    
    def __init__(self,S0,K,r=0.05,T=1,sigma= 0,Smax = 1, M=1,N=1,is_put = False):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T 
        self.sigma = sigma
        self.Smax = Smax
        self.M,self.N = M,N
        self.is_call = not is_put
        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        self.grid = np.zeros(shape =(self.M+1,self.N+1))
        self.boundary_conds = np.linspace(0,Smax,self.M+1)
        
    @abstractmethod
    def setup_boundary_conditions(self):
        raise NotImplementedError('Implementation required!')
        
    @abstractmethod
    def setup_coefficients(self):
        raise NotImplementedError('Implementation required!')
        
    @abstractmethod
    def setup_traverse_grid(self):
        raise NotImplementedError('Implementation required!')
        
    @abstractmethod
    def interpolate(self):
        return np.interp(self.S0,self.boundary_conds,self.grid[:,0])
    
    
    @property
    def dS(self):
        return self.Smax/float(self.M)
    @property
    def dt(self):
        return self.T/float(self.N)
    
    def price(self):
        self.setup_boundary_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()
#EXPLICIT USING FINITE DIFFERENCES
#CAN RESULT IN NEGATIVE PRICING, FLIP TO IMPLICIT IN ORDER TO 
#CORRECT
class FDExplicit(FiniteDifferences):
    def setup_boundary_conditions(self):
        if self.is_call:
            self.grid[:,-1] = np.maximum(0,self.boundary_conds - self.K)
            self.grid[-1,:-1] = (self.Smax-self.K)*np.exp(-self.r*self.dt*(self.N-self.j_values))
        else:
            self.grid[:,-1] = np.maximum(0,self.K-self.boundary_conds)
            self.grid[0,:-1] = (self.K-self.Smax)*np.exp(-self.r*self.dt*(self.N-self.j_values))
            
    def setup_coefficients(self):
        self.a = 0.5*self.dt*((self.sigma**2)*(self.i_values**2)-self.r*self.i_values)
        self.b = 1 - self.dt * ((self.sigma**2)*(self.i_values**2)+self.r)
        self.c = 0.5 *self.dt*((self.sigma**2)*(self.i_values**2)+self.r*self.i_values)
    def traverse_grid(self):
        for j in reversed(self.j_values):
            for i in range(self.M)[2:]:
                self.grid[i,j] = self.a[i]*self.grid[i-1,j+1]+self.b[i]*self.grid[i,j+1]+self.c[i]*self.grid[i+1,j+1]
#IMPLICIT PRICING METHOD


class ImplicitModel(FDExplicit):
    
    def setup_coefficients(self):
        self.a = 0.5*(self.r*self.dt*self.i_values-(self.sigma**2)*self.dt*(self.i_values**2))
        self.b = 1.0 + (self.sigma**2)*self.dt*(self.i_values**2)+ self.r*self.dt
        self.c = -0.5 *(self.r*self.dt*self.i_values+(self.sigma**2)*self.dt*(self.i_values**2))
        self.coeffs = np.diag(self.a[2:self.M],-1)+np.diag(self.b[1:self.M])+np.diag(self.c[1:self.M-1],1)
        
    def traverse_grid(self):
        P,L,U = linalg.lu(self.coeffs)
        aux = np.zeros(self.M-1)
        for j in reversed(range(self.N)):
            aux[0] = np.dot(-self.a[1],self.grid[0,j])
            x1 = linalg.solve(L,self.grid[1:self.M,j+1]+aux)
            x2 = linalg.solve(U,x1)
            self.grid[1:self.M,j]=x2

class FDCnEu(FDExplicit):
    
    
    def setup_coefficients(self):
        self.alpha = 0.25*self.dt*((self.sigma**2)*(self.i_values**2)-self.r*self.i_values)
        self.beta = -self.dt*0.5*((self.sigma**2)*(self.i_values**2)+self.r)
        self.gamma = 0.25*self.dt*((self.sigma**2)*(self.i_values**2)+self.r*self.i_values)
        self.M1 = -np.diag(self.alpha[2:self.M],-1)+np.diag(1-self.beta[1:self.M])-np.diag(self.gamma[1:self.M-1],1)
        self.M2 = np.diag(self.alpha[2:self.M],-1)+np.diag(1+self.beta[1:self.M])+np.diag(self.gamma[1:self.M-1],1)
        """
        Solve using Linear systems of equations
        """
    def traverse_grid(self):
        
        P,L,U = linalg.lu(self.M1)
        
        for j in reversed(range(self.N)):
            x1 = linalg.solve(L,np.dot(self.M2,self.grid[1:self.M,j+1]))
            x2 = linalg.solve(U,x1)
            self.grid[1:self.M,j]=x2

## Cannot get class to function properly, please advise if came across, pricing down and out American options
class FDCnAm(FDCnEu):
    def __init__(self,S0,K,r=0.05,T=1,Smax = 1,M=1, N = 1, omega = 1, tol =0,is_put = False):
        super(FDCnAm,self).__init__(S0,K,r=r,T=T,sigma=sigma,Smax=Smax,M=M,N=N,is_put = is_put)
        self.omega = omega
        self.tol = tol
        self.i_values = np.arange[self.M+1]
        self.j_value = np.arange[self.N+1]
        
    def setup_boundary_conditions(self):
        if self.is_call:
            self.payoff = np.maxium(0,self.boundary_conds[1:self.M]-self.K)
        else:
            self.payoffs = np.maximum(0,self.K-self.boundary_conds[1:self.M])
            
            self.past_values = self.payoffs
            self.boundary_values = self.K * np.exp(-self.r*self.dt*(self.N-self.j_values))
            
    def traverse_grid(self):
        """
        Solving using linear systems of equations
        """
        aux = np.zeros(self.M-1)
        new_values = np.zeros(self.M-1)
        
        for j in reversed(range(self.N)):
            aux[0] = self.alpha[1]*(self.boundary_values[j]+self.boundary_values[j+1])
            rhs = np.dot(self.M2,self.past_values)+aux
            old_values = np.copy(self.past_values)
            error = sys.float_info.max
            
            while self.tol < error:
                new_values[0]=self.calculate_payoff_start_boundary(rhs,old_values)
                
                for k in range(self.M-2)[1:]:
                    new_values[k]=self.calculate_payoff(k,rhs,old_values,new_values)
                    new_values[-1]=self.calculate_payoff_end_boundary(rhs,old_values,new_values)
                    
                    error = np.linalg.norm(new_values - old_values)
                    old_values = np.copy(new_values)
                    
                self.past_values = np.copy(new_values)
                
            self.values = np.concatenate(([self.boundary_values[0]],new_values,[0]))
            
    def calculate_payoff_start_boundary(self):
        payoff = old_values[0]+self.omega/(1-self.beta[1])*(rhs[0]-(1-self.beta[1])*old_values[0]+self.gamma[1]*old_values[1])
        return max(self.payoffs[0],payoff)
    
    def calculate_payoff_end_boundary(self,rhs,old_values,new_values):
        
        payoff = old_values[-1]+self.omega/(1-self.beta[-2])*(rhs[-1]+self.alpha[-2]*new_values[-2]-(1-self.beta[-2])*old_values[-1])
        return max(self.payoffs[-1],payoff)
    def calculate_payoff(self,k,rhs,old_values,new_values):
        payoff = old_values[k]+self.omega/(1-self.beta[k+1])*(rhs[k]+self.alpha[k+1]*new_values[k-1]-(1-self.beta[k+1])*old_values[k]+self.gamma[k+1]*old_values[k+1])
        return max(self.payoff[k],payoff)
    
    def interpolate(self):
        #Linear interpolation on final values as ID array
        
        return np.interp(self.S0,self.boundary_conds,self.values)

  
#This one as well
class ImpliedVolModel(object):
    
    def __init__(self,S0,r=0.05,T=1,div =0,N=1,is_put=False):
        self.S0 = S0
        self.r = r
        self.T = T
        self.div = div
        self.N = N
        self.is_put = is_put
        
    def option_valuation(self, K, sigma):
        
        lr_option = BinomialLROption(self.S0,K,r=self.r,T = self.T,N=self.N,sigma=sigma,div = self.div,is_put =self.is_put)
        return lr_option.price()
   
    
    def get_implied_volatilities(self,Ks,opt_prices):
        impvols = []
        
        for i in range(len(strikes)):
            f = lambda sigma: self.option_valuation(Ks[i],sigma)-opt_prices[i]
            impv = bisection(f,0.01,0.99,0.0001,100)[0]
            impvols.append(impv)
        return impvols
    
    def bisection(f,a,b,tol=0.1,maxiter=10):
        c = (a+b)*0.5
        n = 1
        while n<= maxiter:
            if f(c) == 0 or abs(a-b)*0.5 < tol:
                return c, n

            n+=1
            if f(c)<0:
                a = c
            else:
                b = c 
        return c, n

       
        

