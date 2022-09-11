#import classes necessary
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression 
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, mean_squared_error, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor



#Linear Regression for single asset prediction model
class LinearRegressionModel(object):
    
    def __init__(self):
        self.df_result = pd.DataFrame(columns = ['Actual','Predicted'])
        
    def get_model(self):
        return LinearRegression(fit_intercept=False)
    
    def get_prices_since(self,df,date_since, lookback):
        index = df.index.get_loc(date_since)
        return df.iloc[index-lookback:index]
    def learn(self,df,ys,start_date,end_date,lookback_period = 20):
        model = self.get_model()
        
        for date in df[start_date:end_date].index:
            #fit the model
            x = self.get_prices_since(df,date,lookback_period)
            y = self.get_prices_since(ys,date,lookback_period)
            model.fit(x,y.ravel())
            
            #predict the current period
            x_current = df.loc[date].values
            [y_pred] = model.predict([x_current])
            
            #store predicitions
            
            new_index = pd.to_datetime(date,format = '%Y -%m -%d')
            y_actual = ys.loc[date]
            self.df_result.loc[new_index] = [y_actual,y_pred]
##Different Regression Classes made availabe in a single sheet along with metrics
class RidgeRegression(LinearRegressionModel):
    def get_model(self):
        return Ridge(Alpha = 0.5)

class LassoRegression(LinearRegressionModel):
    def get_model(self):
        return Lasso(alpha = 0.5)

class ElasticNetRegression(LinearRegressionModel):
    def get_model(self):
        return ElasticNet(alpha = 0.5)

class BaggingRegressorModel(LinearRegressionModel):
    def get_model(self):
        return BaggingRegressor(n_estimators=20,random_state = 0)
class GradientBoostModel(LinearRegressionModel):
    def get_model(self):
        return GradientBoostingRegressor(n_estimators=20,random_state = 0)
class RandomForestModel(LinearRegressionModel):
    def get_model(self):
        return RandomForestRegressor(n_estimators=20,random_state = 0)

def print_regression_metrics(df_result):
    actual = list(df_result['Actual'])
    predicted = list(df_result['Predicted'])
    print('mean_absolute_error:', mean_absolute_error(actual,predicted))
    print('mean_squared_error:', mean_squared_error(actual,predicted))
    print('explained_variance_score:', explained_variance_score(actual,predicted))
    print('r2_score:', r2_score(actual,predicted))
    
    
    
#Logistic Regression Models



class LogisticRegressionModel(LinearRegressionModel):
    def get_model(self):
        return LogisticRegression(solver='lbfgs')