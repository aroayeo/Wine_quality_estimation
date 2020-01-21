from data_processing import *
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np


class PolynomialInteractions():
    
    def __init__(self, df, target):
        self.df = df
        self.target = target
        
    def poly(self, df, target):
        
        X_train = self.df.drop(target, axis=1)
        y_train = self.df[target]
        
        # running polynomial model to get interactions
        p = PolynomialFeatures(degree=2).fit(X_train)
        X = pd.DataFrame(p.transform(X_train), columns=p.get_feature_names(X_train.columns))
        return X
    
    def get_interactions(self, df,target):
        X = self.poly(df,target)
        y_train = df[target]
        
        #passing polynomial features through OLS
        X = sm.add_constant(X) 
        model_sm = sm.OLS(np.array(y_train),np.array(X))
        results = model_sm.fit()
        results_summary = results.summary()
        results_as_html = results_summary.tables[1].as_html()
        #getting results into df
        pvalues = pd.read_html(results_as_html, header=0, index_col=0)[0]
        pvalues.index = X.columns
        return pvalues
        
    def new_features(self,df,target):
        
        pvalues = self.get_interactions(df,target)
        
        #filter for significant coefficients
        new_var = pvalues.loc[(pvalues['P>|t|'] <= 0.05)]
#         new_var = new_var.drop('1', axis=0)
        #creating new variables
        interactions = list(new_var.index)
        original = list(self.df.columns)
        names_list = list(set(interactions) - set(original))
        names_list = [x for x in names_list if '^' not in x]  
        names = []
        for name in names_list:
            one = name.split(' ')
            names.append(one)
        return names, names_list
    
    
    def add_to_df(self,df,target,name_csv):
        
        df_mm = df
        names, names_list = self.new_features(df,target)
        
        one = []
        two = []
        for n in names:
            one.append(n[0])
            two.append(n[1])

        for a, b, c in zip(one,two, names_list):
            df_mm[c] = df_mm[a] * df_mm[b]
        
        df_mm.to_csv('{}.csv'.format(name_csv))
        