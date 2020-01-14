from data_processing import *
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np


class PolynomialInteractions():
    
    def __init__(self, csv, target):
        self.csv = csv
        self.target = target
        
    def get_data(self, csv,target):
        
        d = DataClean(csv) # call class to import and process data 
        df_mm, num_scale = d.read_scale(csv)
        return df_mm, num_scale
        
#     def split_data(self,csv,target):
       
#         df_mm, num_scale = self.get_data(csv)
#         X_train, X_test, y_train, y_test = d.split(csv)
#         X_train, X_test, y_train, y_test = d.index(csv) #split and reset index
#         return X_train, X_test, y_train, y_test
        
    def poly(self, csv, target):
        
        df_mm, num_scale = self.get_data(csv,target)
        X_train = num_scale
        y_train = df_mm[target]
        
        # running polynomial model to get interactions
        p = PolynomialFeatures(degree=2).fit(X_train)
        X = pd.DataFrame(p.transform(X_train), columns=p.get_feature_names(X_train.columns))
        return X
    
    def get_interactions(self, csv,target):
        X = self.poly(csv,target)
        df_mm, num_scale = self.get_data(csv,target)
        y_train = df_mm[target]
        
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
        
    def new_features(self,csv,target):
        
        df_mm, num_scale = self.get_data(csv,target)
        X = self.poly(csv,target)
        pvalues = self.get_interactions(csv,target)
        
        #filter for significant coefficients
        new_var = pvalues.loc[(pvalues['P>|t|'] <= 0.05)]
        new_var = new_var.drop('1', axis=0)
        #creating new variables
        interactions = list(new_var.index)
        original = list(num_scale.columns)
        names_list = list(set(interactions) - set(original))
        names_list = [x for x in names_list if '^' not in x]  
        names = []
        for name in names_list:
            one = name.split(' ')
            names.append(one)
        return names, names_list
    
    
    def add_to_df(self,csv,target):
        
        df_mm, num_scale = self.get_data(csv, target)
        names, names_list = self.new_features(csv,target)
        
        one = []
        two = []
        for n in names:
            one.append(n[0])
            two.append(n[1])

        for a, b, c in zip(one,two, names_list):
            df_mm[c] = df_mm[a] * df_mm[b]
        
        df_mm.to_csv('interactions.csv')
        return df_mm