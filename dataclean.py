import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class CleanData():
    
    """preprocessing data for ML modeling: merge, scale and split data"""
    
    def merge_csv(self,csv1,csv2):
        w = pd.read_csv(csv1, delimiter=';')
        r = pd.read_csv(csv2)
        w['type'] = 0 #white wines
        r['type'] = 1 #red wines
        data = pd.concat([r,w])
        data.columns = data.columns.str.replace(' ','')
        data.to_csv('data.csv')
        print(('Class 0 is {} values. Class 1 is {} values').format(csv1,csv2))
        return data
    
    def scale(self, df):
        
        scale = MinMaxScaler()
        num = df.select_dtypes(include=['float64'])
        cat = df.drop(columns = num.columns)
        num_scale = pd.DataFrame(scale.fit_transform(num), columns = num.columns, index = num.index)
        df_mm = pd.concat([num_scale, cat], axis = 1)

        return df_mm


    def split(self, df, target):

        X = df.drop(target,axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        return X_train, X_test, y_train, y_test
    
    class DefineTarget():
    
    # target is the name of your classes target variable 
            # n_c is the number of classes we want to define
            # thres are the limits to define the classes
            # value is the name of your original variable which you want to categorise in classes
            
    
        def __init__(self, data: pd.DataFrame):
            self.data = data
            
        def get_classes(self, target, n_c, thres: list or int, value):
        # Add target variable for classes
            if n_c==2:
                    self.data[target] = self.data[value].apply(lambda x: 0 if x<=thres[0] else 1).astype('int')
            elif n_c==3:
                    self.data[target] = self.data[value].apply(lambda x: 0 if x<=thres[0] else (1 if x <=thres[1] else 2)).astype('int')
            elif n_c==4:
                    self.data[target] = self.data[value].apply(lambda x: 0 if x<=thres[0] else (1 if x <=thres[1] else(2 if x <=thres[2]  else (3)))).astype('int')
            return self.data
