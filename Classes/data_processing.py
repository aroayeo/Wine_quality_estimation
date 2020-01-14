import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np

 
class DataClean():
    
    def __init__(self, csv):
        self.csv = csv
 

    def read_scale(self, csv):
        
        data = pd.read_csv(csv, index_col=0)
        scale = StandardScaler()
        num = data.select_dtypes(include=['float64'])
        cat = data.drop(columns = num.columns)
        num_scale = pd.DataFrame(scale.fit_transform(num), columns = num.columns, index = num.index)
        df_mm = pd.concat([num_scale, cat], axis = 1)

        return df_mm, num_scale


    def split(self,csv,y):

        df_mm, num_scale = self.read_scale(csv)

        X = num_scale
        y = df_mm[y]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        dataframes = [X_train, X_test, y_train, y_test]
        
        for df in [X_train,X_test,y_train,y_test]:
            df.reset_index(inplace=True, drop=True)
            
        return X_train, X_test, y_train, y_test