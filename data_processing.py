import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np

 
class DataClean():


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