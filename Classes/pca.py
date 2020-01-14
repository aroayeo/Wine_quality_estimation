import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_processing import *

class PcaAnalysis():
    
    def __init__(self, csv, target):
        self.csv = csv
        self.target = target

        
    def pca_features(self, csv, target):
        
        d = DataClean(csv) # call class to import and process data 
        df_mm, num_scale = d.read_scale(csv)
        
        pca = PCA(n_components=0.99, whiten=True, svd_solver='full') # Conduct PCA 
        features_pca = pca.fit_transform(num_scale)
        components =['c' + str(n) for n in range(0,features_pca.shape[1])]
        features_df = pd.DataFrame(features_pca, columns=components)
        
        
        return pca, features_df
    
    def get_components(self, csv, target, n_com):
        
        pca, features_pca = self.pca_features(csv, target)
        d = DataClean(csv) # call class to import and process data
        df_mm, num_scale = d.read_scale(csv)
        
        vects_pca = pca.components_[:n_com]
        components =['c' + str(n_com) for n in range(0,n_com)]
        df = []
        for component, i in zip(components, range(0,n_com)):
            component = pd.Series(vects_pca[i], index=num_scale.columns)
            component.sort_values(ascending=False)
            df.append(component)
        df = pd.DataFrame(df).T
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        print("Original number of features:", num_scale.shape[1]) 
        print("Reduced number of features:", features_pca.shape[1]) 
        plt.show()
        return df
        

    
    