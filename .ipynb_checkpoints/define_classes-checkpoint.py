import pandas as pd
import numpy as np


class DefineTarget():
    """
    create target multi classes
    """
    
    def __init__(self, data: pd.DataFrame):
        # target is the name of your classes target variable 
        # n_c is the number of classes we want to define
        # thres are the limits to define the classes
        # value is the name of your original variable which you want to categorise in classes
        
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

          