import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CleanCsv():
    
    def __init__(self, csv1, csv2):
        self.csv1 = csv1
        self.csv2 = csv2
     
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