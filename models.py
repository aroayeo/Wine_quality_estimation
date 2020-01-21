from sklearn.base import clone
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, classification_report, accuracy_score, precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns
from dataclean import *




class RunModels():
    
    def run_models(self, x_train,y_train,x_test,y_test):

        models = [('clf', DecisionTreeClassifier()),
                ('rf', RandomForestClassifier()),
                ('knn', KNeighborsClassifier()),
                ('svm', SVC())]
    #             ('lg', LogisticRegression())]


        spaces = [[{"clf": [DecisionTreeClassifier()],
                            "clf__criterion": ['gini', 'entropy'],
                            "clf__presort": [True, False]}],
                        [{"rf": [RandomForestClassifier()],
                            "rf__n_estimators": [10, 100, 1000],
                            "rf__max_features": [1, 2, 3],
                            'rf__max_features': ['auto', 'sqrt', 'log2']}],
                        [{ "knn" : [KNeighborsClassifier()],
                            'knn__n_neighbors': [1,2,3,4,5,6,7,8,9],
                            'knn__p': [1,2]}],
                        [{"svm" : [SVC()],
                            'svm__kernel': ['linear','rbf', 'sigmoid'],
                            'svm__decision_function_shape': ['ovo', 'ovr']}]]
    #                     [{ "lg" : [LogisticRegression()],
    #                         "lg__penalty": ['l1', 'l2'],
    #                         "lg__C": np.logspace(0, 4, 10)}]]


        g_p = []
        b_e = []
        y_pred_p = []

        for m, s in zip(models,spaces):
            pipeline = Pipeline([m])
            pipeline.fit(x_train, y_train) 
            y_hat = pipeline.predict(x_test)
            print(m[0],classification_report(y_test, y_hat))
            classifier = GridSearchCV(pipeline, s, cv=5, verbose=0).fit(x_train, y_train)
            y_hat = classifier.predict(x_test)
            y_pred_p.append(y_hat)
            bp = classifier.best_estimator_
            gp = classifier.get_params()
            b_e.append(bp)
            g_p.append(gp)


        return g_p, b_e, y_pred_p
    
    class OrdinalClassifier():
        
        def __init__(self, clf):
            self.clf = clf
            self.clfs = {}

        def fit(self, X, y):
            self.unique_class = np.sort(np.unique(y))
            if self.unique_class.shape[0] > 2:
                for i in range(self.unique_class.shape[0]-1):
                    # for each k - 1 ordinal value we fit a binary classification problem
                    binary_y = (y > self.unique_class[i]).astype(np.uint8)
                    clf = clone(self.clf)
                    clf.fit(X, binary_y)
                    self.clfs[i] = clf

        def predict_proba(self, X):
            clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
            predicted = []
            for i,y in enumerate(self.unique_class):
                if i == 0:
                    # V1 = 1 - Pr(y > V1)
                    predicted.append(1 - clfs_predict[y][:,1])
                elif y in clfs_predict:
                    # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                     predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
                else:
                    # Vk = Pr(y > Vk-1)
                    predicted.append(clfs_predict[y-1][:,1])
            return np.vstack(predicted).T

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)


    class PcaAnalysis():


        def pca_features(self, df, target, n_com):

            dc = CleanData() # call class to import and process data 
            df_mm = dc.scale(df)
            num_scale = dc.scale(df).select_dtypes(include=['float64'])
            pca = PCA(n_components=0.99, whiten=True, svd_solver='full') # Conduct PCA 
            features_pca = pca.fit_transform(num_scale)
            components =['c' + str(n) for n in range(0,features_pca.shape[1])]
            features_df = pd.DataFrame(features_pca, columns=components)
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

            return features_df


    
    