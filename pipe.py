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
from ordinal import *
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split as tts
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn import metrics
from sklearn.pipeline import Pipeline

def run_models(x_train,y_train,x_test,y_test):

    models = [('clf', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier()),
            ('knn', KNeighborsClassifier()),
            ('svm', SVC()),
            ('lg', LogisticRegression())]


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
                        'svm__decision_function_shape': ['ovo', 'ovr']}],
                    [{ "lg" : [LogisticRegression()],
                        "lg__penalty": ['l1', 'l2'],
                        "lg__C": np.logspace(0, 4, 10)}]]


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