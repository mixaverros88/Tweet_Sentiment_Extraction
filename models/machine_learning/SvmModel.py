from sklearn import svm
import pickle
import collections
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters
import numpy as np


class SvmModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('Support Vector Machine SVM')
        # Grid Search
        # clf = svm.SVC(kernel='linear', probability=True)
        # scoring = ['f1']
        # param_grid = {'C': np.linspace(start=1000, stop=10000, num=4, endpoint=True)}
        # grid = GridSearchCV(clf, param_grid=param_grid, scoring=scoring, cv=3,
        #                     refit='f1', verbose=42, n_jobs=-1, pre_dispatch=3)
        # grid.fit(self.X_train, self.y_train)
        # # SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='linear')
        # get_models_best_parameters(grid, 'Support Vector Machine SVM')

        model = svm.SVC(kernel='linear')  # Linear Kernel
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        pred = model.predict(self.X_test)
        y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        p = Point(prediction=pred, score=y_score)
        return p
