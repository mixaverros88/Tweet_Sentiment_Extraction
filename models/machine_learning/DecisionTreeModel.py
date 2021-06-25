from sklearn.tree import DecisionTreeClassifier
import pickle
import collections
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters
import numpy as np


class DecisionTreeModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('Support Vector Machine SVM')
        # Grid Search
        # param_grid = {'max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3],
        #               'max_depth': np.arange(3, 6)}  # prone to overfitting
        # model_gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        # get_models_best_parameters(model_gs, 'Support Vector Machine SVM')

        model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=18, min_samples_split=3)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        pred = model.predict(self.X_test)
        # y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        y_score = None
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        p = Point(prediction=pred, score=y_score)
        return p
