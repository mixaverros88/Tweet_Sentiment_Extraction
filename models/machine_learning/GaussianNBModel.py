from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters
import numpy as np


class GaussianNBModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_name):
        self.X_train = X_train.todense()
        self.X_test = X_test.todense()
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('Gaussian')
        # Grid Search
        # param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        # model_gs = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=5, verbose=1, scoring='accuracy')
        # model_gs.fit(self.X_train, self.y_train)
        # get_models_best_parameters(model_gs, 'gaussian')  # GaussianNB(var_smoothing=0.01)

        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        return model.predict(self.X_test)
