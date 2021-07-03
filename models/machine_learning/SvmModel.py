from sklearn import svm
import pickle
import collections
from helper.retrieve.serializedModels import bag_of_words_svm_over_sampling
from sklearn.model_selection import GridSearchCV
from helper.helper_functions.functions import get_models_best_parameters
import numpy as np


def run_on_test_data_set(x, y):
    model = bag_of_words_svm_over_sampling()  # Retrieve Model
    return model.predict(x)


class SvmModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Support Vector Machine SVM')
        # Grid Search
        # clf = svm.SVC(kernel='linear', probability=True)
        # scoring = ['f1']
        # param_grid = {'C': np.linspace(start=1000, stop=10000, num=4, endpoint=True)}
        # grid = GridSearchCV(clf, param_grid=param_grid, scoring=scoring, cv=3,
        #                     refit='f1', verbose=42, n_jobs=-1, pre_dispatch=3)
        # grid.fit(self.X_train, self.y_train)
        # get_models_best_parameters(grid, 'Support Vector Machine SVM')

        model = svm.SVC(kernel=self.param_space.get('kernel'))  # Linear Kernel
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.X_test)
        y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=y_score)
