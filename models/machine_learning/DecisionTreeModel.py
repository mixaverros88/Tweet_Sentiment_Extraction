from sklearn.tree import DecisionTreeClassifier
import pickle
import collections
from helper.retrieve.serializedModels import bag_of_words_decision_tree_over_sampling
from sklearn.model_selection import GridSearchCV
from helper.helper_functions.functions import get_models_best_parameters
import numpy as np


def run_on_test_data_set(x, y):
    model = bag_of_words_decision_tree_over_sampling()  # Retrieve Model
    return model.predict(x)

class DecisionTreeModel:

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
        # param_grid = {'max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3],
        #               'max_depth': np.arange(3, 6)}  # prone to overfitting
        # model_gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        # get_models_best_parameters(model_gs, 'Support Vector Machine SVM')

        model = DecisionTreeClassifier(
            max_depth=self.param_space.get('max_depth'),
            max_leaf_nodes=self.param_space.get('max_leaf_nodes'),
            min_samples_split=self.param_space.get('min_samples_split')
        )
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
