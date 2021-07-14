from sklearn.tree import DecisionTreeClassifier
import pickle
import collections
from definitions import ROOT_DIR


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
        # decision_tree_model_tuning(self.x_train, self.y_train)
        model = DecisionTreeClassifier(
            max_depth=self.param_space.get('max_depth'),
            max_leaf_nodes=self.param_space.get('max_leaf_nodes'),
            min_samples_split=self.param_space.get('min_samples_split')
        )
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
