from sklearn.tree import DecisionTreeClassifier
import collections
from utils.functions import compute_elapsed_time, save_models
from utils.model_tuning import decision_tree_model_tuning
import time


class DecisionTreeModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Decision Tree')
        start_time = time.time()
        # decision_tree_model_tuning(self.X_train, self.y_train)
        model = DecisionTreeClassifier(
            max_depth=self.param_space.get('max_depth'),
            max_leaf_nodes=self.param_space.get('max_leaf_nodes'),
            min_samples_split=self.param_space.get('min_samples_split')
        )
        model.fit(self.x_train, self.y_train)
        save_models(model, self.model_name)
        predictions = model.predict(self.x_test)
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
