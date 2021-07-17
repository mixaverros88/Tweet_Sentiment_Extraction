import collections
from sklearn.linear_model import LogisticRegression
import time

from utils.functions import compute_elapsed_time, save_models


class LogisticRegressionModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Logistic Regression')
        start_time = time.time()
        # logistic_regression_model_tuning(self.x_train, self.y_train)
        model = LogisticRegression(
            C=self.param_space.get('C'),
            penalty=self.param_space.get('penalty'),
            max_iter=self.param_space.get('max_iter'),
            solver=self.param_space.get('solver')
        )
        model.fit(self.x_train, self.y_train)
        save_models(model, self.model_name)
        score = model.fit(self.x_train, self.y_train).decision_function(self.x_test)
        predictions = model.predict(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return Point(prediction=predictions, score=score)
