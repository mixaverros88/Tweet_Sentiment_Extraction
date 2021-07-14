import pickle
import collections
from sklearn.linear_model import LogisticRegression
from utils.serializedModels import bag_of_words_logistic_regression_over_sampling, \
    bag_of_words_logistic_regression_under_sampling
from definitions import ROOT_DIR


def run_on_test_data_set_over_sampling(x):
    model = bag_of_words_logistic_regression_over_sampling()
    return model.predict(x)


def run_on_test_data_set_under_sampling(x):
    model = bag_of_words_logistic_regression_under_sampling()
    return model.predict(x)


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
        # logistic_regression_model_tuning(self.x_train, self.y_train)
        model = LogisticRegression(
            C=self.param_space.get('C'),
            penalty=self.param_space.get('penalty'),
            max_iter=self.param_space.get('max_iter'),
            solver=self.param_space.get('solver')
        )
        model.fit(self.x_train, self.y_train)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        y_score = model.fit(self.x_train, self.y_train).decision_function(self.x_test)
        predictions = model.predict(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=y_score)
