from sklearn.neural_network import MLPClassifier
import collections
import time

from utils.functions import compute_elapsed_time, save_models
from utils.model_tuning import mlp_classifier_model_tuning


class MLPClassifierModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space, *word2vec):
        if word2vec is None:
            self.x_train = x_train.todense()
            self.x_test = x_test.todense()
        else:
            self.x_train = x_train
            self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('MLPClassifier')
        start_time = time.time()
        # mlp_classifier_model_tuning(self.x_train, self.y_train)
        model = MLPClassifier(
            activation=self.param_space.get('activation'),
            alpha=self.param_space.get('alpha'),
            hidden_layer_sizes=self.param_space.get('hidden_layer_sizes'),
            learning_rate=self.param_space.get('learning_rate'),
            max_iter=self.param_space.get('max_iter'))
        model.fit(self.x_train, self.y_train)
        save_models(model, self.model_name)
        predictions = model.predict(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return Point(prediction=predictions, score=None)
