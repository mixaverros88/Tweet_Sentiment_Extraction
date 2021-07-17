from sklearn import svm
import collections
from utils.model_tuning import svm_model_tuning
import time

from utils.functions import compute_elapsed_time, save_models


class SvmModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Support Vector Machine SVM')
        start_time = time.time()
        # svm_model_tuning(self.x_train, self.y_train)
        model = svm.SVC(kernel=self.param_space.get('kernel'))  # Linear Kernel
        model.fit(self.x_train, self.y_train)
        save_models(model, self.model_name)
        predictions = model.predict(self.x_test)
        score = model.fit(self.x_train, self.y_train).decision_function(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return Point(prediction=predictions, score=score)
