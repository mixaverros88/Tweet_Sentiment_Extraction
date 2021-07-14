from sklearn import svm
import pickle
import collections
from definitions import ROOT_DIR


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
        # svm_model_tuning(self.x_train, self.y_train)
        model = svm.SVC(kernel=self.param_space.get('kernel'))  # Linear Kernel
        model.fit(self.x_train, self.y_train)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.x_test)
        y_score = model.fit(self.x_train, self.y_train).decision_function(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=y_score)
