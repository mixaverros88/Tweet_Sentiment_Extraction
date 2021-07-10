from sklearn import svm
import pickle
import collections
from helper.retrieve.serializedModels import bag_of_words_svm_over_sampling


def run_on_test_data_set(x, y):
    model = bag_of_words_svm_over_sampling()  # Retrieve Model
    return model.predict(x)


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
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.x_test)
        y_score = model.fit(self.x_train, self.y_train).decision_function(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=y_score)
