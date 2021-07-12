from sklearn.neural_network import MLPClassifier
import pickle
import collections
from utils.serializedModels import bag_of_words_multi_layer_perceptron_classifier_over_sampling
from definitions import ROOT_DIR


def run_on_test_data_set(x, y):
    model = bag_of_words_multi_layer_perceptron_classifier_over_sampling()
    return model.predict(x)


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
        # mlp_classifier_model_tuning(self.x_train, self.y_train)
        model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(5, 5, 5), learning_rate='adaptive',
                              max_iter=1000)
        model.fit(self.x_train, self.y_train)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
